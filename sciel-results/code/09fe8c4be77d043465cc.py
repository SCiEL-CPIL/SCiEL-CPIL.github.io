"""Matrix multiplication 4x4*4x8 tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 8]
DEADLINE = 145.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - time.time()

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the multiplication tensor."""
    rank = len(U[0])
    # Build tensor T[i,j,k] = 1 for all i,j,k
    # Check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # And for "wrong" indices, should be 0.
    # The multiplication tensor has T[i,j,k]=1 iff the output index is (i,k) given input (i,j) and (j,k).
    # Actually the tensor is: T_{(i,j),(j,k),(k,i)} = 1 in the standard convention.
    # With our indexing: U has rows (i*m+j), V has rows (j*p+k), W has rows (k*n+i).
    # The tensor element T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i should be 1.
    # All other combinations should be 0.
    
    # Efficient check: form the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Check: sum_c U[:,c] * V[:,c] * W[:,c] == T
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct tensor
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    if not np.allclose(recon, T, atol=tol):
        return False
    return True

def _validate_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster validation using random sampling + full check."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Full reconstruction
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    return np.allclose(recon, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def _try_rank(n, m, p, rank, time_budget):
    """Try to find a valid decomposition of given rank using random search + local refinement.
    Returns (U, V, W) if found, else None."""
    start_time = time.time()
    rng = np.random.RandomState(rank * 1000 + 42)
    
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    # Random initialization + gradient-free local search
    max_iters = 5000
    lr = 0.1
    
    for attempt in range(20):
        if time.time() - start_time > time_budget:
            break
        
        U = rng.randn(nm, rank) * 0.5
        V = rng.randn(mp, rank) * 0.5
        W = rng.randn(pn, rank) * 0.5
        
        for it in range(max_iters):
            if time.time() - start_time > time_budget:
                break
            
            # Compute reconstruction error
            recon = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.sum((recon - T)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-12:
                break
            
            # Compute gradients
            # d/dU = (recon - T) contracted with V and W
            diff = recon - T
            grad_U = np.einsum('abd,bc,dc->ad', diff, V, W)
            grad_V = np.einsum('abd,ac,dc->bd', diff, U, W)
            grad_W = np.einsum('abd,ac,bc->cd', diff, U, V)
            
            # Adaptive learning rate
            norm_U = np.linalg.norm(grad_U) + 1e-10
            norm_V = np.linalg.norm(grad_V) + 1e-10
            norm_W = np.linalg.norm(grad_W) + 1e-10
            
            U -= lr * grad_U / norm_U
            V -= lr * grad_V / norm_V
            W -= lr * grad_W / norm_W
        
        if best_err < 1e-12:
            break
    
    if best_err < 1e-9:
        # Convert to list of lists
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return U_list, V_list, W_list
    
    return None

def _strassen_style(n, m, p):
    """Try a structured Strassen-like decomposition.
    For 4x4 * 4x8, we can use block Strassen on 2x2 blocks.
    This gives rank 7 * (number of block multiplications needed).
    4x4 * 4x8 = 2x2 blocks of 2x2 * 2x4.
    Strassen on 2x2 * 2x4: we need to handle rectangular blocks.
    Actually let's think more carefully.
    
    A is 4x4, B is 4x8. Split into 2x2 blocks:
    A = [[A11, A12], [A21, A22]] each 2x2
    B = [[B11, B12, B13, B14], [B21, B22, B23, B24]] each 2x4
    
    C = A*B = [[C11, C12, C13, C14], [C21, C22, C23, C24]] each 2x4
    
    Standard Strassen for square 2x2*2x2 uses 7 multiplications.
    For 2x2 * 2x4, we can still use a Strassen-like scheme.
    
    Let me use the approach: treat B as two 2x2 matrices side by side? No, B is 2x4.
    
    Alternative: use 7 multiplications of 2x2 * 2x2 by splitting B into two 2x2 halves.
    B = [B_left, B_right] where each is 2x2.
    Then A*B = [A*B_left, A*B_right].
    Each A*B_left is 4x2, computed via Strassen (2x2 * 2x2) = 7 mults.
    Total: 14 multiplications of 2x2*2x2.
    Each 2x2*2x2 Strassen has rank 7.
    But we need to compose these into a single tensor decomposition.
    
    Actually, the rank of the composed tensor is at most 14*7 = 98? No, that's not right either.
    The Strassen decomposition for 2x2*2x2 has rank 7, meaning we need 7 scalar multiplications.
    For 4x4*4x8, if we do it as two independent 4x4*4x2 problems (splitting B),
    each 4x4*4x2 can be done via Strassen on 2x2 blocks.
    
    Let me just try rank values and see what works.
    """
    return None

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Known lower bounds: for 4x4*4x4, rank is at most 49 (Lader) or 49.
    # For 4x4*4x8, the rank is at most 4*4*8 = 128 (schoolbook).
    # Let's try ranks from a reasonable lower bound upward.
    # Lower bound for n*m*p matrix mult tensor rank >= max(n*m, m*p, p*n) = max(16, 32, 32) = 32.
    # Also >= n*p = 32 (output size).
    # Try from 32 upward, but focus on promising ranges.
    
    # Try known structured decompositions first
    # For 4x4*4x8: we can view this as 4x4 * 4x8.
    # One approach: use the fact that 4x4*4x4 has rank <= 49 (best known for square).
    # 4x4*4x8 = two copies of 4x4*4x2, or we can use other decompositions.
    
    # Let's try random search for various ranks
    # Be strategic: try ranks that are multiples or near known results.
    
    candidate_ranks = []
    # Start from lower bound
    lb = max(n*m, m*p, p*n, n*p)  # 32
    for r in range(lb, best_rank + 1):
        candidate_ranks.append(r)
    
    # Try in order, but spend more time on lower ranks
    for rank in candidate_ranks:
        if time.time() - start > DEADLINE - 10:
            break
        
        # Time budget for this rank: more for lower ranks
        time_left = DEADLINE - 10 - (time.time() - start)
        if time_left < 2:
            break
        
        # Spend proportional time
        t_budget = min(time_left * 0.3, 15.0)  # at most 15s per rank
        
        result = _try_rank(n, m, p, rank, t_budget)
        if result is not None:
            U_r, V_r, W_r = result
            # Final validation
            if _validate_fast(U_r, V_r, W_r, n, m, p, tol=1e-9):
                best_rank = rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                break  # Found minimal rank, stop
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
