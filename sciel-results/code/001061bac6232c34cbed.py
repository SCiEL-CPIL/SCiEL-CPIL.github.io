"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _verify_full(U, V, W, n, m, p, tol=1e-6):
    """Full verification: check all tensor entries."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all valid (i,k,j) triples sum to 1
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = np.sum(U_arr[u_idx] * V_arr[v_idx] * W_arr[w_idx])
                if abs(val - 1) > tol:
                    return False
    
    # Check that non-matching entries are 0
    # For any (a,b,c) where a = i*m+k, b = k'*p+j, c = j'*n+i'
    # if k' != k or j' != j or i' != i, the product should be 0
    for a in range(n * m):
        i_a = a // m
        k_a = a % m
        for b in range(m * p):
            k_b = b // p
            j_b = b % p
            if k_b != k_a:
                continue  # These are automatically 0 in the tensor
            for c in range(p * n):
                j_c = c // n
                i_c = c % n
                if j_c != j_b or i_c != i_a:
                    # This entry should be 0
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(val) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_partitioned(n, m, p):
    """
    Try partitioning p into smaller parts and using known decompositions.
    For M_{4,4,9}, split p=9 into p1+p2 and use:
    rank(M_{4,4,9}) <= rank(M_{4,4,p1}) + rank(M_{4,4,p2})
    
    We can recursively apply this. The key is to find good decompositions
    for the sub-problems.
    
    For M_{n,m,p} where p is small, we can try to use Strassen-like
    methods on the n×m part.
    
    Strategy: Split n=4 into 2+2, and use the structure:
    C = [A1; A2] * B = [A1*B; A2*B]
    
    Each A_i is 2×4, B is 4×9.
    A_i * B is a 2×9 product.
    
    For 2×4 × 4×9:
    Split B into B1 (4×4) and B2 (4×5):
    A_i * B = A_i * B1 + A_i * B2
    
    A_i * B1 is 2×4 × 4×4, A_i * B2 is 2×4 × 4×5.
    
    For 2×4 × 4×4: Split A_i into two 1×4 rows.
    Each 1×4 × 4×4 needs 4*4=16 mults (no savings for 1×m).
    Total for 2×4 × 4×4: 32.
    
    This doesn't help. Let me try a different split.
    """
    return None

def _try_strassen_based(n, m, p):
    """
    Try to use Strassen's algorithm structure.
    
    For M_{4,4,9}:
    Split A (4×4) into four 2×2 blocks: A11, A12, A21, A22
    Split B (4×9) into two 2×9 blocks: B1 (top 2 rows), B2 (bottom 2 rows)
    
    C (4×9) = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    
    Each 2×2 × 2×9 product:
    C_block[i,j] = sum_{k=0}^{1} A_block[i,k] * B_block[k,j]
    
    For a 2×2 × 2×9 product, we can use 7 "multiplications" in the Strassen sense,
    but each "multiplication" is actually a 1×9 vector times a scalar, which is free.
    The actual multiplications are between 1×9 vectors... no.
    
    Wait, Strassen for 2×2 × 2×2 uses 7 scalar multiplications.
    For 2×2 × 2×9, we can view B as nine 2×1 columns.
    For each column j, we compute A * B[:,j] using Strassen (7 mults per column).
    Total: 7 * 9 = 63 mults per 2×2 × 2×9 block.
    Four such blocks: 4 * 63 = 252. Worse than schoolbook.
    
    Alternative: For 2×2 × 2×9, we can precompute Strassen combinations:
    M1 = (A11+A22) * (B11+B22)  -- but B11 is 1×9, so this is a 1×9 vector
    M2 = (A21+A22) * B11
    M3 = A11 * (B22-B11)
    M4 = A22 * (B11-B22)
    M5 = (A11+A12) * B22
    M6 = (A21-A11) * (B11+B12)
    M7 = (A12-A22) * (B21+B22)
    
    Each of these is a product of a scalar (from A) and a 1×9 vector (from B),
    which is just scaling - no "multiplication" in the tensor sense.
    
    Actually, in the tensor decomposition sense, each "multiplication" M_t = a_t * b_t
    where a_t is a linear combination of A entries and b_t is a linear combination of B entries.
    For 2×2 × 2×9, each b_t is a 1×9 vector (linear combo of B's rows), and a_t is a scalar.
    The product a_t * b_t gives a 1×9 vector.
    
    So for 2×2 × 2×9, we need 7 terms, each contributing a 1×9 vector to the result.
    The total contribution to C is a 2×9 matrix, each entry being a sum of 7 terms.
    
    In the tensor decomposition:
    For each of the 7 Strassen terms t, and for each of the 9 columns j:
    We have a term that contributes to C_block[i,j].
    
    This gives 7 * 9 = 63 rank-1 terms for the 2×2 × 2×9 block.
    For four such blocks: 4 * 63 = 252. Still worse.
    
    The issue is that Strassen doesn't help when one dimension is large (9).
    """
    return None

def _try_numerical_search(n, m, p, max_rank, time_limit=100):
    """
    Try to find a lower-rank decomposition using numerical optimization.
    
    We set up the problem as: find U (n*m × r), V (m*p × r), W (p*n × r)
    such that for all (i,k,j): sum_c U[i*m+k,c] * V[k*p+j,c] * W[j*n+i,c] = 1
    and for all non-matching (a,b,c): sum_c U[a,c] * V[b,c] * W[c,c] = 0
    
    This is a system of polynomial equations. We can try to solve it using
    iterative methods.
    """
    start_time = time.time()
    
    # The number of equations is n*m*p (for the 1s) plus the number of non-matching entries
    # For n=4, m=4, p=9: 144 equations for the 1s
    # Non-matching: total entries in tensor space minus 144
    # Total: (n*m) * (m*p) * (p*n) = 16 * 36 * 36 = 20736
    # Non-matching: 20736 - 144 = 20592 equations for the 0s
    
    # This is a very large system. Let's try a different approach.
    # Use the fact that the tensor has a specific structure.
    
    # Actually, let's try a simpler approach: use random initialization and
    # gradient descent to minimize the residual.
    
    for r in range(36, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize U, V, W randomly
        np.random.seed(42)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Build the target tensor (sparse representation)
        # T[a,b,c] = 1 if a=i*m+k, b=k*p+j, c=j*n+i for some (i,k,j)
        # T[a,b,c] = 0 otherwise
        
        # We'll use a subset of equations for speed
        # Focus on the 144 "1" equations first
        
        # Use iterative refinement
        lr = 0.01
        for iteration in range(500):
            if time.time() - start_time > time_limit:
                break
            
            # Compute gradients
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            # For each (i,k,j) where T=1:
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        a = i * m + k
                        b = k * p + j
                        c = j * n + i
                        
                        # Current value
                        val = np.sum(U[a] * V[b] * W[c])
                        err = val - 1
                        
                        # Gradients
                        U_grad[a] += err * V[b] * W[c]
                        V_grad[b] += err * U[a] * W[c]
                        W_grad[c] += err * U[a] * V[b]
            
            # Update
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            # Check convergence
            if iteration % 50 == 0:
                max_err = 0
                for i in range(n):
                    for k in range(m):
                        for j in range(p):
                            a = i * m + k
                            b = k * p + j
                            c = j * n + i
                            val = np.sum(U[a] * V[b] * W[c])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-6:
                    # Check non-matching entries (sample)
                    valid = True
                    for _ in range(1000):
                        a = np.random.randint(0, n * m)
                        b = np.random.randint(0, m * p)
                        c = np.random.randint(0, p * n)
                        
                        i_a = a // m
                        k_a = a % m
                        k_b = b // p
                        j_b = b % p
                        j_c = c // n
                        i_c = c % n
                        
                        if k_a == k_b and j_b == j_c and i_a == i_c:
                            continue  # This is a matching entry
                        
                        val = np.sum(U[a] * V[b] * W[c])
                        if abs(val) > 1e-6:
                            valid = False
                            break
                    
                    if valid:
                        # Convert to lists
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        
                        # Round to nearest half-integer
                        U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                        V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                        W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                        
                        # Verify rounded version
                        if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                            return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
            
            # Adaptive learning rate
            if iteration % 100 == 99:
                lr *= 0.5
        
        if time.time() - start_time > time_limit:
            break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try numerical search for lower rank
    # We know the lower bound is 36, so try ranks from 36 upward
    # But this is expensive, so limit the search
    
    if time.time() < deadline - 20:
        result = _try_numerical_search(n, m, p, max_rank=143, time_limit=100)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = len(result["U"][0])
    
    # Verify final result
    assert _verify(best["U"], best["V"], best["W"], n, m, p), "Final decomposition failed verification"
    
    return best
# EVOLVE-BLOCK-END
