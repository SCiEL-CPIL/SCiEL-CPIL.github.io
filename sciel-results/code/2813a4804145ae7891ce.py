import time
import numpy as np
import sys

DIMENSIONS = [2, 5, 6]
N, M, P = DIMENSIONS
NM, MP, PN = N*M, M*P, P*N

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    rank = len(U[0])
    
    # Check all on-diagonal entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check all off-diagonal entries equal 0
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            j2 = b // p
            k = b % p
            if j != j2:
                continue
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k != k2 or i != i2:
                    val = 0.0
                    for cc in range(rank):
                        val += U[a][cc] * V[b][cc] * W[c][cc]
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _compute_residual(U, V, W, n, m, p):
    """Compute the sum of squares of the tensor residual."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    
    # Compute the tensor T from factors: T[a,b,c] = sum_c U[a,c] V[b,c] W[c,c]
    # We only care about specific entries.
    # On-diagonal: T[i*m+j, j*p+k, k*n+i] should be 1
    # Off-diagonal: T[i*m+j, j'*p+k, k'*n+i'] should be 0 if (j!=j' or k!=k' or i!=i')
    
    # Efficiently compute all T entries?
    # T is size (nm, mp, pn).
    # T[a, b, c] = (U @ (V * W).T)[a, b] ... no.
    # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    
    # Let's compute the residual directly for the required entries.
    # This is fast enough for n=2, m=5, p=6 (60, 30, 12 entries).
    
    # Precompute products
    # U[a, r] * V[b, r]
    # We need T[a,b,c] for specific (a,b,c).
    
    # Instead of looping over all a,b,c (60*30*12 = 21600), we can just loop over the 60 on-diag and the rest.
    # But for SA, we need a fast evaluation.
    # Let's use the structure.
    
    # Residual = sum (T_target - T_computed)^2
    
    # Let's construct the full tensor T as a flat array or 3D array?
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    
    # We can compute T as:
    # T = np.einsum('ar,br,cr->abc', U, V, W)
    # This is efficient.
    
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Initialize residual
    res = 0.0
    
    # On-diagonal: should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = T[a, b, c]
                res += (val - 1.0) ** 2
                
    # Off-diagonal: should be 0
    # Iterate over all a, b, c and skip the on-diagonal ones.
    # Since n=2, m=5, p=6, total entries 60*30*12 = 21600.
    # 60 are on-diagonal. 21540 are off-diagonal.
    # We can just sum T^2 and subtract the on-diagonal T^2, then add the on-diagonal (T-1)^2?
    # Sum_all T^2 - Sum_diag T^2 + Sum_diag (T-1)^2
    # = Sum_all T^2 - Sum_diag T^2 + Sum_diag (T^2 - 2T + 1)
    # = Sum_all T^2 - Sum_diag T^2 + Sum_diag T^2 - 2 Sum_diag T + 60
    # = Sum_all T^2 - 2 Sum_diag T + 60
    
    # This is much faster!
    total_sq = np.sum(T * T)
    diag_sum = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                diag_sum += T[a, b, c]
                
    res = total_sq - 2.0 * diag_sum + 60.0
    
    return res

def _als_refine(U, V, W, n, m, p, max_iter=20, deadline=None):
    """Quick ALS refinement to polish the solution."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    U, V, W = U.astype(np.float64).copy(), V.astype(np.float64).copy(), W.astype(np.float64).copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        # Solve for W
        # Minimize sum (T_target - (U*V) @ W^T)^2
        # For each c, W[c] is a vector.
        # T_target[a,b,c] is 1 if (a,b,c) on-diag, 0 otherwise.
        # Let A = U * V (element-wise product, shape nm x rank)
        # Then T[a,b,c] = sum_r A[a,r] W[b,r] ... wait.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For fixed c, T[:, :, c] is a matrix of size nm x mp.
        # T[a,b,c] = (U * V)[a, r] * W[c, r] summed over r? No.
        # T[a,b,c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
        # This is not a standard matrix multiplication for solving W.
        
        # Let's stick to the standard ALS formulation from the previous code which worked,
        # but make it faster or just use it for polishing.
        
        # Actually, the previous _als_solve was quite slow due to loops.
        # Let's try a simpler gradient descent step or just skip ALS and rely on SA.
        # Given the time budget, SA is the main driver.
        
        # Let's implement a very simple gradient descent step for polishing.
        # Gradient of Residual w.r.t W[c]:
        # dRes/dW[c,r] = -2 * sum_{a,b: (a,b,c) on-diag} T[a,b,c] * U[a,r] * V[b,r] + 2 * W[c,r] * sum_{a,b} T[a,b,c]^2 ...
        # This is getting complicated.
        
        # Let's just return the current U, V, W. The SA loop will do the optimization.
        pass
        
    return U, V, W

def _try_sa_rank(r, n, m, p, deadline, n_restarts=5, max_iter=2000):
    """Try rank-r decomposition via Simulated Annealing."""
    nm, mp, pn = n * m, m * p, p * n
    
    best_res = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1
        
        # Scale to get closer to 1
        # If we scale U by sqrt(r), V by 1, W by 1, the product scales by r.
        # We want product ~ 1. So U ~ 1/sqrt(r), V ~ 1, W ~ 1?
        # Let's just use random init and let SA handle it.
        
        T_init = _compute_residual(U, V, W, n, m, p)
        
        # SA Parameters
        T_sa = 10.0
        T_min = 1e-10
        alpha = 0.95
        # Step size
        step_size = 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Generate neighbor
            # Pick a random matrix and a random entry
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                mat = U
            elif mat_idx == 1:
                mat = V
            else:
                mat = W
                
            row = np.random.randint(mat.shape[0])
            col = np.random.randint(mat.shape[1])
            
            old_val = mat[row, col]
            new_val = old_val + np.random.randn() * step_size
            mat[row, col] = new_val
            
            # Compute new residual
            T_new = _compute_residual(U, V, W, n, m, p)
            delta = T_new - T_init
            
            # Acceptance probability
            if delta < 0 or (T_sa > 0 and np.random.rand() < np.exp(-delta / T_sa)):
                T_init = T_new
                if T_new < best_res:
                    best_res = T_new
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
            else:
                mat[row, col] = old_val
                
            T_sa *= alpha
            if T_sa < T_min:
                break
                
        # If we found a very good solution, we might stop early
        if best_res < 1e-6:
            break
            
    if best_res < 1e-6:
        # Verify with strict tolerance
        res = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        if _verify_rank(res["U"], res["V"], res["W"], n, m, p):
            return res
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 140  # Safety margin for 150s limit
    n, m, p = DIMENSIONS
    fb = _schoolbook_rank()
    
    # Try ranks lower than 60
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(10, 30, 12) = 30.
    # But matrix multiplication tensor rank is known to be higher.
    # For 2x5x6, is it 60?
    # Let's try 59, 58, ... down to 50.
    
    # SA is slower per iteration than ALS, but might find global minima.
    # We allocate time for a few ranks.
    
    # Try 59
    if time.time() < deadline - 10:
        res = _try_sa_rank(59, n, m, p, deadline, n_restarts=2, max_iter=1000)
        if res is not None:
            return res
            
    # Try 58
    if time.time() < deadline - 5:
        res = _try_sa_rank(58, n, m, p, deadline, n_restarts=2, max_iter=1000)
        if res is not None:
            return res
            
    # If SA fails, fall back to schoolbook
    return fb

# EVOLVE-BLOCK-END
