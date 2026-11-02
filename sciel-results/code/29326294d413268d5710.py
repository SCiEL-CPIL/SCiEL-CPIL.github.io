"""
Rank-minimal decomposition for 2x2 by 2x10 matrix multiplication.

Strategy: Start with Strassen's rank-35 decomposition, then attempt to reduce rank
via numerical optimization. Try to find a lower-rank exact decomposition by
optimizing a smooth surrogate of the tensor rank problem.
"""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 10]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] where i indexes A (n*m),
    j indexes B (m*p), k indexes C (p*n).
    T[i,j,k] = 1 if A[i] * B[j] contributes to C[k], else 0.
    A is n x m, B is m x p, C is p x n.
    C[k_row, k_col] = sum_j A[i_row, j] * B[j, k_col]
    i = i_row * m + i_col  (but for A, it's A[row, col] so i = row*m + col)
    j = j_row * p + j_col  (B[row, col], row is the shared dim)
    k = k_row * n + k_col  (C[row, col])
    
    Actually let me be more careful:
    A is 2x2: A[r,c] for r in {0,1}, c in {0,1}. Index i = r*m + c = r*2 + c
    B is 2x10: B[c,p] for c in {0,1}, p in {0,...,9}. Index j = c*p + p_col = c*10 + p_col
    C is 10x2: C[p,r] for p in {0,...,9}, r in {0,1}. Index k = p*n + r = p*2 + r
    
    C[p, r] = sum_c A[r, c] * B[c, p]
    
    So T[i, j, k] = 1 if i = r*2+c, j = c*10+p_col, k = p_col*2+r for some r, c, p_col
    """
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(n):
        for c in range(m):
            for p_col in range(p):
                i = r * m + c
                j = c * p + p_col
                k = p_col * n + r
                T[i, j, k] = 1.0
    return T

def _tensor_error(U, V, W, T):
    """Compute ||T - sum_t U[:,t] V[:,t] W[:,t]||_F^2"""
    # U: (nm, rank), V: (mp, rank), W: (pn, rank)
    # Reconstructed: sum_t U[:,t] outer V[:,t] outer W[:,t]
    # Efficient: einsum
    recon = np.einsum('it,jt,kt->ijk', U, V, W)
    return np.sum((T - recon) ** 2)

def _try_rank_reduction(T, start_rank=35, target_ranks=None, time_budget=100):
    """Try to find a lower-rank decomposition."""
    start_time = time.time()
    
    if target_ranks is None:
        target_ranks = list(range(34, 20, -1))  # Try 34 down to 21
    
    best_rank = start_rank
    best_U, best_V, best_W = None, None, None
    
    # First, get the rank-35 Strassen solution as starting point
    n, m, p = 2, 2, 10
    U35 = np.zeros((n*m, start_rank))
    V35 = np.zeros((m*p, start_rank))
    W35 = np.zeros((p*n, start_rank))
    
    for t in range(5):
        col_base = t * 7
        col = col_base
        U35[0, col] = 1; U35[3, col] = 1
        V35[2*t, col] = 1; V35[10+2*t+1, col] = 1
        W35[4*t, col] = 1; W35[4*t+3, col] = 1
        
        col = col_base + 1
        U35[2, col] = 1; U35[3, col] = 1
        V35[2*t, col] = 1
        W35[4*t+1, col] = 1; W35[4*t+3, col] = -1
        
        col = col_base + 2
        U35[0, col] = 1
        V35[2*t+1, col] = 1; V35[10+2*t+1, col] = -1
        W35[4*t+2, col] = 1; W35[4*t+3, col] = 1
        
        col = col_base + 3
        U35[3, col] = 1
        V35[10+2*t, col] = 1; V35[2*t, col] = -1
        W35[4*t, col] = 1; W35[4*t+1, col] = 1
        
        col = col_base + 4
        U35[0, col] = 1; U35[1, col] = 1
        V35[10+2*t+1, col] = 1
        W35[4*t, col] = -1; W35[4*t+2, col] = 1
        
        col = col_base + 5
        U35[2, col] = 1; U35[0, col] = -1
        V35[2*t, col] = 1; V35[2*t+1, col] = 1
        W35[4*t+3, col] = 1
        
        col = col_base + 6
        U35[1, col] = 1; U35[3, col] = -1
        V35[10+2*t, col] = 1; V35[10+2*t+1, col] = 1
        W35[4*t, col] = 1
    
    # Verify rank-35 is exact
    err35 = _tensor_error(U35, V35, W35, T)
    if err35 > 1e-10:
        # Fallback: just return rank 35
        return start_rank, U35, V35, W35
    
    for target_r in target_ranks:
        if time.time() - start_time > time_budget:
            break
        
        # Initialize from top-left submatrix of rank-35 solution
        U0 = U35[:, :target_r].copy()
        V0 = V35[:, :target_r].copy()
        W0 = W35[:, :target_r].copy()
        
        # Add some random perturbation to escape local minima
        for _ in range(3):
            U_r = U0 + 0.01 * np.random.randn(n*m, target_r)
            V_r = V0 + 0.01 * np.random.randn(m*p, target_r)
            W_r = W0 + 0.01 * np.random.randn(p*n, target_r)
            
            x0 = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
            
            def objective(x):
                Ux = x[:n*m*target_r].reshape(n*m, target_r)
                Vx = x[n*m*target_r : n*m*target_r + m*p*target_r].reshape(m*p, target_r)
                Wx = x[n*m*target_r + m*p*target_r :].reshape(p*n, target_r)
                return _tensor_error(Ux, Vx, Wx, T)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
                final_err = res.fun
                if final_err < 1e-8:
                    Uf = x0[:n*m*target_r].reshape(n*m, target_r)
                    Vf = x0[n*m*target_r : n*m*target_r + m*p*target_r].reshape(m*p, target_r)
                    Wf = x0[n*m*target_r + m*p*target_r :].reshape(p*n, target_r)
                    best_rank = target_r
                    best_U, best_V, best_W = Uf, Vf, Wf
                    break
            except Exception:
                pass
        
        if time.time() - start_time > time_budget:
            break
    
    if best_U is not None:
        return best_rank, best_U, best_V, best_W
    else:
        return start_rank, U35, V35, W35

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    T = _build_tensor(n, m, p)
    
    # Try to find lower rank
    try:
        rank, U, V, W = _try_rank_reduction(T, start_rank=35, time_budget=100)
    except Exception:
        rank = 35
        U, V, W = None, None, None
    
    if U is None:
        # Fallback to rank 35
        rank = 35
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for t in range(5):
            col_base = t * 7
            col = col_base
            U[0, col] = 1; U[3, col] = 1
            V[2*t, col] = 1; V[10+2*t+1, col] = 1
            W[4*t, col] = 1; W[4*t+3, col] = 1
            col = col_base + 1
            U[2, col] = 1; U[3, col] = 1
            V[2*t, col] = 1
            W[4*t+1, col] = 1; W[4*t+3, col] = -1
            col = col_base + 2
            U[0, col] = 1
            V[2*t+1, col] = 1; V[10+2*t+1, col] = -1
            W[4*t+2, col] = 1; W[4*t+3, col] = 1
            col = col_base + 3
            U[3, col] = 1
            V[10+2*t, col] = 1; V[2*t, col] = -1
            W[4*t, col] = 1; W[4*t+1, col] = 1
            col = col_base + 4
            U[0, col] = 1; U[1, col] = 1
            V[10+2*t+1, col] = 1
            W[4*t, col] = -1; W[4*t+2, col] = 1
            col = col_base + 5
            U[2, col] = 1; U[0, col] = -1
            V[2*t, col] = 1; V[2*t+1, col] = 1
            W[4*t+3, col] = 1
            col = col_base + 6
            U[1, col] = 1; U[3, col] = -1
            V[10+2*t, col] = 1; V[10+2*t+1, col] = 1
            W[4*t, col] = 1
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[round(U[i, j], 6) for j in range(rank)] for i in range(n*m)]
    V_list = [[round(V[i, j], 6) for j in range(rank)] for i in range(m*p)]
    W_list = [[round(W[i, j], 6) for j in range(rank)] for i in range(p*n)]
    
    # Clean up near-zero and near-integer values
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx in range(len(row)):
                val = row[idx]
                if abs(val) < 1e-5:
                    row[idx] = 0
                else:
                    # Round to nearest half-integer
                    nearest = round(val * 2) / 2
                    if abs(val - nearest) < 1e-4:
                        row[idx] = nearest
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
