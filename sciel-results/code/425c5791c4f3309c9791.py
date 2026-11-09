"""Matrix multiplication 2,3,9 tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    deadline = time.time() + 130
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Trivial decomposition: rank = n*m*p = 54
    base_rank = n * m * p
    best_rank = base_rank
    best_U = [[0] * base_rank for _ in range(n * m)]
    best_V = [[0] * base_rank for _ in range(m * p)]
    best_W = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
                
    def verify_sub(U, V, W, n_s, m_s, p_s, tol=1e-9):
        """Verify decomposition for n_s x m_s x p_s."""
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    total = 0
                    for r in range(rank):
                        if isinstance(U, list):
                            u_val = U[i*m_s + j][r]
                            v_val = V[j*p_s + k][r]
                            w_val = W[k*n_s + i][r]
                        else:
                            u_val = U[i*m_s + j, r]
                            v_val = V[j*p_s + k, r]
                            w_val = W[k*n_s + i, r]
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > tol:
                        return False
        return True

    def solve_2x3x3(target_rank, n_trials=2000, seed=42):
        """Find rank decomposition for 2x3x3."""
        n_s, m_s, p_s = 2, 3, 3
        rng = np.random.RandomState(seed)
        
        # Candidate values
        candidates = [-1, -0.5, 0, 0.5, 1]
        
        for trial in range(n_trials):
            if time.time() > deadline:
                return None
            
            U = np.zeros((n_s * m_s, target_rank), dtype=float)
            V = np.zeros((m_s * p_s, target_rank), dtype=float)
            W = np.zeros((p_s * n_s, target_rank), dtype=float)
            
            # Random initialization
            U[:, :] = rng.choice(candidates, size=(n_s*m_s, target_rank))
            V[:, :] = rng.choice(candidates, size=(m_s*p_s, target_rank))
            W[:, :] = rng.choice(candidates, size=(p_s*n_s, target_rank))
            
            if verify_sub(U, V, W, n_s, m_s, p_s):
                return U, V, W
        return None

    # Try to solve 2x3x3 for rank 9, 10, 11...
    # If found, total rank is 3 * r
    for r in [9, 10, 11, 12]:
        if 3 * r >= best_rank:
            break
        if time.time() > deadline:
            break
        
        # Increase trials for lower ranks
        trials = 5000 if r == 9 else 2000
        
        result = solve_2x3x3(r, n_trials=trials, seed=r*100)
        if result is not None:
            U_s, V_s, W_s = result
            total_rank = 3 * r
            if total_rank < best_rank:
                best_rank = total_rank
                # Construct full decomposition
                U_full = np.zeros((n * m, total_rank))
                V_full = np.zeros((m * p, total_rank))
                W_full = np.zeros((p * n, total_rank))
                
                for q in range(3):
                    offset = q * r
                    # U is shared
                    U_full[:, offset:offset+r] = U_s
                    # V: rows 3q to 3q+2
                    V_full[3*q:3*q+3, offset:offset+r] = V_s
                    # W: rows 6q to 6q+4 (k=3q, 3q+1, 3q+2)
                    # W index is k*n + i.
                    # For k=3q, rows are 6q, 6q+1.
                    # For k=3q+1, rows are 6q+2, 6q+3.
                    # For k=3q+2, rows are 6q+4, 6q+5.
                    # W_s rows are k*n_s + i = k*2 + i.
                    # So W_s rows 0,1 correspond to k=0.
                    # W_s rows 2,3 correspond to k=1.
                    # W_s rows 4,5 correspond to k=2.
                    # We need to map W_s rows to W_full rows.
                    for k_s in range(3):
                        for i in range(2):
                            row_s = k_s * 2 + i
                            row_f = (3*q + k_s) * 2 + i
                            W_full[row_f, offset:offset+r] = W_s[row_s, :]
                
                best_U = U_full.tolist()
                best_V = V_full.tolist()
                best_W = W_full.tolist()
                break # Found a good one, stop

    # If no structured solution found, try random search on full problem
    # But this is unlikely to beat 30 if 2x3x3 rank 10 exists.
    # Fallback: return best found
    
    if isinstance(best_U, np.ndarray):
        U_final = best_U.tolist()
        V_final = best_V.tolist()
        W_final = best_W.tolist()
    else:
        U_final = best_U
        V_final = best_V
        W_final = best_W
        
    return {"U": U_final, "V": V_final, "W": W_final}
