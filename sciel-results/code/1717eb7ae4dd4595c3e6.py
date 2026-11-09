import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    rank_sb = n * m * p  # 42
    
    # Build schoolbook decomposition
    i_grid, j_grid, k_grid = np.indices((n, m, p))
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    k_flat = k_grid.flatten()
    col_indices = np.arange(rank_sb)
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    U_sb = np.zeros((nm, rank_sb), dtype=np.int64)
    V_sb = np.zeros((mp, rank_sb), dtype=np.int64)
    W_sb = np.zeros((pn, rank_sb), dtype=np.int64)
    U_sb[u_rows, col_indices] = 1
    V_sb[v_rows, col_indices] = 1
    W_sb[w_rows, col_indices] = 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def is_half_integer(arr, tol=1e-9):
        """Check if all entries are half-integers."""
        doubled = arr * 2
        rounded = np.round(doubled)
        return np.allclose(doubled, rounded, atol=tol)
    
    def snap_to_half_integer(arr, tol=1e-6):
        """Snap entries to nearest half-integer if close."""
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None
    
    def solve_for_W(U, V, rank):
        """Given U and V, solve for W."""
        W = np.zeros((pn, rank))
        for i in range(n):
            for k in range(p):
                # A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A_ik = np.zeros((m, rank))
                for j in range(m):
                    A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve A_ik @ w = ones(m)
                # Use least squares
                w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                
                # Check if solution is valid
                if not np.allclose(A_ik @ w, np.ones(m), atol=1e-9):
                    return None
                
                W[k * n + i, :] = w
        
        return W
    
    def try_rank(rank, num_trials, entry_set):
        """Try to find a decomposition of given rank."""
        for trial in range(num_trials):
            if time.time() > deadline:
                return None, None, None
            
            np.random.seed(trial * 7919 + rank)  # Reproducible
            
            U = np.random.choice(entry_set, size=(nm, rank))
            V = np.random.choice(entry_set, size=(mp, rank))
            
            W = solve_for_W(U, V, rank)
            if W is None:
                continue
            
            # Check if W is half-integer
            if not is_half_integer(W):
                continue
            
            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                continue
            
            if verify_decomposition(U, V, W_snapped):
                return U, V, W_snapped
        
        return None, None, None
    
    # Try ranks from 41 down
    entry_sets = [
        [-1, 0, 1],
        [-1, 0, 1, 2],
        [-2, -1, 0, 1, 2],
        [0, 1],
        [-1, 1],
    ]
    
    for rank in range(rank_sb - 1, max(30, rank_sb - 20), -1):
        if time.time() > deadline:
            break
        
        found = False
        for entry_set in entry_sets:
            if time.time() > deadline:
                break
            
            U_found, V_found, W_found = try_rank(rank, 50, entry_set)
            if U_found is not None:
                best_U, best_V, best_W = U_found, V_found, W_found
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    # Final verification
    if best_rank < rank_sb:
        if not verify_decomposition(best_U, best_V, best_W):
            # Fall back to schoolbook
            best_U, best_V, best_W = U_sb, V_sb, W_sb
            best_rank = rank_sb
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
