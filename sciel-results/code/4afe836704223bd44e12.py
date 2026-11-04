"""
Matrix multiplication tensor decomposition for M_{2,3,10}
Rank-55 base: 5 Strassen blocks (35) + 20 schoolbook terms
With elimination search to find lower rank.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140  # seconds, safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dense 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_exact(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    rank = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _build_rank55(n, m, p):
    """Build the rank-55 decomposition using Strassen + schoolbook."""
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # Local indices: 0=a00/b00/c00, 1=a01/b01/c01, 2=a10/b10/c10, 3=a11/b11/c11
    L = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: 1},       # M2
        {0: 1},             # M3
        {3: 1},             # M4
        {0: 1, 1: 1},       # M5
        {2: 1, 0: -1},      # M6
        {1: 1, 3: -1}       # M7
    ]
    R = [
        {0: 1, 3: 1},       # M1
        {0: 1},             # M2
        {1: 1, 3: -1},      # M3
        {2: 1, 0: -1},      # M4
        {3: 1},             # M5
        {0: 1, 1: 1},       # M6
        {2: 1, 3: 1}        # M7
    ]
    Wc = [
        {0: 1, 3: 1},       # M1: c00, c11
        {2: 1, 3: -1},      # M2: c10, -c11
        {1: 1, 3: 1},       # M3: c01, c11
        {0: 1, 2: 1},       # M4: c00, c10
        {1: 1, 0: -1},      # M5: c01, -c00
        {3: 1},             # M6: c11
        {0: 1}              # M7: c00
    ]
    
    # 5 Strassen blocks for the 2x2 x 2x10 part (j=0,1)
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            # Fill U: local idx -> i=idx//2, j=idx%2, global = i*m+j
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j, r] = val
            
            # Fill V: local idx -> j=idx//2, k=idx%2, global = j*p + k_offset+k
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k), r] = val
            
            # Fill W: local idx -> i=idx//2, k=idx%2, global = (k_offset+k)*n + i
            for local_idx, val in Wc[s].items():
                i = local_idx // 2
                k = local_idx % 2
                W[(k_offset + k) * n + i, r] = val
            
            col += 1
    
    # 20 schoolbook terms for the 2x1 x 1x10 part (j=2)
    for i in range(2):
        for k in range(10):
            U[i * m + 2, col] = 1
            V[2 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    return U, V, W

def _try_eliminate(U, V, W, T, max_eliminations=5, time_limit=120):
    """
    Try to eliminate terms by checking if removing a column still gives exact decomposition.
    Uses iterative greedy elimination with precomputed tensor contributions.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    active = list(range(current_rank))
    T_target = T
    
    # Precompute the contribution of each column to the tensor
    # T_col[r] = U[:, r] * V[:, r] * W[:, r] (outer product)
    # This is expensive to store for large rank, but rank is small.
    # T is 60x30x20 = 36000 entries.
    # 55 columns. 55 * 36000 = 2M floats. OK.
    
    eliminated = 0
    while eliminated < max_eliminations and time.time() - start_time < time_limit - 10:
        found_elimination = False
        
        # Compute current tensor
        T_current = np.einsum('ar,br,cr->abc', current_U[:, active], current_V[:, active], current_W[:, active])
        
        # Try removing each active column
        for idx in range(len(active)):
            if time.time() - start_time > time_limit - 5:
                break
            
            col_idx = active[idx]
            
            # T_without = T_current - T_col[col_idx]
            # T_col = U[:, col_idx] * V[:, col_idx] * W[:, col_idx]
            T_col = np.einsum('a,b,c->abc', current_U[:, col_idx], current_V[:, col_idx], current_W[:, col_idx])
            
            T_test = T_current - T_col
            
            if np.allclose(T_test, T_target, atol=1e-10):
                # This column can be eliminated
                active = [a for i, a in enumerate(active) if i != idx]
                eliminated += 1
                found_elimination = True
                break
        
        if not found_elimination:
            break
    
    if len(active) < current_rank:
        U_final = current_U[:, active]
        V_final = current_V[:, active]
        W_final = current_W[:, active]
        return U_final, V_final, W_final
    
    return current_U, current_V, current_W

def _try_linear_combination_elimination(U, V, W, T, time_limit=120):
    """
    More sophisticated: check if one rank-1 tensor is a linear combination of others.
    For each pair of columns, check if their tensor difference can be absorbed.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    active = list(range(current_rank))
    T_target = T
    
    eliminated = 0
    max_rounds = 10
    
    for round_num in range(max_rounds):
        if time.time() - start_time > time_limit - 10:
            break
        
        found = False
        
        # For each pair of active columns, check if we can merge them
        # by finding a linear combination that preserves the tensor
        for i in range(len(active)):
            if time.time() - start_time > time_limit - 5:
                break
            
            for j in range(i + 1, len(active)):
                if time.time() - start_time > time_limit - 3:
                    break
                
                ai = active[i]
                aj = active[j]
                
                # Check if column j's tensor is a scalar multiple of column i's
                # or if removing j and adjusting i works
                # Simple check: is V[:,aj] proportional to V[:,ai]?
                # And W[:,aj] proportional to W[:,ai]?
                # If so, we can combine into one column
                
                v_i = current_V[:, ai]
                v_j = current_V[:, aj]
                w_i = current_W[:, ai]
                w_j = current_W[:, aj]
                u_i = current_U[:, ai]
                u_j = current_U[:, aj]
                
                # Check if v_j = alpha * v_i
                if np.linalg.norm(v_i) > 1e-10:
                    alpha_v = np.dot(v_j, v_i) / np.dot(v_i, v_i)
                    if np.allclose(v_j, alpha_v * v_i, atol=1e-10):
                        # Check w_j = beta * w_i
                        if np.linalg.norm(w_i) > 1e-10:
                            beta_w = np.dot(w_j, w_i) / np.dot(w_i, w_i)
                            if np.allclose(w_j, beta_w * w_i, atol=1e-10):
                                # Check u_j = gamma * u_i
                                if np.linalg.norm(u_i) > 1e-10:
                                    gamma_u = np.dot(u_j, u_i) / np.dot(u_i, u_i)
                                    if np.allclose(u_j, gamma_u * u_i, atol=1e-10):
                                        # All three are proportional - can merge
                                        # The product u_j*v_j*w_j = gamma*alpha*beta * u_i*v_i*w_i
                                        # So we can just remove j
                                        test_active = [a for idx, a in enumerate(active) if idx != j]
                                        U_test = current_U[:, test_active]
                                        V_test = current_V[:, test_active]
                                        W_test = current_W[:, test_active]
                                        T_recon = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
                                        if np.allclose(T_recon, T_target, atol=1e-10):
                                            active = test_active
                                            eliminated += 1
                                            found = True
                                            break
                if found:
                    break
            if found:
                break
        
        if not found:
            break
    
    if len(active) < current_rank:
        U_final = current_U[:, active]
        V_final = current_V[:, active]
        W_final = current_W[:, active]
        return U_final, V_final, W_final
    
    return current_U, current_V, current_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Build the rank-55 decomposition
    U, V, W = _build_rank55(n, m, p)
    
    # Verify it's correct
    T = _build_tensor(n, m, p)
    if not _check_exact(U, V, W, T):
        # Fallback to trivial rank-60
        trivial_rank = n * m * p
        U_triv = np.zeros((n * m, trivial_rank))
        V_triv = np.zeros((m * p, trivial_rank))
        W_triv = np.zeros((p * n, trivial_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_triv[i * m + j, col] = 1
                    V_triv[j * p + k, col] = 1
                    W_triv[k * n + i, col] = 1
                    col += 1
        return {
            "U": U_triv.tolist(),
            "V": V_triv.tolist(),
            "W": W_triv.tolist()
        }
    
    # Try elimination search
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # First try simple elimination
    if time.time() - start_time < DEADLINE - 20:
        U_elim, V_elim, W_elim = _try_eliminate(U, V, W, T, max_eliminations=10, time_limit=DEADLINE - 20)
        if U_elim.shape[1] < best_rank:
            best_U, best_V, best_W = U_elim, V_elim, W_elim
            best_rank = U_elim.shape[1]
    
    # Try linear combination elimination
    if time.time() - start_time < DEADLINE - 10:
        U_lc, V_lc, W_lc = _try_linear_combination_elimination(U, V, W, T, time_limit=DEADLINE - 10)
        if U_lc.shape[1] < best_rank:
            best_U, best_V, best_W = U_lc, V_lc, W_lc
            best_rank = U_lc.shape[1]
    
    # Final verification
    if not _check_exact(best_U, best_V, best_W, T):
        # Fall back to rank-55
        best_U, best_V, best_W = U, V, W
        best_rank = 55
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
