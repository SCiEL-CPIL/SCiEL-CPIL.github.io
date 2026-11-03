import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    dim_U, dim_V, dim_W = n * m, m * p, p * n

    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # Build target tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Baseline: rank-17 solution
    U17, V17, W17 = _build_rank17()

    best_rank = 17
    best_U, best_V, best_W = U17, V17, W17

    # Try to find rank-16 solution
    if time.time() < deadline - 5:
        r16 = _search_rank16(T, U17, V17, W17, deadline)
        if r16 is not None:
            best_rank = 16
            best_U, best_V, best_W = r16

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _build_rank17():
    n, m, p = 2, 3, 3
    dim_U, dim_V, dim_W = 6, 9, 6
    rank = 17
    U = np.zeros((dim_U, rank), dtype=np.int64)
    V = np.zeros((dim_V, rank), dtype=np.int64)
    W = np.zeros((dim_W, rank), dtype=np.int64)

    # Strassen for A[0:2,0:2] * B[0:2,0:2]
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)*B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4 = A11*(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1; W[0, 4] = -1

    # M6 = (A10-A00)*(B00+B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Schoolbook: A[:,2]*B[2,:] -> C[:,0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1

    # Schoolbook: A[:,0:2]*B[:,2] -> C[:,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1

    # Schoolbook: A[:,2]*B[2,2] -> C[:,2]
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1

    return U, V, W

def _search_rank16(T, U17, V17, W17, deadline):
    """
    Search for rank-16 by fixing the first 7 terms to Strassen
    and optimizing the remaining 9 terms. Uses fast matrix-multiply-based error.
    """
    dim_U, dim_V, dim_W = 6, 9, 6
    rank = 16
    num_strassen = 7
    num_free = rank - num_strassen  # 9 terms

    U_strassen = U17[:, :num_strassen].astype(np.float64).copy()
    V_strassen = V17[:, :num_strassen].astype(np.float64).copy()
    W_strassen = W17[:, :num_strassen].astype(np.float64).copy()

    # Compute T_strassen via matrix products for each rank term
    T_strassen = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
    for r in range(num_strassen):
        T_strassen += np.einsum('a,b,c->abc', U_strassen[:, r], V_strassen[:, r], W_strassen[:, r])
    
    T_free = T - T_strassen
    T_free_flat = T_free.reshape(-1)

    def compute_error(U_f, V_f, W_f):
        """Fast error: build tensor via outer products, compare flat."""
        S = np.einsum('ar,br,cr->abc', U_f, V_f, W_f)
        return np.sum((S.reshape(-1) - T_free_flat) ** 2)

    # Precompute: for each free rank r, the contribution to error when we change
    # one entry. We use a smarter approach: compute the "residual" after removing
    # rank term r, then the error delta from changing U[a,r] is:
    # delta = 2 * U[a,r] * sum_{b,c} V[b,r]*W[c,r]*T_free[a,b,c] 
    #         - 2 * new_val * sum_{b,c} V[b,r]*W[c,r]*T_free[a,b,c]
    #         + (new_val^2 - old_val^2) * sum_{b,c} V[b,r]^2 * W[c,r]^2
    # Actually simpler: just recompute. With 6*9*6=324 entries and 9 terms,
    # einsum is fast enough.

    np.random.seed(42)
    
    # Starting point 1: first 9 schoolbook terms from rank-17
    U_init1 = U17[:, 7:7+num_free].astype(np.float64).copy()
    V_init1 = V17[:, 7:7+num_free].astype(np.float64).copy()
    W_init1 = W17[:, 7:7+num_free].astype(np.float64).copy()
    
    # Starting point 2: random sparse init
    U_init2 = np.zeros((dim_U, num_free), dtype=np.float64)
    V_init2 = np.zeros((dim_V, num_free), dtype=np.float64)
    W_init2 = np.zeros((dim_W, num_free), dtype=np.float64)
    for r in range(num_free):
        U_init2[np.random.randint(dim_U), r] = np.random.choice([-1.0, 1.0])
        V_init2[np.random.randint(dim_V), r] = np.random.choice([-1.0, 1.0])
        W_init2[np.random.randint(dim_W), r] = np.random.choice([-1.0, 1.0])
    
    # Starting point 3: structured init based on which tensor entries remain
    U_init3 = np.zeros((dim_U, num_free), dtype=np.float64)
    V_init3 = np.zeros((dim_V, num_free), dtype=np.float64)
    W_init3 = np.zeros((dim_W, num_free), dtype=np.float64)
    # T_free has specific pattern; try to cover remaining entries
    free_entries = np.argwhere(T_free_flat != 0)
    for r in range(num_free):
        if r < len(free_entries):
            a, b, c = free_entries[r]
            U_init3[a, r] = 1.0
            V_init3[b, r] = 1.0
            W_init3[c, r] = 1.0
        else:
            U_init3[np.random.randint(dim_U), r] = np.random.choice([-1.0, 1.0])
            V_init3[np.random.randint(dim_V), r] = np.random.choice([-1.0, 1.0])
            W_init3[np.random.randint(dim_W), r] = np.random.choice([-1.0, 1.0])

    initializations = [
        (U_init1, V_init1, W_init1),
        (U_init2, V_init2, W_init2),
        (U_init3, V_init3, W_init3),
    ]

    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None

    for init_idx, (U_init, V_init, W_init) in enumerate(initializations):
        if time.time() > deadline - 5:
            break
        
        current_U = U_init.copy()
        current_V = V_init.copy()
        current_W = W_init.copy()
        current_err = compute_error(current_U, current_V, current_W)
        
        if current_err < best_err:
            best_err = current_err
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
        
        # Check if already exact
        if current_err < 1e-9:
            U_int = np.round(np.hstack([U_strassen, current_U])).astype(np.int64)
            V_int = np.round(np.hstack([V_strassen, current_V])).astype(np.int64)
            W_int = np.round(np.hstack([W_strassen, current_W])).astype(np.int64)
            S = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
            for r in range(rank):
                S += np.einsum('a,b,c->abc', U_int[:, r].astype(np.float64), V_int[:, r].astype(np.float64), W_int[:, r].astype(np.float64))
            if np.allclose(S, T, atol=1e-9):
                return U_int, V_int, W_int

        # Coordinate descent: systematically try all positions for each rank term
        no_improve = 0
        max_no_improve = 500
        steps = 0
        max_steps = 200000  # More steps possible with faster error
        
        while time.time() < deadline - 5 and steps < max_steps and no_improve < max_no_improve:
            steps += 1
            
            # Cycle through matrices and rank terms more systematically
            r = steps % num_free
            mat = (steps // num_free) % 3
            
            if mat == 0:
                idx = (steps // (num_free * 3)) % dim_U
            elif mat == 1:
                idx = (steps // (num_free * 3)) % dim_V
            else:
                idx = (steps // (num_free * 3)) % dim_W
            
            if mat == 0:
                old_val = current_U[idx, r]
            elif mat == 1:
                old_val = current_V[idx, r]
            else:
                old_val = current_W[idx, r]
            
            # Try deltas in order of likelihood
            for delta in [1, -1, 2, -2]:
                new_val = old_val + delta
                if abs(new_val) > 4:
                    continue
                
                if mat == 0:
                    current_U[idx, r] = new_val
                elif mat == 1:
                    current_V[idx, r] = new_val
                else:
                    current_W[idx, r] = new_val
                
                new_err = compute_error(current_U, current_V, current_W)
                
                if new_err < current_err - 1e-12:
                    current_err = new_err
                    no_improve = 0
                    break  # Accept this delta, move to next position
                else:
                    # Revert
                    if mat == 0:
                        current_U[idx, r] = old_val
                    elif mat == 1:
                        current_V[idx, r] = old_val
                    else:
                        current_W[idx, r] = old_val
            
            else:
                # No delta improved
                no_improve += 1
            
            if current_err < best_err:
                best_err = current_err
                best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
            
            # Check for exact solution periodically
            if current_err < 1e-9:
                U_int = np.round(np.hstack([U_strassen, current_U])).astype(np.int64)
                V_int = np.round(np.hstack([V_strassen, current_V])).astype(np.int64)
                W_int = np.round(np.hstack([W_strassen, current_W])).astype(np.int64)
                S = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
                for r in range(rank):
                    S += np.einsum('a,b,c->abc', U_int[:, r].astype(np.float64), V_int[:, r].astype(np.float64), W_int[:, r].astype(np.float64))
                if np.allclose(S, T, atol=1e-9):
                    return U_int, V_int, W_int
    
    # Final verification of best found
    if best_err < 1e-9 and best_U is not None:
        U_int = np.round(np.hstack([U_strassen, best_U])).astype(np.int64)
        V_int = np.round(np.hstack([V_strassen, best_V])).astype(np.int64)
        W_int = np.round(np.hstack([W_strassen, best_W])).astype(np.int64)
        S = np.zeros((dim_U, dim_V, dim_W), dtype=np.float64)
        for r in range(rank):
            S += np.einsum('a,b,c->abc', U_int[:, r].astype(np.float64), V_int[:, r].astype(np.float64), W_int[:, r].astype(np.float64))
        if np.allclose(S, T, atol=1e-9):
            return U_int, V_int, W_int

    return None
