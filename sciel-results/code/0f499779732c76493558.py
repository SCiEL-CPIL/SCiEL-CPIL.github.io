"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if len(U) == 0:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check: sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = np.dot(U_arr[i*m+j, :], V_arr[jp*p+k, :] * W_arr[kp*n+ip, :])
                            rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(lhs - rhs) > 1e-6:
                                return False
    return True

def _als_decompose(r, n, m, p, max_iters=200, n_restarts=15, deadline=None, seed_base=42):
    """
    Alternating Least Squares for tensor decomposition.
    """
    if deadline is not None and time.time() > deadline - 5:
        return None
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        np.random.seed(seed_base + restart * 7919 + r * 131)
        
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Update W: for each (ip, kp), solve linear system
            for ip in range(n):
                for kp in range(p):
                    n_eq = n * m * m * p
                    A = np.zeros((n_eq, r))
                    b = np.zeros(n_eq)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    A[idx, :] = U[i*m+j, :] * V[jp*p+k, :]
                                    if i == ip and j == jp and k == kp:
                                        b[idx] = 1.0
                                    idx += 1
                    try:
                        W[kp*n+ip, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W[kp*n+ip, :] = np.zeros(r)
            
            # Update U: for each (i, j), solve linear system
            for i in range(n):
                for j in range(m):
                    n_eq = m * p * p * n
                    A = np.zeros((n_eq, r))
                    b = np.zeros(n_eq)
                    idx = 0
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    A[idx, :] = V[jp*p+k, :] * W[kp*n+ip, :]
                                    if j == jp and k == kp and i == ip:
                                        b[idx] = 1.0
                                    idx += 1
                    try:
                        U[i*m+j, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U[i*m+j, :] = np.zeros(r)
            
            # Update V: for each (jp, k), solve linear system
            for jp in range(m):
                for k in range(p):
                    n_eq = n * m * p * n
                    A = np.zeros((n_eq, r))
                    b = np.zeros(n_eq)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for kp in range(p):
                                for ip in range(n):
                                    A[idx, :] = U[i*m+j, :] * W[kp*n+ip, :]
                                    if j == jp and k == kp and i == ip:
                                        b[idx] = 1.0
                                    idx += 1
                    try:
                        V[jp*p+k, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V[jp*p+k, :] = np.zeros(r)
            
            # Compute loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    lhs = np.dot(U[i*m+j, :], V[jp*p+k, :] * W[kp*n+ip, :])
                                    loss += (lhs - rhs) ** 2
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
        
        if best_loss < 1e-8:
            # Try to round to rational numbers with various denominators
            for denom in [1, 2, 4, 10, 20, 100, 200, 1000]:
                U_rounded = np.round(best_U * denom) / denom
                V_rounded = np.round(best_V * denom) / denom
                W_rounded = np.round(best_W * denom) / denom
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
            
            # If rounding didn't work, try with more precision
            U_rounded = np.round(best_U * 10000) / 10000
            V_rounded = np.round(best_V * 10000) / 10000
            W_rounded = np.round(best_W * 10000) / 10000
            
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin before 120s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # For 2x4x6, try ranks from best_rank-1 down to a reasonable minimum
    min_rank_to_try = max(25, best_rank - 20)
    
    for r in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        result = _als_decompose(r, n, m, p, max_iters=150, n_restarts=10, deadline=deadline)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = r
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            print(f"Found rank {r} decomposition")
            break
    
    return best_result
# EVOLVE-BLOCK-END
