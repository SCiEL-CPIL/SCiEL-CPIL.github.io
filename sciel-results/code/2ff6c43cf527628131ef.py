"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col

def _build_decomp(j_pairs, k_pairs, remaining_j, n, m, p):
    """Build a decomposition from block structure."""
    rank = len(j_pairs) * len(k_pairs) * 7 + len(remaining_j) * 2 * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (j0, j1) in j_pairs:
        for (k0, k1) in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    for j in remaining_j:
        for k in range(p):
            U[0*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
    
    if col != rank:
        return None
    
    if not _verify_decomp(U, V, W, n, m, p, rank):
        return None
    
    return (U, V, W, rank)

def _try_eliminate_terms(U, V, W, n, m, p, rank, max_iterations=10, deadline=None):
    """Try to eliminate terms greedily."""
    cur_U = [row[:] for row in U]
    cur_V = [row[:] for row in V]
    cur_W = [row[:] for row in W]
    cur_rank = rank
    
    for iteration in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for term_idx in range(cur_rank - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                break
            rows = []
            rhs = []
            for i in range(n):
                for l in range(m):
                    for k in range(p):
                        rhs_val = cur_U[i*m+l][term_idx] * cur_V[l*p+k][term_idx] * cur_W[k*n+i][term_idx]
                        if abs(rhs_val) > 1e-12:
                            row = []
                            for r in range(cur_rank):
                                if r == term_idx:
                                    continue
                                row.append(cur_U[i*m+l][r] * cur_V[l*p+k][r] * cur_W[k*n+i][r])
                            rows.append(row)
                            rhs.append(rhs_val)
            
            if not rows:
                continue
            
            A_mat = np.array(rows, dtype=float)
            b_vec = np.array(rhs, dtype=float)
            
            try:
                coeffs, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
            except:
                continue
            
            if len(residuals) > 0 and residuals[0] > 1e-8 * max(1.0, np.linalg.norm(b_vec)):
                continue
            
            if np.max(np.abs(coeffs)) > 100:
                continue
            
            new_rank = cur_rank - 1
            keep = [r for r in range(cur_rank) if r != term_idx]
            
            new_U = [[0] * new_rank for _ in range(n * m)]
            new_V = [[0] * new_rank for _ in range(m * p)]
            new_W = [[0] * new_rank for _ in range(p * n)]
            
            for new_idx, r in enumerate(keep):
                for i in range(n):
                    for l in range(m):
                        new_U[i*m+l][new_idx] = cur_U[i*m+l][r] + coeffs[new_idx] * cur_U[i*m+l][term_idx]
                for l in range(m):
                    for k in range(p):
                        new_V[l*p+k][new_idx] = cur_V[l*p+k][r] + coeffs[new_idx] * cur_V[l*p+k][term_idx]
                for k in range(p):
                    for i in range(n):
                        new_W[k*n+i][new_idx] = cur_W[k*n+i][r] + coeffs[new_idx] * cur_W[k*n+i][term_idx]
            
            if not _verify_decomp(new_U, new_V, new_W, n, m, p, new_rank):
                continue
            
            cur_U, cur_V, cur_W, cur_rank = new_U, new_V, new_W, new_rank
            improved = True
            break
        
        if not improved:
            break
    
    return (cur_U, cur_V, cur_W, cur_rank)

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140
    
    best_rank = float('inf')
    best_U = best_V = best_W = None
    
    j_list = list(range(m))
    k_list = list(range(p))
    
    k_matchings = [
        ((0,1), (2,3), (4,5)),
        ((0,2), (1,3), (4,5)),
        ((0,3), (1,2), (4,5)),
        ((0,1), (2,4), (3,5)),
        ((0,2), (1,4), (3,5)),
        ((0,3), (1,4), (2,5)),
        ((0,1), (2,5), (3,4)),
        ((0,2), (1,5), (3,4)),
        ((0,3), (1,5), (2,4)),
        ((0,4), (1,2), (3,5)),
        ((0,4), (1,3), (2,5)),
        ((0,5), (1,2), (3,4)),
        ((0,5), (1,3), (2,4)),
        ((0,5), (1,4), (2,3)),
    ]
    
    configs = []
    
    # 2 j-pairs, 3 k-pairs (best starting point: rank 54)
    for j0 in range(m):
        for j1 in range(j0+1, m):
            for j2 in range(m):
                if j2 in (j0, j1):
                    continue
                for j3 in range(j2+1, m):
                    if j3 in (j0, j1, j2):
                        continue
                    remaining_j = [j for j in j_list if j not in (j0, j1, j2, j3)]
                    for kp in k_matchings:
                        configs.append(((j0,j1),(j2,j3)), kp, remaining_j)
    
    # 1 j-pair, 3 k-pairs (rank 57)
    for j0 in range(m):
        for j1 in range(j0+1, m):
            remaining_j = [j for j in j_list if j not in (j0, j1)]
            for kp in k_matchings:
                configs.append(((j0,j1),), kp, remaining_j)
    
    # Sort by expected rank
    configs.sort(key=lambda cfg: len(cfg[0]) * len(cfg[1]) * 7 + len(cfg[2]) * 2 * p)
    
    for jp, kp, rj in configs:
        if time.time() > _deadline - 60:
            break
        
        result = _build_decomp(jp, kp, rj, n, m, p)
        if result is None:
            continue
        
        U, V, W, rank = result
        
        if rank >= best_rank:
            continue
        
        U2, V2, W2, rank2 = _try_eliminate_terms(U, V, W, n, m, p, rank, max_iterations=5, deadline=_deadline - 30)
        
        if rank2 < best_rank:
            best_rank = rank2
            best_U = U2
            best_V = V2
            best_W = W2
    
    if best_U is None:
        best_rank = n * m * p
        best_U = [[0] * best_rank for _ in range(n * m)]
        best_V = [[0] * best_rank for _ in range(m * p)]
        best_W = [[0] * best_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    best_U[i*m+l][col] = 1
                    best_V[l*p+k][col] = 1
                    best_W[k*n+i][col] = 1
                    col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
