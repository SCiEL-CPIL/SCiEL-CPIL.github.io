"""Matrix multiplication 5x5x5 tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify_float(n, m, p, U, V, W, tol=1e-9):
    """Verify using floating point."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                dot = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if abs(dot - 1.0) > tol:
                    return False
    return True

def _verify_exact(n, m, p, U, V, W):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    U_f = [[Fraction(x) for x in row] for row in U]
    V_f = [[Fraction(x) for x in row] for row in V]
    W_f = [[Fraction(x) for x in row] for row in W]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_f[i * m + j][c] * V_f[j * p + k][c] * W_f[k * n + i][c]
                if total != Fraction(1):
                    return False
    return True

def _strassen_2x2():
    """
    Classic Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7).
    U indexes (i,j), V indexes (j,k), W indexes (k,i).
    
    C11 = M1 + M4 - M3 - M7
    C12 = M5 + M7
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M5
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1: (A11-A22)*(B22-B11)
    U[0][0] = 1; U[3][0] = -1
    V[0][0] = 1; V[3][0] = -1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: A11*(B11-B22)
    U[0][2] = 1
    V[0][2] = 1; V[3][2] = -1
    W[0][2] = -1; W[3][2] = 1
    
    # M4: A22*(B22-B11)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[3][4] = 1
    
    # M6: (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[1][6] = 1
    
    return {"U": U, "V": V, "W": W}

def _embed_strassen_2x2_into_5x5():
    """
    Embed the 2x2 Strassen decomposition into 5x5 by using it for the top-left
    2x2 block and schoolbook for the rest.
    """
    n = 5
    strassen = _strassen_2x2()
    
    # Rank calculation:
    # 2x2x2 block: rank 7
    # Remaining 117 indices: rank 117
    # Total rank = 124
    rank = 7 + 117
    
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    # Embed Strassen for top-left 2x2: indices i,j,k in {0,1}
    for c in range(7):
        for ii in range(4):
            if strassen["U"][ii][c] != 0:
                i, j = divmod(ii, 2)
                U[i * 5 + j][c] = strassen["U"][ii][c]
        for jj in range(4):
            if strassen["V"][jj][c] != 0:
                j, k = divmod(jj, 2)
                V[j * 5 + k][c] = strassen["V"][jj][c]
        for kk in range(4):
            if strassen["W"][kk][c] != 0:
                k, i = divmod(kk, 2)
                W[k * 5 + i][c] = strassen["W"][kk][c]
    
    # Schoolbook for remaining (i,j,k) where not all in {0,1}
    col = 7
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i < 2 and j < 2 and k < 2:
                    continue
                U[i * 5 + j][col] = 1
                V[j * 5 + k][col] = 1
                W[k * 5 + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _construct_block_decomposition():
    """
    Construct a block decomposition using partition 5 = 2 + 2 + 1.
    Uses Strassen for 2x2x2 blocks and schoolbook for others.
    This yields rank 121 (3 Strassen blocks + 114 schoolbook).
    """
    n, m, p = 5, 5, 5
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Pre-compute Strassen factors for 2x2x2 blocks
    strassen = _strassen_2x2()
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                size_a = parts[a]
                size_b = parts[b]
                size_c = parts[c]
                
                off_a = offsets[a]
                off_b = offsets[b]
                off_c = offsets[c]
                
                if size_a == 2 and size_b == 2 and size_c == 2:
                    # Strassen block
                    for c_idx in range(7):
                        u_vec = [0] * 25
                        v_vec = [0] * 25
                        w_vec = [0] * 25
                        
                        # Map local indices to global
                        for ii in range(4):
                            if strassen["U"][ii][c_idx] != 0:
                                lu, lv = divmod(ii, 2)
                                gi = off_a + lu
                                gj = off_b + lv
                                u_vec[gi * 5 + gj] = strassen["U"][ii][c_idx]
                        
                        for jj in range(4):
                            if strassen["V"][jj][c_idx] != 0:
                                lv, lw = divmod(jj, 2)
                                gj = off_b + lv
                                gk = off_c + lw
                                v_vec[gj * 5 + gk] = strassen["V"][jj][c_idx]
                        
                        for kk in range(4):
                            if strassen["W"][kk][c_idx] != 0:
                                lw, lu = divmod(kk, 2)
                                gk = off_c + lw
                                gi = off_a + lu
                                w_vec[gk * 5 + gi] = strassen["W"][kk][c_idx]
                        
                        add_col(u_vec, v_vec, w_vec)
                    
                else:
                    # Schoolbook block
                    for u in range(size_a):
                        for v in range(size_b):
                            for w in range(size_c):
                                u_vec = [0] * 25
                                v_vec = [0] * 25
                                w_vec = [0] * 25
                                
                                gi = off_a + u
                                gj = off_b + v
                                gk = off_c + w
                                
                                u_vec[gi * 5 + gj] = 1
                                v_vec[gj * 5 + gk] = 1
                                w_vec[gk * 5 + gi] = 1
                                add_col(u_vec, v_vec, w_vec)
                                
    rank = len(U_cols)
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}

def _try_direct_search(n, m, p, target_rank, timeout=30):
    """
    Try to find a decomposition of exactly target_rank using a structured
    random search with verification.
    """
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    
    # Use a structured initialization based on Strassen-like patterns
    for attempt in range(50):
        if time.time() - start_time > timeout:
            break
        
        np.random.seed(attempt * 12345 + target_rank)
        
        U = np.zeros((nm, target_rank), dtype=float)
        V = np.zeros((mp, target_rank), dtype=float)
        W = np.zeros((pn, target_rank), dtype=float)
        
        # Initialize with a schoolbook-like base for the first min(target_rank, 25) entries
        base = min(target_rank, n * m * p)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= target_rank:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        if col < target_rank:
            # Add random perturbation for extra columns
            for c in range(col, target_rank):
                U[:, c] = np.random.choice([-1, 0, 1], size=nm)
                V[:, c] = np.random.choice([-1, 0, 1], size=mp)
                W[:, c] = np.random.choice([-1, 0, 1], size=pn)
        
        # Verify
        valid = True
        for i in range(n):
            if not valid:
                break
            for j in range(m):
                if not valid:
                    break
                for k in range(p):
                    dot = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    if abs(dot - 1.0) > 1e-8:
                        valid = False
                        break
        
        if valid:
            return {
                "U": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U],
                "V": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V],
                "W": [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
            }
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with single Strassen embedding (rank 124)
    best = _embed_strassen_2x2_into_5x5()
    best_rank = len(best["U"][0])
    
    # Try block decomposition (rank 121)
    if time.time() < deadline - 20:
        block_decomp = _construct_block_decomposition()
        U, V, W = block_decomp["U"], block_decomp["V"], block_decomp["W"]
        rank = len(U[0])
        if rank < best_rank and _verify_float(n, m, p, U, V, W):
            if _verify_exact(n, m, p, U, V, W):
                best = block_decomp
                best_rank = rank
    
    # Try direct search for lower ranks (unreliable but worth a shot)
    if time.time() < deadline - 30 and best_rank > 100:
        for target_rank in range(best_rank - 1, max(90, best_rank - 10), -1):
            if time.time() > deadline - 30:
                break
            result = _try_direct_search(n, m, p, target_rank, timeout=5)
            if result is not None:
                U, V, W = result["U"], result["V"], result["W"]
                if _verify_float(n, m, p, U, V, W) and _verify_exact(n, m, p, U, V, W):
                    best = result
                    best_rank = target_rank
                    break
    
    return best
# EVOLVE-BLOCK-END
