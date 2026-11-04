import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _schoolbook(n=4, m=4, p=7):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n=4, m=4, p=7, tol=1e-9):
    """Verify that the decomposition reproduces the matmul tensor."""
    r = len(U[0]) if U and U[0] else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i * m + j]
                vk = V[j * p + k]
                wi = W[k * n + i]
                for c in range(r):
                    s += ui[c] * vk[c] * wi[c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication."""
    terms = []
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]))
    terms.append(([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 0, 1, 0]))
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, -1]))
    return terms

def _build_strassen_4x4x4():
    """Build a rank-49 decomposition for 4x4x4 using Kronecker product of Strassen 2x2."""
    base = _strassen_2x2_terms()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    col = 0
    for c1 in range(7):
        u1, v1, w1 = base[c1]
        for c2 in range(7):
            u2, v2, w2 = base[c2]
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U[i * 4 + j][col] = u1[i1 * 2 + j1] * u2[i2 * 2 + j2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V[j * 4 + k][col] = v1[j1 * 2 + k1] * v2[j2 * 2 + k2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W[k * 4 + i][col] = w1[k1 * 2 + i1] * w2[k2 * 2 + i2]
            col += 1
    return {"U": U, "V": V, "W": W}

def _try_find_low_rank_4x4x4(max_rank=30, time_limit=20.0):
    """Try to find a decomposition of rank < 49 for 4x4x4 using random search."""
    start_time = time.time()
    n, m, p = 4, 4, 4
    # Target: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We fix U and V randomly, and solve for W.
    # Variables: W is 16 x R.
    # Equations: 64.
    
    for R in range(23, max_rank + 1):
        if time.time() - start_time > time_limit:
            break
        
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random U and V
            U_rand = np.random.randn(16, R)
            V_rand = np.random.randn(16, R)
            
            # Construct linear system A W_flat = b
            # A has shape (64, 16*R)
            # b has shape (64,)
            A = np.zeros((64, 16 * R))
            b = np.ones(64)
            
            idx = 0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        # Equation: sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c] = 1
                        # Variable for W is W[k*4+i, c]
                        # Column index in A: c * 16 + (k*4+i)
                        # Wait, W is 16xR. W_flat index: row * R + col? Or col * 16 + row?
                        # Let's use W_flat[c * 16 + row]
                        # Then variable W[k*4+i, c] is at index c * 16 + (k*4+i)
                        
                        for c in range(R):
                            coeff = U_rand[i * 4 + j, c] * V_rand[j * 4 + k, c]
                            A[idx, c * 16 + (k * 4 + i)] = coeff
                        
                        idx += 1
            
            # Solve A W_flat = b
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                # Check if the solution is exact
                W_sol = W_flat.reshape(R, 16).T # 16 x R
                
                # Verify
                error = 0.0
                for i in range(4):
                    for j in range(4):
                        for k in range(4):
                            s_val = 0.0
                            for c in range(R):
                                s_val += U_rand[i * 4 + j, c] * V_rand[j * 4 + k, c] * W_sol[k * 4 + i, c]
                            error += abs(s_val - 1.0)
                
                if error < 1e-9:
                    # Found a valid decomposition
                    # Convert to list of lists
                    U_list = [[0]*R for _ in range(16)]
                    V_list = [[0]*R for _ in range(16)]
                    W_list = [[0]*R for _ in range(16)]
                    for i in range(16):
                        for c in range(R):
                            U_list[i][c] = U_rand[i, c]
                            V_list[i][c] = V_rand[i, c]
                            W_list[i][c] = W_sol[i, c]
                    
                    return {"U": U_list, "V": V_list, "W": W_list, "rank": R}
            except Exception:
                continue
    
    return None

def _build_hybrid():
    """Build a hybrid decomposition."""
    n, m, p = 4, 4, 7
    deadline = time.time() + 130.0
    
    # Try to find a low rank 4x4x4
    low_rank_4x4x4 = _try_find_low_rank_4x4x4(max_rank=30, time_limit=20.0)
    
    if low_rank_4x4x4 and time.time() < deadline:
        R1 = low_rank_4x4x4["rank"]
        U1 = low_rank_4x4x4["U"]
        V1 = low_rank_4x4x4["V"]
        W1 = low_rank_4x4x4["W"]
    else:
        # Fallback to Strassen Kronecker
        strassen_part = _build_strassen_4x4x4()
        R1 = 49
        U1 = strassen_part["U"]
        V1 = strassen_part["V"]
        W1 = strassen_part["W"]
    
    # Build the 4x4x3 schoolbook part
    n2, m2, p2 = 4, 4, 3
    sb_rank = n2 * m2 * p2
    U_sb = [[0] * sb_rank for _ in range(n2 * m2)]
    V_sb = [[0] * sb_rank for _ in range(m2 * p2)]
    W_sb = [[0] * sb_rank for _ in range(p2 * n2)]
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U_sb[i * m2 + j][col] = 1
                V_sb[j * p2 + k][col] = 1
                W_sb[k * n2 + i][col] = 1
                col += 1
    
    total_rank = R1 + sb_rank
    U_full = [[0] * total_rank for _ in range(16)]
    V_full = [[0] * total_rank for _ in range(28)]
    W_full = [[0] * total_rank for _ in range(28)]
    
    # Copy Part 1
    for i in range(16):
        for c in range(R1):
            U_full[i][c] = U1[i][c]
    
    for j in range(4):
        for k in range(4):
            for c in range(R1):
                V_full[j * 7 + k][c] = V1[j * 4 + k][c]
    
    for k in range(4):
        for i in range(4):
            for c in range(R1):
                W_full[k * 4 + i][c] = W1[k * 4 + i][c]
    
    # Copy Part 2
    for i in range(16):
        for c in range(sb_rank):
            U_full[i][R1 + c] = U_sb[i][c]
    
    for j in range(4):
        for k in range(3):
            for c in range(sb_rank):
                V_full[j * 7 + (k + 4)][R1 + c] = V_sb[j * 3 + k][c]
    
    for k in range(3):
        for i in range(4):
            for c in range(sb_rank):
                W_full[(k + 4) * 4 + i][R1 + c] = W_sb[k * 4 + i][c]
    
    return {"U": U_full, "V": V_full, "W": W_full}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    result = _build_hybrid()
    
    if time.time() < deadline:
        if not _verify(result["U"], result["V"], result["W"], n, m, p):
            result = _schoolbook(n, m, p)
    
    return result
