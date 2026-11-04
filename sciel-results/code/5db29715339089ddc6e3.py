import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 130.0

# EVOLVE-BLOCK-START

def _strassen_4x4x4_UV():
    """Return U, V for Strassen 4x4x4 (rank 49)."""
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    
    # Re-use the logic from the current program to generate U, V for 4x4x4
    # But we only need U and V to be fixed.
    
    U2 = [[0] * 7 for _ in range(4)]
    V2 = [[0] * 7 for _ in range(4)]
    W2 = [[0] * 7 for _ in range(4)]
    
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    U2[0][2] = 1
    V2[2][2] = 1; V2[0][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    U2[3][3] = 1
    V2[1][3] = 1; V2[3][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[2][4] = 1; W2[0][4] = -1
    
    U2[1][5] = 1; U2[3][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    U2[0][6] = 1; U2[2][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(1,0): 1, (0,0): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(0,1): 1, (1,1): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for r, (a_coeffs, b_coeffs, c_coeffs) in enumerate(steps):
        for s in range(7):
            col = r * 7 + s
            
            for bi in range(2):
                for bj in range(2):
                    coeff_a = a_coeffs.get((bi, bj), 0)
                    if coeff_a == 0: continue
                    for li in range(2):
                        for lj in range(2):
                            local_u = U2[li*2+lj][s]
                            if local_u == 0: continue
                            global_i = bi*2 + li
                            global_j = bj*2 + lj
                            U[global_i*4 + global_j][col] += coeff_a * local_u
            
            for bj in range(2):
                for bk in range(2):
                    coeff_b = b_coeffs.get((bj, bk), 0)
                    if coeff_b == 0: continue
                    for lj in range(2):
                        for lk in range(2):
                            local_v = V2[lj*2+lk][s]
                            if local_v == 0: continue
                            global_j = bj*2 + lj
                            global_k = bk*2 + lk
                            V[global_j*4 + global_k][col] += coeff_b * local_v
                            
    return U, V

def _construct_445_rank49():
    n, m, p = 4, 4, 5
    U, V = _strassen_4x4x4_UV()
    rank = 49
    
    W = [[0] * rank for _ in range(p * n)]
    
    # For each k, solve for W_k
    for k in range(p):
        # W_k is a 4 x rank matrix (4 rows for i, rank columns for r)
        # We solve for each j separately, but the unknowns W_k[i, r] are shared across j?
        # No, the equation is:
        # sum_r W_k[i, r] * U[i*m+j, r] * V[j*p+k, r] = T[i, j, k]
        # T[i, j, k] is 1 if (i,j,k) is a valid index in the tensor.
        # In our case, T[i, j, k] = 1 for all i, j, k.
        
        # Let's form the system for all i, j.
        # Unknowns: W_k[i, r] for i=0..3, r=0..48. Total 196 unknowns.
        # Equations: for each i, j. Total 16 equations.
        
        # We can use np.linalg.lstsq.
        # Construct A matrix (16 x 196) and b vector (16).
        # Indexing: eq_idx = i * m + j.
        # var_idx = i * rank + r.
        
        A = np.zeros((n * m, n * rank))
        b = np.zeros(n * m)
        
        for i in range(n):
            for j in range(m):
                eq_idx = i * m + j
                b[eq_idx] = 1.0 # T[i, j, k] is 1
                
                for r in range(rank):
                    var_idx = i * rank + r
                    A[eq_idx, var_idx] = U[i * m + j][r] * V[j * p + k][r]
        
        # Solve A x = b
        # x is the vector of W_k[i, r]
        x, residuals, rank_sv, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Fill W
        for i in range(n):
            for r in range(rank):
                W[k * n + i][r] = x[i * rank + r]
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p):
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0])
    
    # Check all valid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
                    
    # Check invalid triples (sample)
    import random
    random.seed(42)
    for _ in range(100):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue
        val = 0.0
        for r in range(rank):
            val += U[a][r] * V[b][r] * W[c][r]
        if abs(val) > 1e-9:
            return False
            
    return True

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    try:
        result = _construct_445_rank49()
        if _verify_decomposition(result, n, m, p):
            return result
    except Exception as e:
        pass
    
    # Fallback to rank 50 (previous method)
    # ... (copy previous fallback logic) ...
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Re-use Strassen 4x4x4
    U4, V4 = _strassen_4x4x4_UV()
    # Note: _strassen_4x4x4_UV returns U, V for 4x4x4.
    # We need to map them to 4x4x5 indices.
    
    for i in range(16):
        for r in range(49):
            U[i][r] = U4[i][r]
            
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j*5 + k][r] = V4[j*4 + k][r]
                
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k*4 + i][r] = 0 # W from 4x4x4 is not directly usable if we change U, V?
                # Actually, if we use the same U, V, the W for k=0..3 should be the same as 4x4x4 W.
                # But we don't have W from _strassen_4x4x4_UV.
                # We can re-derive it or just use the rank 50 extension logic.
                
    # Simpler fallback: Rank 80 schoolbook
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

# EVOLVE-BLOCK-END
