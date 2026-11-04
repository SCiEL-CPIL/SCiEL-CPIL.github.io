import time
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """Return a valid tensor decomposition for 4x4 by 4x5 matrix multiplication."""
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # Construct Strassen 4x4x4 decomposition (rank 49) and extend to 4x4x5
    try:
        result_444 = _strassen_4x4x4()
        result = _extend_to_445(result_444, n, m, p)
        
        if not _verify_decomposition(result, n, m, p):
            return _schoolbook(n, m, p)
        
        # Try to reduce rank by finding and removing redundant terms
        if time.time() - start_time < 120:
            reduced = _reduce_rank_greedy(result, n, m, p, start_time)
            if reduced is not None:
                return reduced
        
        return result
    except Exception:
        return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
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

def _reduce_rank_greedy(decomp, n, m, p, start_time):
    """
    Greedily try to remove terms from the decomposition if they are redundant.
    A term r is redundant if its contribution can be expressed as a linear combination
    of other terms. We check this by seeing if removing term r still allows
    reconstruction of the tensor.
    """
    U = [row[:] for row in decomp["U"]]
    V = [row[:] for row in decomp["V"]]
    W = [row[:] for row in decomp["W"]]
    rank = len(U[0])
    
    # Build target tensor values
    n_rows_u = n * m
    n_rows_v = m * p
    n_rows_w = p * n
    
    # Store the tensor as a dict for efficient lookup
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i*m+j, j*p+k, k*n+i)] = 1.0
    
    # Try removing each term (in order of last added first, as they're more likely redundant)
    for r_remove in range(rank - 1, -1, -1):
        if time.time() - start_time > 135:
            break
        
        # Check if term r_remove is needed
        # Compute tensor without this term
        # T_recon[a,b,c] = sum_{r != r_remove} U[a,r]*V[b,r]*W[c,r]
        # We need T_recon[a,b,c] == T[a,b,c] for all valid (a,b,c)
        
        # Only check valid positions (where T=1) and a sample of invalid positions
        valid = True
        for (a, b, c), target_val in T.items():
            recon = 0.0
            for r in range(rank):
                if r == r_remove:
                    continue
                recon += U[a][r] * V[b][r] * W[c][r]
            if abs(recon - target_val) > 1e-9:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check a few invalid positions
        import random
        rng = random.Random(r_remove)
        for _ in range(20):
            a = rng.randint(0, n_rows_u - 1)
            b = rng.randint(0, n_rows_v - 1)
            c = rng.randint(0, n_rows_w - 1)
            i = a // m; j = a % m
            j2 = b // p; k = b % p
            k2 = c // n; i2 = c % n
            if i == i2 and j == j2 and k == k2:
                continue
            recon = 0.0
            for r in range(rank):
                if r == r_remove:
                    continue
                recon += U[a][r] * V[b][r] * W[c][r]
            if abs(recon) > 1e-9:
                valid = False
                break
        
        if valid:
            # Remove term r_remove
            for a in range(n_rows_u):
                U[a].pop(r_remove)
            for b in range(n_rows_v):
                V[b].pop(r_remove)
            for c in range(n_rows_w):
                W[c].pop(r_remove)
            rank -= 1
    
    if rank < len(decomp["U"][0]):
        return {"U": U, "V": V, "W": W}
    return None

def _strassen_2x2x2():
    """Return U, V, W (each 4x7) for Strassen 2x2x2 matrix multiplication."""
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    U[0][2] = 1
    V[2][2] = 1; V[0][2] = -1
    W[2][2] = 1; W[3][2] = 1
    U[3][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1
    U[1][5] = 1; U[3][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    U[0][6] = 1; U[2][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W

def _strassen_4x4x4():
    """Construct rank 49 decomposition for 4x4x4."""
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    U2, V2, W2 = _strassen_2x2x2()
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
            for ci in range(2):
                for ck in range(2):
                    coeff_c = c_coeffs.get((ci, ck), 0)
                    if coeff_c == 0: continue
                    for li in range(2):
                        for lk in range(2):
                            local_w = W2[lk*2+li][s] 
                            if local_w == 0: continue
                            global_k = ck*2 + lk
                            global_i = ci*2 + li
                            W[global_k*4 + global_i][col] += coeff_c * local_w
    return {"U": U, "V": V, "W": W}

def _extend_to_445(decomp_444, n, m, p):
    """Extend 4x4x4 decomposition to 4x4x5."""
    U4 = decomp_444["U"]
    V4 = decomp_444["V"]
    W4 = decomp_444["W"]
    rank_444 = len(U4[0])
    new_rank = rank_444 + 4
    U = [[0] * new_rank for _ in range(n * m)]
    V = [[0] * new_rank for _ in range(m * p)]
    W = [[0] * new_rank for _ in range(p * n)]
    for i in range(16):
        for r in range(rank_444):
            U[i][r] = U4[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(rank_444):
                V[j*5 + k][r] = V4[j*4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(rank_444):
                W[k*4 + i][r] = W4[k*4 + i][r]
    for j in range(4):
        col = rank_444 + j
        for i in range(4):
            U[i*4 + j][col] = 1
        V[j*5 + 4][col] = 1
        for i in range(4):
            W[4*4 + i][col] = 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p, num_checks=100):
    """Verify the decomposition."""
    import random
    U = decomp["U"]
    V = decomp["V"]
    W = decomp["W"]
    rank = len(U[0]) if U else 0
    
    # Check all valid (i,j,k) triples give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check some invalid positions give 0
    random.seed(42)
    for _ in range(num_checks):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        i = a // m; j = a % m
        j2 = b // p; k = b % p
        k2 = c // n; i2 = c % n
        if i == i2 and j == j2 and k == k2:
            continue
        val = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True

def _try_optimized_extension(decomp_444, n, m, p):
    """Try to find a lower-rank extension by sharing terms."""
    # This is a placeholder for optimization attempts
    # For now, return the standard extension
    return _extend_to_445(decomp_444, n, m, p)
# EVOLVE-BLOCK-END
