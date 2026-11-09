"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time

DEADLINE_SECONDS = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _deadline_exceeded(margin=5.0):
    return _elapsed() > (DEADLINE_SECONDS - margin)

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x4 matrix.
    
    We use the Strassen algorithm for 2x2 matrix multiplication as a building block.
    Each 2x2 multiplication has rank 7. We decompose the 2x4 * 4x4 multiplication
    into four 2x2 * 2x2 multiplications, giving total rank 4*7 = 28.
    
    U: (n*m, rank) = (8, 28)
    V: (m*p, rank) = (16, 28)  
    W: (p*n, rank) = (8, 28)
    """
    # Deadline guard: if we're close to the limit, return immediately with the
    # deterministic Strassen result (which is always valid and fast).
    if _deadline_exceeded(margin=10.0):
        # Fallback: return a simple schoolbook rank-32 decomposition
        n, m, p = 2, 4, 4
        rank = n * m * p  # 32
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
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # For matrices A (2x2) and B (2x2), C = A*B:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # Index conventions:
    # U[i*m + j, c] for i in [0,2), j in [0,4)
    # V[j*p + k, c] for j in [0,4), k in [0,4)
    # W[k*n + i, c] for k in [0,4), i in [0,2)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen terms: each term has (u_terms, v_terms, w_terms)
    # u_terms: list of (local_i, local_j, coeff) where local_i in {0,1}, local_j in {0,1}
    # v_terms: list of (local_j, local_k, coeff) where local_j in {0,1}, local_k in {0,1}
    # w_terms: list of (local_k, local_i, coeff) where local_k in {0,1}, local_i in {0,1}
    
    # M1 = (a+d)(e+h)
    # U: a at (0,0), d at (1,1)
    # V: e at (0,0), h at (1,1)
    # W: M1 contributes to C[0,0] and C[1,1]
    # C[0,0] = M1 + M4 - M5 + M7
    # C[1,1] = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        {
            'U': [(0, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]  # contributes to W[0*2+0]=W[0] and W[1*2+1]=W[3]
        },
        # M2 = (c+d)*e
        {
            'U': [(1, 0, 1), (1, 1, 1)],
            'V': [(0, 0, 1)],
            'W': [(0, 1, 1), (1, 1, -1)]  # C[1,0] += M2, C[1,1] -= M2
        },
        # M3 = a*(f-h)
        {
            'U': [(0, 0, 1)],
            'V': [(0, 1, 1), (1, 1, -1)],
            'W': [(1, 0, 1), (1, 1, 1)]  # C[0,1] += M3, C[1,1] += M3
        },
        # M4 = d*(g-e)
        {
            'U': [(1, 1, 1)],
            'V': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (0, 1, 1)]  # C[0,0] += M4, C[1,0] += M4
        },
        # M5 = (a+b)*h
        {
            'U': [(0, 0, 1), (0, 1, 1)],
            'V': [(1, 1, 1)],
            'W': [(0, 0, -1), (1, 0, 1)]  # C[0,0] -= M5, C[0,1] += M5
        },
        # M6 = (c-a)*(e+f)
        {
            'U': [(1, 0, 1), (0, 0, -1)],
            'V': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]  # C[1,1] += M6
        },
        # M7 = (b-d)*(g+h)
        {
            'U': [(0, 1, 1), (1, 1, -1)],
            'V': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]  # C[0,0] += M7
        }
    ]
    
    # The 2x4 matrix A has rows i=0,1 and columns j=0,1,2,3
    # The 4x4 matrix B has rows j=0,1,2,3 and columns k=0,1,2,3
    # We split A into two 2x2 blocks: A_left (j=0,1) and A_right (j=2,3)
    # We split B into four 2x2 blocks: B_00 (j=0,1; k=0,1), B_01 (j=0,1; k=2,3),
    #                                   B_10 (j=2,3; k=0,1), B_11 (j=2,3; k=2,3)
    # C = A*B = A_left * B_00 + A_left * B_01 + A_right * B_10 + A_right * B_11
    # Each term is a 2x2 * 2x2 multiplication
    
    # Block definitions: (j_offset, k_offset)
    # j_offset: which 2x2 block of A (0 or 2)
    # k_offset: which 2x2 block of B columns (0 or 2)
    blocks = [
        (0, 0),  # A_left * B_00
        (0, 2),  # A_left * B_01
        (2, 0),  # A_right * B_10
        (2, 2),  # A_right * B_11
    ]
    
    col = 0
    for j_off, k_off in blocks:
        for term in strassen_terms:
            # Add to U
            for (li, lj, coeff) in term['U']:
                global_i = li
                global_j = lj + j_off
                U[global_i * m + global_j][col] += coeff
            
            # Add to V
            for (lj, lk, coeff) in term['V']:
                global_j = lj + j_off
                global_k = lk + k_off
                V[global_j * p + global_k][col] += coeff
            
            # Add to W
            for (lk, li, coeff) in term['W']:
                global_k = lk + k_off
                global_i = li
                W[global_k * n + global_i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
