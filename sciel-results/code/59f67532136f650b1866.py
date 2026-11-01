"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def construct():
    """Strassen-based rank-42 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 6 pairs of 2. Apply 7-product Strassen to each 2x2 block.
    Total rank = 6 * 7 = 42.
    Note: Theoretical lower bound is 24. Rank-12 for 2x2x4 blocks could yield rank 36.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2  # 6
    
    rank = num_pairs * 7  # 42
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    # Strassen 7 products: (A_coeffs, B_coeffs)
    # A: (i,j) -> U row i*m+j
    # B: (j,k_rel) -> V row j*p + (base_col + k_rel)
    products = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),   # M1=(A11+A22)(B11+B22)
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),                 # M2=(A21+A22)B11
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),                 # M3=A11(B12-B22)
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),                 # M4=A22(B21-B11)
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),                  # M5=(A11+A12)B22
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),    # M6=(A21-A11)(B11+B12)
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),    # M7=(A12-A22)(B21+B22)
    ]
    
    # Output: for product s, list of ((ci, ck_rel), coeff)
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],    # M1: C11+=1, C22+=1
        [((1,0), 1), ((1,1), -1)],   # M2: C21+=1, C22+=-1
        [((0,1), 1), ((1,1), 1)],    # M3: C12+=1, C22+=1
        [((0,0), 1), ((1,0), 1)],    # M4: C11+=1, C21+=1
        [((0,0), -1), ((0,1), 1)],   # M5: C11+=-1, C12+=1
        [((1,1), 1)],                 # M6: C22+=1
        [((0,0), 1)],                 # M7: C11+=1
    ]
    
    col = 0
    for t in range(num_pairs):
        if time.time() > deadline:
            return {"U": U, "V": V, "W": W}
        base_col = 2 * t
        for s in range(7):
            for (ai, aj), coeff in products[s][0]:
                U[ai * m + aj][col] = coeff
            for (bj, bk_rel), coeff in products[s][1]:
                V[bj * p + base_col + bk_rel][col] = coeff
            for (ci, ck_rel), coeff in output_contribs[s]:
                W[(base_col + ck_rel) * n + ci][col] = coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
