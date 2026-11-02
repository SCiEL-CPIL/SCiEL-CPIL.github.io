DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import time

def construct():
    """Rank-42 Strassen decomposition for 2x2 * 2x12 matrix multiplication.
    
    Uses the tensor product structure M_{2,2,12} \cong M_{2,2,2} \otimes M_{1,1,6}.
    Rank(M_{2,2,2}) = 7 (Strassen).
    Rank(M_{1,1,6}) = 6 (Dot product).
    Total rank = 7 * 6 = 42.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=12
    # A is n x m = 2x2, B is m x p = 2x12, C is n x p = 2x12
    # Partition p columns into pairs: 6 pairs
    num_pairs = p // 2  # 6
    
    rank = num_pairs * 7  # 42
    U = [[0] * rank for _ in range(n * m)]   # 4 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 24 x 42
    
    # A element to U row mapping: A[i][j] -> row i*m+j
    # A[0][0]->0, A[0][1]->1, A[1][0]->2, A[1][1]->3
    
    # B element to V row mapping: B[j][k] -> row j*p+k
    # B[0][k]->k, B[1][k]->p+k = 12+k
    
    # C element to W row mapping: C[i][k] -> W[k*n+i] (trace convention)
    # C[0][k]->k*2+0 = 2k, C[1][k]->k*2+1 = 2k+1
    
    # Strassen 7 products for 2x2 * 2x2
    # A elements: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B elements within block: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    
    products = [
        ([( (0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),       # M1=(A11+A22)(B11+B22)
        ([( (1,0), 1), ((1,1), 1)], [((0,0), 1)]),                    # M2=(A21+A22)B11
        ([( (0,0), 1)], [((0,1), 1), ((1,1), -1)]),                   # M3=A11(B12-B22)
        ([( (1,1), 1)], [((1,0), 1), ((0,0), -1)]),                   # M4=A22(B21-B11)
        ([( (0,0), 1), ((0,1), 1)], [((1,1), 1)]),                    # M5=(A11+A12)B22
        ([( (1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),       # M6=(A21-A11)(B11+B12)
        ([( (0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),       # M7=(A12-A22)(B21+B22)
    ]
    
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],     # M1: C11+=1, C22+=1
        [((1,0), 1), ((1,1), -1)],    # M2: C21+=1, C22+=-1
        [((0,1), 1), ((1,1), 1)],     # M3: C12+=1, C22+=1
        [((0,0), 1), ((1,0), 1)],     # M4: C11+=1, C21+=1
        [((0,0), -1), ((0,1), 1)],    # M5: C11+=-1, C12+=1
        [((1,1), 1)],                 # M6: C22+=1
        [((0,0), 1)],                 # M7: C11+=1
    ]
    
    col = 0
    for t in range(num_pairs):
        base_col = 2 * t
        for s in range(7):
            for (ai, aj), coeff in products[s][0]:
                u_row = ai * m + aj
                U[u_row][col] = coeff
            
            for (bj, bk_rel), coeff in products[s][1]:
                b_col = base_col + bk_rel
                v_row = bj * p + b_col
                V[v_row][col] = coeff
            
            for (ci, ck_rel), coeff in output_contribs[s]:
                c_col = base_col + ck_rel
                w_row = c_col * n + ci
                W[w_row][col] = coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
