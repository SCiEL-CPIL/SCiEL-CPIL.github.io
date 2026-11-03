import random
import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Use 3x3x3 partition with Laderman for 3x3x3 blocks
    # Partition: i=[0,3), j=[0,3) U [3,5), k=[0,3) U [3,6) U [6,7)
    # Blocks:
    #   3x3x3 at (i[0:3], j[0:3], k[0:3]) -> Laderman rank 23
    #   3x3x3 at (i[0:3], j[0:3], k[3:6]) -> Laderman rank 23
    #   3x2x3 at (i[0:3], j[3:5], k[0:3]) -> schoolbook rank 18
    #   3x2x3 at (i[0:3], j[3:5], k[3:6]) -> schoolbook rank 18
    #   3x3x1 at (i[0:3], j[0:3], k[6:7]) -> schoolbook rank 9
    #   3x2x1 at (i[0:3], j[3:5], k[6:7]) -> schoolbook rank 6
    # Total rank: 2*23 + 2*18 + 9 + 6 = 46 + 36 + 9 + 6 = 97
    
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]  # 15 x 97
    V = [[0] * rank for _ in range(m * p)]  # 35 x 97
    W = [[0] * rank for _ in range(p * n)]  # 21 x 97
    
    col = 0
    
    def add_laderman(i0, j0, k0):
        """Add Laderman 3x3x3 decomposition starting at (i0, j0, k0)."""
        nonlocal col
        
        # Local indices: a in {0,1,2}, b in {0,1,2}, c in {0,1,2}
        # Global: i = i0+a, j = j0+b, k = k0+c
        # U index: i*m + j = (i0+a)*m + (j0+b)
        # V index: j*p + k = (j0+b)*p + (k0+c)
        # W index: k*n + i = (k0+c)*n + (i0+a)
        
        # Define the 23 products as (U_terms, V_terms)
        # U_terms: list of (a, b, coeff)
        # V_terms: list of (b, c, coeff)
        
        products = [
            # p1
            ([(0,0,1), (1,1,1), (2,2,1)], [(0,0,1), (1,1,1), (2,2,1)]),
            # p2
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # p3
            ([(0,0,1), (1,0,1), (2,0,1)], [(0,0,1), (1,0,1), (2,0,1)]),
            # p4
            ([(1,0,1), (1,1,1), (1,2,1)], [(1,0,1), (1,1,1), (1,2,1)]),
            # p5
            ([(2,0,1), (2,1,1), (2,2,1)], [(2,0,1), (2,1,1), (2,2,1)]),
            # p6
            ([(0,1,1), (1,2,1), (2,0,1)], [(0,1,1), (1,2,1), (2,0,1)]),
            # p7
            ([(0,2,1), (1,0,1), (2,1,1)], [(0,2,1), (1,0,1), (2,1,1)]),
            # p8
            ([(0,0,1), (1,1,-1), (2,2,1)], [(0,0,1), (1,1,-1), (2,2,1)]),
            # p9
            ([(0,0,1), (1,1,-1)], [(0,0,1), (1,1,-1)]),
            # p10
            ([(1,0,1), (2,0,-1)], [(1,0,1), (2,0,-1)]),
            # p11
            ([(0,0,1), (0,1,1), (0,2,1)], [(0,0,1), (1,0,1), (2,0,1)]),
            # p12
            ([(1,0,1), (1,1,1), (1,2,1)], [(0,1,1), (1,1,1), (2,1,1)]),
            # p13
            ([(2,0,1), (2,1,1), (2,2,1)], [(0,2,1), (1,2,1), (2,2,1)]),
            # p14
            ([(0,1,1), (0,2,1), (1,0,1)], [(0,1,1), (1,2,1), (2,0,1)]),
            # p15
            ([(0,2,1), (1,1,1), (2,0,1)], [(0,2,1), (1,0,1), (2,1,1)]),
            # p16
            ([(0,0,1), (1,2,1), (2,1,1)], [(0,0,1), (1,2,1), (2,1,1)]),
            # p17
            ([(1,0,1), (2,2,1), (0,1,1)], [(1,0,1), (2,2,1), (0,1,1)]),
            # p18
            ([(2,0,1), (0,1,1), (1,2,1)], [(2,0,1), (0,1,1), (1,2,1)]),
            # p19
            ([(0,0,1), (1,2,-1), (2,1,1)], [(0,0,1), (1,2,-1), (2,1,1)]),
            # p20
            ([(1,0,1), (2,2,-1), (0,1,1)], [(1,0,1), (2,2,-1), (0,1,1)]),
            # p21
            ([(2,0,1), (0,1,-1), (1,2,1)], [(2,0,1), (0,1,-1), (1,2,1)]),
            # p22
            ([(0,1,1), (0,2,-1), (1,0,1)], [(0,1,1), (1,2,-1), (2,0,1)]),
            # p23
            ([(0,2,1), (1,0,-1), (2,1,1)], [(0,2,1), (1,0,-1), (2,1,1)]),
        ]
        
        # W coefficients for each output c[a][k] = sum of products
        # c00 = p1 - p2 + p3 - p8 + p9 - p10 + p11 - p16 + p19
        # c01 = p6 - p7 + p8 - p9 + p10 - p11 + p12 - p17 + p20
        # c02 = p7 - p6 + p9 - p8 + p11 - p10 + p13 - p18 + p21
        # c10 = p3 - p4 + p8 - p9 + p10 - p13 + p14 - p15 + p22
        # c11 = p1 - p2 + p4 - p5 + p6 - p7 + p14 - p15 + p16 - p17 + p18 - p19 + p20 - p21
        # c12 = p7 - p6 + p5 - p4 + p9 - p8 + p13 - p12 + p15 - p14 + p17 - p16 + p18 - p19 + p20 - p21
        # c20 = p1 - p3 + p4 - p8 + p9 - p10 + p13 - p14 + p15
        # c21 = p1 - p3 + p4 - p8 + p9 - p10 + p12 - p13 + p14 - p15
        # c22 = p1 - p5 + p3 - p4 + p8 - p9 + p10 - p11 + p16 - p17 + p18
        
        w_coeffs = {
            (0,0): [(0,1), (1,-1), (2,1), (7,-1), (8,1), (9,-1), (10,1), (15,-1), (18,1)],
            (0,1): [(5,1), (6,-1), (7,1), (8,-1), (9,1), (10,-1), (11,1), (16,-1), (19,1)],
            (0,2): [(6,1), (5,-1), (8,1), (7,-1), (10,1), (9,-1), (12,1), (17,-1), (20,1)],
            (1,0): [(2,1), (3,-1), (7,1), (8,-1), (9,1), (12,-1), (13,1), (14,-1), (21,1)],
            (1,1): [(0,1), (1,-1), (3,1), (4,-1), (5,1), (6,-1), (13,1), (14,-1), (15,1), (16,-1), (17,1), (18,-1), (19,1), (20,-1)],
            (1,2): [(6,1), (5,-1), (4,1), (3,-1), (8,1), (7,-1), (12,1), (11,-1), (14,1), (13,-1), (16,1), (15,-1), (17,1), (18,-1), (19,1), (20,-1)],
            (2,0): [(0,1), (2,-1), (3,1), (7,-1), (8,1), (9,-1), (12,1), (13,-1), (14,1)],
            (2,1): [(0,1), (2,-1), (3,1), (7,-1), (8,1), (9,-1), (11,1), (12,-1), (13,1), (14,-1)],
            (2,2): [(0,1), (4,-1), (2,1), (3,-1), (7,1), (8,-1), (9,1), (10,-1), (15,1), (16,-1), (17,1)],
        }
        
        for prod_idx in range(23):
            u_terms, v_terms = products[prod_idx]
            
            for (a, b, coeff) in u_terms:
                u_idx = (i0 + a) * m + (j0 + b)
                U[u_idx][col] += coeff
            
            for (b, c, coeff) in v_terms:
                v_idx = (j0 + b) * p + (k0 + c)
                V[v_idx][col] += coeff
            
            col += 1
        
        # Now set W coefficients
        for (a, k_local), coeff_list in w_coeffs.items():
            for prod_idx, coeff in coeff_list:
                w_idx = (k0 + k_local) * n + (i0 + a)
                W[w_idx][prod_idx + (col - 23)] += coeff
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        nonlocal col
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    U[idx_u][col] = 1
                    V[idx_v][col] = 1
                    W[idx_w][col] = 1
                    col += 1
    
    # Add Laderman blocks
    add_laderman(0, 0, 0)   # i[0:3], j[0:3], k[0:3]
    add_laderman(0, 0, 3)   # i[0:3], j[0:3], k[3:6]
    
    # Add schoolbook blocks
    add_schoolbook(0, 3, 3, 5, 0, 3)   # i[0:3], j[3:5], k[0:3] -> 18
    add_schoolbook(0, 3, 3, 5, 3, 6)   # i[0:3], j[3:5], k[3:6] -> 18
    add_schoolbook(0, 3, 0, 3, 6, 7)   # i[0:3], j[0:3], k[6:7] -> 9
    add_schoolbook(0, 3, 3, 5, 6, 7)   # i[0:3], j[3:5], k[6:7] -> 6
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    
    assert rank == 97, f"Expected rank 97, got {rank}"
    assert len(result['U']) == n * m
    assert len(result['V']) == m * p
    assert len(result['W']) == p * n
    
    import random
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_decomp = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            val = 0
            for r in range(rank):
                ua = 0
                for jj in range(m):
                    ua += result['U'][i * m + jj][r] * A[i][jj]
                vb = 0
                for jj in range(m):
                    vb += result['V'][jj * p + k][r] * B[jj][k]
                val += result['W'][k * n + i][r] * ua * vb
            C_decomp[i][k] = val
            
    if C_decomp != C:
        print("Verification failed!", file=sys.stderr)
        for i in range(n):
            for k in range(p):
                if C_decomp[i][k] != C[i][k]:
                    print(f"  C[{i}][{k}]: expected {C[i][k]}, got {C_decomp[i][k]}", file=sys.stderr)
    else:
        print("Verification passed.", file=sys.stderr)
