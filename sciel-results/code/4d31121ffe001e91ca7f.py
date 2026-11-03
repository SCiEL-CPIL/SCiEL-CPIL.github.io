"""Matrix multiplication 4,4,7 - rank-minimal tensor decomposition."""
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Construct a valid decomposition for 4x4 * 4x7 = 4x7.
    
    Strategy: Strassen for 4x4*4x4 (rank 49) + schoolbook for 4x4*4x3 (rank 48)
    Total rank: 97
    """
    n, m, p = 4, 4, 7
    rank = 97
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    col = 0
    
    # Part 1: Strassen 4x4*4x4 for B columns 0-3
    # 7 Strassen block-products, each 2x2*2x2 with rank 7
    
    # Strassen 2x2*2x2 rank-1 terms (U_pat, V_pat, W_pat)
    strassen_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,1]),   # M2
        ([1,0,0,0], [0,0,1,-1], [0,1,0,1]),  # M3
        ([0,0,0,1], [1,0,-1,0], [1,0,1,0]),  # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # M6
        ([0,1,-1,0], [0,0,1,1], [1,0,0,0]),  # M7
    ]
    
    def add_2x2_block(A_defs, B_defs, C_defs):
        """Add 2x2*2x2 Strassen. A_defs: [(bi,bj,sign)], B_defs: [(bj,sign)], C_defs: [(ci,cj,sign)]"""
        nonlocal col
        for term in strassen_2x2:
            Up, Vp, Wp = term
            for i2 in range(4):
                if Up[i2]:
                    ii, ij = i2//2, i2%2
                    for bi, bj, sa in A_defs:
                        U[(bi*2+ii)*4 + (bj*2+ij)][col] += Up[i2]*sa
            for j2 in range(4):
                if Vp[j2]:
                    jj, jk = j2//2, j2%2
                    for bj, sb in B_defs:
                        V[(bj*2+jj)*7 + jk][col] += Vp[j2]*sb
            for k2 in range(4):
                if Wp[k2]:
                    kk, ki = k2//2, k2%2
                    for ci, cj, sc in C_defs:
                        W[(ci*2+kk)*4 + (cj*2+ki)][col] += Wp[k2]*sc
            col += 1
    
    # M1=(A00+A11)(B00+B11) -> C00+=, C11+=
    add_2x2_block([(0,0,1),(1,1,1)], [(0,1),(1,1)], [(0,0,1),(1,1,1)])
    # M2=(A10+A11)B00 -> C10+=, C11+=
    add_2x2_block([(1,0,1),(1,1,1)], [(0,1)], [(1,0,1),(1,1,1)])
    # M3=A00(B10-B11) -> C01+=, C11+=
    add_2x2_block([(0,0,1)], [(1,1),(1,-1)], [(0,1,1),(1,1,1)])
    # M4=A11(B00-B10) -> C00+=, C10+=
    add_2x2_block([(1,1,1)], [(0,1),(1,-1)], [(0,0,1),(1,0,1)])
    # M5=(A00+A01)B11 -> C00-=, C01+=
    add_2x2_block([(0,0,1),(0,1,1)], [(1,1)], [(0,0,-1),(0,1,1)])
    # M6=(A10-A00)(B00+B01) -> C11+=
    add_2x2_block([(1,0,1),(0,0,-1)], [(0,1),(0,1)], [(1,1,1)])
    # M7=(A01-A11)(B10+B11) -> C00+=
    add_2x2_block([(0,1,1),(1,1,-1)], [(1,1),(1,1)], [(0,0,1)])
    
    # Part 2: Schoolbook for B columns 4-6
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4+j][col] = 1
                V[j*7+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
