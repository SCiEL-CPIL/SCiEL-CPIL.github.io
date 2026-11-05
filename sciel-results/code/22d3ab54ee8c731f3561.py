import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # 7 columns for one 2x2 * 2x2 block
        
        # m0: (x00+x11)(y00+y11) -> Z00+=1, Z11+=1
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m1: (x10+x11)y00 -> Z10+=1, Z11-=1
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # m2: x00(y01-y11) -> Z01+=1, Z11+=1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m3: x11(y10-y00) -> Z00+=1, Z10+=1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # m4: (x00+x01)y11 -> Z00-=1, Z01+=1
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # m5: (x10-x00)(y00+y01) -> Z11+=1
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m6: (x01-x11)(y10+y11) -> Z00+=1
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_schoolbook_2x1x2(i_base, j_base, k_base):
        nonlocal col
        # 4 columns for 2x1 * 1x2 block
        # A is 2x1 at (i_base, j_base)
        # B is 1x2 at (j_base, k_base)
        # C is 2x2 at (k_base, i_base)
        
        # C[0, k_base] += A[0, j_base] * B[j_base, k_base]
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        # C[1, k_base] += A[1, j_base] * B[j_base, k_base]
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # C[0, k_base+1] += A[0, j_base] * B[j_base, k_base+1]
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # C[1, k_base+1] += A[1, j_base] * B[j_base, k_base+1]
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Part 1: j in {0,1,2}, k in {0,1,2,3}
    # Sub-part 1a: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    
    # Sub-part 1b: j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    
    # Sub-part 1c: j=2, k in {0,1}
    add_schoolbook_2x1x2(0, 2, 0)
    
    # Sub-part 1d: j=2, k in {2,3}
    add_schoolbook_2x1x2(0, 2, 2)
    
    # Part 2: j in {3,4}, k in {4,5}
    # Sub-part 2a: j in {3,4}, k in {4,5}
    add_strassen_block(0, 3, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
