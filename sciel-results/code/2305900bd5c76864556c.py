import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    
    # Rank 56 decomposition based on 8 independent 2x2x2 Strassen blocks.
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * total_rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * total_rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen 2x2x2 decomposition (rank 7)
    # Local indices: A: i*2+j, B: j*2+k, C: i*2+k
    # U_local[4][7], V_local[4][7], W_local[4][7]
    U_local = [[0]*7 for _ in range(4)]
    V_local = [[0]*7 for _ in range(4)]
    W_local = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U_local[0][0]=1; U_local[3][0]=1
    V_local[0][0]=1; V_local[3][0]=1
    W_local[0][0]=1; W_local[3][0]=1
    
    # M2: (A10+A11)B00
    U_local[2][1]=1; U_local[3][1]=1
    V_local[0][1]=1
    W_local[2][1]=1; W_local[3][1]=-1
    
    # M3: A00(B10-B00)
    U_local[0][2]=1
    V_local[1][2]=1; V_local[3][2]=-1
    W_local[1][2]=1; W_local[3][2]=1
    
    # M4: A11(B01-B11)
    U_local[3][3]=1
    V_local[2][3]=1; V_local[0][3]=-1
    W_local[0][3]=1; W_local[2][3]=1
    
    # M5: (A00+A01)B11
    U_local[0][4]=1; U_local[1][4]=1
    V_local[3][4]=1
    W_local[0][4]=-1; W_local[1][4]=1
    
    # M6: (A10-A00)(B00+B01)
    U_local[2][5]=1; U_local[0][5]=-1
    V_local[0][5]=1; V_local[1][5]=1
    W_local[3][5]=1
    
    # M7: (A01-A11)(B10+B11)
    U_local[1][6]=1; U_local[3][6]=-1
    V_local[2][6]=1; V_local[3][6]=1
    W_local[0][6]=1

    # 8 Strassen blocks: j=0..3 (col-block), s=0..1 (row-block of B / col-block of A)
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            # A mapping: local (i,j_loc) -> global (i, 2*j + s*2 + j_loc)
            a_map = [0]*4
            for i in range(2):
                for jl in range(2):
                    a_map[i*2+jl] = i * 4 + (2*j + s*2 + jl)
            
            # B mapping: local (j_loc,k_loc) -> global (s*2+j_loc, 2*j+k_loc)
            b_map = [0]*4
            for jl in range(2):
                for kl in range(2):
                    b_map[jl*2+kl] = (s*2 + jl) * 8 + (2*j + kl)
            
            # W mapping: local (i,k_loc) -> global (2*j+k_loc, i) [W index = k*n+i]
            c_map = [0]*4
            for i in range(2):
                for kl in range(2):
                    c_map[i*2+kl] = (2*j + kl) * 2 + i

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val:
                        U[a_map[la]][r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val:
                        V[b_map[lb]][r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val:
                        W[c_map[lc]][r] += val

    if time.time() - start_time > 140:
        pass
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
