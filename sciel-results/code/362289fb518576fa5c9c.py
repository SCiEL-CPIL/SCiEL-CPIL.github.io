"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    # Rank breakdown:
    # j=2 part: rank 2
    # j=0,1 part:
    #   k=0,1: Strassen 2x2, rank 7
    #   k=2,3: Strassen 2x2, rank 7
    #   k=4: Schoolbook 2x2x2x1, rank 4
    total_rank = 2 + 7 + 7 + 4
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # 1. Handle j=2 part (A[:, 2] * B[2, :])
    # Rank 2 decomposition
    # r=0 covers i=0, r=1 covers i=1
    # U[i, 2, r] = 1 if i==r else 0? No, we need sum to be 1 for all i.
    # My previous logic:
    # r=0: U[0,2,0]=1, U[1,2,0]=1. V[2,k,0]=1. W[k,0,0]=1, W[k,1,0]=0.
    # r=1: U[0,2,1]=0, U[1,2,1]=1. V[2,k,1]=1. W[k,0,1]=0, W[k,1,1]=1.
    # Check:
    # i=0: 1*1*1 + 0*1*0 = 1.
    # i=1: 1*1*0 + 1*1*1 = 1.
    # This works for all k.
    
    for k in range(p):
        # r=0
        U[u_idx(0, 2)][0] = 1
        U[u_idx(1, 2)][0] = 1
        V[v_idx(2, k)][0] = 1
        W[w_idx(k, 0)][0] = 1
        W[w_idx(k, 1)][0] = 0
        
        # r=1
        U[u_idx(0, 2)][1] = 0
        U[u_idx(1, 2)][1] = 1
        V[v_idx(2, k)][1] = 1
        W[w_idx(k, 0)][1] = 0
        W[w_idx(k, 1)][1] = 1
        
    # 2. Handle j=0,1 part
    # A_L is 2x2, B_L is 2x5
    # Split B_L into B_0 (cols 0,1), B_1 (cols 2,3), B_2 (col 4)
    
    # Strassen coefficients for 2x2
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    def add_strassen_block(base_r, k_start):
        # k_start is 0 or 2
        # B block is cols k_start, k_start+1
        # A block is cols 0, 1 (j=0,1)
        # A indices: i in 0,1; j in 0,1
        # B indices: j in 0,1; k in k_start, k_start+1
        
        # P1
        r = base_r
        # U: a+d -> U[0,0]=1, U[1,1]=1
        U[u_idx(0, 0)][r] += 1
        U[u_idx(1, 1)][r] += 1
        # V: e+h -> V[0, k_start]=1, V[1, k_start+1]=1
        V[v_idx(0, k_start)][r] += 1
        V[v_idx(1, k_start + 1)][r] += 1
        # W: P1+P4-P5+P7 for C[0,0] (local), P1-P2+P3+P6 for C[1,0] (local)?
        # Local C[0,0] corresponds to global C[0, k_start]
        # Local C[0,1] corresponds to global C[0, k_start+1]
        # Local C[1,0] corresponds to global C[1, k_start]
        # Local C[1,1] corresponds to global C[1, k_start+1]
        
        # C[0,0] = P1 + P4 - P5 + P7
        W[w_idx(k_start, 0)][r] += 1
        W[w_idx(k_start, 1)][r] += 0 # P1 contributes to C[0,0] only? No.
        # Wait, P1 is a scalar. It contributes to multiple C entries.
        # C[0,0] gets P1. C[1,1] gets P1.
        # C[0,1] gets P3+P5.
        # C[1,0] gets P2+P4.
        
        # Let's map W contributions for each P
        # P1: C[0,0]+=1, C[1,1]+=1
        W[w_idx(k_start, 0)][r] += 1
        W[w_idx(k_start + 1, 1)][r] += 1
        
        # P2: C[1,0]+=1
        r = base_r + 1
        # U: c+d -> U[1,0]=1, U[1,1]=1
        U[u_idx(1, 0)][r] += 1
        U[u_idx(1, 1)][r] += 1
        # V: e -> V[0, k_start]=1
        V[v_idx(0, k_start)][r] += 1
        # W: C[1,0]+=1
        W[w_idx(k_start, 1)][r] += 1
        
        # P3: C[0,1]+=1
        r = base_r + 2
        # U: a -> U[0,0]=1
        U[u_idx(0, 0)][r] += 1
        # V: f-h -> V[1, k_start]=1, V[1, k_start+1]=-1
        V[v_idx(1, k_start)][r] += 1
        V[v_idx(1, k_start + 1)][r] += -1
        # W: C[0,1]+=1
        W[w_idx(k_start + 1, 0)][r] += 1
        
        # P4: C[0,0]-=1, C[1,0]+=1
        r = base_r + 3
        # U: d -> U[1,1]=1
        U[u_idx(1, 1)][r] += 1
        # V: g-e -> V[1, k_start]=1, V[0, k_start]=-1
        V[v_idx(1, k_start)][r] += 1
        V[v_idx(0, k_start)][r] += -1
        # W: C[0,0]-=1, C[1,0]+=1
        W[w_idx(k_start, 0)][r] += -1
        W[w_idx(k_start, 1)][r] += 1
        
        # P5: C[0,0]-=1, C[0,1]+=1
        r = base_r + 4
        # U: a+b -> U[0,0]=1, U[0,1]=1
        U[u_idx(0, 0)][r] += 1
        U[u_idx(0, 1)][r] += 1
        # V: h -> V[1, k_start+1]=1
        V[v_idx(1, k_start + 1)][r] += 1
        # W: C[0,0]-=1, C[0,1]+=1
        W[w_idx(k_start, 0)][r] += -1
        W[w_idx(k_start + 1, 0)][r] += 1
        
        # P6: C[1,1]+=1
        r = base_r + 5
        # U: c-a -> U[1,0]=1, U[0,0]=-1
        U[u_idx(1, 0)][r] += 1
        U[u_idx(0, 0)][r] += -1
        # V: e+f -> V[0, k_start]=1, V[1, k_start]=1
        V[v_idx(0, k_start)][r] += 1
        V[v_idx(1, k_start)][r] += 1
        # W: C[1,1]+=1
        W[w_idx(k_start + 1, 1)][r] += 1
        
        # P7: C[0,0]+=1, C[1,1]-=1
        r = base_r + 6
        # U: b-d -> U[0,1]=1, U[1,1]=-1
        U[u_idx(0, 1)][r] += 1
        U[u_idx(1, 1)][r] += -1
        # V: g+h -> V[1, k_start]=1, V[1, k_start+1]=1
        V[v_idx(1, k_start)][r] += 1
        V[v_idx(1, k_start + 1)][r] += 1
        # W: C[0,0]+=1, C[1,1]-=1
        W[w_idx(k_start, 0)][r] += 1
        W[w_idx(k_start + 1, 1)][r] += -1

    add_strassen_block(2, 0) # k=0,1
    add_strassen_block(9, 2) # k=2,3
    
    # 3. Handle k=4 part (B_L col 4)
    # Schoolbook 2x2 x 2x1
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    # Rank 4
    
    # r=16: A[0,0]B[0,4] -> C[0,4]
    r = 16
    U[u_idx(0, 0)][r] += 1
    V[v_idx(0, 4)][r] += 1
    W[w_idx(4, 0)][r] += 1
    
    # r=17: A[0,1]B[1,4] -> C[0,4]
    r = 17
    U[u_idx(0, 1)][r] += 1
    V[v_idx(1, 4)][r] += 1
    W[w_idx(4, 0)][r] += 1
    
    # r=18: A[1,0]B[0,4] -> C[1,4]
    r = 18
    U[u_idx(1, 0)][r] += 1
    V[v_idx(0, 4)][r] += 1
    W[w_idx(4, 1)][r] += 1
    
    # r=19: A[1,1]B[1,4] -> C[1,4]
    r = 19
    U[u_idx(1, 1)][r] += 1
    V[v_idx(1, 4)][r] += 1
    W[w_idx(4, 1)][r] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
