import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 0
    U = []
    V = []
    W = []

    def add_block(i_start, i_end, j_start, j_end, k_start, k_end, rank_offset):
        nonlocal rank
        i_size = i_end - i_start
        j_size = j_end - j_start
        k_size = k_end - k_start
        
        # Schoolbook for this block
        for i in range(i_size):
            for j in range(j_size):
                for k in range(k_size):
                    col = rank_offset + rank
                    # U index: (i_start + i) * m + (j_start + j)
                    u_idx = (i_start + i) * m + (j_start + j)
                    # V index: (j_start + j) * p + (k_start + k)
                    v_idx = (j_start + j) * p + (k_start + k)
                    # W index: (k_start + k) * n + (i_start + i)
                    w_idx = (k_start + k) * n + (i_start + i)
                    
                    U.append([0] * (rank + 1))
                    V.append([0] * (rank + 1))
                    W.append([0] * (rank + 1))
                    
                    # This is inefficient. Let's build lists of columns.
        
    # Re-implement with column lists
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)
                    
    def add_rank7_2x2x2(i_start, j_start, k_start):
        # i_start, j_start, k_start are the top-left indices of the 2x2x2 block
        # U indices: i in {i_start, i_start+1}, j in {j_start, j_start+1}
        # V indices: j in {j_start, j_start+1}, k in {k_start, k_start+1}
        # W indices: k in {k_start, k_start+1}, i in {i_start, i_start+1}
        
        # 1. (1,0,0,0) x (1,0,0,0) x (1,0,0,0)
        u1 = [0]*(n*m); v1 = [0]*(m*p); w1 = [0]*(p*n)
        u1[i_start*m + j_start] = 1
        v1[j_start*p + k_start] = 1
        w1[k_start*n + i_start] = 1
        U_cols.append(u1); V_cols.append(v1); W_cols.append(w1)
        
        # 2. (0,0,1,0) x (1,0,0,0) x (0,1,0,0)
        u2 = [0]*(n*m); v2 = [0]*(m*p); w2 = [0]*(p*n)
        u2[(i_start+1)*m + j_start] = 1
        v2[j_start*p + k_start] = 1
        w2[k_start*n + (i_start+1)] = 1
        U_cols.append(u2); V_cols.append(v2); W_cols.append(w2)
        
        # 3. (0,1,0,0) x (0,1,0,0) x (0,0,1,0)
        u3 = [0]*(n*m); v3 = [0]*(m*p); w3 = [0]*(p*n)
        u3[i_start*m + (j_start+1)] = 1
        v3[(j_start+1)*p + k_start] = 1
        w3[(k_start+1)*n + i_start] = 1
        U_cols.append(u3); V_cols.append(v3); W_cols.append(w3)
        
        # 4. (0,0,0,1) x (0,1,0,0) x (0,0,0,1)
        u4 = [0]*(n*m); v4 = [0]*(m*p); w4 = [0]*(p*n)
        u4[(i_start+1)*m + (j_start+1)] = 1
        v4[(j_start+1)*p + (k_start+1)] = 1
        w4[(k_start+1)*n + (i_start+1)] = 1
        U_cols.append(u4); V_cols.append(v4); W_cols.append(w4)
        
        # 5. (1,1,1,1) x (1,0,0,0) x (1,1,1,1)
        u5 = [0]*(n*m); v5 = [0]*(m*p); w5 = [0]*(p*n)
        u5[i_start*m + j_start] = 1
        u5[i_start*m + (j_start+1)] = 1
        u5[(i_start+1)*m + j_start] = 1
        u5[(i_start+1)*m + (j_start+1)] = 1
        v5[j_start*p + k_start] = 1
        v5[j_start*p + (k_start+1)] = 1
        v5[(j_start+1)*p + k_start] = 1
        v5[(j_start+1)*p + (k_start+1)] = 1
        w5[k_start*n + i_start] = 1
        w5[k_start*n + (i_start+1)] = 1
        w5[(k_start+1)*n + i_start] = 1
        w5[(k_start+1)*n + (i_start+1)] = 1
        U_cols.append(u5); V_cols.append(v5); W_cols.append(w5)
        
        # 6. (1,1,1,1) x (0,1,0,0) x (1,1,1,1)
        u6 = [0]*(n*m); v6 = [0]*(m*p); w6 = [0]*(p*n)
        u6[i_start*m + j_start] = 1
        u6[i_start*m + (j_start+1)] = 1
        u6[(i_start+1)*m + j_start] = 1
        u6[(i_start+1)*m + (j_start+1)] = 1
        v6[(j_start+1)*p + k_start] = 1
        v6[(j_start+1)*p + (k_start+1)] = 1
        v6[(j_start+1)*p + k_start] = 1 # Duplicate? No, v6 is (0,1,0,0) in local coords?
        # v6 local: (0,1,0,0) means j=1, k=0 and j=1, k=1?
        # Local V indices: 0: j=0,k=0; 1: j=0,k=1; 2: j=1,k=0; 3: j=1,k=1.
        # (0,1,0,0) means index 1 is 1. So j=0, k=1.
        # My previous mapping: v5 was (1,0,0,0) -> j=0,k=0.
        # Let's re-verify the local index mapping.
        # U local: 0: i=0,j=0; 1: i=0,j=1; 2: i=1,j=0; 3: i=1,j=1.
        # V local: 0: j=0,k=0; 1: j=0,k=1; 2: j=1,k=0; 3: j=1,k=1.
        # W local: 0: k=0,i=0; 1: k=0,i=1; 2: k=1,i=0; 3: k=1,i=1.
        
        # Term 1: u=(1,0,0,0) -> i=0,j=0. v=(1,0,0,0) -> j=0,k=0. w=(1,0,0,0) -> k=0,i=0. OK.
        # Term 2: u=(0,0,1,0) -> i=1,j=0. v=(1,0,0,0) -> j=0,k=0. w=(0,1,0,0) -> k=0,i=1. OK.
        # Term 3: u=(0,1,0,0) -> i=0,j=1. v=(0,1,0,0) -> j=0,k=1. w=(0,0,1,0) -> k=1,i=0. OK.
        # Term 4: u=(0,0,0,1) -> i=1,j=1. v=(0,1,0,0) -> j=0,k=1. w=(0,0,0,1) -> k=1,i=1. OK.
        # Term 5: u=(1,1,1,1). v=(1,0,0,0) -> j=0,k=0. w=(1,1,1,1).
        # Term 6: u=(1,1,1,1). v=(0,1,0,0) -> j=0,k=1. w=(1,1,1,1).
        # Term 7: u=(1,1,1,1). v=(1,1,1,1). w=(1,1,1,1).
        
        # Correcting v6:
        v6 = [0]*(m*p)
        v6[j_start*p + (k_start+1)] = 1 # j=0, k=1
        U_cols.append(u6); V_cols.append(v6); W_cols.append(w6)
        
        # 7. (1,1,1,1) x (1,1,1,1) x (1,1,1,1)
        u7 = [0]*(n*m); v7 = [0]*(m*p); w7 = [0]*(p*n)
        u7[i_start*m + j_start] = 1
        u7[i_start*m + (j_start+1)] = 1
        u7[(i_start+1)*m + j_start] = 1
        u7[(i_start+1)*m + (j_start+1)] = 1
        v7[j_start*p + k_start] = 1
        v7[j_start*p + (k_start+1)] = 1
        v7[(j_start+1)*p + k_start] = 1
        v7[(j_start+1)*p + (k_start+1)] = 1
        w7[k_start*n + i_start] = 1
        w7[k_start*n + (i_start+1)] = 1
        w7[(k_start+1)*n + i_start] = 1
        w7[(k_start+1)*n + (i_start+1)] = 1
        U_cols.append(u7); V_cols.append(v7); W_cols.append(w7)

    # Partition 4x5x6 into 2x2x2 blocks where possible
    # I: 0,1; 2,3
    # J: 0,1; 2,3; 4
    # K: 0,1; 2,3; 4,5
    
    # Block (0,0,0): 2x2x2 -> Rank 7
    add_rank7_2x2x2(0, 0, 0)
    # Block (0,0,1): 2x2x2 -> Rank 7
    add_rank7_2x2x2(0, 0, 2)
    # Block (0,0,2): 2x2x2 -> Rank 7
    add_rank7_2x2x2(0, 0, 4)
    
    # Block (0,1,0): 2x2x2 -> Rank 7
    add_rank7_2x2x2(0, 2, 0)
    # Block (0,1,1): 2x2x2 -> Rank 7
    add_rank7_2x2x2(0, 2, 2)
    # Block (0,1,2): 2x2x2 -> Rank 7
    add_rank7_2x2x2(0, 2, 4)
    
    # Block (0,2,0): 2x1x2 -> Schoolbook 4
    add_schoolbook(0, 2, 4, 5, 0, 2)
    # Block (0,2,1): 2x1x2 -> Schoolbook 4
    add_schoolbook(0, 2, 4, 5, 2, 4)
    # Block (0,2,2): 2x1x2 -> Schoolbook 4
    add_schoolbook(0, 2, 4, 5, 4, 6)
    
    # Block (1,0,0): 2x2x2 -> Rank 7
    add_rank7_2x2x2(2, 0, 0)
    # Block (1,0,1): 2x2x2 -> Rank 7
    add_rank7_2x2x2(2, 0, 2)
    # Block (1,0,2): 2x2x2 -> Rank 7
    add_rank7_2x2x2(2, 0, 4)
    
    # Block (1,1,0): 2x2x2 -> Rank 7
    add_rank7_2x2x2(2, 2, 0)
    # Block (1,1,1): 2x2x2 -> Rank 7
    add_rank7_2x2x2(2, 2, 2)
    # Block (1,1,2): 2x2x2 -> Rank 7
    add_rank7_2x2x2(2, 2, 4)
    
    # Block (1,2,0): 2x1x2 -> Schoolbook 4
    add_schoolbook(2, 4, 4, 5, 0, 2)
    # Block (1,2,1): 2x1x2 -> Schoolbook 4
    add_schoolbook(2, 4, 4, 5, 2, 4)
    # Block (1,2,2): 2x1x2 -> Schoolbook 4
    add_schoolbook(2, 4, 4, 5, 4, 6)

    rank = len(U_cols)
    U = [list(col) for col in zip(*U_cols)] if U_cols else []
    V = [list(col) for col in zip(*V_cols)] if V_cols else []
    W = [list(col) for col in zip(*W_cols)] if W_cols else []
    
    return {"U": U, "V": V, "W": W}
