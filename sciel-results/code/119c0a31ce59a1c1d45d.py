import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Define partitions
    # n=3 -> [2, 1]
    # m=5 -> [2, 2, 1]
    # p=5 -> [2, 2, 1]
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Calculate offsets
    n_offsets = [0]
    for sz in n_parts[:-1]:
        n_offsets.append(n_offsets[-1] + sz)
        
    m_offsets = [0]
    for sz in m_parts[:-1]:
        m_offsets.append(m_offsets[-1] + sz)
        
    p_offsets = [0]
    for sz in p_parts[:-1]:
        p_offsets.append(p_offsets[-1] + sz)
        
    # Initialize U, V, W with zeros
    # We don't know the final rank yet, so we'll build lists of rows and then combine.
    # Or we can estimate max rank and trim.
    # Max rank is sum of block ranks.
    
    total_rank = 0
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Helper to add a block
    def add_block(n_idx, m_idx, p_idx):
        nonlocal total_rank
        n_sz = n_parts[n_idx]
        m_sz = m_parts[m_idx]
        p_sz = p_parts[p_idx]
        
        n_off = n_offsets[n_idx]
        m_off = m_offsets[m_idx]
        p_off = p_offsets[p_idx]
        
        if n_sz == 2 and m_sz == 2 and p_sz == 2:
            rank = 7
            # Strassen decomposition
            # Local U (4x7), V (4x7), W (4x7)
            U_loc = np.zeros((4, 7))
            V_loc = np.zeros((4, 7))
            W_loc = np.zeros((4, 7))
            
            # Indices in local block:
            # U row: i*m_sz + j. i in {0,1}, j in {0,1}
            # V row: j*p_sz + k. j in {0,1}, k in {0,1}
            # W row: k*n_sz + i. k in {0,1}, i in {0,1}
            
            # M1: (A11+A22)(B11+B22)
            # A11: i=0,j=0. A22: i=1,j=1.
            # B11: j=0,k=0. B22: j=1,k=1.
            # U: col 0
            U_loc[0, 0] = 1 # A11
            U_loc[3, 0] = 1 # A22 (i=1, j=1 -> 1*2+1=3)
            # V: col 0
            V_loc[0, 0] = 1 # B11
            V_loc[3, 0] = 1 # B22 (j=1, k=1 -> 1*2+1=3)
            # W: col 0
            # C11 += M1, C22 += M1
            # C11: i=0, k=0 -> W[0,0]
            # C22: i=1, k=1 -> W[3,0]
            W_loc[0, 0] = 1
            W_loc[3, 0] = 1
            
            # M2: (A21+A22)B11
            # A21: i=1, j=0 -> 1*2+0=2. A22: i=1, j=1 -> 3.
            # B11: j=0, k=0 -> 0.
            # U: col 1
            U_loc[2, 1] = 1
            U_loc[3, 1] = 1
            # V: col 1
            V_loc[0, 1] = 1
            # W: col 1
            # C21 += M2, C22 -= M2
            # C21: i=1, k=0 -> W[2,1] (k=0, i=1 -> 0*2+1=1? No. W index is k*n+i. n=2. k=0, i=1 -> 1. Wait.
            # W row index is k*n_sz + i.
            # C21: i=1, k=0. Index = 0*2 + 1 = 1.
            # C22: i=1, k=1. Index = 1*2 + 1 = 3.
            W_loc[1, 1] = 1
            W_loc[3, 1] = -1
            
            # M3: A11(B12-B22)
            # A11: i=0, j=0 -> 0.
            # B12: j=0, k=1 -> 1. B22: j=1, k=1 -> 3.
            # U: col 2
            U_loc[0, 2] = 1
            # V: col 2
            V_loc[1, 2] = 1
            V_loc[3, 2] = -1
            # W: col 2
            # C12 += M3, C22 += M3
            # C12: i=0, k=1. Index = 1*2 + 0 = 2.
            # C22: i=1, k=1. Index = 3.
            W_loc[2, 2] = 1
            W_loc[3, 2] = 1
            
            # M4: A22(B21-B11)
            # A22: i=1, j=1 -> 3.
            # B21: j=1, k=0 -> 2. B11: j=0, k=0 -> 0.
            # U: col 3
            U_loc[3, 3] = 1
            # V: col 3
            V_loc[2, 3] = 1
            V_loc[0, 3] = -1
            # W: col 3
            # C11 += M4, C21 += M4
            # C11: i=0, k=0. Index = 0.
            # C21: i=1, k=0. Index = 1.
            W_loc[0, 3] = 1
            W_loc[1, 3] = 1
            
            # M5: (A11+A12)B22
            # A11: i=0, j=0 -> 0. A12: i=0, j=1 -> 1.
            # B22: j=1, k=1 -> 3.
            # U: col 4
            U_loc[0, 4] = 1
            U_loc[1, 4] = 1
            # V: col 4
            V_loc[3, 4] = 1
            # W: col 4
            # C11 -= M5, C12 += M5
            # C11: i=0, k=0. Index = 0.
            # C12: i=0, k=1. Index = 2.
            W_loc[0, 4] = -1
            W_loc[2, 4] = 1
            
            # M6: (A21-A11)(B11+B12)
            # A21: i=1, j=0 -> 2. A11: i=0, j=0 -> 0.
            # B11: j=0, k=0 -> 0. B12: j=0, k=1 -> 1.
            # U: col 5
            U_loc[2, 5] = 1
            U_loc[0, 5] = -1
            # V: col 5
            V_loc[0, 5] = 1
            V_loc[1, 5] = 1
            # W: col 5
            # C22 += M6
            # C22: i=1, k=1. Index = 3.
            W_loc[3, 5] = 1
            
            # M7: (A12-A22)(B21+B22)
            # A12: i=0, j=1 -> 1. A22: i=1, j=1 -> 3.
            # B21: j=1, k=0 -> 2. B22: j=1, k=1 -> 3.
            # U: col 6
            U_loc[1, 6] = 1
            U_loc[3, 6] = -1
            # V: col 6
            V_loc[2, 6] = 1
            V_loc[3, 6] = 1
            # W: col 6
            # C11 += M7
            # C11: i=0, k=0. Index = 0.
            W_loc[0, 6] = 1
            
            # Place into global U, V, W
            # U global rows: n_off + i*m_sz + j
            # V global rows: m_off + j*p_sz + k
            # W global rows: p_off + k*n_sz + i
            
            U_glob = np.zeros((n*m, rank))
            V_glob = np.zeros((m*p, rank))
            W_glob = np.zeros((p*n, rank))
            
            for c in range(rank):
                for r in range(4):
                    if U_loc[r, c] != 0:
                        # r is local U index
                        # U local index = i*m_sz + j
                        # i = r // m_sz
                        # j = r % m_sz
                        i = r // m_sz
                        j = r % m_sz
                        U_glob[n_off + i*m_sz + j, c] = U_loc[r, c]
                        
                    if V_loc[r, c] != 0:
                        # r is local V index
                        # V local index = j*p_sz + k
                        # j = r // p_sz
                        # k = r % p_sz
                        j = r // p_sz
                        k = r % p_sz
                        V_glob[m_off + j*p_sz + k, c] = V_loc[r, c]
                        
                    if W_loc[r, c] != 0:
                        # r is local W index
                        # W local index = k*n_sz + i
                        # k = r // n_sz
                        # i = r % n_sz
                        k = r // n_sz
                        i = r % n_sz
                        W_glob[p_off + k*n_sz + i, c] = W_loc[r, c]
                        
            U_blocks.append(U_glob)
            V_blocks.append(V_glob)
            W_blocks.append(W_glob)
            total_rank += rank
            
        else:
            # Schoolbook
            rank = n_sz * m_sz * p_sz
            U_glob = np.zeros((n*m, rank))
            V_glob = np.zeros((m*p, rank))
            W_glob = np.zeros((p*n, rank))
            
            col = 0
            for i in range(n_sz):
                for j in range(m_sz):
                    for k in range(p_sz):
                        U_glob[n_off + i*m_sz + j, col] = 1
                        V_glob[m_off + j*p_sz + k, col] = 1
                        W_glob[p_off + k*n_sz + i, col] = 1
                        col += 1
                        
            U_blocks.append(U_glob)
            V_blocks.append(V_glob)
            W_blocks.append(W_glob)
            total_rank += rank

    for ni in range(len(n_parts)):
        for mi in range(len(m_parts)):
            for pi in range(len(p_parts)):
                add_block(ni, mi, pi)
                
    U = np.hstack(U_blocks)
    V = np.hstack(V_blocks)
    W = np.hstack(W_blocks)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
