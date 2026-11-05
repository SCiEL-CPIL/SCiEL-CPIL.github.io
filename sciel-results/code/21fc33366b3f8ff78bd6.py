import time
import numpy as np

DIMENSIONS = [5, 5, 5]
N = 5

def _block_decomposition():
    """
    Returns the rank-117 decomposition of the 5x5x5 matrix multiplication tensor.
    Partition 5 into 2+2+1.
    Blocks:
    - 2x2x2: Strassen (rank 7). There are 8 such blocks? No.
    The indices are (a,b,c) where a,b,c in {0,1,2} corresponding to parts [2,2,1].
    (0,0,0): 2x2x2 -> Rank 7
    (0,0,1): 2x2x1 -> Rank 4 (Schoolbook)
    (0,1,0): 2x1x2 -> Rank 4 (Schoolbook)
    (0,1,1): 2x1x1 -> Rank 2 (Schoolbook)
    (1,0,0): 2x2x2 -> Rank 7 (Wait, parts[1]=2, parts[0]=2, parts[2]=1? No, a,b,c indices map to dims i,j,k)
    
    Let's re-verify the block sizes.
    Parts = [2, 2, 1].
    a=0 (size 2), a=1 (size 2), a=2 (size 1).
    b=0 (size 2), b=1 (size 2), b=2 (size 1).
    c=0 (size 2), c=1 (size 2), c=2 (size 1).
    
    Block (a,b,c) has size parts[a] x parts[b] x parts[c].
    
    (0,0,0): 2x2x2 -> Strassen (7)
    (0,0,1): 2x2x2 -> Strassen (7)
    (0,0,2): 2x2x1 -> Schoolbook (4)
    
    (0,1,0): 2x2x2 -> Strassen (7)
    (0,1,1): 2x2x2 -> Strassen (7)
    (0,1,2): 2x2x1 -> Schoolbook (4)
    
    (0,2,0): 2x1x2 -> Schoolbook (4)
    (0,2,1): 2x1x2 -> Schoolbook (4)
    (0,2,2): 2x1x1 -> Schoolbook (2)
    
    (1,0,0): 2x2x2 -> Strassen (7)
    (1,0,1): 2x2x2 -> Strassen (7)
    (1,0,2): 2x2x1 -> Schoolbook (4)
    
    (1,1,0): 2x2x2 -> Strassen (7)
    (1,1,1): 2x2x2 -> Strassen (7)
    (1,1,2): 2x2x1 -> Schoolbook (4)
    
    (1,2,0): 2x1x2 -> Schoolbook (4)
    (1,2,1): 2x1x2 -> Schoolbook (4)
    (1,2,2): 2x1x1 -> Schoolbook (2)
    
    (2,0,0): 1x2x2 -> Schoolbook (4)
    (2,0,1): 1x2x2 -> Schoolbook (4)
    (2,0,2): 1x2x1 -> Schoolbook (2)
    
    (2,1,0): 1x2x2 -> Schoolbook (4)
    (2,1,1): 1x2x2 -> Schoolbook (4)
    (2,1,2): 1x2x1 -> Schoolbook (2)
    
    (2,2,0): 1x1x2 -> Schoolbook (2)
    (2,2,1): 1x1x2 -> Schoolbook (2)
    (2,2,2): 1x1x1 -> Schoolbook (1)
    
    Sum:
    Strassen (2x2x2): 8 blocks * 7 = 56
    Schoolbook 2x2x1: 4 blocks * 4 = 16
    Schoolbook 2x1x2: 4 blocks * 4 = 16
    Schoolbook 1x2x2: 4 blocks * 4 = 16
    Schoolbook 2x1x1: 2 blocks * 2 = 4
    Schoolbook 1x2x1: 2 blocks * 2 = 4
    Schoolbook 1x1x2: 2 blocks * 2 = 4
    Schoolbook 1x1x1: 1 block * 1 = 1
    
    Total: 56 + 16 + 16 + 16 + 4 + 4 + 4 + 1 = 117.
    """
    n, m, p = DIMENSIONS
    parts = [2, 2, 1]
    offsets = [0, 2, 4]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_col(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
        
    # Strassen factors for 2x2x2
    # M1 = (A11-A22)(B21+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A22)(B11+B22)
    # M6 = (A12-A22)(B21+B22)
    # M7 = (A11+A12)(B11-B12)
    # C11 = M5+M4-M3-M7
    # C12 = M1+M7
    # C21 = M2+M3
    # C22 = M5+M1-M2-M4
    
    # Indices in 2x2 block:
    # A: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    # C: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    strassen_factors = [
        # M1
        {
            'U': [(0,0,1), (1,1,1)], # A11, A22 (with signs)
            'V': [(1,0,1), (1,1,1)], # B21, B22
            'W': [(0,0,1), (0,1,1)]  # C11, C12
        },
        # M2
        {
            'U': [(1,0,1), (1,1,1)], # A21, A22
            'V': [(0,0,1)],           # B11
            'W': [(1,0,1)]            # C21
        },
        # M3
        {
            'U': [(0,0,1)],           # A11
            'V': [(0,1,1), (1,1,-1)], # B12, B22
            'W': [(0,0,-1), (1,0,1)]  # C11, C21
        },
        # M4
        {
            'U': [(1,1,1)],           # A22
            'V': [(1,0,1), (0,0,-1)], # B21, B11
            'W': [(0,0,-1), (1,1,1)]  # C11, C22
        },
        # M5
        {
            'U': [(0,0,1), (1,1,1)],  # A11, A22
            'V': [(0,0,1), (1,1,1)],  # B11, B22
            'W': [(0,0,1), (1,1,1)]   # C11, C22
        },
        # M6
        {
            'U': [(0,1,1), (1,1,-1)], # A12, A22
            'V': [(1,0,1), (1,1,1)],  # B21, B22
            'W': [(0,1,1)]            # C12
        },
        # M7
        {
            'U': [(0,0,1), (0,1,1)],  # A11, A12
            'V': [(0,0,1), (0,1,-1)], # B11, B12
            'W': [(0,0,-1), (1,1,-1)] # C11, C22
        }
    ]
    
    for a in range(3):
        for b in range(3):
            for c in range(3):
                sa = parts[a]
                sb = parts[b]
                sc = parts[c]
                oa = offsets[a]
                ob = offsets[b]
                oc = offsets[c]
                
                if sa == 2 and sb == 2 and sc == 2:
                    for f in strassen_factors:
                        u_vec = [0]*25
                        v_vec = [0]*25
                        w_vec = [0]*25
                        
                        for (li, lj, val) in f['U']:
                            gi = oa + li
                            gj = ob + lj
                            u_vec[gi*5 + gj] += val
                            
                        for (li, lj, val) in f['V']:
                            gi = ob + li
                            gj = oc + lj
                            v_vec[gi*5 + gj] += val
                            
                        for (li, lj, val) in f['W']:
                            gi = oc + li
                            gj = oa + lj
                            w_vec[gi*5 + gj] += val
                            
                        add_col(u_vec, v_vec, w_vec)
                else:
                    # Schoolbook
                    for u in range(sa):
                        for v in range(sb):
                            for w in range(sc):
                                u_vec = [0]*25
                                v_vec = [0]*25
                                w_vec = [0]*25
                                
                                gi = oa + u
                                gj = ob + v
                                gk = oc + w
                                
                                u_vec[gi*5 + gj] = 1
                                v_vec[gj*5 + gk] = 1
                                w_vec[gk*5 + gi] = 1
                                
                                add_col(u_vec, v_vec, w_vec)
                                
    rank = len(U_cols)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for col_idx in range(rank):
        u_col = U_cols[col_idx]
        v_col = V_cols[col_idx]
        w_col = W_cols[col_idx]
        for row_idx in range(25):
            U[row_idx][col_idx] = u_col[row_idx]
            V[row_idx][col_idx] = v_col[row_idx]
            W[row_idx][col_idx] = w_col[row_idx]
            
    return {"U": U, "V": V, "W": W}

def construct():
    """
    Returns the rank-117 decomposition for 5x5 matrix multiplication.
    """
    return _block_decomposition()
