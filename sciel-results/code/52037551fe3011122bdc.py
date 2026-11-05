import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Define blocks
    # Rows: 0,1 (size 2), 2 (size 1)
    row_blocks = [(0, 1), (2, 2)] # (start, end) inclusive? Let's use lists of indices
    row_blocks = [[0, 1], [2]]
    
    # Mids: 0,1 (size 2), 2,3 (size 2), 4 (size 1)
    mid_blocks = [[0, 1], [2, 3], [4]]
    
    # Outs: 0,1 (size 2), 2,3 (size 2), 4,5 (size 2), 6 (size 1)
    out_blocks = [[0, 1], [2, 3], [4, 5], [6]]
    
    U = []
    V = []
    W = []
    
    # We will build the columns one by one
    # First, determine the total rank
    total_rank = 0
    block_decompositions = []
    
    for rb in row_blocks:
        for mb in mid_blocks:
            for ob in out_blocks:
                nr = len(rb)
                nm = len(mb)
                no = len(ob)
                
                if nr == 2 and nm == 2 and no == 2:
                    # Use Strassen
                    rank_block = 7
                    # We need to generate the 7 terms for this block
                    # The indices in the global matrices are:
                    # U rows: rb indices, V rows: mb indices, W rows: ob indices (with k*n+i convention)
                    # Strassen terms
                    # M1: (A11+A22)(B11+B22)
                    # M2: (A21+A22)B11
                    # M3: A11(B12-B22)
                    # M4: A22(B21-B11)
                    # M5: (A11+A12)B22
                    # M6: (A21-A11)(B11+B12)
                    # M7: (A12-A22)(B21+B22)
                    
                    # We will add these 7 columns to U, V, W
                    block_decompositions.append(('strassen', rb, mb, ob, rank_block))
                    total_rank += rank_block
                else:
                    # Use Schoolbook
                    rank_block = nr * nm * no
                    block_decompositions.append(('schoolbook', rb, mb, ob, rank_block))
                    total_rank += rank_block
                    
    # Initialize U, V, W with zeros
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    for btype, rb, mb, ob, rank_block in block_decompositions:
        if btype == 'schoolbook':
            for i in rb:
                for j in mb:
                    for k in ob:
                        U[i * m + j][col_idx] = 1
                        V[j * p + k][col_idx] = 1
                        W[k * n + i][col_idx] = 1
                        col_idx += 1
        else: # strassen
            # rb = [r1, r2], mb = [m1, m2], ob = [o1, o2]
            r1, r2 = rb
            m1, m2 = mb
            o1, o2 = ob
            
            # M1: (A11+A22)(B11+B22)
            # U: A11=1, A22=1
            # V: B11=1, B22=1
            # W: C11=1, C22=1
            U[r1 * m + m1][col_idx] = 1
            U[r2 * m + m2][col_idx] = 1
            V[m1 * p + o1][col_idx] = 1
            V[m2 * p + o2][col_idx] = 1
            W[o1 * n + r1][col_idx] = 1
            W[o2 * n + r2][col_idx] = 1
            col_idx += 1
            
            # M2: (A21+A22)B11
            U[r2 * m + m1][col_idx] = 1
            U[r2 * m + m2][col_idx] = 1
            V[m1 * p + o1][col_idx] = 1
            W[o1 * n + r2][col_idx] = 1
            col_idx += 1
            
            # M3: A11(B12-B22)
            U[r1 * m + m1][col_idx] = 1
            V[m1 * p + o2][col_idx] = 1
            V[m2 * p + o2][col_idx] = -1
            W[o2 * n + r1][col_idx] = 1
            col_idx += 1
            
            # M4: A22(B21-B11)
            U[r2 * m + m2][col_idx] = 1
            V[m2 * p + o1][col_idx] = 1
            V[m1 * p + o1][col_idx] = -1
            W[o1 * n + r2][col_idx] = 1
            col_idx += 1
            
            # M5: (A11+A12)B22
            U[r1 * m + m1][col_idx] = 1
            U[r1 * m + m2][col_idx] = 1
            V[m2 * p + o2][col_idx] = 1
            W[o2 * n + r1][col_idx] = 1
            col_idx += 1
            
            # M6: (A21-A11)(B11+B12)
            U[r2 * m + m1][col_idx] = 1
            U[r1 * m + m1][col_idx] = -1
            V[m1 * p + o1][col_idx] = 1
            V[m1 * p + o2][col_idx] = 1
            W[o1 * n + r2][col_idx] = 1
            W[o2 * n + r2][col_idx] = 1
            col_idx += 1
            
            # M7: (A12-A22)(B21+B22)
            U[r1 * m + m2][col_idx] = 1
            U[r2 * m + m2][col_idx] = -1
            V[m2 * p + o1][col_idx] = 1
            V[m2 * p + o2][col_idx] = 1
            W[o1 * n + r1][col_idx] = 1
            W[o2 * n + r1][col_idx] = 1
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
