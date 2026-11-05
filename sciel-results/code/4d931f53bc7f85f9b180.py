import numpy as np
import random
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    
    # We decompose the 5x5 multiplication into 8 block multiplications
    # A = [[A11, A12], [A21, A22]]
    # A11 is 2x2, A12 is 2x3, A21 is 3x2, A22 is 3x3
    
    # Ranks used:
    # 2x2x2: 7 (Strassen)
    # 2x3x2: 12 (Schoolbook)
    # 2x2x3: 12 (Schoolbook)
    # 2x3x3: 18 (Schoolbook)
    # 3x2x2: 12 (Schoolbook)
    # 3x3x2: 18 (Schoolbook)
    # 3x2x3: 18 (Schoolbook)
    # 3x3x3: 27 (Schoolbook)
    
    total_rank = 7 + 12 + 12 + 18 + 12 + 18 + 18 + 27
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    def add_decomposition(i_start, i_end, j_start, j_end, k_start, k_end, decomp_func):
        nonlocal col_offset
        # decomp_func returns U_sub, V_sub, W_sub
        # U_sub shape: (i_len * j_len, r_sub)
        # V_sub shape: (j_len * k_len, r_sub)
        # W_sub shape: (k_len * i_len, r_sub)
        U_sub, V_sub, W_sub = decomp_func(i_end-i_start, j_end-j_start, k_end-k_start)
        r_sub = len(U_sub[0]) if U_sub else 0
        for r in range(r_sub):
            for i in range(i_end - i_start):
                for j in range(j_end - j_start):
                    U[i_start + i][j_start + j] # This mapping is wrong for global indices
                    # Global U index is i*m + j.
                    # Local U index is i_local * j_len + j_local.
                    pass
        # This manual mapping is error prone. Let's do it directly.
        
    # I will just write the loops for each block.
    
    # Block 1: A11 B11 (2x2x2) Strassen
    # i in [0,2), j in [0,2), k in [0,2)
    # Strassen 2x2x2 decomposition
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # We need to map these to the global U, V, W.
    # For each term (e.g., A11*B11), we add to U, V, W.
    # But Strassen combines them.
    # It is easier to just use schoolbook for 2x2x2 too? Rank 8.
    # Total rank 125. No improvement.
    # I must use Strassen.
    
    # Let's define the Strassen terms for 2x2x2.
    # The indices for 2x2x2 are local i,j,k in {0,1}.
    # Global i = 0+i, j = 0+j, k = 0+k.
    
    # Term 1: (A11+A22) * (B11+B22)
    # A11 is at (0,0), A22 is at (1,1).
    # B11 is at (0,0), B22 is at (1,1).
    # U entries: U[0,0]+=1, U[1,1]+=1.
    # V entries: V[0,0]+=1, V[1,1]+=1.
    # W entries: W[0,0]+=1, W[1,1]+=1.
    
    # This is getting complicated to do manually for all 7 terms and then add the other 7 blocks.
    # I will use a helper to add a rank-1 tensor.
    
    def add_rank1(i, j, k, u_val, v_val, w_val):
        nonlocal col_offset
        U[i*m + j][col_offset] += u_val
        V[j*p + k][col_offset] += v_val
        W[k*n + i][col_offset] += w_val
        col_offset += 1
        
    # But Strassen has linear combinations.
    # We can expand them into rank-1 tensors.
    # (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
    # This is 4 rank-1 tensors.
    # Strassen saves multiplications, not necessarily the number of rank-1 terms in the final decomposition?
    # Yes, Strassen uses 7 multiplications. Each multiplication is a scalar product.
    # In the tensor decomposition, each multiplication corresponds to a set of rank-1 tensors?
    # No, a scalar multiplication $x \cdot y$ is one rank-1 tensor if $x,y$ are scalars.
    # Here $x, y$ are matrices.
    # The "multiplication" in Strassen is the product of two $1 \times 1$ "matrices" (scalars) formed by linear combinations of entries.
    # So each of the 7 Strassen multiplications corresponds to 1 rank-1 tensor in the decomposition of the $2 \times 2 \times 2$ tensor?
    # No. The $2 \times 2 \times 2$ tensor has rank 7.
    # So there are 7 rank-1 tensors.
    # Each rank-1 tensor is defined by vectors $u, v, w \in \mathbb{R}^2$ (flattened).
    
    # Let's get the 7 rank-1 tensors for $2 \times 2 \times 2$.
    # From Strassen:
    # 1. u = (1,0,0,1) [A11,A12,A21,A22]? No, flattened A is [A11, A12, A21, A22].
    # A = [a00, a01, a10, a11].
    # M1 = (a00+a11)(b00+b11).
    # u1 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # w1 = [1, 0, 0, 1] (C11 gets +1, C22 gets +1? No, C11=M1+..., C22=M1-...)
    # The W vector for M1 contributes to C11 and C22.
    # C11 coeff 1, C22 coeff 1.
    # w1 = [1, 0, 0, 1] (C11, C12, C21, C22).
    
    # M2 = (a10+a11)b00.
    # u2 = [0, 0, 1, 1]
    # v2 = [1, 0, 0, 0]
    # w2 = [0, 0, 1, 0] (C21 gets M2).
    
    # M3 = a00(b01-b11).
    # u3 = [1, 0, 0, 0]
    # v3 = [0, 1, 0, -1]
    # w3 = [0, 1, 0, 0] (C12 gets M3).
    
    # M4 = a11(b10-b00).
    # u4 = [0, 0, 0, 1]
    # v4 = [0, 0, 1, -1]
    # w4 = [1, 0, 1, 0] (C11 gets M4, C21 gets M4).
    
    # M5 = (a00+a01)b11.
    # u5 = [1, 1, 0, 0]
    # v5 = [0, 0, 0, 1]
    # w5 = [-1, 1, 0, 0] (C11 gets -M5, C12 gets M5).
    
    # M6 = (a10-a00)(b00+b01).
    # u6 = [-1, 0, 1, 0]
    # v6 = [1, 1, 0, 0]
    # w6 = [0, 0, 0, 1] (C22 gets M6).
    
    # M7 = (a01-a11)(b10+b11).
    # u7 = [0, 1, 0, -1]
    # v7 = [0, 0, 1, 1]
    # w7 = [1, 0, 0, -1] (C11 gets M7, C22 gets -M7).
    
    # These are the 7 rank-1 tensors for the 2x2x2 block.
    # I can add these to the global U, V, W.
    
    # For the other blocks, I will use schoolbook.
    # Schoolbook for $n \times m \times p$ has $n \cdot m \cdot p$ rank-1 tensors.
    # Each tensor is $u = e_{ij}, v = e_{jk}, w = e_{ki}$.
    
    # Block 1: 2x2x2 (Strassen)
    # i in [0,2), j in [0,2), k in [0,2)
    # Global indices: i, j, k.
    # Local indices: il, jl, kl.
    # Global i = il, j = jl, k = kl.
    
    strassen_tensors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, -1])
    ]
    
    for u, v, w in strassen_tensors:
        for il in range(2):
            for jl in range(2):
                for kl in range(2):
                    # u is flattened A (2x2). Index il*2+jl.
                    # v is flattened B (2x2). Index jl*2+kl.
                    # w is flattened C (2x2). Index kl*2+il.
                    u_val = u[il*2 + jl]
                    v_val = v[jl*2 + kl]
                    w_val = w[kl*2 + il]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        # Add to global
                        # This is NOT a rank-1 tensor in the global sense if we loop like this.
                        # The Strassen decomposition is a sum of 7 rank-1 tensors.
                        # Each rank-1 tensor is a triple of vectors.
                        # The vectors are in R^4 (flattened 2x2).
                        # My loop above is expanding the vectors into global indices?
                        # No. The rank-1 tensor is defined by the vectors u, v, w.
                        # The contribution to U[i,j] is u[i,j].
                        # So I just need to add u[i,j] to U[i,j] for the column.
                        
                        pass # I will just add the vector values directly to the column.
                    
        # For each of the 7 tensors, I add 1 column to U, V, W.
        # U_col[i,j] = u[i,j]
        # V_col[j,k] = v[j,k]
        # W_col[k,i] = w[k,i]
        pass
        
    # This is getting messy. I will just use schoolbook for ALL blocks.
    # Rank 125.
    # But I need to beat 125.
    
    # I will use Strassen for 2x2x2 and schoolbook for others.
    # Rank 124.
    
    # Let's implement Strassen for 2x2x2 correctly.
    # There are 7 terms.
    # For each term, I add a column to U, V, W.
    
    # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
    # U[0,0]+=1, U[1,1]+=1
    # V[0,0]+=1, V[1,1]+=1
    # W[0,0]+=1, W[1,1]+=1
    
    # I will write a helper to add a column from vectors.
    
    def add_column(u_vec, v_vec, w_vec, i_off, j_off, k_off, i_len, j_len, k_len):
        nonlocal col_offset
        for i in range(i_len):
            for j in range(j_len):
                U[i_off+i][j_off+j] = u_vec[i*j_len+j]
        for j in range(j_len):
            for k in range(k_len):
                V[j_off+j][k_off+k] = v_vec[j*k_len+k]
        for k in range(k_len):
            for i in range(i_len):
                W[k_off+k][i_off+i] = w_vec[k*i_len+i]
        col_offset += 1

    # Block 1: 2x2x2 Strassen
    # i_off=0, j_off=0, k_off=0, i_len=2, j_len=2, k_len=2
    # The vectors u,v,w are length 4.
    # My add_column assumes u_vec is flattened row-major.
    
    # I need to be careful with the W vector indexing.
    # W[k,i]. Flattened W is w[k*i_len + i].
    
    # Let's test add_column logic.
    # U[i,j] = u_vec[i*j_len+j]. Correct.
    # V[j,k] = v_vec[j*k_len+k]. Correct.
    # W[k,i] = w_vec[k*i_len+i]. Correct.
    
    # Now add the 7 Strassen terms.
    for u, v, w in strassen_tensors:
        add_column(u, v, w, 0, 0, 0, 2, 2, 2)
        
    # Block 2: 2x3x2 Schoolbook
    # i in [0,2), j in [2,5), k in [0,2)
    # i_off=0, j_off=2, k_off=0, i_len=2, j_len=3, k_len=2
    # Rank 12.
    for i in range(2):
        for j in range(3):
            for k in range(2):
                # Rank 1 tensor: U[i,j]=1, V[j,k]=1, W[k,i]=1
                # This is 12 columns.
                # I can just call add_column with basis vectors.
                # But add_column adds 1 column.
                # I need to add 12 columns.
                pass # I will just loop and add 1s.
                
    # This is too slow to write out.
    # I will just use the schoolbook loop for the remaining blocks.
    
    # Block 2: 2x3x2
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i][j] = 1
                V[j][k] = 1
                W[k][i] = 1
                col_offset += 1
                
    # Block 3: 2x2x3
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i][j] = 1
                V[j][k] = 1
                W[k][i] = 1
                col_offset += 1
                
    # Block 4: 2x3x3
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i][j] = 1
                V[j][k] = 1
                W[k][i] = 1
                col_offset += 1
                
    # Block 5: 3x2x2
    for i in range(2, 5):
        for j in range(2):
            for k in range(2):
                U[i][j] = 1
                V[j][k] = 1
                W[k][i] = 1
                col_offset += 1
                
    # Block 6: 3x3x2
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2):
                U[i][j] = 1
                V[j][k] = 1
                W[k][i] = 1
                col_offset += 1
                
    # Block 7: 3x2x3
    for i in range(2, 5):
        for j in range(2):
            for k in range(2, 5):
                U[i][j] = 1
                V[j][k] = 1
                W[k][i] = 1
                col_offset += 1
                
    # Block 8: 3x3x3
    for i in range(2, 5):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i][j] = 1
                V[j][k] = 1
                W[k][i] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
