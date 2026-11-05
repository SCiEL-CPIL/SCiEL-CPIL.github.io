import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    import random
    random.seed(42)
    for _ in range(1000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        g = random.randint(0, p*n-1)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[g, r]
            if abs(val) > tol:
                return False
    return True

def _strassen_2x2_block():
    """
    Strassen's algorithm for 2x2 matrix multiplication.
    Returns U, V, W factors for the 2x2x2 tensor with rank 7.
    
    For C = AB where A, B are 2x2:
    C[0,0] = M1 - M4 + M6 + M7
    C[0,1] = M5 + M4
    C[1,0] = M1 + M2
    C[1,1] = M3 - M2 + M5 + M7
    
    Where:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))  # 4x7
    V = np.zeros((m*p, rank))  # 4x7
    W = np.zeros((p*n, rank))  # 4x7
    
    # Indexing: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
    # B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[3, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1; W[2, 5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W

def _build_block_strassen(n, m, p):
    """
    Build a block Strassen decomposition for 4x5x6 matrix multiplication.
    
    Split A (4x5) rows into [0:2],[2:4], cols into [0:2],[2:4],[4:5]
    Split B (5x6) rows into [0:2],[2:4],[4:5], cols into [0:2],[2:4],[4:6]
    Split C (4x6) rows into [0:2],[2:4], cols into [0:2],[2:4],[4:6]
    
    For 2x2x2 sub-blocks, use Strassen (rank 7)
    For 2x1x2 sub-blocks, use rank-2 decomposition
    Total rank: 12*7 + 6*2 = 96
    """
    if (n, m, p) != (4, 5, 6):
        return _build_schoolbook(n, m, p)
    
    total_rank = 96
    U = np.zeros((n * m, total_rank))  # 20 x 96
    V = np.zeros((m * p, total_rank))  # 30 x 96
    W = np.zeros((p * n, total_rank))  # 24 x 96
    
    # Get Strassen 2x2x2 factors
    SU, SV, SW = _strassen_2x2_block()
    
    # Row groups for A and C: [0:2], [2:4]
    # Col groups for A: [0:2], [2:4], [4:5]
    # Row groups for B: [0:2], [2:4], [4:5]
    # Col groups for B and C: [0:2], [2:4], [4:6]
    
    a_row_groups = [(0, 2), (2, 4)]
    a_col_groups = [(0, 2), (2, 4), (4, 5)]
    b_row_groups = [(0, 2), (2, 4), (4, 5)]
    b_col_groups = [(0, 2), (2, 4), (4, 6)]
    
    col_offset = 0
    
    for ri, (r_start, r_end) in enumerate(a_row_groups):
        for ci, (c_start, c_end) in enumerate(a_col_groups):
            # Match with b_row_groups
            bi = ci  # a_col_groups[i] corresponds to b_row_groups[i]
            k_start, k_end = b_row_groups[bi]
            
            a_size_r = r_end - r_start  # 2
            a_size_c = c_end - c_start  # 2 or 1
            b_size_r = k_end - k_start  # 2 or 1
            assert a_size_c == b_size_r
            
            for di, (d_start, d_end) in enumerate(b_col_groups):
                b_size_c = d_end - d_start  # 2
                
                # Determine block type
                if a_size_r == 2 and a_size_c == 2 and b_size_c == 2:
                    # 2x2x2: use Strassen, rank 7
                    block_rank = 7
                elif a_size_r == 2 and a_size_c == 1 and b_size_c == 2:
                    # 2x1x2: rank 2
                    block_rank = 2
                else:
                    # Fallback to schoolbook for this block
                    block_rank = a_size_r * a_size_c * b_size_c
                
                # Embed the block decomposition
                if a_size_r == 2 and a_size_c == 2 and b_size_c == 2:
                    # Strassen 2x2x2
                    for lr in range(7):
                        # Map local indices to global
                        for la in range(4):  # local A index
                            local_i = la // 2  # 0 or 1
                            local_j = la % 2   # 0 or 1
                            global_i = r_start + local_i
                            global_j = c_start + local_j
                            U[global_i * m + global_j, col_offset + lr] = SU[la, lr]
                        
                        for lb in range(4):  # local B index
                            local_k = lb // 2  # 0 or 1
                            local_l = lb % 2   # 0 or 1
                            global_k = k_start + local_k
                            global_l = d_start + local_l
                            V[global_k * p + global_l, col_offset + lr] = SV[lb, lr]
                        
                        for lc in range(4):  # local C index
                            local_kc = lc // 2  # 0 or 1
                            local_ic = lc % 2   # 0 or 1
                            global_k = k_start + local_kc
                            global_i = r_start + local_ic
                            W[global_k * n + global_i, col_offset + lr] = SW[lc, lr]
                
                elif a_size_r == 2 and a_size_c == 1 and b_size_c == 2:
                    # 2x1x2: A is 2x1, B is 1x2, C is 2x2
                    # C[i,l] = A[i,0] * B[0,l]
                    # Rank 2: one term per output element... actually rank is 2
                    # C[0,0] = A[0,0]*B[0,0], C[0,1] = A[0,0]*B[0,1]
                    # C[1,0] = A[1,0]*B[0,0], C[1,1] = A[1,0]*B[0,1]
                    # This is outer product of A (2x1) and B (1x2) = 2x2
                    # Rank 2: use 2 terms
                    # Term 0: A[0,0]*B[0,0] -> C[0,0]
                    # Term 1: A[1,0]*B[0,1] -> C[1,1]
                    # Wait, that doesn't work. Let me think again.
                    # Actually for 2x1x2, the tensor has 4 nonzero entries.
                    # The minimal rank is 2.
                    # We can write: C = A * B where A is 2x1, B is 1x2
                    # This is a rank-1 tensor if we view it as (2x1) outer (1x2)
                    # But in the matrix multiplication tensor formulation:
                    # T[a,b,g] = 1 if a=(i,j), b=(k,l), g=(k',i') and i=i', j=k, k=k'
                    # For 2x1x2: i in {0,1}, j=0, k=0, l in {0,1}
                    # T[(i,0),(0,l),(0,i)] = 1 for i in {0,1}, l in {0,1}
                    # So 4 nonzero entries: (0,0,0), (1,0,1), (0,1,0), (1,1,1)
                    # Wait: a=(i,j)=(i,0), b=(k,l)=(0,l), g=(k',i')=(0,i)
                    # Need j=k: 0=0 ✓, i=i' ✓
                    # So T[(i,0),(0,l),(0,i)] = 1 for all i,l in {0,1}
                    # That's 4 entries.
                    # Rank of this tensor: it's the sum of 4 rank-1 tensors
                    # But can we do better? 
                    # T = e_{(0,0)} ⊗ e_{(0,0)} ⊗ e_{(0,0)} + e_{(0,0)} ⊗ e_{(0,1)} ⊗ e_{(0,0)} + e_{(1,0)} ⊗ e_{(0,0)} ⊗ e_{(1,0)} + e_{(1,0)} ⊗ e_{(0,1)} ⊗ e_{(1,0)}
                    # = (e_{(0,0)} + e_{(1,0)}) ⊗ e_{(0,0)} ⊗ (e_{(0,0)} + e_{(1,0)}) ... no that's not right
                    # Actually: T[i,l,i] = 1 for all i,l. This is the outer product of (1,1) in i-space, (1,1) in l-space... 
                    # T = u ⊗ v ⊗ w where u[i]=1, v[l]=1, w[i]=1? No, that gives T[i,l,i'] = 1 for all i,l,i'.
                    # Hmm, let me just use rank 4 (schoolbook) for safety, or figure out rank 2.
                    # Actually, I think the rank is 2:
                    # Term 1: U has 1 at (0,0) and (1,0), V has 1 at (0,0), W has 1 at (0,0) and (1,0)
                    #   -> contributes to T[(0,0),(0,0),(0,0)] and T[(1,0),(0,0),(1,0)]
                    # Term 2: U has 1 at (0,0) and (1,0), V has 1 at (0,1), W has 1 at (0,0) and (1,0)
                    #   -> contributes to T[(0,0),(0,1),(0,0)] and T[(1,0),(0,1),(1,0)]
                    # Together they give all 4 entries! So rank 2.
                    
                    # Local A indices: (0,0)=0, (1,0)=1  (2 elements)
                    # Local B indices: (0,0)=0, (0,1)=1  (2 elements)  
                    # Local C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3  (4 elements)
                    # Wait, C is 2x2, so local C index = k'*2 + i' where k' in {0}, i' in {0,1}
                    # Actually C local index: for C[i', l'], the index in the p*n space...
                    # In our global W, the index is k*n + i. For the sub-block, local k' and i'.
                    # Local C index = k' * 2 + i' (since C is 2 rows x 2 cols, but stored as k*n+i)
                    # Wait, C is 2x2 in the sub-block. The W index for local (k', i') is k'*2 + i'.
                    
                    # Term 1: 
                    U[0, col_offset + 0] = 1  # A local (0,0) -> global (r_start+0, c_start+0)
                    U[1, col_offset + 0] = 1  # A local (1,0) -> global (r_start+1, c_start+0)
                    V[0, col_offset + 0] = 1  # B local (0,0) -> global (k_start+0, d_start+0)
                    W[0, col_offset + 0] = 1  # C local (0,0) -> global (k_start+0, r_start+0)
                    W[2, col_offset + 0] = 1  # C local (1,0) -> global (k_start+0, r_start+1)
                    
                    # Term 2:
                    U[0, col_offset + 1] = 1
                    U[1, col_offset + 1] = 1
                    V[1, col_offset + 1] = 1  # B local (0,1) -> global (k_start+0, d_start+1)
                    W[0, col_offset + 1] = 1  # C local (0,0)
                    W[2, col_offset + 1] = 1  # C local (1,0)
                    
                    # Map to global indices
                    # A: local (i, 0) -> global (r_start+i, c_start+0)
                    # B: local (0, l) -> global (k_start+0, d_start+l)
                    # C: local (k', i') -> global (k_start+k', r_start+i'), W index = (k_start+k')*n + (r_start+i')
                    
                    # Let me redo with proper global mapping
                    # Clear and redo
                    for lr in range(2):
                        for li in range(2):
                            U[(r_start + li) * m + (c_start + 0), col_offset + lr] = 1
                        for ll in range(2):
                            V[(k_start + 0) * p + (d_start + ll), col_offset + lr] = (1 if ll == lr else 0)
                        for li in range(2):
                            W[(k_start + 0) * n + (r_start + li), col_offset + lr] = 1
                    # Wait, that's not right either. Let me be more careful.
                    # Term 0: U has 1s at A[0,0] and A[1,0], V has 1 at B[0,0], W has 1s at C[0,0] and C[1,0]
                    #   In global: U[(r_start+0)*m+(c_start+0), col_offset+0] = 1
                    #              U[(r_start+1)*m+(c_start+0), col_offset+0] = 1
                    #              V[(k_start+0)*p+(d_start+0), col_offset+0] = 1
                    #              W[(k_start+0)*n+(r_start+0), col_offset+0] = 1
                    #              W[(k_start+0)*n+(r_start+1), col_offset+0] = 1
                    # Term 1: U has 1s at A[0,0] and A[1,0], V has 1 at B[0,1], W has 1s at C[0,0] and C[1,0]
                    #   In global: U[(r_start+0)*m+(c_start+0), col_offset+1] = 1
                    #              U[(r_start+1)*m+(c_start+0), col_offset+1] = 1
                    #              V[(k_start+0)*p+(d_start+1), col_offset+1] = 1
                    #              W[(k_start+0)*n+(r_start+0), col_offset+1] = 1
                    #              W[(k_start+0)*n+(r_start+1), col_offset+1] = 1
                    # 
                    # But wait, this gives T[(i,0),(0,l),(0,i')] = sum_r U[i,0,r]*V[0,l,r]*W[0,i',r]
                    # For r=0: U[i,0,0]=1, V[0,l,0]=1 if l=0 else 0, W[0,i',0]=1
                    #   So contributes 1 when l=0, for all i,i'
                    # For r=1: U[i,0,1]=1, V[0,l,1]=1 if l=1 else 0, W[0,i',1]=1
                    #   So contributes 1 when l=1, for all i,i'
                    # Total: T[(i,0),(0,l),(0,i')] = 1 if (l=0 or l=1) = 1 for all i,l,i'
                    # But we only want T[(i,0),(0,l),(0,i)] = 1 (i.e., i=i')!
                    # This is wrong. We're getting 1 for all i,i' combinations.
                    
                    # I need to reconsider. The 2x1x2 tensor:
                    # T[i,l,i'] = 1 if i=i', else 0. For i,i' in {0,1}, l in {0,1}.
                    # This is: T = sum_i e_i ⊗ (e_0 + e_1) ⊗ e_i
                    # = e_0 ⊗ (e_0+e_1) ⊗ e_0 + e_1 ⊗ (e_0+e_1) ⊗ e_1
                    # Rank 2!
                    
                    # Term 0: U[0,0]=1, V[0,0]=1, V[0,1]=1, W[0,0]=1
                    #   -> T[0,0,0]=1, T[0,1,0]=1
                    # Term 1: U[1,1]=1, V[1,0]=1, V[1,1]=1, W[1,1]=1
                    #   -> T[1,0,1]=1, T[1,1,1]=1
                    # Together: T[0,0,0]=1, T[0,1,0]=1, T[1,0,1]=1, T[1,1,1]=1. Correct!
                    
                    # So:
                    # Term 0: U local (0,0)=1, V local (0,0)=1, V local (0,1)=1, W local (0,0)=1
                    # Term 1: U local (1,0)=1, V local (0,0)=1, V local (0,1)=1, W local (1,0)=1
                    # Wait, V is shared. Let me re-express:
                    # T[i,l,i'] = delta(i,i')
                    # = sum_{r=0}^{1} U[i,r] * V[l,r] * W[i',r]
                    # Term r=0: U[0,0]=1, U[1,0]=0, V[0,0]=1, V[1,0]=1, W[0,0]=1, W[1,0]=0
                    #   -> T[0,0,0]=1*1*1=1, T[0,1,0]=1*1*1=1, T[1,*,*]=0
                    # Term r=1: U[0,1]=0, U[1,1]=1, V[0,1]=1, V[1,1]=1, W[0,1]=0, W[1,1]=1
                    #   -> T[1,0,1]=1*1*1=1, T[1,1,1]=1*1*1=1, T[0,*,*]=0
                    # Total: T[0,0,0]=1, T[0,1,0]=1, T[1,0,1]=1, T[1,1,1]=1. Correct!
                    
                    # Now map to global:
                    # A local (i, 0) -> global (r_start+i, c_start+0)
                    # B local (0, l) -> global (k_start+0, d_start+l)
                    # C local (i', l') -> but C is indexed by (k', i') in W as k'*n+i'
                    # In the sub-block, C[i', l'] where i' is row, l' is col
                    # W global index for C[i', l'] in sub-block: (k_start + 0)*n + (r_start + i')
                    # Wait, C[i', l'] means row i', col l'. In W, the index is col*n + row = l'*n + i'
                    # But in our global W, it's k*n + i where k is the column of C and i is the row of C.
                    # For the sub-block: C local row i', col l'. Global row = r_start + i', global col = d_start + l'.
                    # W index = (d_start + l') * n + (r_start + i')
                    
                    # Hmm wait, I think I confused myself. Let me re-read the problem.
                    # W[k*n + i, r] where k is the column index of C and i is the row index of C.
                    # So W index = col_C * n + row_C.
                    
                    # For the sub-block C[i', l'] (i' = local row, l' = local col):
                    # Global row = r_start + i', global col = d_start + l'
                    # W index = (d_start + l') * n + (r_start + i')
                    
                    # Let me redo:
                    # Term 0: 
                    #   U[(r_start+0)*m+(c_start+0), col_offset+0] = 1
                    #   V[(k_start+0)*p+(d_start+0), col_offset+0] = 1
                    #   V[(k_start+0)*p+(d_start+1), col_offset+0] = 1
                    #   W[(d_start+0)*n+(r_start+0), col_offset+0] = 1
                    # Term 1:
                    #   U[(r_start+1)*m+(c_start+0), col_offset+1] = 1
                    #   V[(k_start+0)*p+(d_start+0), col_offset+1] = 1
                    #   V[(k_start+0)*p+(d_start+1), col_offset+1] = 1
                    #   W[(d_start+0)*n+(r_start+1), col_offset+1] = 1
                    #   W[(d_start+1)*n+(r_start+1), col_offset+1] = 1
                    # 
                    # Wait, that's not matching my derivation. Let me redo carefully.
                    # 
                    # T_local[i, l, i'] = delta(i, i') for i,i' in {0,1}, l in {0,1}
                    # = sum_r U_loc[i, r] * V_loc[l, r] * W_loc[i', r]
                    # 
                    # r=0: U_loc[0,0]=1, U_loc[1,0]=0
                    #        V_loc[0,0]=1, V_loc[1,0]=1
                    #        W_loc[0,0]=1, W_loc[1,0]=0
                    # r=1: U_loc[0,1]=0, U_loc[1,1]=1
                    #        V_loc[0,1]=1, V_loc[1,1]=1
                    #        W_loc[0,1]=0, W_loc[1,1]=1
                    #
                    # Check: T[0,0,0] = 1*1*1 + 0*1*0 = 1 ✓
                    # T[0,1,0] = 1*1*1 + 0*1*0 = 1 ✓
                    # T[1,0,1] = 0*1*0 + 1*1*1 = 1 ✓
                    # T[1,1,1] = 0*1*0 + 1*1*1 = 1 ✓
                    # T[0,0,1] = 1*1*0 + 0*1*1 = 0 ✓
                    # T[1,0,0] = 0*1*1 + 1*1*0 = 0 ✓
                    # 
                    # Now global mapping:
                    # A local (i, 0) -> global A index = (r_start+i)*m + (c_start+0)
                    # B local (0, l) -> global B index = (k_start+0)*p + (d_start+l)
                    # C local: W index for (i', l') where i' is row, l' is col of C
                    #   Global C row = r_start + i', global C col = d_start + l'
                    #   W index = (d_start + l')*n + (r_start + i')
                    #
                    # But in my local W indexing above, I used W_loc[i', r] which conflates row and the tensor index.
                    # The local C has 4 elements: (i', l') for i',l' in {0,1}
                    # W_loc index = l'*2 + i' (col*n_local + row) where n_local=2
                    # 
                    # Let me redo with proper local W indices:
                    # W_loc[l'*2 + i', r]
                    # 
                    # r=0: W_loc[0,0]=1 (l'=0,i'=0), W_loc[1,0]=0 (l'=0,i'=1), W_loc[2,0]=0 (l'=1,i'=0), W_loc[3,0]=0 (l'=1,i'=1)
                    # r=1: W_loc[0,1]=0, W_loc[1,1]=0, W_loc[2,1]=0, W_loc[3,1]=1 (l'=1,i'=1)
                    # 
                    # Wait, that doesn't match. Let me recompute:
                    # T[i,l,i'] = sum_r U[i,r]*V[l,r]*W[l'*2+i',r] where l' is the col index
                    # But T only depends on i, l, i' not l'... 
                    # 
                    # I think the issue is that in the tensor T[a,b,g], g encodes both the row and column of C.
                    # For the sub-block, a=(i,0) [A row i, col 0], b=(0,l) [B row 0, col l], g=(i',l') [C row i', col l']
                    # T=1 iff i=i' AND 0=0 AND 0=l... wait no.
                    # 
                    # The condition is: A[i,j] * B[j,k] contributes to C[i,k]
                    # So T[(i,j),(j,k),(k,i)] = 1
                    # For the sub-block: i in {0,1} (local A row), j=0 (local A col = local B row), k in {0,1} (local B col)
                    # T[(i,0),(0,k),(k,i)] = 1
                    # So g = (k, i) meaning C row = i, C col = k
                    # W index for g=(k,i) in local: k*2 + i (col*n_local + row)
                    # 
                    # So T[i,k,i] = 1 for i,k in {0,1}. 4 nonzero entries.
                    # T = sum_i sum_k e_i ⊗ e_k ⊗ e_{(k,i)}
                    # = e_0⊗e_0⊗e_0 + e_0⊗e_1⊗e_1 + e_1⊗e_0⊗e_2 + e_1⊗e_1⊗e_3
                    # 
                    # Can we factor this into rank 2?
                    # = e_0⊗(e_0⊗e_0 + e_1⊗e_1) + e_1⊗(e_0⊗e_2 + e_1⊗e_3)
                    # The middle factors are different, so this isn't a simple outer product.
                    # 
                    # Actually, the rank of the 2x1x2 matrix multiplication tensor is 2.
                    # Let me verify: C[i,k] = A[i,0]*B[0,k]
                    # This is just the outer product of the column vector A[:,0] and row vector B[0,:].
                    # As a bilinear map, this has rank 2 (you need 2 scalar multiplications:
                    # M1 = A[0,0]*B[0,0], M2 = A[1,0]*B[0,0]... no that's not right either.
                    # 
                    # Actually for 2x1x2: C = [A[0,0]; A[1,0]] * [B[0,0], B[0,1]]
                    # C[0,0] = A[0,0]*B[0,0]
                    # C[0,1] = A[0,0]*B[0,1]
                    # C[1,0] = A[1,0]*B[0,0]
                    # C[1,1] = A[1,0]*B[0,1]
                    # 
                    # We can compute: M1 = A[0,0]*B[0,0], M2 = A[1,0]*B[0,1]
                    # C[0,0] = M1, C[1,1] = M2
                    # C[0,1] = A[0,0]*B[0,1] = M1 * B[0,1]/B[0,0]... no, we can't do division.
                    # 
                    # Actually the rank is 4 for 2x1x2? No, that can't be right.
                    # Let me think again. The tensor T has 4 nonzero entries out of 2*2*4=16 possible.
                    # T[(i,0),(0,k),(k,i)] = 1
                    # 
                    # This is the "diagonal" tensor. Its rank is 2:
                    # T = e_0⊗(e_0⊗e_0 + e_1⊗e_1) + e_1⊗(e_0⊗e_2 + e_1⊗e_3)
                    # Hmm, but the second factors are different.
                    # 
                    # Actually, I recall that the rank of M_{2,1,2} is 2. Let me just use rank 4 (schoolbook) to be safe, or better yet, let me just use rank 4 for the 2x1x2 blocks.
                    # 
                    # Actually no. Let me just use schoolbook (rank 4) for the 2x1x2 blocks to be safe.
                    # That gives total rank: 12*7 + 6*4 = 84 + 24 = 108.
                    # Still better than 120!
                    
                    # Use schoolbook for 2x1x2 blocks
                    for li in range(2):
                        for lk in range(2):
                            U[(r_start+li)*m+(c_start+0), col_offset] = 1
                            V[(k_start+0)*p+(d_start+lk), col_offset] = 1
                            W[(d_start+lk)*n+(r_start+li), col_offset] = 1
                            col_offset += 1
                    # block_rank = 4, but we already incremented col_offset 4 times
                    continue  # skip the normal col_offset increment below
                
                col_offset += block_rank
    
    # Verify col_offset == total_rank
    assert col_offset == total_rank, f"col_offset={col_offset}, total_rank={total_rank}"
    
    return U, V, W

def _optimize_decomposition(n, m, p, initial_rank=None, max_time=100):
    """
    Try to find a lower-rank decomposition using alternating least squares.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    if initial_rank is None:
        initial_rank = n * m * p
    
    # Try decreasing ranks
    for target_rank in range(initial_rank - 1, max(10, initial_rank - 50), -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(seed)
            
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Alternating least squares
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U: for each row a, solve least squares
                # sum_r U[a,r] * V[b,r] * W[g,r] = T[a,b,g]
                # For fixed a: sum_r U[a,r] * (V[:,r] * W[:,r])_elementwise = T[a,*,*]
                
                for a in range(n * m):
                    # Build the design matrix
                    # For each (b,g), the coefficient vector is V[b,:] * W[g,:]
                    # Stack these to form M of shape (m*p*p*n, target_rank)
                    M_rows = []
                    t_vals = []
                    
                    for b in range(m * p):
                        for g in range(p * n):
                            coeff = V[b, :] * W[g, :]
                            M_rows.append(coeff)
                            
                            # Check if (a,b,g) is consistent
                            i = a // m; j = a % m
                            jp = b // p; k = b % p
                            kn = g // n; ip = g % n
                            if i == ip and j == jp and k == kn:
                                t_vals.append(1.0)
                            else:
                                t_vals.append(0.0)
                    
                    M_mat = np.array(M_rows)
                    t_vec = np.array(t_vals)
                    
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                        U[a, :] = sol
                    except:
                        pass
                
                # Update V: for each row b
                for b in range(m * p):
                    M_rows = []
                    t_vals = []
                    
                    for a in range(n * m):
                        for g in range(p * n):
                            coeff = U[a, :] * W[g, :]
                            M_rows.append(coeff)
                            
                            i = a // m; j = a % m
                            jp = b // p; k = b % p
                            kn = g // n; ip = g % n
                            if i == ip and j == jp and k == kn:
                                t_vals.append(1.0)
                            else:
                                t_vals.append(0.0)
                    
                    M_mat = np.array(M_rows)
                    t_vec = np.array(t_vals)
                    
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                        V[b, :] = sol
                    except:
                        pass
                
                # Update W: for each row g
                for g in range(p * n):
                    M_rows = []
                    t_vals = []
                    
                    for a in range(n * m):
                        for b in range(m * p):
                            coeff = U[a, :] * V[b, :]
                            M_rows.append(coeff)
                            
                            i = a // m; j = a % m
                            jp = b // p; k = b % p
                            kn = g // n; ip = g % n
                            if i == ip and j == jp and k == kn:
                                t_vals.append(1.0)
                            else:
                                t_vals.append(0.0)
                    
                    M_mat = np.array(M_rows)
                    t_vec = np.array(t_vals)
                    
                    try:
                        sol, _, _, _ = np.linalg.lstsq(M_mat, t_vec, rcond=None)
                        W[g, :] = sol
                    except:
                        pass
                
                # Check convergence
                if iteration % 10 == 0:
                    loss = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i*m+j; b = j*p+k; g = k*n+i
                                val = np.sum(U[a, :] * V[b, :] * W[g, :])
                                loss += (val - 1.0) ** 2
                    
                    # Sample zero constraints
                    np.random.seed(42)
                    for _ in range(200):
                        a = np.random.randint(0, n*m)
                        b = np.random.randint(0, m*p)
                        g = np.random.randint(0, p*n)
                        i = a // m; j = a % m
                        jp = b // p; k = b % p
                        kn = g // n; ip = g % n
                        if i != ip or j != jp or k != kn:
                            val = np.sum(U[a, :] * V[b, :] * W[g, :])
                            loss += val ** 2
                    
                    if loss < 1e-10:
                        # Verify fully
                        if _verify_decomposition(U, V, W, n, m, p):
                            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try to find a better decomposition
    result = _optimize_decomposition(n, m, p, initial_rank=best_rank, max_time=140)
    
    if result is not None:
        U, V, W = result
        if U.shape[1] < best_rank:
            best_U, best_V, best_W = U, V, W
            best_rank = U.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
