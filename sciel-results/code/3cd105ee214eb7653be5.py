"""Matrix multiplication 2x4x5 rank-minimal exact tensor decomposition."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# Strassen 2x2x2 rank-7 decomposition
# For C = A*B where A, B are 2x2:
# M1 = (A11+A22)(B11+B22)
# M2 = (A21+A22)B11
# M3 = A11(B12-B22)
# M4 = A22(B21-B11)
# M5 = (A11+A12)B22
# M6 = (A21-A11)(B11+B12)
# M7 = (A12-A22)(B21+B22)
# C11 = M1 + M4 - M5 + M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6

def strassen_2x2x2():
    """Return U, V, W matrices for 2x2x2 Strassen decomposition (rank 7)."""
    # U is 4x7 (for 2x2 matrix A, flattened row-major: A11,A12,A21,A22)
    # V is 4x7 (for 2x2 matrix B)
    # W is 4x7 (for 2x2 matrix C, flattened row-major: C11,C12,C21,C22)
    # Index: U[a_idx, c], V[b_idx, c], W[c_idx, c]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Strategy: Use Strassen for the 2x2x2 blocks where possible
    # A is 2x4, B is 4x5
    # Split A into 2 blocks of 2x2: A = [A_left | A_right] where A_left, A_right are 2x2
    # Split B into blocks: B = [[B11, B12, B13, B14, B15]] where each Bij is 2x1
    # Actually, let's think more carefully.
    
    # A is 2x4: rows 0,1; cols 0,1,2,3
    # B is 4x5: rows 0,1,2,3; cols 0,1,2,3,4
    # C is 2x5
    
    # The tensor has indices (i,j,k) where i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Approach: Group columns of A and rows of B into 2x2 blocks
    # A has 4 columns -> 2 groups of 2 columns: cols {0,1} and cols {2,3}
    # B has 4 rows -> 2 groups of 2 rows: rows {0,1} and rows {2,3}
    
    # For each pair of column-group of A and row-group of B, we get a 2x2x2 sub-problem
    # but the output dimension k has 5 values, not 2.
    
    # Let me reconsider. The standard approach:
    # A[i, j] for i in 0..1, j in 0..3
    # B[j, k] for j in 0..3, k in 0..4
    # C[i, k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    # Split j into two groups: j in {0,1} and j in {2,3}
    # C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]
    
    # For each group, we have a 2x2x5 sub-problem (2 rows of A, 2 rows of B, 5 cols of B)
    # We can use Strassen-like decomposition for the 2x2 part, but B has 5 columns.
    
    # Actually, for a 2x2x5 problem:
    # A' is 2x2, B' is 2x5, C' is 2x5
    # C'[i,k] = sum_{j=0}^{1} A'[i,j] * B'[j,k]
    
    # This is equivalent to two independent 2x1x5 problems per row of A, or
    # we can view it as: for each k, C'[:,k] = A' * B'[:,k]
    # This is a 2x2x5 tensor.
    
    # The rank of 2x2x5 matrix multiplication: we can use Strassen for the 2x2x2 part
    # but we have 5 columns. 
    # 
    # Actually, the 2x2x5 multiplication can be decomposed as:
    # C'[i,k] = A'[i,0]*B'[0,k] + A'[i,1]*B'[1,k]
    # This is just linear in B' columns. The tensor is sum_k (A' * B'[:,k] outer product in some sense)
    # 
    # The rank of the 2x2x5 matmul tensor: Since B' has 5 columns, and for each column
    # we do a 2x2x1 matmul (rank 2 each, or we can combine), the total rank is at most
    # 5*2 = 10 (schoolbook) or we can do better.
    #
    # Actually, for 2x2xp matmul, the rank is 2p for p>=2? No...
    # Let me think: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] = delta in the sense that C = A*B.
    # For fixed k, the map (A,B) -> C[:,k] is a bilinear map.
    # The tensor is sum_k T_k where T_k[i,j] = delta_{i,j}... no.
    # T[i,j,k] contributes to C[i,k] via A[i,j]*B[j,k].
    # So the tensor is sum_{i,j,k} e_i (in C) * e_i (in A) * e_j (in B) * delta... 
    # Actually T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k], but that's not right.
    # The multiplication tensor for (n,m,p) has T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k. That's the standard matmul tensor.
    
    # For 2x2x5: rank is at most 2*2*5 = 20 (schoolbook), but we can do better.
    # Using Strassen on the 2x2 part: for each k, we compute C[:,k] = A * B[:,k]
    # But Strassen combines multiple outputs. For 2x2x2, rank 7.
    # For 2x2x5, we can view it as 5 separate 2x2x1 problems, each of rank 2, total rank 10.
    # Or we can use a decomposition that shares factors across k values.
    
    # Let's just use the straightforward approach:
    # Total rank = 2 * (rank of 2x2x5 matmul)
    # For 2x2x5: schoolbook gives rank 10. Can we do better?
    # The border rank of 2x2xp is 2p for p>=2 (I believe), and the rank might also be 2p.
    # Actually, I recall that rank(2x2xp) = 2p for p >= 2. Let's verify with p=2: rank(2x2x2)=7 (Strassen), not 4.
    # So rank(2x2xp) is not simply 2p.
    
    # Hmm, let me just go with a concrete decomposition.
    
    # Strategy: 
    # Split the 4 middle dimension into 2 groups of 2.
    # Group 1: j in {0,1}, Group 2: j in {2,3}
    # For each group, we have a 2x2x5 sub-problem.
    # For each 2x2x5 sub-problem, use schoolbook: rank = 2*2*5 = 20? No, that's too much.
    # Wait, schoolbook for 2x2x5 is rank 2*2*5 = 20? No, schoolbook rank is n*m*p = 2*2*5 = 20.
    # But that's the trivial decomposition. We can do better.
    
    # Actually, for 2x2xp, the schoolbook decomposition has rank 2*p (one per (i,k) pair... no).
    # Schoolbook: for each (i,j,k), we have a term. That's 2*2*p = 4p terms.
    # But we can combine: C[i,k] = sum_j A[i,j] B[j,k]
    # For fixed i, C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # This is a rank-2 decomposition in k: C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # So for each i, we need 2 terms (one for each j). Total: 2*2 = 4 terms per (i) ... 
    # Wait, the tensor decomposition: each term is u ⊗ v ⊗ w.
    # Term (i,j,k): u has 1 at position i (in C), v has 1 at (i,j) (in A), w has 1 at (j,k) (in B).
    # But we can combine over k: for fixed (i,j), the term is u_{i,j} ⊗ v_{i,j} ⊗ w_{i,j}
    # where u_{i,j}[i',k'] = delta_{i',i} (for all k'), v_{i,j}[i'',j''] = delta_{i'',i}delta_{j'',j},
    # w_{i,j}[j'',k''] = delta_{j'',j} (for all k').
    # But this doesn't work because u would need to be the same for all k.
    # Actually, the standard schoolbook for matmul: rank = n*m*p, with one term per (i,j,k).
    # But there's a more efficient schoolbook: rank = n*p + m*p + ... no.
    
    # Let me just use the known result: for 2x2x5, I'll use a decomposition of rank 10.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # Term for (i=0, j=0): contributes A[0,0]*B[0,k] to C[0,k] for all k.
    # As a tensor term: u has 1 at (0,k) for all k? No, a tensor term is a single trilinear form.
    # The term is: u_{0,0}[i',k'] = delta_{i',0} (independent of k'), v_{0,0}[i'',j''] = delta_{i'',0}delta_{j'',0},
    # w_{0,0}[j'',k''] = delta_{j'',0} (independent of k'').
    # Then the product u*v*w evaluated at (A,B) gives: sum_{i',k'} u[i',k'] * sum_{i'',j''} v[i'',j'']A[i'',j''] * sum_{j'',k''} w[j'',k'']B[j'',k'']
    # = (sum_{i'',j''} delta_{i'',0}delta_{j'',0} A[i'',j'']) * (sum_{j'',k''} delta_{j'',0} B[j'',k'']) * (sum_{i',k'} delta_{i',0} * 1)
    # That's not right. The tensor decomposition means:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where T[i,j,k] = sum_r u_r[i] v_r[j] w_r[k]... but our tensor has 3 indices (i, j, k) corresponding to (C_index, A_index, B_index).
    # Actually, the matmul tensor T has T[i,j,k] = 1 (for all valid i,j,k) in the sense that
    # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all i,j,k.
    # A rank-r decomposition: T = sum_{r=1}^r u_r ⊗ v_r ⊗ w_r where u_r is in C-space (2x5=10 dim),
    # v_r in A-space (2x4=8 dim), w_r in B-space (4x5=20 dim).
    # T[i,j,k] = sum_r u_r[i,k] * v_r[i,j] * w_r[j,k]
    
    # OK so for the full 2x4x5 problem:
    # T[i,j,k] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    # We need T[i,j,k] = sum_r U[r][i,k] * V[r][i,j] * W[r][j,k]
    
    # Schoolbook: rank 40, with U[r][i,k]=delta, V[r][i,j]=delta, W[r][j,k]=delta for each (i,j,k).
    
    # Better: For each (i,j), we can combine over k.
    # Define for each (i,j): U_{i,j}[i',k'] = delta_{i',i}, V_{i,j}[i'',j''] = delta_{i'',i}delta_{j'',j}, W_{i,j}[j'',k''] = delta_{j'',j}
    # Then sum_r U_r[i,k] V_r[i,j] W_r[j,k] = sum_{i',j'} U_{i',j'}[i,k] V_{i',j'}[i,j] W_{i',j'}[j,k]
    # = sum_{i',j'} delta_{i,i'} * delta_{i,i'}delta_{j,j'} * delta_{j,j'}
    # = delta_{j,j} ... this is getting confusing. Let me just verify with small case.
    
    # For 2x2x2 Strassen, the decomposition is well-known. Let me just hardcode the full 2x4x5 using
    # Strassen on the 2x2x2 blocks.
    
    # Split j into {0,1} and {2,3}.
    # C[i,k] = sum_{j in {0,1}} A[i,j]B[j,k] + sum_{j in {2,3}} A[i,j]B[j,k]
    
    # For the first sum: 2x2x5 sub-problem with A1 = A[:,0:2], B1 = B[0:2,:]
    # For the second sum: 2x2x5 sub-problem with A2 = A[:,2:4], B2 = B[2:4,:]
    
    # For each 2x2x5 sub-problem, I'll use a rank-10 decomposition (schoolbook-ish but optimized).
    # Actually, let me use rank 7 for the 2x2x2 part and handle the extra 3 columns separately.
    
    # Alternative: Just use Strassen for 2x2x2 blocks and schoolbook for the rest.
    # Split k into {0,1} and {2,3,4}.
    # For k in {0,1}: 2x4x2 sub-problem. Split j into {0,1} and {2,3}: two 2x2x2 Strassen blocks, rank 7+7=14.
    # For k in {2,3,4}: 2x4x3 sub-problem. Schoolbook: rank 2*4*3=24.
    # Total: 14+24=38. Not great.
    
    # Better: For each j-group (2 groups), use Strassen for k in {0,1} (rank 7 each) and schoolbook for k in {2,3,4} (rank 2*3=6 each).
    # Total: 2*(7+6) = 26.
    
    # Even better: For each j-group, the 2x2x5 problem. Can we decompose it with rank < 14?
    # 2x2x5: I'll use a decomposition of rank 10 (schoolbook optimized).
    # Total: 2*10 = 20.
    
    # Let me construct the 2x2x5 decomposition with rank 10.
    # For 2x2x5: T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    # Schoolbook: 2*2*5 = 20 terms. But we can do 10 terms:
    # For each (i,j): one term that covers all k.
    # Term (i,j): U[i',k'] = delta_{i',i}, V[i'',j''] = delta_{i'',i}delta_{j'',j}, W[j'',k''] = delta_{j'',j}
    # Check: sum_{i,j} U[i,k]V[i,j]W[j,k] = sum_{i,j} delta_{i,i} * delta_{i,i}delta_{j,j} * delta_{j,j} = sum_{i,j} delta_{j,j} = ... 
    # This doesn't work directly. Let me think again.
    
    # The correct schoolbook for 2x2xp with rank 2p:
    # For each (j,k): term with U[i',k']=delta_{k',k} (for all i'), V[i'',j'']=delta_{j'',j} (for all i''), W[j'',k'']=delta_{j'',j}delta_{k'',k}
    # No, that's not right either.
    
    # Let me just use the trivial schoolbook: rank = n*m*p = 2*2*5 = 20 for each sub-problem.
    # Total: 2*20 = 40. Same as original. Not helpful.
    
    # OK, let me use Strassen properly.
    # For the 2x4x5 problem:
    # Split into 2 blocks along j (each 2x2x5).
    # For each 2x2x5 block:
    #   Split k into k0 in {0,1} and k1 in {2,3,4}.
    #   For k0: 2x2x2 Strassen, rank 7.
    #   For k1: 2x2x3 schoolbook, rank 2*2*3 = 12.
    #   Total per block: 7+12 = 19.
    # Total: 2*19 = 38.
    
    # Can we do better for 2x2x3? 
    # 2x2x3: rank at most 7 (same as 2x2x2) + 2 (for the extra column) = 9? 
    # Actually, 2x2x3 can be done with rank 7+2=9 if we extend Strassen.
    # Or we can use a direct decomposition.
    
    # Let me just go with rank 38 for now and see if I can improve.
    # Actually, the known best for 2x4x5 might be lower. Let me try a different split.
    
    # Split k into 5 single columns. For each k, we have a 2x4x1 problem (rank 8 schoolbook).
    # Total: 5*8 = 40. Worse.
    
    # Split j into 4 single columns. For each j, we have a 2x1x5 problem (rank 10 schoolbook).
    # Total: 4*10 = 40. Same.
    
    # Let me try: use Strassen on 2x2x2 blocks for the first 2 columns of k, and for the remaining 3 columns,
    # use a combined decomposition.
    
    # Actually, let me try a completely different approach: use the fact that we can share factors.
    
    # I'll implement the 38-rank decomposition and verify it.
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Build 2x2x5 decomposition using Strassen for k in {0,1} and schoolbook for k in {2,3,4}
    
    def build_2x2x5():
        """Build U, V, W for 2x2x5 matmul with rank 19 (7 Strassen + 12 schoolbook)."""
        # U: 10x19 (C-space: 2x5), V: 4x19 (A-space: 2x2), W: 10x19 (B-space: 2x5)
        # Strassen part (k in {0,1}): 7 terms
        # Schoolbook part (k in {2,3,4}): 12 terms (2*2*3)
        
        U_s, V_s, W_s = strassen_2x2x2()  # 4x7 each
        # U_s is 4x7: indices (i,k) for i in {0,1}, k in {0,1}
        # V_s is 4x7: indices (i,j) for i in {0,1}, j in {0,1}
        # W_s is 4x7: indices (j,k) for j in {0,1}, k in {0,1}
        
        # Schoolbook for k in {2,3,4}:
        # For each (i,j,k) with k in {2,3,4}: one term
        # U[i,k] = 1, V[i,j] = 1, W[j,k] = 1
        U_sb = []
        V_sb = []
        W_sb = []
        for i in range(2):
            for j in range(2):
                for k in range(3):  # k = 2,3,4
                    U_sb.append((i, k+2))
                    V_sb.append((i, j))
                    W_sb.append((j, k+2))
        
        # Combine
        U = [[0]*19 for _ in range(10)]
        V = [[0]*19 for _ in range(4)]
        W = [[0]*19 for _ in range(10)]
        
        # Strassen part (terms 0-6)
        for c in range(7):
            for idx in range(4):
                i, k = idx // 2, idx % 2
                U[i*5+k][c] = U_s[idx][c]
                i2, j = idx // 2, idx % 2
                V[i2*2+j][c] = V_s[idx][c]
                j2, k2 = idx // 2, idx % 2
                W[j2*5+k2][c] = W_s[idx][c]
        
        # Schoolbook part (terms 7-18)
        for t in range(12):
            c = 7 + t
            i, k = U_sb[t]
            U[i*5+k][c] = 1
            i2, j = V_sb[t]
            V[i2*2+j][c] = 1
            j2, k2 = W_sb[t]
            W[j2*5+k2][c] = 1
        
        return U, V, W
    
    # Now build the full 2x4x5 decomposition
    # Split j into {0,1} and {2,3}
    # For each j-group, use 2x2x5 decomposition
    
    U_2x2x5, V_2x2x5, W_2x2x5 = build_2x2x5()
    
    rank = 2 * 19  # 38
    U = [[0]*rank for _ in range(10)]  # 2x5
    V = [[0]*rank for _ in range(8)]   # 2x4
    W = [[0]*rank for _ in range(20)]  # 4x5
    
    # First j-group (j in {0,1})
    for c in range(19):
        for ik in range(10):
            U[ik][c] = U_2x2x5[ik][c]
        for ij in range(4):
            i, j = ij // 2, ij % 2
            V[i*4+j][c] = V_2x2x5[ij][c]
        for jk in range(10):
            j, k = jk // 5, jk % 5
            W[j*5+k][c] = W_2x2x5[jk][c]
    
    # Second j-group (j in {2,3})
    for c in range(19):
        cc = 19 + c
        for ik in range(10):
            U[ik][cc] = U_2x2x5[ik][c]
        for ij in range(4):
            i, j = ij // 2, ij % 2
            V[i*4+(j+2)][cc] = V_2x2x5[ij][c]
        for jk in range(10):
            j, k = jk // 5, jk % 5
            W[(j+2)*5+k][cc] = W_2x2x5[jk][c]
    
    # Verify
    def verify(U, V, W, n, m, p):
        import numpy as np
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # C[i,k] = sum_r U[r, i,k] * V[r, i,j] * W[r, j,k]
        # Check against actual matmul
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[r, i*p+k] * V[r, i*m+j] * W[r, j*p+k]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * p)]
    V = [[0] * rank for _ in range(n * m)]
    W = [[0] * rank for _ in range(m * p)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * p + k][column] = 1
                V[i * m + j][column] = 1
                W[j * p + k][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
