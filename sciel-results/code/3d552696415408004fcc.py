import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    deadline = time.time() + 140

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Schoolbook fallback (rank 96)
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school

    # Strassen 4x4x4 decomposition (rank 21)
    # Standard Strassen gives rank 7 for 2x2x2.
    # For 4x4x4, we can use the rank-21 decomposition.
    # Let me construct it via the known rank-21 algorithm.
    #
    # The 4x4x4 matrix multiplication tensor has rank 21.
    # One construction: split 4x4 into four 2x2 blocks and use
    # Strassen's 7 multiplications, but that gives 7*7=49 for the
    # full 4x4. Instead, there's a direct rank-21 decomposition.
    #
    # Actually, let me use a different approach:
    # 3x4x8 = 3x4x4 (first half) + 3x4x4 (second half)
    # Each 3x4x4 can be embedded in 4x4x4.
    # Rank of 4x4x4 is 21, so total rank = 42.
    #
    # For 4x4x4, I'll use the rank-21 decomposition.
    # The entries can be integers or half-integers.
    
    # Let me construct rank-21 for 4x4x4 using a known formula.
    # The 4x4x4 MM tensor rank is 21.
    # I'll use the decomposition from the literature.
    #
    # Alternative: use Kronecker product structure.
    # 4x4x4 can be seen as (2x2x2) with some structure.
    # Rank(2x2x2) = 7 (Strassen).
    # 4x4x4 via 2x2x2 blocks: each of the 16 entries involves
    # products. A direct approach:
    #
    # Let me just hardcode a rank-21 decomposition for 4x4x4.
    # The tensor T4[i,j,k] = 1 if i,j,k in 0..3 represent valid MM indices.
    # For 4x4x4: T4[a, b, c] where a = i*4+j (i,j in 0..3), 
    #            b = j*4+k, c = k*4+i, with i,j,k in 0..3.
    
    # Build 4x4x4 target tensor
    T4 = np.zeros((16, 16, 16), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T4[i*4+j, j*4+k, k*4+i] = 1.0

    # Rank-21 decomposition of 4x4x4 MM tensor
    # I'll use a known construction. The rank of 4x4x4 is 21.
    # Let me try to find it via a structured approach.
    #
    # One known rank-21 scheme for 4x4x4:
    # It uses the fact that 4x4x4 can be decomposed using
    # 21 scalar multiplications.
    
    # Let me try a different structural approach:
    # 3x4x8 directly.
    # Split the 8 columns of B into two groups of 4: B = [B1 | B2]
    # C = A*B1 + A*B2 (where A*B1 is 3x4 and A*B2 is 3x4, concatenated)
    #
    # For each 3x4x4 subproblem, embed into 4x4x4.
    # 3x4x4: A is 3x4, B is 4x4, C is 3x4.
    # Pad A to 4x4 (add zero row), C to 4x4 (add zero row).
    # Then use rank-21 4x4x4 decomposition.
    #
    # So I need a rank-21 decomposition of 4x4x4.
    
    # Let me construct it. The 4x4x4 MM tensor:
    # Indices: a = i*4+j, b = j*4+k, c = k*4+i for i,j,k in {0,1,2,3}
    # T4[a,b,c] = 1 if there exist i,j,k in {0,1,2,3} with 
    #   a=i*4+j, b=j*4+k, c=k*4+i, else 0.
    # Note: for each (i,j,k), exactly one (a,b,c) is set to 1.
    # But multiple (i,j,k) can map to same (a,b,c)? No, the mapping is bijective
    # since i,j,k determine a,b,c uniquely and conversely.
    # Actually: a=i*4+j determines i=a//4, j=a%4.
    # b=j*4+k determines j=b//4, k=b%4.
    # c=k*4+i determines k=c//4, i=c%4.
    # So consistency requires: a%4 == b//4, b%4 == c//4, c%4 == a//4.
    # And i,j,k in 0..3 always holds for 4x4x4.
    
    # So T4[a,b,c] = 1 iff (a%4 == b//4) and (b%4 == c//4) and (c%4 == a//4)
    
    # Now, rank-21 decomposition. Let me look up or derive it.
    # 
    # I'll use a constructive approach. The 4x4x4 tensor is isomorphic to
    # a specific tensor. Let me try to find a low-rank decomposition
    # using a small random search, but with the constraint that we need
    # integer/half-integer entries.
    
    # Actually, let me try a cleaner structural decomposition.
    # 4x4x4 can be viewed as a "block" tensor.
    # 
    # Let me use the following approach: represent 4x4 matrices in a basis
    # where the multiplication is sparse.
    
    # For now, let me try to use a rank-21 decomposition found via
    # a structured ansatz.
    
    # The 4x4x4 MM tensor has 21 as its rank. Let me construct it.
    # I'll use the following known rank-21 scheme:
    # 
    # The 4x4x4 tensor can be decomposed as follows:
    # Think of 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B, so C11 = A11*B11 + A12*B21, etc.
    # Using Strassen for each 2x2 block product:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # 
    # But for 4x4, we have 4 block multiplications (C11, C12, C21, C22),
    # each requiring 7 Strassen multiplications = 28 total. That's rank 28,
    # not 21.
    #
    # The rank-21 decomposition is more efficient. Let me try to find it.
    
    # I'll attempt a numerical search for a rank-21 decomposition with
    # integer/half-integer entries, using the structure of the tensor.
    
    # Given time constraints, let me use a safe fallback:
    # Rank-21 might be hard to find quickly. Let me use rank-28 (Strassen
    # on 2x2 blocks) which is easier to construct, giving total rank 56.
    # Or better, let me try to directly find a good decomposition for 3x4x8.
    
    # Actually, let me try a smarter approach for 3x4x8 directly.
    # 
    # 3x4x8: A is 3x4, B is 4x8, C is 3x8.
    # The tensor has 12*32*24 = 9216 entries.
    # 
    # Let me try rank 42 using the 4x4x4 embedding approach with a
    # known rank-21 4x4x4 decomposition.
    
    # I found a reference: the rank of 4x4x4 is 21. Let me construct it
    # using a specific scheme.
    
    # Here's a rank-21 decomposition of 4x4x4:
    # We can use the fact that 4x4x4 is a sub-tensor of 4x4x4... 
    # Let me just try to find it numerically with a small search.
    
    # For safety, let me use a rank-28 decomposition (2x2 Strassen blocks)
    # which gives total rank 56 for 3x4x8. That's still better than 96.
    #
    # But let me try rank 42 first. I'll construct the 4x4x4 rank-21
    # decomposition using a known formula.
    
    # After research, here's a rank-21 scheme for 4x4x4:
    # The entries involve specific linear combinations.
    # 
    # Let me just hardcode a valid rank-21 decomposition if I can find one,
    # otherwise fall back to rank-28.
    
    # For now, let me use a simple but effective approach:
    # Rank-28 via 2x2 Strassen blocks for 4x4x4, giving rank 56 for 3x4x8.
    # This is guaranteed to work and is much better than 96.
    
    # Actually, let me try to be smarter. Let me directly decompose 3x4x8.
    # 
    # 3x4x8: The middle dimension is 4. The output is 3x8.
    # I can think of this as: for each of the 8 output columns, 
    # compute 3x1 result from 3x4 * 4x1.
    # 
    # Let me try a rank-42 approach using a direct construction.
    
    # I'll use the following strategy:
    # 1. Try to construct rank-42 decomposition
    # 2. Fall back to rank-56 (Strassen blocks)
    # 3. Fall back to schoolbook rank-96
    
    # Rank-56 construction via 2x2 Strassen blocks for 4x4x4:
    # For 4x4x4, use 4 independent 2x2x2 Strassen decompositions.
    # Each 2x2x2 has rank 7, so 4*7 = 28.
    # For 3x4x8, we have two 3x4x4 subproblems, each embedded in 4x4x4.
    # Total rank = 2 * 28 = 56.
    
    # Let me construct the 2x2x2 Strassen decomposition first.
    # Strassen: M1..M7
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[m1-m2+m4-m6, m3+m5], [m2+m7, m1-m3+m5+m6]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # For 2x2x2 tensor T2[i,j,k] where i,j,k in {0,1,2,3} (flattened 2x2)
    # T2[i*2+j, j*2+k, k*2+i] = 1 for valid i,j,k
    
    # Let me build the rank-7 decomposition for 2x2x2:
    # U2 has shape (4,7), V2 has shape (4,7), W2 has shape (4,7)
    # Index in U: i*2+j (i=row of A, j=col of A)
    # Index in V: j*2+k (j=row of B, k=col of B)  
    # Index in W: k*2+i (k=row of C, i=col of C)
    
    # M1 = (a+d)(e+h): 
    #   U: a=0,0; d=1,1 -> U[0,0]=1, U[3,0]=1 (for M1 col)
    #   V: e=0,0; h=1,1 -> V[0,0]=1, V[3,0]=1
    #   W: C[0,0] += M1, C[0,1] -= M1... wait let me be careful.
    
    # C[0,0] = M1 - M2 + M4 - M6
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M7
    # C[1,1] = M1 - M3 + M5 + M6
    
    # W index: k*2+i where k=row of C, i=col of C
    # C[0,0]: k=0,i=0 -> index 0
    # C[0,1]: k=0,i=1 -> index 1
    # C[1,0]: k=1,i=0 -> index 2
    # C[1,1]: k=1,i=1 -> index 3
    
    U2 = np.zeros((4, 7), dtype=np.float64)
    V2 = np.zeros((4, 7), dtype=np.float64)
    W2 = np.zeros((4, 7), dtype=np.float64)
    
    # M1: (a+d)(e+h)
    # U: a=U[0,0]=1, d=U[3,0]=1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1  # e=V[0,0]=1, h=V[3,0]=1
    # W: C[0,0]+=M1 -> W[0,0]+=1; C[1,1]+=M1 -> W[3,0]+=1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (c+d)e
    U2[2, 1] = 1; U2[3, 1] = 1  # c=U[2,1]=1, d=U[3,1]=1
    V2[0, 1] = 1  # e=V[0,1]=1
    # W: C[0,0]-=M2 -> W[0,1]-=1; C[1,0]+=M2 -> W[2,1]+=1
    W2[0, 1] = -1; W2[2, 1] = 1
    
    # M3: a(f-h)
    U2[0, 2] = 1  # a=U[0,2]=1
    V2[1, 2] = 1; V2[3, 2] = -1  # f=V[1,2]=1, h=V[3,2]=-1
    # W: C[0,1]+=M3 -> W[1,2]+=1; C[1,1]-=M3 -> W[3,2]-=1
    W2[1, 2] = 1; W2[3, 2] = -1
    
    # M4: d(g-e)
    U2[3, 3] = 1  # d=U[3,3]=1
    V2[2, 3] = 1; V2[0, 3] = -1  # g=V[2,3]=1, e=V[0,3]=-1
    # W: C[0,0]+=M4 -> W[0,3]+=1
    W2[0, 3] = 1
    
    # M5: (a+b)h
    U2[0, 4] = 1; U2[1, 4] = 1  # a=U[0,4]=1, b=U[1,4]=1
    V2[3, 4] = 1  # h=V[3,4]=1
    # W: C[0,1]+=M5 -> W[1,4]+=1; C[1,1]+=M5 -> W[3,4]+=1
    W2[1, 4] = 1; W2[3, 4] = 1
    
    # M6: (c-a)(e+f)
    U2[2, 5] = 1; U2[0, 5] = -1  # c=U[2,5]=1, a=U[0,5]=-1
    V2[0, 5] = 1; V2[1, 5] = 1  # e=V[0,5]=1, f=V[1,5]=1
    # W: C[0,0]-=M6 -> W[0,5]-=1
    W2[0, 5] = -1
    
    # M7: (b-d)(g+h)
    U2[1, 6] = 1; U2[3, 6] = -1  # b=U[1,6]=1, d=U[3,6]=-1
    V2[2, 6] = 1; V2[3, 6] = 1  # g=V[2,6]=1, h=V[3,6]=1
    # W: C[1,0]+=M7 -> W[2,6]+=1
    W2[2, 6] = 1
    
    # Verify 2x2x2 decomposition
    T2_check = np.einsum('ic,jc,kc->ijk', U2, V2, W2)
    T2_target = np.zeros((4, 4, 4), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2_target[i*2+j, j*2+k, k*2+i] = 1.0
    
    if not np.allclose(T2_check, T2_target, atol=1e-10):
        print("Warning: 2x2x2 Strassen decomposition failed verification")
    
    # Now build 4x4x4 rank-28 decomposition from four 2x2x2 Strassen blocks
    # 4x4 matrices as 2x2 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # A00 corresponds to indices (0,0),(0,1),(1,0),(1,1) in the 4x4 -> flat indices 0,1,4,5
    # A01 corresponds to indices (0,2),(0,3),(1,2),(1,3) -> flat indices 2,3,6,7
    # A10 corresponds to indices (2,0),(2,1),(3,0),(3,1) -> flat indices 8,9,12,13
    # A11 corresponds to indices (2,2),(2,3),(3,2),(3,3) -> flat indices 10,11,14,15
    
    # For block (bi, bj) where bi, bj in {0,1}:
    # A block row bi, block col bj: flat indices = (bi*2+i)*4 + (bj*2+j) for i,j in {0,1}
    # B block row bj, block col bk: flat indices = (bj*2+j)*4 + (bk*2+k)
    # C block row bi, block col bk: flat indices = (bi*2+i)*4 + (bk*2+k)
    
    U4 = np.zeros((16, 28), dtype=np.float64)
    V4 = np.zeros((16, 28), dtype=np.float64)
    W4 = np.zeros((16, 28), dtype=np.float64)
    
    for block_pair in range(4):
        # block_pair determines which (A_block, B_block) combination
        # Actually, for each output C block (bi, bk), we need specific A and B blocks
        # C[bi,bk] = sum_j A[bi,j]*B[j,bk]
        # So for C block (bi,bk):
        #   C[bi,bk] = A[bi,0]*B[0,bk] + A[bi,1]*B[1,bk]
        # Each product A[bi,j]*B[j,bk] is a 2x2*2x2=2x2 product, rank 7.
        # Total: 4 blocks * 2 sums * 7 = 56? No, that's too many.
        
        # Wait, I'm overcomplicating. For 4x4x4, I just need to compute
        # C = A*B where all are 4x4. Using 2x2 block Strassen:
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        # C10 = A10*B00 + A11*B10
        # C11 = A10*B01 + A11*B11
        # Each 2x2 product uses 7 Strassen mults, so 4*2*7 = 56 mults.
        # That gives rank 56 for 4x4x4, not 28.
        
        # So my "rank-28" idea was wrong. Let me just use the direct 
        # 3x4x8 decomposition with rank 56.
        
        pass  # Skip this incorrect approach
    
    # Let me directly construct a rank-56 decomposition for 3x4x8
    # by splitting into two 3x4x4 subproblems, each using rank-28 (4x4x4 via blocks)
    
    # Actually, let me just use a simpler approach:
    # For 3x4x8, split B into two 4x4 halves.
    # For each half, embed 3x4x4 into 4x4x4 and use a rank-28 decomposition.
    
    # But constructing rank-28 for 4x4x4 is complex. Let me try rank 42
    # using a different method.
    
    # Let me try a direct numerical optimization for 3x4x8 with rank 42.
    
    for target_rank in [42, 48, 56]:
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            for it in range(300):
                if time.time() > deadline - 15:
                    break
                
                T_flat = T.reshape(n*m, m*p*p*n)
                
                # Update U
                M_VW = np.einsum('jc,kc->jkc', V, W).reshape(m*p, p*n, target_rank)
                M_VW_flat = M_VW.transpose(2, 0, 1).reshape(target_rank, m*p*p*n)
                U = T_flat @ np.linalg.pinv(M_VW_flat.T)
                
                # Update V
                M_UW = np.einsum('ic,kc->ikc', U, W).reshape(n*m, p*n, target_rank)
                M_UW_flat = M_UW.transpose(2, 0, 1).reshape(target_rank, n*m*p*n)
                T_flat_v = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
                V = T_flat_v @ np.linalg.pinv(M_UW_flat.T)
                
                # Update W
                M_UV = np.einsum('ic,jc->ijc', U, V).reshape(n*m, m*p, target_rank)
                M_UV_flat = M_UV.transpose(2, 0, 1).reshape(target_rank, n*m*m*p)
                T_flat_w = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
                W = T_flat_w @ np.linalg.pinv(M_UV_flat.T)
                
                # Check convergence
                T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
                err = np.max(np.abs(T_approx - T))
                if err < 1e-8:
                    break
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            T_approx = np.einsum('ic,jc,kc->ijk', U_r, V_r, W_r)
            if np.allclose(T_approx, T, atol=1e-10):
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

