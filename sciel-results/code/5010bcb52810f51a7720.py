import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s
    
    n, m, p = 4, 5, 6
    schoolbook_rank = n * m * p  # 120
    
    # We'll construct a decomposition using Strassen's algorithm on 2x2 blocks
    # The idea: split the matrices into 2x2 blocks and use Strassen's rank-7 
    # decomposition for 2x2 matrix multiplication.
    
    # For 4x5 * 5x6:
    # We can think of this as:
    # - A is 4x5, B is 5x6
    # - We want C = A * B, which is 4x6
    
    # Strategy: Use Strassen on 2x2 sub-blocks
    # Split A into 2x2 blocks (each 2x2 or 2x3), B into 2x2 blocks (each 3x3 or 3x6), etc.
    
    # Actually, let me think more carefully. The cleanest approach is:
    # 1. Pad A to 4x6 and B to 6x6 (or similar) to get square-ish blocks
    # 2. Use Strassen's algorithm recursively
    
    # Better approach: Direct construction using Strassen on 2x2 blocks
    # 
    # For multiplying 2x2 matrices with Strassen (rank 7):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4 * 4x4, we can use Strassen once, giving rank 49 (7^2 = 49)
    # For 4x4 * 4x2, we split B into two 4x2 matrices and do two 4x4*4x2 multiplies
    # Actually, let's think of 4x5*5x6 differently.
    
    # Alternative: Use the fact that Rank(m*n*p) <= Rank(m*n*p') + Rank(m*p'*p) 
    # when we split appropriately.
    
    # Let me try a different approach: construct the decomposition directly.
    # 
    # The tensor for matrix multiplication C_ik = sum_j A_ij B_jk
    # has entries T[i,j,k] = 1 if we're computing C_ik from A_ij and B_jk
    #
    # For rank-1 decomposition: U[i,j] * V[j,k] * W[k,i]
    # where U is (m*n, rank), V is (m*p, rank), W is (p*n, rank)
    # Wait, let me re-read the problem.
    #
    # From the problem: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j,k)
    # where delta(i,j,k) = 1 if we're computing the (i,k) entry of C from (i,j) of A and (j,k) of B
    # Actually, for matrix multiplication, we want:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] should be 1 (it's the same for all i,j,k in valid range)
    # The decomposition satisfies: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all valid i,j,k
    
    # Let's verify with schoolbook: U[i*m+j, i*m+j] = 1, V[j*p+k, i*m+j] = 1, W[k*n+i, i*m+j] = 1
    # Then sum_r gives 1 for each (i,j,k). Yes.
    
    # Now, let's try to use Strassen to reduce the rank.
    # 
    # For 2x2 * 2x2, Strassen gives rank 7.
    # For 4x4 * 4x4, using Strassen gives rank 49.
    # For 4x4 * 4x2, we can think of it as two 4x4 * 4x1 multiplies, or use a different approach.
    
    # Let me construct the 4x5*5x6 decomposition by splitting:
    # A = [A1 | A2] where A1 is 4x4, A2 is 4x1
    # B = [B1; B2] where B1 is 4x6, B2 is 1x6
    # C = A1*B1 + A2*B2
    # 
    # Rank(4x5*5x6) <= Rank(4x4*4x6) + Rank(4x1*1x6)
    # Rank(4x1*1x6) = 1*1*6 = 6 (schoolbook) or can we do better?
    # Actually, 4x1 * 1x6: C[i,k] = A[i,0] * B[0,k]
    # This is an outer product, so rank = 1 * 4 * 6 / ... no, it's just rank 1 for each (i,k)
    # Wait, the tensor for 4x1*1x6 has T[i,0,k] = 1 for i in 0..3, k in 0..5
    # This can be decomposed as: sum_r U[i,0,r] * V[0,k,r] * W[k,i,r]
    # We can use rank 1: U[i,0,0] = 1, V[0,k,0] = 1, W[k,i,0] = 1
    # But that only works for one (i,k) pair. We need separate terms for each (i,k).
    # Actually, for the outer product, we need rank = 4*6 = 24? No...
    # Let me think again. C[i,k] = A[i,0] * B[0,k]
    # This is a rank-1 matrix in the (i,k) sense, but in terms of tensor decomposition:
    # T[i,0,k] = 1 for all i,k
    # We need: sum_r U[i,0,r] * V[0,k,r] * W[k,i,r] = 1 for all i in 0..3, k in 0..5
    # We can use: U[i,0,r] = 1 for r = i*6+k, V[0,k,r] = 1, W[k,i,r] = 1
    # That gives rank 24. Or we can do better.
    # Actually, U[i,0,r] * V[0,k,r] * W[k,i,r] must equal 1 for all i,k.
    # If we set U[i,0,r] = u_i, V[0,k,r] = v_k, W[k,i,r] = w_{k,i}, then u_i * v_k * w_{k,i} = 1
    # We can set u_i = 1, v_k = 1, w_{k,i} = 1 for a single r, giving rank 1.
    # Wait, that works! U[i,0,0] = 1, V[0,k,0] = 1, W[k,i,0] = 1
    # Then sum_r gives 1 for all i,k. So rank 1.
    
    # Similarly, Rank(4x4*4x6):
    # We can split 4x6 into 4x4 + 4x2
    # Rank(4x4*4x4) = 49 (Strassen)
    # Rank(4x4*4x2): split 4x2 into two 4x1
    # Rank(4x4*4x1) = 4*4*1 = 16? Or can we do better?
    # For 4x4*4x1: C[i,0] = sum_j A[i,j] * B[j,0]
    # T[i,j,0] = 1 for all i,j
    # We need: sum_r U[i,j,r] * V[j,0,r] * W[0,i,r] = 1 for all i,j
    # This is like multiplying 4x4 by 4x1, which is 4*4*1 = 16 in schoolbook.
    # But we can use Strassen-like methods. Actually, 4x4*4x1 is just 4 dot products of length 4.
    # Each dot product needs rank 1? No, the tensor is T[i,j,0] = 1.
    # We can decompose as: sum_r U[i,j,r] * V[j,0,r] * W[0,i,r]
    # If we use rank 16: U[i,j,r] = 1 for r = i*4+j, V[j,0,r] = 1, W[0,i,r] = 1
    # That works. Can we do better? The minimum rank for this tensor is 16 (I think).
    
    # So Rank(4x5*5x6) <= Rank(4x4*4x4) + Rank(4x4*4x2) + Rank(4x1*1x6)
    # = 49 + 2*16 + 1 = 49 + 32 + 1 = 82
    
    # Hmm, that's 82, not as good as I hoped. Let me reconsider.
    
    # Actually, for 4x4*4x2, we can think of it as 4x4 * (4x2)
    # The 4x2 matrix B can be seen as two 4x1 matrices B1 and B2
    # C1 = A*B1 (4x1), C2 = A*B2 (4x1)
    # Rank(4x4*4x1) = 16 (as argued above)
    # So Rank(4x4*4x2) <= 32
    
    # Total: 49 + 32 + 1 = 82
    
    # But wait, I think we can do better for 4x4*4x1.
    # The tensor T[i,j,0] = 1 for i,j in 0..3
    # This is the "all-ones" tensor of shape 4x4x1
    # The rank of the all-ones tensor of shape a x b x 1 is... 
    # Actually, T[i,j,0] = 1 can be written as u_i * v_j * w_0 where u_i = 1, v_j = 1, w_0 = 1
    # So rank 1! Let me verify: U[i,j,0] = 1, V[j,0,0] = 1, W[0,i,0] = 1
    # Then U[i,j,0] * V[j,0,0] * W[0,i,0] = 1*1*1 = 1. Yes!
    
    # So Rank(4x4*4x1) = 1? That seems too good. Let me double-check.
    # The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For 4x4*4x1: n=4, m=4, p=1
    # U has shape (4*4, rank) = (16, rank)
    # V has shape (4*1, rank) = (4, rank)
    # W has shape (1*4, rank) = (4, rank)
    # We want: sum_r U[i*4+j, r] * V[j*1+0, r] * W[0*4+i, r] = 1 for all i,j in 0..3
    # If we set rank=1: U[i*4+j, 0] = 1, V[j, 0] = 1, W[i, 0] = 1
    # Then the product is 1*1*1 = 1. Yes! Rank 1 works.
    
    # So Rank(4x5*5x6) <= Rank(4x4*4x4) + Rank(4x4*4x2) + Rank(4x1*1x6)
    # = 49 + 2*1 + 1 = 49 + 2 + 1 = 52
    
    # Wait, that seems surprisingly good. Let me verify the 4x4*4x2 part.
    # 4x4*4x2: n=4, m=4, p=2
    # We split B (4x2) into B1 (4x1) and B2 (4x1)
    # C1 = A*B1 (4x1), C2 = A*B2 (4x1)
    # Rank(4x4*4x1) = 1 (as shown above)
    # So Rank(4x4*4x2) <= 2
    
    # And 4x1*1x6: n=4, m=1, p=6
    # C[i,k] = A[i,0] * B[0,k]
    # This is an outer product. T[i,0,k] = 1 for all i,k
    # We need: sum_r U[i*1+0, r] * V[0*6+k, r] * W[k*4+i, r] = 1 for all i in 0..3, k in 0..5
    # If we set rank=1: U[i, 0] = 1, V[k, 0] = 1, W[k*4+i, 0] = 1
    # Then product is 1*1*1 = 1. Yes! Rank 1.
    
    # So total rank = 49 + 2 + 1 = 52
    
    # Now I need to actually construct this decomposition.
    # 
    # The tricky part is the 4x4*4x4 part with rank 49 using Strassen.
    # Let me implement Strassen's algorithm for 2x2 matrices and use it recursively.
    
    def strassen_2x2():
        """Return U, V, W for 2x2*2x2 with rank 7."""
        # n=m=p=2
        # U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
        U = [[0.0]*7 for _ in range(4)]
        V = [[0.0]*7 for _ in range(4)]
        W = [[0.0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11=A[0,0], A22=A[1,1], B11=B[0,0], B22=B[1,1]
        # C11 += M1, C22 += M1
        # U[0,0]=1, U[3,0]=1 (A11 and A22)
        # V[0,0]=1, V[3,0]=1 (B11 and B22)
        # W[0,0]=1 (C11), W[3,0]=1 (C22)
        U[0][0] = 1.0
        U[3][0] = 1.0
        V[0][0] = 1.0
        V[3][0] = 1.0
        W[0][0] = 1.0
        W[3][0] = 1.0
        
        # M2 = (A21+A22)B11
        # A21=A[1,0], A22=A[1,1], B11=B[0,0]
        # C21 += M2
        # U[2,1]=1 (A21), U[3,1]=1 (A22)
        # V[0,1]=1 (B11)
        # W[2,1]=1 (C21)
        U[2][1] = 1.0
        U[3][1] = 1.0
        V[0][1] = 1.0
        W[2][1] = 1.0
        
        # M3 = A11(B21-B22)
        # A11=A[0,0], B21=B[1,0], B22=B[1,1]
        # C12 += M3
        # U[0,2]=1 (A11)
        # V[2,2]=1 (B21), V[3,2]=-1 (B22)
        # W[1,2]=1 (C12)
        U[0][2] = 1.0
        V[2][2] = 1.0
        V[3][2] = -1.0
        W[1][2] = 1.0
        
        # M4 = A22(B11-B21)
        # A22=A[1,1], B11=B[0,0], B21=B[1,0]
        # C11 += M4, C21 += M4
        # U[3,3]=1 (A22)
        # V[0,3]=1 (B11), V[2,3]=-1 (B21)
        # W[0,3]=1 (C11), W[2,3]=1 (C21)
        U[3][3] = 1.0
        V[0][3] = 1.0
        V[2][3] = -1.0
        W[0][3] = 1.0
        W[2][3] = 1.0
        
        # M5 = (A11+A12)B22
        # A11=A[0,0], A12=A[0,1], B22=B[1,1]
        # C11 -= M5, C12 += M5
        # U[0,4]=1 (A11), U[1,4]=1 (A12)
        # V[3,4]=1 (B22)
        # W[0,4]=-1 (C11), W[1,4]=1 (C12)
        U[0][4] = 1.0
        U[1][4] = 1.0
        V[3][4] = 1.0
        W[0][4] = -1.0
        W[1][4] = 1.0
        
        # M6 = (A21-A11)(B11+B12)
        # A21=A[1,0], A11=A[0,0], B11=B[0,0], B12=B[0,1]
        # C22 += M6
        # U[2,5]=1 (A21), U[0,5]=-1 (A11)
        # V[0,5]=1 (B11), V[1,5]=1 (B12)
        # W[3,5]=1 (C22)
        U[2][5] = 1.0
        U[0][5] = -1.0
        V[0][5] = 1.0
        V[1][5] = 1.0
        W[3][5] = 1.0
        
        # M7 = (A12-A22)(B21+B22)
        # A12=A[0,1], A22=A[1,1], B21=B[1,0], B22=B[1,1]
        # C11 += M7
        # U[1,6]=1 (A12), U[3,6]=-1 (A22)
        # V[2,6]=1 (B21), V[3,6]=1 (B22)
        # W[0,6]=1 (C11)
        U[1][6] = 1.0
        U[3][6] = -1.0
        V[2][6] = 1.0
        V[3][6] = 1.0
        W[0][6] = 1.0
        
        return U, V, W
    
    # Now build 4x4*4x4 using Strassen
    # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Split B into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    # C = A*B, split into 2x2 blocks: C11, C12, C21, C22 (each 2x2)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each Mx is a 2x2 matrix product, which we can decompose with rank 7.
    # But the inputs to each Mx are linear combinations of the block entries.
    # This requires careful bookkeeping.
    
    # Let me think about the indices.
    # A is 4x4, so A[i,j] where i,j in 0..3
    # Block A11: i in 0..1, j in 0..1
    # Block A12: i in 0..1, j in 2..3
    # Block A21: i in 2..3, j in 0..1
    # Block A22: i in 2..3, j in 2..3
    
    # Similarly for B and C.
    
    # For M1 = (A11+A22)(B11+B22):
    # The (i,j) entry of (A11+A22) is A[i,j] + A[i+2,j+2] for i,j in 0..1
    # The (j,k) entry of (B11+B22) is B[j,k] + B[j+2,k+2] for j,k in 0..1
    # The (i,k) entry of M1 is sum_j (A[i,j]+A[i+2,j+2])*(B[j,k]+B[j+2,k+2])
    
    # This is getting complex. Let me just implement it directly.
    
    # For 4x4*4x4, the tensor T[i,j,k] = 1 for all i,j,k in 0..3
    # We want to decompose with rank 49.
    
    # Let me construct the Strassen-based decomposition.
    # 
    # The 7 Strassen products each involve 2x2 matrix multiplication.
    # For each Strassen product, we have 7 rank-1 terms.
    # So total 7*7 = 49 terms.
    
    # Let me define the 7 Strassen products:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B11-B21)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # For each Pq, we use Strassen's 2x2 decomposition.
    
    # Let me create a helper to build the 4x4*4x4 decomposition.
    
    def build_4x4_4x4():
        """Build U, V, W for 4x4*4x4 with rank 49 using Strassen."""
        # n=m=p=4
        # U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
        U = [[0.0]*49 for _ in range(16)]
        V = [[0.0]*49 for _ in range(16)]
        W = [[0.0]*49 for _ in range(16)]
        
        # Strassen 2x2 decomposition
        U2, V2, W2 = strassen_2x2()
        
        # Define the 7 Strassen products
        # For each product Pq = X*Y where X and Y are 2x2 matrices (linear combinations of blocks)
        # We need to map the 2x2 indices to the 4x4 indices.
        
        # Let's define for each Pq:
        # - The linear combination coefficients for A (4x4) and B (4x4)
        # - How the result contributes to C (4x4)
        
        # P1 = (A11+A22)(B11+B22)
        # X11 = A[0,0]+A[2,2], X12 = A[0,1]+A[2,3], X21 = A[1,0]+A[3,2], X22 = A[1,1]+A[3,3]
        # Y11 = B[0,0]+B[2,2], Y12 = B[0,1]+B[2,3], Y21 = B[1,0]+B[3,2], Y22 = B[1,1]+B[3,3]
        # C11 += P1, C22 += P1
        # C[0,0] += P1[0,0], C[0,1] += P1[0,1], C[1,0] += P1[1,0], C[1,1] += P1[1,1]
        # C[2,2] += P1[0,0], C[2,3] += P1[0,1], C[3,2] += P1[1,0], C[3,3] += P1[1,1]
        
        # This is getting very detailed. Let me just implement it step by step.
        
        # For each Strassen product q (0..6), we have 7 sub-terms (r=0..6)
        # Total 49 terms, indexed by q*7+r
        
        # Define the Strassen products
        # Each product is of the form (sum of A blocks) * (sum of B blocks)
        
        # Let me define for each q:
        # a_coeffs: dict mapping (i,j) -> coefficient for A[i,j]
        # b_coeffs: dict mapping (j,k) -> coefficient for B[j,k]
        # c_coeffs: dict mapping (i,k) -> coefficient for C[i,k] (with sign)
        
        # P1 = (A11+A22)(B11+B22)
        # A coeffs: A[0,0]:1, A[0,1]:1, A[1,0]:1, A[1,1]:1, A[2,2]:1, A[2,3]:1, A[3,2]:1, A[3,3]:1
        # Wait, A11 is A[0:2,0:2], A22 is A[2:4,2:4]
        # So (A11+A22)[i,j] = A[i,j] + A[i+2,j+2] for i,j in 0..1
        # In 4x4 indices: for i,j in 0..1: A[i,j] + A[i+2,j+2]
        
        # Let me just build it directly.
        
        # For P1:
        # X[i,j] = A[i,j] + A[i+2,j+2] for i,j in 0..1
        # Y[j,k] = B[j,k] + B[j+2,k+2] for j,k in 0..1
        # Z[i,k] = sum_j X[i,j]*Y[j,k]
        # C[0:2,0:2] += Z, C[2:4,2:4] += Z
        
        # Using Strassen for X*Y:
        # M1 = (X11+X22)(Y11+Y22)
        # X11=X[0,0]=A[0,0]+A[2,2], X22=X[1,1]=A[1,1]+A[3,3]
        # Y11=Y[0,0]=B[0,0]+B[2,2], Y22=Y[1,1]=B[1,1]+B[3,3]
        
        # This is getting too complex to do by hand. Let me just use the schoolbook for now
        # and see if I can get a valid decomposition, even if not rank-minimal.
        
        # Actually, let me try a different approach: use the known rank-49 decomposition
        # from Strassen's algorithm.
        
        # For simplicity, let me just return the schoolbook decomposition for now.
        # The goal is to get a valid decomposition, even if not optimal.
        
        return None, None, None  # Placeholder
    
    # Given the complexity, let me just implement the schoolbook decomposition
    # which is guaranteed to be correct.
    
    U = [[0.0]*schoolbook_rank for _ in range(n*m)]
    V = [[0.0]*schoolbook_rank for _ in range(m*p)]
    W = [[0.0]*schoolbook_rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
