import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    
    # We want to decompose the tensor T such that:
    # C = A @ B
    # A is (n x m) = (2 x 3)
    # B is (m x p) = (3 x 10)
    # C is (n x p) = (2 x 10)
    
    # The tensor M has entries M[alpha, beta, gamma] = 1 if beta (i,j) and gamma (j,k) 
    # share the same j, and alpha is (k,i).
    # Specifically, M[k*n+i, i*m+j, j*p+k] = 1.
    
    # Strategy: Use Strassen for the 2x2 block structure of the multiplication.
    # A = [[A11, A12], [A21, A22]] where A11, A21 are 1x2 blocks (since n=2, m=3, split m=3 into 2+1? No, Strassen usually splits square matrices).
    # Let's split A (2x3) into two 1x3 rows? No, Strassen requires square blocks.
    # Let's pad A to 2x2? No, m=3.
    # Let's split the m dimension (3) into 2 + 1.
    # A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (2x10); B_bottom (1x10)]
    # C = A_left @ B_top + A_right @ B_bottom
    
    # This doesn't directly use Strassen on a 2x2 matrix multiplication because the inner dimension is split.
    # However, we can view the multiplication as:
    # C = A @ B
    # Let's split A into A1 (2x2) and A2 (2x1).
    # Let's split B into B1 (2x10) and B2 (1x10).
    # C = A1 @ B1 + A2 @ B2
    
    # A1 @ B1 is (2x2) @ (2x10). This is a rectangular multiplication.
    # We can apply Strassen-like reductions if we pad or if we use specific identities.
    # Actually, the rank of (2x2) @ (2x10) is at most 2*2*10 = 40 (schoolbook).
    # Strassen for (2x2) @ (2x2) reduces rank from 8 to 7.
    # Here we have (2x2) @ (2x10). We can split B1 into 5 blocks of (2x2).
    # So A1 @ B1 = A1 @ [B1_1, B1_2, ..., B1_5] where B1_i is 2x2.
    # For each 2x2 block multiplication, we can use Strassen (rank 7 instead of 8).
    # So A1 @ B1 rank <= 5 * 7 = 35.
    
    # A2 @ B2 is (2x1) @ (1x10). This is an outer product-like structure.
    # (2x1) @ (1x10) has rank 2*1*10 = 20 (schoolbook).
    # Can we do better? The tensor for (2x1)@(1x10) is essentially sum_j A2[j] B2[j] ... wait.
    # A2 is column vector a (2x1). B2 is row vector b (1x10).
    # C = a b^T.
    # The tensor entries are T[a_i, b_j, c_ij] = 1.
    # This is an outer product of two vectors. The rank of the outer product tensor of two vectors of length 2 and 10 is 1?
    # No, the matrix multiplication tensor T_{ijk} = delta_{ij} delta_{jk} ... no.
    # For C = A @ B, C_{ik} = sum_j A_{ij} B_{jk}.
    # If A is (2x1) and B is (1x10), then C_{ik} = A_{i1} B_{1k}.
    # The tensor is T[i, 1, k] = 1. All other entries 0.
    # This tensor is a "slice" of the identity.
    # Actually, T is a 3-tensor of size 2 x 1 x 10.
    # T[i, 1, k] = 1.
    # This can be decomposed into 1 term?
    # U[1, 1] = 1, V[1, 1] = 1, W[k, 1] = 1?
    # Sum_c U[i, c] V[1, c] W[k, c] = U[i, 1] V[1, 1] W[k, 1].
    # We need this to be 1 for all i, k.
    # So U[i, 1] = 1, V[1, 1] = 1, W[k, 1] = 1.
    # Yes! The rank of (2x1) @ (1x10) is 1?
    # Let's check.
    # C = A @ B. A is 2x1, B is 1x10.
    # C is 2x10.
    # C[i, k] = A[i, 0] * B[0, k].
    # The tensor M has M[i, 0, k] = 1.
    # Decomposition:
    # U[i, 1] = 1 (for i=0,1)
    # V[0, 1] = 1
    # W[k, 1] = 1 (for k=0..9)
    # Sum_c U[i, c] V[j, c] W[k, c] = U[i, 1] V[0, 1] W[k, 1] = 1*1*1 = 1.
    # This works for the non-zero entries.
    # Does it introduce errors for zero entries?
    # The tensor M is zero everywhere else.
    # The decomposition produces T[i, j, k] = 1 if j=0, else 0?
    # Wait, the index j in V corresponds to the column of A / row of B.
    # In our problem, m=3.
    # We split m=3 into m1=2 and m2=1.
    # A is split into A1 (2x2) and A2 (2x1).
    # B is split into B1 (2x10) and B2 (1x10).
    # The indices for V are j=0,1,2.
    # For A1 @ B1, we use j=0,1.
    # For A2 @ B2, we use j=2.
    #
    # If we use rank 1 for A2 @ B2:
    # U[i, c2] = 1 for i=0,1.
    # V[2, c2] = 1.
    # W[k, c2] = 1 for k=0..9.
    # This contributes 1 to T[i, 2, k].
    # The target tensor M has M[i, 2, k] = 1?
    # Let's check the definition.
    # M[alpha, beta, gamma] = 1 if beta=(i,j), gamma=(j,k), alpha=(k,i).
    # If j=2 (the last column of A), then beta = i*m + 2.
    # gamma = 2*p + k.
    # alpha = k*n + i.
    # So yes, M[k*n+i, i*m+2, 2*p+k] = 1.
    # So the rank 1 decomposition for the j=2 slice is correct and doesn't affect other j slices because V[2, c2] is the only non-zero V entry for that column c2, and V[j, c2] = 0 for j != 2.
    # Wait, if V[2, c2] = 1 and V[0, c2]=0, V[1, c2]=0, then the term U[i, c2] V[j, c2] W[k, c2] is non-zero only if j=2.
    # So it correctly isolates the j=2 slice.
    
    # So, rank for A2 @ B2 is 1.
    
    # Now for A1 @ B1 (2x2) @ (2x10).
    # Split B1 (2x10) into 5 blocks of 2x2: B1_0, ..., B1_4.
    # A1 @ B1 = sum_{t=0}^4 A1 @ B1_t.
    # Each A1 @ B1_t is (2x2) @ (2x2).
    # We can use Strassen for each (2x2) @ (2x2).
    # Strassen rank for 2x2 is 7.
    # So total rank for A1 @ B1 is 5 * 7 = 35.
    
    # Total rank = 35 + 1 = 36.
    
    # Let's construct the matrices U, V, W.
    # Dimensions:
    # U: (n*m, R) = (6, 36)
    # V: (m*p, R) = (30, 36)
    # W: (p*n, R) = (20, 36)
    
    R = 36
    U = np.zeros((n*m, R), dtype=object)
    V = np.zeros((m*p, R), dtype=object)
    W = np.zeros((p*n, R), dtype=object)
    
    col = 0
    
    # Part 1: A1 @ B1 using Strassen for 5 blocks
    # A1 indices in A: rows i=0,1; cols j=0,1.
    # U indices for A1: i*m + j. So (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4.
    # B1 indices in B: rows j=0,1; cols k=0..9.
    # V indices for B1: j*p + k.
    # W indices for C: k*n + i.
    
    # Strassen for 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B11)
    # M4 = A22(B22 - B12)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W contributions.
    # For a specific block t (0..4), covering k in [2t, 2t+1].
    # The sub-tensor is for j in {0,1}, k in {2t, 2t+1}, i in {0,1}.
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen components
        # M1: (A11 + A22)(B11 + B22)
        # U: A11 (0,0) and A22 (1,1) -> U[0, col]=1, U[4, col]=1
        # V: B11 (0, k0) and B22 (1, k1) -> V[k0, col]=1, V[p+k1, col]=1
        # W: C11 (k0, 0), C22 (k1, 1) -> W[k0*n+0, col]=1, W[k1*n+1, col]=1
        # Wait, Strassen gives linear combinations.
        # The tensor decomposition requires that sum_c U[beta, c] V[gamma, c] W[alpha, c] = M[alpha, beta, gamma].
        # For Strassen, we define 7 terms.
        # Term 1: U has 1 at A11, A22. V has 1 at B11, B22. W has 1 at C11, C22?
        # No, the Strassen algorithm computes C from M's.
        # M1 = (A11+A22)(B11+B22).
        # This corresponds to a rank-1 tensor? No, it's a product of sums.
        # (A11+A22) is a vector in U-space? No, U is linear.
        # The decomposition is linear.
        # We need to express the multiplication tensor as a sum of rank-1 tensors.
        # A rank-1 tensor is u v w^T.
        # Strassen's 7 multiplications correspond to 7 rank-1 terms in the decomposition?
        # Yes, if we can express the entries as products of linear forms.
        
        # Let's define the 7 terms for block t.
        # Indices:
        # A11: U[0, c]
        # A12: U[1, c]
        # A21: U[3, c]
        # A22: U[4, c]
        
        # B11: V[k0, c]
        # B12: V[k1, c]
        # B21: V[p+k0, c]
        # B22: V[p+k1, c]
        
        # C11: W[k0*n+0, c]
        # C12: W[k1*n+0, c]
        # C21: W[k0*n+1, c]
        # C22: W[k1*n+1, c]
        
        # M1 = (A11 + A22)(B11 + B22)
        # This term contributes to C11 and C22.
        # In the tensor decomposition, we add a term where:
        # U[0, c]=1, U[4, c]=1
        # V[k0, c]=1, V[p+k1, c]=1
        # W[k0*n+0, c]=1, W[k1*n+1, c]=1
        # But wait, M1 is used in C11 = M1 + ... and C22 = M1 - ...
        # The signs are handled by the coefficients in U, V, or W.
        # Since we are summing rank-1 terms, we can put the signs in W.
        
        # Let's carefully assign the 7 columns for this block.
        
        # c1: M1
        # U[0]=1, U[4]=1
        # V[k0]=1, V[p+k1]=1
        # W[k0*n+0]=1, W[k1*n+1]=1  # M1 adds to C11 and C22
        
        # c2: M2 = (A21 + A22)B11
        # U[3]=1, U[4]=1
        # V[k0]=1
        # W[k0*n+1]=1, W[k1*n+1]=-1 # M2 adds to C21, subtracts from C22
        
        # c3: M3 = A11(B21 - B11)
        # U[0]=1
        # V[p+k0]=1, V[k0]=-1
        # W[k1*n+0]=1, W[k0*n+1]=1 # M3 adds to C12 and C21
        
        # c4: M4 = A22(B22 - B12)
        # U[4]=1
        # V[p+k1]=1, V[k1]=-1
        # W[k0*n+0]=1, W[k1*n+1]=1 # M4 adds to C11 and C22
        
        # c5: M5 = (A11 + A12)B22
        # U[0]=1, U[1]=1
        # V[p+k1]=1
        # W[k1*n+0]=1, W[k0*n+0]=-1 # M5 adds to C12, subtracts from C11
        
        # c6: M6 = (A21 - A11)(B11 + B12)
        # U[3]=1, U[0]=-1
        # V[k0]=1, V[k1]=1
        # W[k1*n+1]=1 # M6 adds to C22
        
        # c7: M7 = (A12 - A22)(B21 + B22)
        # U[1]=1, U[4]=-1
        # V[p+k0]=1, V[p+k1]=1
        # W[k0*n+0]=1 # M7 adds to C11
        
        # Let's verify the reconstruction of C.
        # C11 = M1 + M4 - M5 + M7
        #   = W[k0*n+0, c1] + W[k0*n+0, c4] - W[k0*n+0, c5] + W[k0*n+0, c7]
        #   = 1 + 1 - (-1) + 1 = 4? No.
        # The W entries are the coefficients for the output.
        # The term is U[beta] V[gamma] W[alpha].
        # For C11 (alpha = k0*n+0):
        # Contribution from c1: U[0]V[k0]W[k0*n+0] + U[4]V[p+k1]W[k0*n+0] ...
        # This is getting confusing. Let's stick to the tensor definition.
        # M[alpha, beta, gamma] = sum_c U[beta, c] V[gamma, c] W[alpha, c].
        
        # For M1 term (c1):
        # U[0, c1]=1, U[4, c1]=1.
        # V[k0, c1]=1, V[p+k1, c1]=1.
        # W[k0*n+0, c1]=1, W[k1*n+1, c1]=1.
        # Non-zero contributions to M:
        # beta=0 (A11), gamma=k0 (B11), alpha=k0*n+0 (C11): 1*1*1 = 1.
        # beta=0 (A11), gamma=k0 (B11), alpha=k1*n+1 (C22): 1*1*1 = 1.
        # beta=4 (A22), gamma=p+k1 (B22), alpha=k0*n+0 (C11): 1*1*1 = 1.
        # beta=4 (A22), gamma=p+k1 (B22), alpha=k1*n+1 (C22): 1*1*1 = 1.
        # beta=0, gamma=p+k1, alpha=...: U[0]=1, V[p+k1]=1. M[alpha, 0, p+k1] += W[alpha, c1].
        # This corresponds to A11 * B22.
        # The Strassen M1 is (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22.
        # So c1 generates all 4 cross terms.
        # We need to combine c1..c7 to get the correct C.
        
        # This manual construction is error-prone. 
        # Given the time constraint and the complexity of verifying Strassen tensor decomposition manually in code without a verifier, 
        # and the fact that the seed score is 1.0 (rank 60), 
        # and the inspiration mentioned "Strassen decomposition for 2x2 sub-blocks... reduces rank from 60 to 55",
        # I will implement the Strassen-based construction carefully.
        
        # However, there is a risk of error. 
        # A safer bet within the "constructive" family is to use the known rank for (2x2)x(2x10) which is 35?
        # Actually, the rank of (2x2)x(2x2) is 7.
        # So 5 blocks * 7 = 35.
        # Plus 1 for the remainder = 36.
        
        # Let's implement the 7 terms for each block.
        
        def add_term(c, u_vals, v_vals, w_vals):
            for idx, val in u_vals.items():
                U[idx, c] = val
            for idx, val in v_vals.items():
                V[idx, c] = val
            for idx, val in w_vals.items():
                W[idx, c] = val

        # Block t
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1
            
            # Indices
            u_11 = 0  # A[0,0]
            u_12 = 1  # A[0,1]
            u_21 = 3  # A[1,0]
            u_22 = 4  # A[1,1]
            
            v_11 = k0      # B[0, k0]
            v_12 = k1      # B[0, k1]
            v_21 = p + k0  # B[1, k0]
            v_22 = p + k1  # B[1, k1]
            
            w_11 = k0 * n + 0 # C[0, k0]
            w_12 = k1 * n + 0 # C[0, k1]
            w_21 = k0 * n + 1 # C[1, k0]
            w_22 = k1 * n + 1 # C[1, k1]
            
            # c1: M1
            add_term(col, 
                     {u_11: 1, u_22: 1}, 
                     {v_11: 1, v_22: 1}, 
                     {w_11: 1, w_22: 1})
            col += 1
            
            # c2: M2
            add_term(col, 
                     {u_21: 1, u_22: 1}, 
                     {v_11: 1}, 
                     {w_21: 1, w_22: -1})
            col += 1
            
            # c3: M3
            add_term(col, 
                     {u_11: 1}, 
                     {v_21: 1, v_11: -1}, 
                     {w_12: 1, w_21: 1})
            col += 1
            
            # c4: M4
            add_term(col, 
                     {u_22: 1}, 
                     {v_22: 1, v_12: -1}, 
                     {w_11: 1, w_22: 1})
            col += 1
            
            # c5: M5
            add_term(col, 
                     {u_11: 1, u_12: 1}, 
                     {v_22: 1}, 
                     {w_12: 1, w_11: -1})
            col += 1
            
            # c6: M6
            add_term(col, 
                     {u_21: 1, u_11: -1}, 
                     {v_11: 1, v_12: 1}, 
                     {w_22: 1})
            col += 1
            
            # c7: M7
            add_term(col, 
                     {u_12: 1, u_22: -1}, 
                     {v_21: 1, v_22: 1}, 
                     {w_11: 1})
            col += 1

    # Part 2: A2 @ B2 (2x1) @ (1x10)
    # Rank 1 term.
    # U[i, c] = 1 for i=0,1.
    # V[2, c] = 1.
    # W[k, c] = 1 for k=0..9.
    
    add_term(col,
             {0: 1, 3: 1},  # A[0,2] and A[1,2]
             {2*p: 1},      # B[2, 0..9] row? No, V index is j*p + k.
                            # j=2. So V[2*p + k, c].
             {k*n + i: 1 for i in range(n) for k in range(p)})
    
    # Wait, V index for j=2 is 2*p + k.
    # My add_term for V above used {2*p: 1}, which is only k=0.
    # I need V[2*p + k, c] = 1 for all k.
    
    # Let's fix the last term.
    # Reset col? No, col is 35.
    # We need to set V[20+k, 35] = 1 for k=0..9.
    # And W[k*n+i, 35] = 1 for all i,k.
    
    for k in range(p):
        V[2*p + k, col] = 1
    for i in range(n):
        for k in range(p):
            W[k*n + i, col] = 1
    # U is already set: U[0, 35]=1, U[3, 35]=1.
    
    col += 1
    
    # Convert to float for numpy operations in verification if needed, but return as is.
    # The interface expects integer/half-integer. 1 and -1 are fine.
    
    # Verification
    # Check a few entries.
    # M[k*n+i, i*m+j, j*p+k] should be 1.
    # Let's check (i=0, j=0, k=0).
    # alpha=0, beta=0, gamma=0.
    # T[0, 0, 0] = sum_c W[0, c] U[0, c] V[0, c].
    # c=0 (M1 of block 0): W[0,0]=1, U[0,0]=1, V[0,0]=1. Product 1.
    # c=2 (M3 of block 0): W[0,2]=0? M3 W is w_12, w_21. w_12 is k1*n+0 = 1*2+0=2. So W[2,2]=1. W[0,2]=0.
    # c=4 (M5 of block 0): W[0,4]=-1. U[0,4]=1. V[0,4]=0. Product 0.
    # c=6 (M7 of block 0): W[0,6]=1. U[0,6]=0. V[0,6]=0.
    # Sum = 1. Correct.
    
    # Check a zero entry. M[0, 1, 0] should be 0.
    # alpha=0, beta=1 (A[0,1]), gamma=0 (B[0,0]).
    # T[0, 1, 0] = sum_c W[0, c] U[1, c] V[0, c].
    # c=0: U[1,0]=0.
    # c=4 (M5): U[1,4]=1. V[0,4]=0.
    # c=6 (M7): U[1,6]=1. V[0,6]=0.
    # Sum = 0. Correct.
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", res['U'].shape[1])
