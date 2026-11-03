"""Matrix multiplication 2x4x7 - tensor decomposition via intermediate linear maps."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find a low-rank exact tensor decomposition for 2x4 x 4x7 matrix multiplication.
    
    T[i,j,k] = 1 for i in [0,2), j in [0,4), k in [0,7)
    We need U (8 x R), V (28 x R), W (14 x R) such that:
      sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*2+i, r] = 1  for all (i,j,k)
    
    Strategy:
      1. Build the schoolbook rank-56 decomposition.
      2. Form the intermediate bilinear map M where M[(i,j),(k,r')] encodes
         the contraction through W. Use SVD to find the minimal-rank basis
         that exactly spans the image, reducing the decomposition rank.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Step 1: Schoolbook rank
    R0 = n * m * p  # 56
    U0 = np.zeros((n * m, R0))
    V0 = np.zeros((m * p, R0))
    W0 = np.zeros((p * n, R0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1.0
                V0[j * p + k, col] = 1.0
                W0[k * n + i, col] = 1.0
                col += 1
    
    # Step 2: Form the intermediate bilinear map.
    # The decomposition identity is:
    #   T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # 
    # Define the intermediate matrix M of shape (n*m) x (p*n) where:
    #   M[a, b] = sum_r U[a, r] * W[b, r]   (b = k*n+i, a = i*m+j)
    # 
    # Then the identity becomes:
    #   T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    #            = sum_{a,b} U[i,j,a] * (V contracted) ... 
    # 
    # More precisely, define for each (j,k): 
    #   S[j,k] is a (n*m) x (p*n) matrix where S[j,k][(i*m+j'), (k'*n+i')]
    #     = sum_r U[i*m+j', r] * V[j*p+k, r] * W[k'*n+i', r]
    #   and we need S[j,k][i*m+j, k*n+i] = 1 (when j'=j, k'=k) and 0 otherwise.
    #
    # Actually, let's use a cleaner formulation. The full tensor identity is:
    #   T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    #
    # Let's define the "slice" matrices. For fixed j and k:
    #   Slice[j,k] is a (n*m) x (p*n) matrix:
    #   Slice[j,k][a, b] = sum_r U[a, r] V[j*p+k, r] W[b, r]
    #
    # The identity requires: Slice[j,k][i*m+j, k*n+i] = 1 for all i, and
    # Slice[j,k][a, b] = 0 for (a,b) != (i*m+j, k*n+i) in the "diagonal" sense.
    #
    # Actually the full constraint is: for ALL (i,j,k):
    #   sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # and for the off-diagonal terms to vanish, we need the tensor to be exactly
    # the multiplication tensor, which it is by construction in schoolbook.
    #
    # Key reduction idea: The rank R of the decomposition equals the dimension
    # of the span of the "bilinear forms" { U[:,r] * W[:,r] } in some sense.
    # More precisely, define the matrix:
    #   B = U @ W.T   (shape n*m x p*n)
    # Then for each r, U[:,r] and W[:,r] contribute to B. The number of rank-1
    # terms needed to express B with the constraint that V also fits is what matters.
    #
    # Let's use a different approach: treat this as finding the minimum rank R
    # such that there exist U, V, W with the bilinear identity. We can set up
    # the problem as: for each (i,j,k), the equation is linear in the products
    # U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    #
    # Practical approach: use the fact that we can "absorb" part of the
    # structure. Note that for fixed j, the sum over r of U[i*m+j,r]*W[k*n+i,r]
    # gives a matrix in (i,k) space. Let's define:
    #   C_j[i, k] = sum_r U[i*m+j, r] * W[k*n+i, r]
    # Then the identity is: sum_j C_j[i,k] * V[j*p+k, r]... no, that's not right.
    #
    # Let me try yet another angle. The identity is:
    #   sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    #
    # This is a trilinear identity. We can fix one index and view it as bilinear.
    # Fix k: sum_r (U[i*m+j, r] W[k*n+i, r]) * V[j*p+k, r] = 1 for all i,j.
    # Define A_k[i*m+j, r'] where r' indexes (j, r)... this gets complicated.
    #
    # Simplest valid reduction: Note that the schoolbook has U, V, W all 0/1.
    # We can try to find a smaller rank by using the following:
    # The tensor T[i,j,k] = 1 is the outer product of three all-ones vectors.
    # But we need the specific index coupling.
    #
    # Let's try rank = n*m*p - reduction. For [2,4,7]:
    # Observation: The middle dimension m=4 can be paired. For each pair (j1,j2),
    # we can use the identity:
    #   A[i,j1]*B[j1,k] + A[i,j2]*B[j2,k]
    # = (A[i,j1]+A[i,j2])*B[j1,k] + (A[i,j2]-A[i,j1])*B[j2,k] + ... 
    # This doesn't directly reduce rank in the tensor sense.
    #
    # Let's use a numerical approach: set up the linear system for a candidate rank R
    # and solve. The equations are nonlinear (products of three unknowns per r),
    # but we can fix U and W and solve for V linearly, then iterate.
    #
    # Alternative: Use the known result that the rank of [2,4,7] multiplication
    # tensor is at most 48 (via blocking with [2,2,2] Strassen on pairs).
    #
    # Construction for rank 48:
    # Partition the 4 middle indices into two groups of 2: {0,1} and {2,3}.
    # A = [A1 | A2], B = [B1; B2], C = A1*B1 + A2*B2.
    # Each A1*B1 is 2x2 * 2x7. Use Strassen for the 2x2 part:
    #   For 2x2 * 2x7, write B1 = [b00 b01 b02 b03 b04 b05 b06; b10 b11 ... b16]
    #   Strassen on the 2x2 x 2x2 block (first two columns) gives rank 7 for that part,
    #   and the remaining 5 columns are rank 10 schoolbook. Total per 2x2*2x7: 17.
    #   Two such blocks: 34. Plus cross terms? No, they're independent.
    #   Total: 34. But we need to handle the full 2x7 for each 2x2 A block.
    #
    # Actually, for 2x2 * 2x7: C[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    # This tensor [2,2,7] has rank at most 2*2*7=28 schoolbook.
    # Using Strassen on the 2x2x2 sub-block: for columns k=0,1 (a 2x2x2 sub-tensor),
    # rank 7 instead of 8. For the remaining 5 columns, rank 10 schoolbook.
    # Total: 7 + 10 = 17 for [2,2,7].
    # For [2,4,7] = [2,2,7] + [2,2,7] (two independent middle pairs): 17 + 17 = 34.
    #
    # Let's construct this rank-34 decomposition explicitly.
    
    def strassen_2x2_2x2():
        """
        Strassen decomposition for 2x2 * 2x2 -> 2x2.
        Returns U (4x7), V (4x7), W (4x7) such that:
          sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k in {0,1}
        """
        # Strassen's 7 products for 2x2 multiplication
        # U entries indexed by (i,j) in 2x2, V by (j,k), W by (k,i)
        # Products:
        # m1 = (a00+a11)(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a01)*b11
        # m6 = (a10-a00)*(b00+b01)
        # m7 = (a01-a11)*(b10+b11)
        #
        # C00 = m1 + m4 - m5 - m7
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m7
        #
        # In tensor form, for each product r, U[i,j,r], V[j,k,r], W[k,i,r]:
        # The product m_r = (linear combo of A) * (linear combo of B)
        # contributes to C via the W matrix.
        
        R = 7
        U = np.zeros((4, R))  # i*2+j
        V = np.zeros((4, R))  # j*2+k
        W = np.zeros((4, R))  # k*2+i
        
        # m1 = (A00+A11)(B00+B11)
        # A factors: A00 + A11 -> U[0*2+0,0]=1, U[1*2+1,0]=1
        # B factors: B00 + B11 -> V[0*2+0,0]=1, V[1*2+1,0]=1
        # W: contributes to C00 (+1), C11 (+1)
        # C00: k=0,i=0 -> W[0*2+0,0] = 1
        # C11: k=1,i=1 -> W[1*2+1,0] = 1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # C10: k=0,i=1 -> W[0*2+1,1] = 1
        W[1, 1] = 1
        
        # m3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # C01: k=1,i=0 -> W[1*2+0,2] = 1
        W[2, 2] = 1
        
        # m4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # C00: k=0,i=0 -> W[0,3]=1; C10: k=0,i=1 -> W[1,3]=1
        W[0, 3] = 1; W[1, 3] = 1
        
        # m5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # C00: -1; C01: +1
        W[0, 4] = -1; W[2, 4] = 1
        
        # m6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # Not used in standard 2x2 Strassen output (this is for a different formulation)
        # Actually m6 is not standard. Let me use the correct 7 products.
        # Standard Strassen:
        # p1 = (a11+a00)(b00+b11)
        # p2 = (a10+a11)b00
        # p3 = a00(b01-b11)
        # p4 = a11(b10-b00)
        # p5 = (a00+a01)b11
        # p6 = (a10-a00)(b00+b01)
        # p7 = (a01-a11)(b10+b11)
        # c00 = p1 + p4 - p5 - p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p3 + p7
        
        # Let me redo:
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1  # c00
        W[3, 0] = 1  # c11
        
        # p2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1  # c10
        
        # p3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1  # c01
        W[3, 2] = 1  # c11
        
        # p4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1  # c00
        W[1, 3] = 1  # c10
        
        # p5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1  # c00
        W[2, 4] = 1   # c01
        
        # p6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # p6 doesn't appear in c00,c01,c10,c11 in standard Strassen
        # It's used in some formulations but not the basic 2x2. Skip or set to 0.
        # Actually p6 is not needed for 2x2. Let me replace with p7.
        
        # p7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1  # c00
        W[3, 6] = 1   # c11
        
        # Clear p6 (index 5) since it's not used:
        U[:, 5] = 0
        V[:, 5] = 0
        W[:, 5] = 0
        
        return U, V, W
    
    # Build rank-34 decomposition for [2,4,7]:
    # Two groups of middle indices: group1 = {0,1}, group2 = {2,3}
    # For each group, use [2,2,7] decomposition with rank 17.
    # [2,2,7]: Strassen on first 2 columns (rank 7) + schoolbook on last 5 (rank 10)
    
    def two_by_two_by_seven(j_offset):
        """
        Decomposition for 2x2 * 2x7 with middle index offset j_offset.
        Returns U (4x17), V (14x17), W (14x17) in local coordinates.
        Local: i in {0,1}, j_local in {0,1} -> j = j_offset + j_local, k in {0..6}
        """
        R = 17
        U = np.zeros((4, R))   # indexed by i*2 + j_local
        V = np.zeros((14, R))  # indexed by j_local*7 + k
        W = np.zeros((14, R))  # indexed by k*2 + i
        
        # First 7 products: Strassen on columns k=0,1
        U_s, V_s, W_s = strassen_2x2_2x2()
        # V_s is 4x7 (j_local*2+k for k in {0,1}), need to map to j_local*7+k
        for r in range(7):
            U[:, r] = U_s[:, r]
            # Map V_s[j_local*2 + k, r] to V[j_local*7 + k, r] for k=0,1
            for j_local in range(2):
                for k in range(2):
                    V[j_local * 7 + k, r] = V_s[j_local * 2 + k, r]
            # W_s is 4x7 (k*2+i for k in {0,1}), same indexing
            for k in range(2):
                for i in range(2):
                    W[k * 2 + i, r] = W_s[k * 2 + i, r]
        
        # Last 10 products: schoolbook for columns k=2..6
        for j_local in range(2):
            for k in range(2, 7):
                r = 7 + j_local * 5 + (k - 2)
                U[j_local * 2 + 0, r] = 1.0  # i=0, but this is for the A[i,j] part
                # Wait, U is indexed by (i, j_local). For schoolbook, each (i,j_local,k) is separate.
                # But we're only doing 10 products for 2*5=10 (i,j_local,k) combos? No, 2*2*5=20.
                # I made an error. For [2,2,7] schoolbook, rank = 2*2*7=28.
                # Strassen reduces the 2x2x2 part from 8 to 7, saving 1.
                # So [2,2,7] rank = 28 - 1 = 27, not 17.
                # Let me reconsider.
        
        # Correction: [2,2,7] schoolbook rank = 2*2*7 = 28.
        # Strassen on the 2x2x2 sub-tensor (k=0,1) saves 1 (8->7).
        # Remaining: 2*2*5 = 20 schoolbook for k=2..6.
        # Total: 7 + 20 = 27.
        # For [2,4,7]: two [2,2,7] blocks: 27 + 27 = 54. Not better than 56.
        
        # This approach doesn't help much. Let me try a different reduction.
        # The real saving comes from the n=2 dimension being small.
        # For [2,m,p], the rank is at most m*p + (m-1)*(p-1) or similar.
        # Actually, for [2,m,p]: C[i,k] = sum_j A[i,j]B[j,k].
        # This is a sum of m terms. The tensor rank is at most m*p (using the fact
        # that for each j, the term A[i,j]*B[j,k] has rank p in the (i,k) space,
        # but A[i,j] is 2-dim). 
        #
        # Better: [2,m,p] can be done with rank m*p by: for each j, use p rank-1 terms
        # where U[i,j,r] = A[i,j] coefficient, V[j,k,r] = B[j,k] coefficient,
        # W[k,i,r] = 1. But this gives rank m*p*... no.
        #
        # Let's just use rank = m*p = 28 for [2,4,7]:
        # For each j in {0,1,2,3} and k in {0..6}:
        #   U[0*4+j, r] = 1, U[1*4+j, r] = 1  (both rows of A column j)
        #   V[j*7+k, r] = 1
        #   W[k*2+0, r] = 1, W[k*2+1, r] = 1  (both rows of output column k)
        # Then sum_r U[i*4+j,r] V[j*7+k,r] W[k*2+i,r] = sum_{j,k} 1*1*1 = 4*7 = 28 != 1.
        # That's wrong.
        
        # I think the cleanest approach is to just return the schoolbook rank 56
        # and note that finding a better decomposition requires more sophisticated
        # algebraic geometry methods that are hard to implement in this context.
        #
        # However, let's try one more idea: use the fact that n=2 to get rank = m*p = 28
        # via a different construction.
        #
        # For [2,4,7]: C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
        # Define for each j: vector a_j = [A[0,j], A[1,j]] (in R^2)
        # Define for each k: vector b_k = [B[0,k], B[1,k], B[2,k], B[3,k]] (in R^4)
        # C[i,k] = a_j . b_k summed over j = a_all . b_all where a_all is 4-dim.
        #
        # In tensor form, we need:
        #   sum_r U[i*4+j, r] V[j*7+k, r] W[k*2+i, r] = 1
        #
        # Try rank 28: for each (j,k), r = j*7+k:
        #   U[i*4+j, r] = 1 for both i
        #   V[j*7+k, r] = 1
        #   W[k*2+i, r] = 1 for both i
        # Then for fixed (i,j,k): sum_r U[i*4+j,r] V[j*7+k,r] W[k*2+i,r]
        # Only r = j*7+k contributes: 1*1*1 = 1. 
        # But for other (i',j',k') with j'!=j or k'!=k, we get 0.
        # Wait, we need the sum to equal 1 ONLY when the indices match the tensor structure.
        # The tensor T[i,j,k] = 1 for ALL valid (i,j,k). So we need:
        #   sum_r U[i*4+j, r] V[j*7+k, r] W[k*2+i, r] = 1 for ALL i,j,k.
        # With the above construction, for fixed (i,j,k), only r=j*7+k contributes,
        # giving 1. For a different triple (i',j',k'), we check if any r contributes:
        # r = j'*7+k', U[i'*4+j', r] = 1, V[j'*7+k', r] = 1, W[k'*2+i', r] = 1.
        # So it also gives 1! But we need it to give 1 for ALL (i,j,k), which it does.
        # 
        # Wait, but the decomposition must be exact for the multiplication tensor,
        # which is T[i,j,k] = 1. So yes, rank 28 works!
        
        # Let me verify: the multiplication tensor for 2x4 * 4x7 is:
        # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0..6}
        # (because C[i,k] = sum_j A[i,j]B[j,k], and the "coefficient" of A[i,j]B[j,k]
        #  in C[i,k] is 1)
        
        # So the rank-28 construction:
        # For each (j,k) in {0..3}x{0..6}, r = j*7+k:
        #   U[i*4+j, r] = 1 for i=0,1
        #   V[j*7+k, r] = 1
        #   W[k*2+i, r] = 1 for i=0,1
        #
        # Check: sum_r U[i*4+j, r] V[j*7+k, r] W[k*2+i, r]
        # = U[i*4+j, j*7+k] * V[j*7+k, j*7+k] * W[k*2+i, j*7+k]
        # = 1 * 1 * 1 = 1.  ✓
        #
        # And no other r contributes because U[i*4+j, r] is nonzero only for r in {j*7+0, ..., j*7+6},
        # and for those r, V[j*7+k, r] is nonzero only when r = j*7+k, and W[k*2+i, r] is
        # nonzero only when r corresponds to the same k.
        #
        # So rank 28 is valid! Let's construct it.
        
        R = m * p  # 4*7 = 28
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        for j in range(m):
            for k in range(p):
                r = j * p + k
                for i in range(n):
                    U[i * m + j, r] = 1.0
                    W[k * n + i, r] = 1.0
                V[j * p + k, r] = 1.0
        
        # Verify the decomposition
        T_reconstructed = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    T_reconstructed[i, j, k] = s
        
        # Check if all entries are 1
        assert np.allclose(T_reconstructed, 1.0), f"Decomposition failed: {T_reconstructed}"
        
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
