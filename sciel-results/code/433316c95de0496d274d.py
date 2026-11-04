"""Tensor network contraction for 4x5x6 matrix multiplication."""
import time

DIMENSIONS = [4, 5, 6]

def construct():
    deadline = time.time() + 130
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Use Strassen on 2x2x2 sub-blocks
    # Split i into groups of 2: {0,1}, {2,3}
    # Split k into groups of 2: {0,1}, {2,3}, {4,5}
    # Split j into {0,1}, {2,3}, {4}
    
    # For each (i_group, k_group, j_pair) where j_pair is {0,1} or {2,3}:
    #   Use Strassen (rank 7)
    # For each (i_group, k_group) with j={4}:
    #   Use schoolbook (rank 2)
    
    # Strassen formulas for 2x2 matrices:
    # Given A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We'll collect terms as (U_entries, V_entries, W_entries, coeff)
    # where U_entries is a dict mapping (i,j) -> coeff
    # V_entries maps (j,k) -> coeff
    # W_entries maps (k,i) -> coeff
    
    terms = []  # list of (U_dict, V_dict, W_dict)
    
    # Strassen sub-problems for j pairs
    j_pairs = [(0, 1), (2, 3)]
    i_groups = [(0, 1), (2, 3)]
    k_groups = [(0, 1), (2, 3), (4, 5)]
    
    for ig in i_groups:
        for kg in k_groups:
            i0, i1 = ig
            k0, k1 = kg
            
            # For each j pair, use Strassen
            for (j0, j1) in j_pairs:
                # A sub-matrix: rows i0,i1; cols j0,j1
                # B sub-matrix: rows j0,j1; cols k0,k1
                # We need to express each Strassen M as a rank-1 term
                
                # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
                # This contributes to C[i0,k0], C[i1,k1] via M1
                # In tensor form: U has entries at (i0,j0)=1, (i1,j1)=1
                # V has entries at (j0,k0)=1, (j1,k1)=1
                # W has entries at (k0,i0)=1, (k1,i1)=1
                # But wait, this gives product of sums, which is correct for M1
                
                # Actually, each M is a product of two linear forms.
                # M1 = (a00+a11)(b00+b11)
                # This is NOT a single rank-1 term in the tensor sense.
                # It's the product of (sum of two A entries) and (sum of two B entries).
                # In the tensor decomposition, this corresponds to:
                # U[i0,j0]=1, U[i1,j1]=1, V[j0,k0]=1, V[j1,k1]=1, W[k0,i0]=1, W[k1,i1]=1
                # The contribution is (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
                # which expands to A[i0,j0]B[j0,k0] + A[i0,j0]B[j1,k1] + A[i1,j1]B[j0,k0] + A[i1,j1]B[j1,k1]
                # But we only want A[i0,j0]B[j0,k0] + A[i1,j1]B[j1,k1] for the diagonal terms!
                
                # I'm making an error. Let me reconsider.
                
                # In the standard matrix multiplication tensor, we want:
                # C[i,k] = sum_j A[i,j] B[j,k]
                # A rank-1 term is U[i,j] V[j,k] W[k,i] where each is 0 or 1 (or coefficients)
                # For schoolbook, each term has U[i,j]=1 for one (i,j), V[j,k]=1 for one (j,k), W[k,i]=1 for one (k,i)
                
                # For Strassen, we use linear combinations. The key is that each "product" in Strassen
                # is between a linear combination of A entries and a linear combination of B entries.
                # This linear combination can be represented as a single rank-1 term in the tensor
                # by setting the appropriate U and V entries to the coefficients.
                
                # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
                # U entries: U[i0,j0]=1, U[i1,j1]=1
                # V entries: V[j0,k0]=1, V[j1,k1]=1
                # W entries: W[k0,i0]=1, W[k1,i1]=1
                # This term contributes to C[i0,k0] and C[i1,k1] (and cross terms that cancel)
                
                # Wait, no. The contribution of this rank-1 term to C[i,k] is:
                # sum_j U[i,j] V[j,k] W[k,i]
                # For (i,k) = (i0,k0): sum_j U[i0,j] V[j,k0] W[k0,i0]
                # = U[i0,j0]*V[j0,k0]*W[k0,i0] + U[i0,j1]*V[j1,k0]*W[k0,i0] + ...
                # = 1*1*1 + 0 + 0 + 0 = 1 (only the (i0,j0,k0) term)
                # Plus U[i0,j0]*V[j0,k0]... wait I need to be more careful.
                
                # The term contributes U[i0,j0]*V[j0,k0]*W[k0,i0] to C[i0,k0] = 1*1*1 = 1
                # And U[i0,j1]*V[j1,k0]*W[k0,i0] = 0 (U[i0,j1]=0)
                # And U[i1,j0]*V[j0,k0]*W[k0,i1] = 0 (W[k0,i1]=0)
                # And U[i1,j1]*V[j1,k1]*W[k1,i1] contributes to C[i1,k1] = 1
                # But also U[i0,j0]*V[j1,k0] = 0, etc.
                
                # Hmm, this only gives the diagonal terms. The off-diagonal terms in the expansion
                # of (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1]) are A[i0,j0]*B[j1,k0] and A[i1,j1]*B[j0,k0]
                # These are NOT captured by this single rank-1 term.
                
                # I think I'm confusing myself. Let me look at this differently.
                
                # In Strassen's algorithm, M1 = (A00+A11)(B00+B11) is computed as a single
                # matrix product of 1x1 matrices. In the tensor decomposition context,
                # this means we have a rank-1 term where:
                # - The "A factor" is the linear form A00+A11
                # - The "B factor" is the linear form B00+B11
                # - The "C factor" distributes to C00 and C11 (and other entries via the combination)
                
                # So for M1:
                # U[i0,j0] = 1, U[i1,j1] = 1  (the A linear form)
                # V[j0,k0] = 1, V[j1,k1] = 1  (the B linear form)
                # W[k0,i0] = 1, W[k1,i1] = 1  (contributes to C[i0,k0] and C[i1,k1])
                
                # This term's contribution to C[i,k] is:
                # sum_j U[i,j] V[j,k] W[k,i]
                # For (i0,k0): U[i0,j0]V[j0,k0]W[k0,i0] + U[i0,j1]V[j1,k0]W[k0,i0] + ...
                # = 1*1*1 + 0 + 0 + 0 = 1
                # For (i1,k1): U[i1,j0]V[j0,k1]W[k1,i1] + U[i1,j1]V[j1,k1]W[k1,i1] + ...
                # = 0 + 1*1*1 = 1
                # For (i0,k1): U[i0,j0]V[j0,k1]W[k1,i0] + ... = 0 (W[k1,i0]=0)
                # For (i1,k0): U[i1,j0]V[j0,k0]W[k0,i1] + ... = 0 (W[k0,i1]=0)
                
                # So this term only contributes to C[i0,k0] and C[i1,k1], each with coefficient 1.
                # But M1 should contribute to C00 with +1 and C11 with +1 in the Strassen formula.
                # That matches!
                
                # Now the issue is that M1 = (A00+A11)(B00+B11) expands to:
                # A00*B00 + A00*B11 + A11*B00 + A11*B11
                # But our rank-1 term only captures A00*B00 and A11*B11.
                # The cross terms A00*B11 and A11*B00 are missing!
                
                # This is the fundamental issue. Strassen's algorithm works because the
                # cross terms cancel out when you combine all 7 M's. But in the tensor
                # decomposition, we're not doing that combination - we're just summing rank-1 terms.
                
                # I think the correct interpretation is different. In Strassen, each M is a
                # scalar product, and the final C entries are linear combinations of the M's.
                # To convert to a tensor decomposition, we need to "absorb" the linear
                # combinations into the W matrix.
                
                # Let me reconsider. The Strassen algorithm computes:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                # where M1..M7 are as defined.
                
                # Each M is a product of two linear forms. To make this a tensor decomposition,
                # we can think of each M as a "virtual" product, and then the C entries are
                # linear combinations of these virtual products.
                
                # In the tensor decomposition, we have rank-1 terms. Each term has U, V, W vectors.
                # The key insight is that we can set W to have coefficients that implement the
                # Strassen combination.
                
                # For M1: U has 1 at (i0,j0) and (i1,j1), V has 1 at (j0,k0) and (j1,k1)
                # W should be set so that the contribution to C[i,k] is correct.
                # But M1 contributes +1 to C00 and +1 to C11.
                # So W[k0,i0] = 1, W[k1,i1] = 1.
                
                # But as I showed, this only gives A00*B00 and A11*B11, not the full M1.
                
                # I think the resolution is that in Strassen, M1 is not (A00+A11)(B00+B11)
                # in the sense of a single multiplication. Rather, it's computed as a product
                # of two scalars, where each scalar is a linear combination. The "product"
                # is the actual multiplication of these two scalars.
                
                # In the tensor decomposition, a rank-1 term U⊗V⊗W computes:
                # C[i,k] += sum_j U[i,j] V[j,k] W[k,i]
                # This is NOT the same as (sum of some A entries) * (sum of some B entries).
                # It's a sum of products, not a product of sums.
                
                # So Strassen's algorithm cannot be directly converted to a tensor decomposition
                # in this way. The rank-7 Strassen algorithm uses 7 scalar multiplications,
                # but each "multiplication" is between two linear combinations, which is
                # not a rank-1 term in the tensor sense.
                
                # I need to abandon the Strassen approach and just use schoolbook, or find
                # a different decomposition.
                
                pass  # Skip Strassen, it doesn't work directly
            
            # For j=4, use schoolbook: 2 terms
            for j in [4]:
                for i in [i0, i1]:
                    for k in [k0, k1]:
                        U_dict = {(i, j): 1}
                        V_dict = {(j, k): 1}
                        W_dict = {(k, i): 1}
                        terms.append((U_dict, V_dict, W_dict))
    
    # For the j pairs, we need to use schoolbook since Strassen doesn't directly apply
    for ig in i_groups:
        for kg in k_groups:
            i0, i1 = ig
            k0, k1 = kg
            for (j0, j1) in j_pairs:
                for i in [i0, i1]:
                    for j in [j0, j1]:
                        for k in [k0, k1]:
                            U_dict = {(i, j): 1}
                            V_dict = {(j, k): 1}
                            W_dict = {(k, i): 1}
                            terms.append((U_dict, V_dict, W_dict))
    
    # Now build the matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(terms):
        for (i, j), coeff in U_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in V_dict.items():
            V[j * p + k][col] = coeff
        for (k, i), coeff in W_dict.items():
            W[k * n + i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
