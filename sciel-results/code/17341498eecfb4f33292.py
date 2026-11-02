import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    nm, mp, pn = n*m, m*p, p*n  # 6, 27, 18
    
    # Build the 54 schoolbook rank-1 vectors
    U_school = np.zeros((nm, nm*mp))  # 6 x 54
    V_school = np.zeros((mp, nm*mp))  # 27 x 54
    W_school = np.zeros((pn, nm*mp))  # 18 x 54
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    # The tensor is T = sum_{r=1}^{54} u_r ⊗ v_r ⊗ w_r
    # We want to find 4 columns that are linear combinations of the others
    # (with the same linear relation holding across U, V, and W)
    
    # Approach: use the fact that for fixed (j,k), varying i gives linear relations
    # For fixed j,k: T[i,j,k] = A[i,j]*B[j,k] for i=0,1
    # This means for each (j,k), the pair of U-entries (U[0,j], U[1,j]) 
    # with the same V and W should satisfy a relation.
    
    # More systematically: the tensor T_{ijk} = δ_{i,j} * δ_{j,k} in some sense
    # Actually T_{ijk} = 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k]
    
    # Let's think differently. The 54 terms are:
    # For each (i,j,k): u has 1 at position (i*m+j), v has 1 at (j*p+k), w has 1 at (k*n+i)
    
    # Key insight: For a fixed j, the terms with that j have:
    # U entries at positions (0,j) and (1,j)
    # V entries at positions (j,0)..(j,8)
    # W entries at positions (0,0),(0,1),(1,0),(1,1),(2,0),(2,1),...,(8,0),(8,1)
    
    # The sub-tensor for fixed j is a 2x9x9 tensor (indices i, k, and the "j-dimension" is fixed)
    # Actually it's rank-1: T[i,_,k] = A[i,j]*B[j,k]
    # So for fixed j, the 18 terms (i in {0,1}, k in {0..8}) form a rank-2 sub-tensor
    # (rank 2 because it's outer product of A[:,j] and B[j,:])
    
    # Wait, that's not quite right either. Let me reconsider.
    
    # The full tensor is: T[i,j,k] = 1 (it's the matmul tensor, all entries are 1 in the sense that
    # each (i,j,k) triple corresponds to a product term)
    
    # Actually, the matmul tensor M_{ijk} = 1 for all i,j,k (it's the structure tensor)
    # And C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So the tensor is sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where e_i is unit vector in C-space at position (i,k)
    # More precisely: T = sum_{i,j,k} u_{i,j} ⊗ v_{j,k} ⊗ w_{i,k}
    # where u_{i,j} has 1 at (i,j) in A-space, v_{j,k} has 1 at (j,k) in B-space,
    # w_{i,k} has 1 at (i,k) in C-space.
    
    # The rank is the minimum number of terms needed to represent this tensor.
    
    # Strategy: Use bilinear reduction. 
    # For fixed i and k, C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a sum of m=3 products.
    
    # We can use the fact that for 1x3x1 (i.e., dot product of length 3), the rank is 3 (can't reduce).
    # But for 2x3x9, we have more freedom.
    
    # Let me try a direct approach: find linear dependencies among the 54 rank-1 terms.
    # If we can express 4 of the terms as linear combinations of the other 50, we get rank 50.
    
    # The condition is: there exist coefficients c_1,...,c_54 (not all zero) such that
    # sum_r c_r * u_r = 0, sum_r c_r * v_r = 0, sum_r c_r * w_r = 0
    # AND for the tensor identity to hold, we need:
    # sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0
    
    # Actually, if sum c_r u_r = 0 and sum c_r v_r = 0 and sum c_r w_r = 0, that's not sufficient.
    # We need the trilinear form to vanish.
    
    # Let me try a simpler approach: use numpy to find the null space of the "flattened" tensor.
    
    # Flatten the tensor: T is a 6x27x18 tensor. Reshape to 6x(27*18) = 6x486 matrix.
    # The rank-1 terms u_r ⊗ v_r ⊗ w_r become, when flattened, vectors of length 486.
    # Specifically, term r corresponds to a matrix (or vector) that has 1 at position (i_r, j_r*pn + k_r)
    # where i_r is the A-index, j_r is the B-index, k_r is the C-index.
    
    # Wait, let me be more careful. The tensor T has entries T[a,b,c] where a in 0..5 (A-index),
    # b in 0..26 (B-index), c in 0..17 (C-index).
    # T[a,b,c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' and i=i' and j=j' and k=k'
    # Otherwise T[a,b,c] = 0.
    
    # So T is a 6x27x18 tensor with exactly 54 non-zero entries, each equal to 1.
    # The non-zero entries are at:
    # For i in {0,1}, j in {0,1,2}, k in {0,...,8}:
    #   T[i*3+j, j*9+k, k*2+i] = 1
    
    # Now, the rank of this tensor is the minimum R such that
    # T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    
    # Let's build the 54 rank-1 vectors and see if we can find 4 that are redundant.
    
    # Build the 6x486 matrix (flatten T along last two dimensions)
    T_flat = np.zeros((nm, mp*pn))  # 6 x 486
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T_flat[a_idx, b_idx*pn + c_idx] = 1.0
    
    # The 54 rank-1 terms, when flattened to 6x486, are:
    # Term (i,j,k): has 1 at position (i*m+j, (j*p+k)*pn + (k*n+i))
    # So it's a matrix with a single 1.
    
    # For the tensor decomposition T = sum_r u_r ⊗ v_r ⊗ w_r,
    # flattening gives T_flat = sum_r u_r ⊗ (v_r ⊗ w_r)
    # where v_r ⊗ w_r is a vector of length mp*pn = 486.
    
    # So we need to express T_flat as a sum of rank-1 matrices (outer products of u_r and (v_r⊗w_r)).
    
    # The 54 schoolbook terms give:
    # u_{ijk} = e_{i*m+j} (standard basis in R^6)
    # (v⊗w)_{ijk} = e_{(j*p+k)*pn + (k*n+i)} (standard basis in R^486)
    
    # So T_flat = sum_{i,j,k} e_{i*m+j} ⊗ e_{(j*p+k)*pn + (k*n+i)}
    
    # This is a 6x486 matrix. Its rank is at most 6 (since it's 6 rows).
    # Actually, let's check: each row i of T_flat has non-zero entries at positions
    # corresponding to all (j,k) with that i.
    # Row i has non-zeros at: for j in {0,1,2}, k in {0..8}: position (j*9+k)*18 + (k*2+i)
    
    # Let's just compute the SVD or find dependencies directly.
    
    # Alternative approach: Since the problem is small, let me try to construct a rank-50 
    # decomposition by finding 4 linear relations.
    
    # Key observation: For fixed i and k, the equation C[i,k] = sum_j A[i,j]*B[j,k] 
    # involves 3 products. We can't reduce this below 3 in general.
    
    # But for fixed i, varying k: C[i,k] = sum_j A[i,j]*B[j,k]
    # This is a 1x3x9 matmul, which has rank 3*9 = 27 (schoolbook) but maybe less?
    # The rank of 1xm x p is m*p = 3*9 = 27 (can't reduce, since it's a dot product of length 3 for each of 9 outputs).
    
    # Hmm, actually for 1x3x1, the rank is 3 (you need 3 products for a dot product of length 3).
    # For 1x3x9, the rank is 27 (each output needs 3 products, and they're independent).
    
    # So the total rank for 2x3x9 is at least 2*27 = 54? No, that's not right either,
    # because the two rows share the same B matrix.
    
    # Let me just try to find a rank-50 decomposition computationally.
    
    # I'll use a greedy approach: start with 54 terms, try to merge/remove terms.
    
    # Actually, let me try a different structural approach.
    # Use the decomposition: split A into A1 (2x2) and A2 (2x1), B into B1 (2x9) and B2 (1x9).
    # C = A1*B1 + A2*B2
    
    # For A2*B2: 2x1 * 1x9, rank = 2*9 = 18 (each of the 9 outputs needs 2 products, but they share B2)
    # Actually, C2[i,k] = A2[i]*B2[k], so it's rank 2*9 = 18? No, it's rank 2 (it's a rank-2 matrix:
    # A2 is 2x1, B2 is 1x9, so C2 = A2*B2 is rank 1 as a matrix, but as a bilinear map, 
    # the tensor rank is 2*1*9 = 18).
    
    # Hmm, I'm going in circles. Let me just implement the schoolbook and return rank 54.
    
    rank = 54
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
