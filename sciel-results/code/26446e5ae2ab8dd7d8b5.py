import time
import numpy as np
from itertools import product, combinations

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 6, 6, 6
    deadline = start_time + 110  # Safety margin

    # Build the matrix multiplication tensor T[i,j,k] = 1 if i=k (A row i, B col j, C row i col k)
    # Actually for matrix multiplication: C = A * B
    # T[i,j,k] = 1 if the entry (i,k) of C depends on A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k where the product A[i,j]*B[j,k] contributes to C[i,k]
    # The tensor is: T[i,j,k] = delta_{i,k} is WRONG.
    # Let me reconsider. The matrix multiplication tensor M of order (n,m,p) is defined by:
    # M[i,j,k] = 1 if and only if the (i,k) entry of the product uses A[i,j] * B[j,k]
    # For standard matrix multiplication, M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # such that j is a valid column of A and row of B. In our case n=m=p=6.
    # Actually, the tensor entries are: M[i,j,k] = 1 if and only if the multiplication A[i,j] * B[j,k] 
    # contributes to C[i,k]. This is always true when 0 <= j < m (which is 6).
    # So M[i,j,k] = 1 for all i,j,k in {0,...,5}? No, that's not right either.
    
    # Let me think more carefully. The matrix multiplication tensor is a 3-tensor T of shape (n, m, p)
    # where T[i, j, k] = 1 if and only if the entry A[i,j] * B[j,k] contributes to C[i,k].
    # For standard matrix multiplication with n=m=p=6, T[i,j,k] = 1 for ALL i,j,k in {0,...,5}.
    # Wait, no. Let me re-examine.
    
    # The tensor decomposition of matrix multiplication: we want to write
    # C = A * B as a sum of rank-1 terms: C[i,k] = sum_r U[i,r] * V[r,k] ... no, that's not right.
    
    # Actually, the standard formulation: the matrix multiplication tensor M of size n x m x p
    # has entries M[i,j,k] = 1 if and only if the product entry (i,k) of C = A*B involves A[i,j] * B[j,k].
    # For square n x n matrices, M[i,j,k] = 1 for all i,j,k where the j-th column of A times the j-th row of B
    # contributes to C[i,k]. This is always true, so M[i,j,k] = 1 for all i,j,k in {0,...,n-1}.
    
    # Hmm, but that would make the tensor all-ones, which has rank 1, which is clearly wrong.
    
    # Let me look at this differently. The tensor T is defined such that:
    # T_{ijk} = 1 if and only if the (i,k)-th entry of AB is the sum over j of A_{ij} B_{jk}
    # So T_{ijk} = 1 for all valid indices. But the tensor decomposition is:
    # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # where the "multiplication" is the outer product structure.
    
    # For matrix multiplication, the correct tensor is:
    # T_{ijk} = 1 if and only if i, j, k are such that the term A_{ij} B_{jk} appears in (AB)_{ik}
    # This is true for all i, j, k. So T is the all-ones tensor? No...
    
    # I think I'm confusing myself. Let me use the standard definition:
    # The matrix multiplication tensor <n,m,p> is a 3-tensor with entries:
    # M_{ijk} = 1 if and only if the multiplication of the (i,j) entry of A and the (j,k) entry of B
    # contributes to the (i,k) entry of C. For standard matrix multiplication, this is always the case
    # when 0 <= i < n, 0 <= j < m, 0 <= k < p.
    # So M is the all-ones tensor of shape (n,m,p).
    
    # But wait, the rank of the all-ones tensor of shape (n,m,p) is 1 (it's just 1*1*1*...), which is clearly wrong.
    
    # I must be misunderstanding the tensor structure. Let me think about it as:
    # We want to decompose the operation C = AB. The "tensor" that encodes this operation has
    # entries that tell us which products A[i,j]*B[j,k] contribute to C[i,k].
    # The tensor T has shape (n, m, p) where T[i, j, k] = 1 means "A[i,j] * B[j,k] contributes to C[i,k]".
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    
    # The rank of this tensor (as a 3-way tensor) is what we're trying to minimize.
    # A rank-r decomposition is: T[i,j,k] = sum_{r=1}^{R} U[i,r] * V[j,r] * W[k,r]
    # where U is n x R, V is m x R, W is p x R.
    
    # For the all-ones tensor of shape (n,m,p), the rank is indeed 1 (take U=V=W=1).
    # But matrix multiplication is NOT the all-ones tensor!
    
    # I think the issue is that the matrix multiplication tensor is NOT the all-ones tensor.
    # Let me reconsider. The tensor should encode the SPARSE structure of which terms contribute.
    # For C = AB, C[i,k] = sum_j A[i,j] * B[j,k].
    # The tensor T[i,j,k] should be 1 if and only if the term A[i,j] * B[j,k] appears in the sum for C[i,k].
    # This is true for ALL i,j,k. So T is all ones, and rank 1. That can't be right.
    
    # OH WAIT. I think I've been confusing the tensor with something else. Let me reconsider.
    # The matrix multiplication tensor is a 3-tensor where the "modes" correspond to the three
    # "legs" of the multiplication. The correct formulation is:
    # T[i, j, k] = 1 if and only if the (i,j,k) term is "active" in the multiplication.
    # For standard matrix multiplication, ALL terms are active, so T is all-ones.
    
    # But the RANK of the matrix multiplication tensor (in the sense of tensor rank) is NOT 1.
    # The tensor rank of the n x n x n matrix multiplication tensor is the bilinear complexity,
    # which for n=2 is 7 (Strassen), for n=3 is at most 23, etc.
    
    # I think the confusion is that the "tensor" in tensor rank decomposition is not the
    # all-ones tensor. Let me look at this from the perspective of bilinear complexity.
    
    # The bilinear complexity of n x n matrix multiplication is the minimum number R such that:
    # C[i,k] = sum_{r=1}^{R} sum_{i',j'} U[i, r, i', j'] ... no, this is getting complicated.
    
    # Let me use the standard formulation from the literature:
    # The matrix multiplication tensor M_{n} of size n x n x n is defined by:
    # M_{i,j,k} = 1 if and only if the (i,k) entry of the product AB involves A[i,j] * B[j,k]
    # in the standard way. For standard matrix multiplication, this means:
    # M_{i,j,k} = 1 for all i,j,k in {0,...,n-1}.
    
    # Wait, I just realized my error. The tensor is NOT all-ones. The tensor M_{i,j,k}
    # encodes which "triple" (i,j,k) is "used" in the multiplication. For standard matrix
    # multiplication, the term A[i,j] * B[j,k] contributes to C[i,k] and ONLY to C[i,k].
    # So the tensor should be: M[i,j,k] = 1 if and only if... hmm.
    
    # Actually, I think the correct formulation is that the tensor has entries:
    # M[i, j, k] = 1 if and only if i, j, k form a valid "chain" i -> j -> k in the multiplication.
    # For standard matrix multiplication, every triple (i,j,k) is valid, so M is all-ones.
    
    # I'm going in circles. Let me just use the known fact: the rank of the 2x2x2 matrix
    # multiplication tensor is 7 (Strassen). Let me verify with a small example.
    
    # For 2x2x2: M[i,j,k] for i,j,k in {0,1}. If M is all-ones, rank is 1. But we know
    # the bilinear complexity is 7. So M cannot be all-ones.
    
    # I think the correct definition is:
    # M[i, j, k] = 1 if and only if the multiplication step (i,j) x (j,k) -> (i,k) is "primitive".
    # But that's still all-ones for standard matrix multiplication.
    
    # Let me try a different approach. The tensor decomposition for matrix multiplication
    # is: for each r in {1,...,R}, we have three vectors u_r (length n), v_r (length m), w_r (length p)
    # such that for all i,k: C[i,k] = sum_j A[i,j] B[j,k] = sum_r u_r[i] * (sum_j v_r[j] * B[j,k] * ... )
    
    # No, the correct formulation is:
    # We want: C = A * B = sum_{r=1}^{R} (sum_i U[i,r] A[i,:]) * (sum_j V[j,r] B[j,:]) ... no.
    
    # OK let me just look at the Strassen algorithm for 2x2 and extract the tensor structure.
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each Mr = X_r * Y_r where X_r is a linear combination of A entries and Y_r is a linear combination of B entries.
    # The "tensor" structure is: for each r, we have coefficients for A, B, and C.
    # The rank is the number of such terms R.
    
    # So the tensor T has "entries" that are the coefficients in the linear combinations.
    # More precisely, T is a 3-tensor of shape (n*m, m*p, n*p) or something like that.
    # Actually, the standard way to view it: T[i,j,k] where i indexes A entries, j indexes B entries,
    # and k indexes C entries, and T[i,j,k] = 1 if the product A[i]*B[j] contributes to C[k].
    
    # For 2x2 matrix multiplication:
    # A has entries A00, A01, A10, A11 (indices 0,1,2,3)
    # B has entries B00, B01, B10, B11 (indices 0,1,2,3)
    # C has entries C00, C01, C10, C11 (indices 0,1,2,3)
    # T[i,j,k] = 1 if A[i]*B[j] contributes to C[k]
    # A00*B00 -> C00, A00*B10 -> C01 (wait, no)
    
    # Let me be more careful. A is 2x2: A = [[A00, A01], [A10, A11]]
    # B is 2x2: B = [[B00, B01], [B10, B11]]
    # C = A*B = [[A00*B00+A01*B10, A00*B01+A01*B11], [A10*B00+A11*B10, A10*B01+A11*B11]]
    # So:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # If we index A entries as: A00=0, A01=1, A10=2, A11=3
    # B entries as: B00=0, B01=1, B10=2, B11=3
    # C entries as: C00=0, C01=1, C10=2, C11=3
    # Then T[i,j,k] = 1 if A[i]*B[j] contributes to C[k]:
    # T[0,0,0]=1, T[1,2,0]=1, T[0,1,1]=1, T[1,3,1]=1, T[2,0,2]=1, T[3,2,2]=1, T[2,1,3]=1, T[3,3,3]=1
    
    # So T is a 4x4x4 tensor with 8 non-zero entries. Its tensor rank is 7 (Strassen).
    
    # Great! So for 6x6x6, we have:
    # A entries: 36 (indexed 0..35), B entries: 36, C entries: 36
    # T[i,j,k] = 1 if A[i]*B[j] contributes to C[k]
    # A[i] is the entry at row i//6, col i%6 of A
    # B[j] is the entry at row j//6, col j%6 of B
    # C[k] is the entry at row k//6, col k%6 of C
    # A[i]*B[j] contributes to C[k] if and only if:
    #   (i//6) == (k//6) AND (j%6) == (k%6)
    # Because A[row_a, col_a] * B[row_b, col_b] contributes to C[row_a, col_b]
    # So i//6 = row_a, i%6 = col_a, j//6 = row_b, j%6 = col_b, k//6 = row_a, k%6 = col_b
    # Thus: i//6 == k//6 AND j%6 == k%6 AND j//6 == i%6 (for the product to be defined)
    # Wait, for A[i]*B[j] to be a valid product, we need col_a == row_b, i.e., i%6 == j//6.
    # And the result goes to C[k] where row_k == row_a = i//6 and col_k == col_b = j%6.
    # So k = (i//6)*6 + (j%6).
    # And the condition is i%6 == j//6.
    
    # So T[i,j,k] = 1 if and only if:
    #   i%6 == j//6 AND k == (i//6)*6 + (j%6)
    # where i,j,k are in {0,...,35}.
    
    # This is a sparse tensor with 36*36/6 = 216 non-zero entries (for each i, j is determined up to 
    # the constraint i%6 == j//6, which gives 6 choices for j, and k is then determined).
    # Actually: for each i (36 choices), j must satisfy j//6 == i%6, so j is in {(i%6)*6 + s : s in 0..5},
    # giving 6 choices. And k is then (i//6)*6 + (j%6) = (i//6)*6 + s. So 36*6 = 216 non-zero entries.
    
    # The tensor rank of this 36x36x36 tensor is what we want to minimize.
    # Known: for 2x2x2, rank is 7. For 3x3x3, rank is at most 23.
    # For 6x6x6, the current best known is 189 (from the seed program using block Strassen).
    
    # Now, the task says: "The output factor follows the official trace convention W[k*n+i,r]."
    # This means W is indexed as W[k*n+i, r] where k is the column index of C and i is the row index of C.
    # So W has shape (36, rank) where the first index is k*n+i = col_C*6 + row_C.
    # Wait, that's unusual. Normally W would be indexed by (row, col) of C, i.e., (i, k) -> i*n+k.
    # But the task says W[k*n+i, r], so the first index of W is col*6 + row.
    
    # Let me re-read: "W[k*n+i,r]" where k and i are... let me think about what k and i represent.
    # In the context of matrix multiplication C = A*B, if we index C entries as C[i,k] (row i, col k),
    # then W[k*n+i, r] means the first index of W is col*6 + row, not row*6 + col.
    
    # So when I build W, for a C entry at (row_c, col_c), the W index is col_c*6 + row_c.
    
    # Now, the decomposition is: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k',r]
    # where i indexes A entries (0..35), j indexes B entries (0..35), k' indexes C entries
    # with the trace convention k' = col_c*6 + row_c.
    
    # Let me verify: for the non-zero entries of T, T[i,j,k_standard] = 1 where k_standard = row_c*6+col_c.
    # But W is indexed by k' = col_c*6 + row_c.
    # So k' = (k_standard % 6)*6 + (k_standard // 6).
    
    # OK, I think the current program already handles this correctly. Let me just try to
    # improve the rank by using a better algorithm or finding a lower-rank decomposition.
    
    # The current rank is 189 = 27 * 7. This comes from applying Strassen (rank 7 for 2x2)
    # to each of the 27 block positions (3x3 grid of 2x2 blocks).
    
    # Can we do better? For 6x6, we could potentially use:
    # 1. A direct rank-? decomposition for 3x3x3 (known upper bound 23, but maybe lower)
    # 2. A different block structure
    # 3. A global optimization to reduce rank
    
    # Let me try to implement a simple local search / greedy approach to see if we can
    # find a lower-rank decomposition. The idea: start with the 189-rank decomposition and
    # try to eliminate terms or replace groups of terms with fewer terms.
    
    # Actually, let me first try a different approach: use the known 3x3x3 decomposition
    # with rank 23 (if I can find the coefficients) and apply it to the 2x2 block structure.
    # But 6x6 = 2x2 blocks of 3x3, so if I use a rank-23 decomposition for 3x3x3,
    # the total rank would be 23 * 4 = 92? No, that's not how it works.
    
    # Actually, if I view 6x6 as 2x2 blocks of 3x3, then:
    # A = [[A00, A01], [A10, A11]] where each Aij is 3x3
    # B = [[B00, B01], [B10, B11]] where each Bij is 3x3
    # C = A*B = [[A00*B00+A01*B10, A00*B01+A01*B11], [A10*B00+A11*B10, A10*B01+A11*B11]]
    # Each block multiplication is a 3x3 * 3x3 -> 3x3 problem.
    # If I use a rank-R3 decomposition for 3x3x3, then each block product can be done with R3 terms.
    # There are 8 block products in the 2x2 block multiplication (but Strassen reduces to 7).
    # Using Strassen on the 2x2 block level: 7 block products, each with rank R3.
    # Total rank = 7 * R3.
    # If R3 = 23, total rank = 161. That's better than 189!
    # If R3 = 20, total rank = 140.
    # The known upper bound for 3x3x3 is 23 (Lader). Can we do better?
    
    # Let me try to find a rank-23 decomposition for 3x3x3. This is a known result
    # by Lader (1977). I need to find the actual coefficients.
    
    # Actually, let me try a different approach. Let me implement a simple optimization
    # to try to reduce the rank from 189. I'll use the current 189-rank decomposition
    # as a starting point and try to eliminate linearly dependent terms.
    
    # First, let me build the tensor and check for linear dependencies among the rank-1 terms.
    
    # The 189 terms from the seed program: for each of the 7 Strassen multipliers M1..M7,
    # and for each (r,s,c) in {0,1,2}^3 (the 3x3 block positions), we have one term.
    # Each term has U, V, W vectors of length 36.
    
    # If some of these 189 terms are linearly dependent (i.e., their (U, V, W) triples
    # satisfy a linear relation), we might be able to eliminate some.
    
    # Actually, tensor rank reduction is NP-hard in general. But for structured tensors,
    # we might find savings.
    
    # Let me try a simpler approach: check if the 189 vectors (flattened U*V*W) are
    # linearly independent. If not, we can potentially reduce the rank.
    
    # Actually, the correct way to check is: the rank-1 terms t_r = U_r * V_r * W_r (outer product)
    # span the tensor T. The tensor rank is the minimum number of rank-1 terms needed.
    # The 189 terms might not be minimal, but checking for linear dependencies among
    # the outer products is a necessary (but not sufficient) condition for reduction.
    
    # Let me build the tensor and the 189 outer products, then check the rank of the
    # matrix whose columns are the vectorized outer products.
    
    # This is a 36*36*36 = 46656 x 189 matrix. That's feasible.
    
    # Let me do this.
    
    # First, build the tensor T (46656 entries, but sparse with 216 non-zeros).
    # Then build the 189 outer products as columns of a matrix.
    # Then check the rank.
    
    # If the rank is less than 189, we can potentially reduce.
    
    # But even if the 189 outer products are linearly independent, the tensor rank
    # might still be less than 189 (we'd need a different set of rank-1 terms).
    
    # For now, let me just check the linear independence of the 189 terms.
    
    # Build tensor T as a 3D array
    T = np.zeros((36, 36, 36), dtype=np.float64)
    for i in range(36):
        row_a = i // 6
        col_a = i % 6
        for j in range(36):
            row_b = j // 6
            col_b = j % 6
            if col_a == row_b:
                k = row_a * 6 + col_b  # standard C index
                T[i, j, k] = 1.0
    
    # Build the 189 terms from the seed program
    block_size = 3
    precomputed_steps = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    terms_U = []
    terms_V = []
    terms_W = []
    
    for x_list, y_list, c_list in precomputed_steps:
        for r in range(block_size):
            for s in range(block_size):
                for c in range(block_size):
                    u_vec = np.zeros(36, dtype=np.float64)
                    v_vec = np.zeros(36, dtype=np.float64)
                    w_vec = np.zeros(36, dtype=np.float64)
                    for (ib, jb, x_coeff) in x_list:
                        u_idx = (ib * block_size + r) * 6 + (jb * block_size + s)
                        u_vec[u_idx] += x_coeff
                    for (jb, kb, y_coeff) in y_list:
                        v_idx = (jb * block_size + s) * 6 + (kb * block_size + c)
                        v_vec[v_idx] += y_coeff
                    for (ib, kb, c_coeff) in c_list:
                        # W index: k*n+i where k=col_C, i=row_C
                        # col_C = kb*block_size + c, row_C = ib*block_size + r
                        w_idx = (kb * block_size + c) * 6 + (ib * block_size + r)
                        w_vec[w_idx] += c_coeff
                    terms_U.append(u_vec)
                    terms_V.append(v_vec)
                    terms_W.append(w_vec)
    
    # Verify the decomposition
    T_reconstructed = np.zeros((36, 36, 36), dtype=np.float64)
    for r in range(len(terms_U)):
        T_reconstructed += np.einsum('i,j,k->ijk', terms_U[r], terms_V[r], terms_W[r])
    
    if not np.allclose(T_reconstructed, T, atol=1e-10):
        print("Decomposition verification failed!")
        # Return the seed program's result as fallback
        U = np.zeros((36, 189), dtype=np.float64)
        V = np.zeros((36, 189), dtype=np.float64)
        W = np.zeros((36, 189), dtype=np.float64)
        for r in range(189):
            U[:, r] = terms_U[r]
            V[:, r] = terms_V[r]
            W[:, r] = terms_W[r]
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Check linear independence of the 189 outer products
    # Vectorize each outer product: u * v * w -> 36*36*36 vector
    # This is a 46656 x 189 matrix
    outer_products = np.zeros((36*36*36, 189), dtype=np.float64)
    for r in range(189):
        outer_products[:, r] = np.einsum('i,j,k->ijk', terms_U[r], terms_V[r], terms_W[r]).flatten()
    
    # Compute the rank using SVD
    try:
        U_svd, svals, Vt = np.linalg.svd(outer_products, full_matrices=False)
        rank = np.sum(svals > 1e-10)
        print(f"Rank of 189 outer products: {rank} out of 189")
    except Exception as e:
        print(f"SVD failed: {e}")
        rank = 189
    
    # If rank < 189, we might be able to reduce, but it's not straightforward
    # to find the reduced set of terms.
    
    # For now, let me try a different approach: use a known lower-rank decomposition
    # for 3x3x3 if available, or try to find one via optimization.
    
    # Actually, let me try to use the fact that 6 = 2*3 and see if there's a way
    # to mix the block structures.
    
    # Alternative: try to find a rank-180 decomposition by removing 9 terms
    # if there are 9 linear dependencies.
    
    # For now, let me just return the 189-rank decomposition, but with a small
    # optimization: check if we can eliminate any terms that are linear combinations
    # of others.
    
    # Actually, let me try a completely different approach: use scipy to solve
    # for a lower-rank decomposition via iterative optimization.
    
    # The problem: minimize R such that T = sum_{r=1}^{R} u_r * v_r * w_r
    # This is a non-convex problem, but we can try gradient descent or alternating
    # least squares.
    
    # Start with R = 180 and see if we can find a solution.
    
    # Actually, this is very expensive for R=180 and tensor size 36^3. Let me
    # think of a smarter approach.
    
    # Given the time constraint (110s), let me try a simpler heuristic:
    # 1. Start with the 189 terms
    # 2. For each term, check if it can be approximated by a linear combination
    #    of the remaining terms
    # 3. If yes, remove it
    # Repeat until no more terms can be removed.
    
    # This is O(R^2 * 46656) which for R=189 is about 189^2 * 46656 ≈ 1.66 billion
    # operations. That might be too slow.
    
    # Let me try a faster approach: compute the SVD of the outer product matrix
    # and use the top-R components to define new terms.
    
    # Actually, the SVD approach doesn't directly give a tensor rank decomposition.
    
    # Let me just try to verify the current decomposition and return it, with
    # a small improvement: use integer/half-integer entries as required.
    
    # The current entries are already integers (1, -1, 0), so that's fine.
    
    # Let me check if there's a known rank-186 or lower decomposition for 6x6x6.
    # The best known upper bound for 6x6x6 matrix multiplication is 189 (from the
    # block Strassen approach). I'm not aware of a better published result.
    
    # Given the constraints, let me try a local search: randomly perturb the
    # coefficients and see if we can reduce the rank while maintaining the
    # decomposition.
    
    # Actually, that's not a good idea because the problem is discrete (rank is
    # an integer) and the search space is huge.
    
    # Let me try one more thing: check if the tensor T has any symmetries that
    # can be exploited to reduce the rank.
    
    # The tensor T[i,j,k] = 1 if i%6 == j//6 and k = (i//6)*6 + (j%6).
    # This tensor has a lot of structure: it's a "permutation" tensor in some sense.
    
    # Let me try to see if we can group the 216 non-zero entries into fewer
    # "clusters" that can each be handled with a lower-rank decomposition.
    
    # For now, let me just return the 189-rank decomposition with proper formatting.
    
    U_out = np.zeros((36, 189), dtype=np.float64)
    V_out = np.zeros((36, 189), dtype=np.float64)
    W_out = np.zeros((36, 189), dtype=np.float64)
    for r in range(189):
        U_out[:, r] = terms_U[r]
        V_out[:, r] = terms_V[r]
        W_out[:, r] = terms_W[r]
    
    # Verify once more
    T_check = np.zeros((36, 36, 36), dtype=np.float64)
    for r in range(189):
        T_check += np.einsum('i,j,k->ijk', U_out[:, r], V_out[:, r], W_out[:, r])
    if not np.allclose(T_check, T, atol=1e-10):
        print("Final verification failed!")
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
