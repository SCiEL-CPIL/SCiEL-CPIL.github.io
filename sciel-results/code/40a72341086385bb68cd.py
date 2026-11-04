"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # The multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T has indices (i, j, k) with T[i,j,k] = 1
    # We need to decompose T as sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    
    # Schoolbook: rank = n*m*p = 48
    # But we can do better using bilinear algorithms
    
    # Let's use a greedy compression approach:
    # 1. Start with schoolbook terms
    # 2. Try to find linear dependencies and merge
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # This is actually the outer product structure: T = sum_i (e_i ⊗ I_m ⊗ e_i)
    # where for each i, we have the m*p terms
    
    # Key insight: For each row i of A, C[i,:] = A[i,:] @ B
    # This is a matrix-vector multiplication (1x3 times 3x8)
    # The rank for 1x3x8 multiplication can be analyzed
    
    # For 1xm x px: the minimum rank is known to be at most m*p - min(m,p) + 1
    # For 1x3x8: at most 3*8 - min(3,8) + 1 = 24 - 3 + 1 = 22
    # But we have 2 such rows, so total could be 2*22 = 44, but they share the B matrix
    
    # Let me try a different approach: use the fact that we can share computations
    
    # Actually, let me implement a proper search using linear algebra
    
    # Build the schoolbook decomposition first
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    # Now try to compress: find if any column is a linear combination of others
    # We work in the tensor space: each term is a vector in R^{n*m * m*p * p*n}
    # But that's huge. Instead, we can check dependencies using the structure.
    
    # Alternative: use a greedy algorithm that tries to remove terms
    # by checking if the remaining terms can still span the tensor
    
    # Let me try a different strategy: use known results for small matrix multiplication
    # For 2x3 x 3x8, we can try to find a decomposition of rank < 48
    
    # Let me implement a randomized/greedy compression:
    # 1. Start with all 48 terms
    # 2. For each term, check if it's in the span of the others
    # 3. If yes, remove it
    
    def tensor_to_vector(U_col, V_col, W_col):
        """Convert a rank-1 term to a vector in the tensor space."""
        # The tensor space is R^{(n*m) x (m*p) x (p*n)}
        # But we can represent it as a matrix by flattening two indices
        # For checking dependencies, we can use the matricization
        pass
    
    # Better approach: use the fact that the multiplication tensor satisfies
    # T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i} (with appropriate indexing)
    
    # Let me try to find a lower rank decomposition using a constructive approach
    # Based on the idea that we can share the B matrix multiplications
    
    # For C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # For C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    # We can compute:
    # T_k = B[0,k] * [A[0,0], A[1,0]] + B[1,k] * [A[0,1], A[1,1]] + B[2,k] * [A[0,2], A[1,2]]
    
    # This suggests we can group by k: for each k, we have a 2x3 matrix-vector product
    # The rank for computing 2x3 * 3x1 is at most 2*3 - 1 = 5 (by known results)
    # So total rank could be 8 * 5 = 40
    
    # Let me implement this: for each k, decompose the 2x3x1 multiplication
    
    def decompose_2x3x1():
        """Find rank decomposition for multiplying 2x3 by 3x1."""
        # C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
        # C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
        # 
        # This is a tensor in R^{6} x R^{3} x R^{2}
        # Minimum rank for 2x3x1 is 5 (since 2*3=6, and we can save 1)
        # 
        # Construction: Use the fact that we can write:
        # Let S = A[0,0]*B[0] + A[1,0]*B[1] + A[0,2]*B[2]  (some combination)
        # Actually, let me use a known decomposition.
        # 
        # For 2x3x1, one decomposition of rank 5:
        # We can use:
        # Term 1: A[0,0]*B[0] -> contributes to C[0]
        # Term 2: A[0,1]*B[1] -> contributes to C[0]  
        # Term 3: A[0,2]*B[2] -> contributes to C[0]
        # Term 4: A[1,0]*B[0] -> contributes to C[1]
        # Term 5: A[1,1]*B[1] -> contributes to C[1]
        # Term 6: A[1,2]*B[2] -> contributes to C[1]
        # 
        # To save one term, we can use a bilinear trick.
        # Let me try: define T = A[0,0]*A[1,2] - A[0,2]*A[1,0] (but that's not right for this)
        # 
        # Actually, for matrix-vector multiplication, the minimum rank is known.
        # For 2x3 x 3x1, the rank is 5.
        # 
        # One decomposition:
        # Use the identity: x*y = ((x+y)^2 - x^2 - y^2)/4 but that doesn't help directly
        # 
        # Let me just use rank 6 for now and see if we can compress globally.
        
        # For 2x3x1, schoolbook is rank 6. Let's use that.
        terms = []
        for i in range(2):
            for j in range(3):
                terms.append((i, j))
        return terms
    
    # Let me try a different approach: use a global compression
    # The full tensor has 48 terms. Let me see if I can find dependencies.
    
    # Represent each schoolbook term as a vector in a suitable space
    # The tensor T is in R^{(n*m)} ⊗ R^{(m*p)} ⊗ R^{(p*n)}
    # We can flatten this to a vector of length n*m*m*p*p*n, but that's 48*24*16 = huge
    
    # Instead, let's use the matricization: fix two indices, vary the third
    # T_{i,j,k} = 1 for all i,j,k
    # This is actually the all-ones tensor!
    
    # Wait, no. T[i,j,k] = 1 means the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So the tensor is indeed all ones in the (i,j,k) space.
    
    # The all-ones tensor of size 2x3x8 has rank... 
    # For an all-ones tensor of size a x b x c, the rank is min(a,b,c) if we're talking about 
    # the standard tensor rank? No, that's not right either.
    
    # Actually, the tensor T where T[i,j,k] = 1 for all i,j,k can be written as:
    # T = sum_{r=1}^{min(n,m,p)} u_r ⊗ v_r ⊗ w_r
    # where u_r[i] = 1, v_r[j] = 1, w_r[k] = 1 for all r
    # But that gives T[i,j,k] = min(n,m,p), not 1.
    
    # To get T[i,j,k] = 1, we need:
    # T = (1/min(n,m,p)) * sum_{r=1}^{min(n,m,p)} (sqrt(min(n,m,p)) * 1_n) ⊗ ... 
    # This doesn't work simply.
    
    # Let me reconsider. The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # This is the sum of all basis tensors, which is (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k)
    # So T = 1_n ⊗ 1_m ⊗ 1_p, which is a rank-1 tensor!
    
    # Wait, that can't be right for matrix multiplication. Let me re-examine.
    
    # The multiplication tensor for C = A*B where A is n×m and B is m×p is:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p E_{i,j} ⊗ E_{j,k} ⊗ E_{k,i}
    # where E_{i,j} is the basis element for A[i,j], E_{j,k} for B[j,k], E_{k,i} for C[i,k]
    
    # So T[i,j,k] = 1 if we're looking at the coefficient, but the tensor is a sum of 48 rank-1 terms.
    # The question is whether these 48 rank-1 terms are linearly independent in the tensor space.
    
    # They are NOT linearly independent in general, so we can potentially compress.
    
    # Let me implement a proper linear algebra approach:
    # 1. Create a matrix where each column is one rank-1 term (flattened)
    # 2. Find the rank of this matrix
    # 3. That gives the minimum number of terms needed
    
    # The tensor space has dimension n*m * m*p * p*n = 48 * 24 * 16 = 18432
    # We have 48 vectors in this space. Let me check their linear independence.
    
    # But 18432 is manageable. Let me do this.
    
    def term_to_vector(i, j, k):
        """Convert term (i,j,k) to a vector in R^{n*m * m*p * p*n}."""
        vec = np.zeros(n*m * m*p * p*n)
        # The term is e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
        idx_a = i * m + j
        idx_b = j * p + k
        idx_c = k * n + i
        # Flatten: index = idx_a * (m*p) * (p*n) + idx_b * (p*n) + idx_c
        flat_idx = idx_a * (m*p) * (p*n) + idx_b * (p*n) + idx_c
        vec[flat_idx] = 1.0
        return vec
    
    # Build the matrix of all schoolbook terms
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append((i, j, k))
    
    M = np.array([term_to_vector(*t) for t in terms]).T  # Shape: (18432, 48)
    
    # Find the rank
    rank = np.linalg.matrix_rank(M)
    
    # Get a basis for the column space
    # Use QR with column pivoting to find a basis
    Q, R, P = np.linalg.qr(M, mode='complete', pivoting=True) if hasattr(np.linalg, 'qr') else (None, None, None)
    
    # numpy's qr doesn't support pivoting directly, let me use SVD
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    
    # The rank is the number of non-zero singular values
    tol = 1e-10
    rank_svd = np.sum(s > tol)
    
    # Vt[0:rank_svd, :] gives us the basis vectors
    # We can reconstruct the decomposition using these basis vectors
    
    # Actually, for the decomposition, we need to express each original term as a linear combination
    # of the basis vectors, then use that to build U, V, W.
    
    # Let me use a different approach: use the SVD to get a low-rank approximation
    # But we need EXACT decomposition, so we need the exact rank.
    
    # For exact arithmetic, let me use rational numbers or check dependencies exactly.
    
    # Since all entries are 0 or 1, we can use integer linear algebra.
    
    # Let me try a greedy approach: start with all terms, and try to remove one at a time
    # if it's in the span of the others.
    
    active = list(range(len(terms)))
    
    def is_in_span(vec, other_vecs, tol=1e-10):
        """Check if vec is in the span of other_vecs."""
        if len(other_vecs) == 0:
            return np.allclose(vec, 0, atol=tol)
        M_sub = np.array(other_vecs).T
        # Solve M_sub * x = vec
        try:
            x, residuals, rank_sub, s_vals = np.linalg.lstsq(M_sub, vec, rcond=tol)
            residual = np.linalg.norm(M_sub @ x - vec)
            return residual < tol * max(1.0, np.linalg.norm(vec))
        except:
            return False
    
    # This is O(n^2 * dim) which might be slow, but let's try
    # Actually, let me use a more efficient method: maintain a basis incrementally
    
    basis = []
    basis_indices = []
    
    for idx in active:
        vec = term_to_vector(*terms[idx])
        # Check if vec is in the span of current basis
        if len(basis) == 0:
            basis.append(vec)
            basis_indices.append(idx)
        else:
            B = np.array(basis).T
            # Project vec onto the basis
            try:
                coeffs, residuals, rank_b, s_vals = np.linalg.lstsq(B, vec, rcond=1e-10)
                proj = B @ coeffs
                residual = np.linalg.norm(vec - proj)
                if residual > 1e-10 * max(1.0, np.linalg.norm(vec)):
                    # Not in span, add to basis
                    basis.append(vec)
                    basis_indices.append(idx)
            except:
                basis.append(vec)
                basis_indices.append(idx)
    
    min_rank = len(basis_indices)
    
    # Now build the decomposition using these basis terms
    # But we need to express ALL terms as combinations of the basis
    # For the final answer, we just need a valid decomposition of rank = min_rank
    
    # Use the basis terms directly as our decomposition
    U = np.zeros((n*m, min_rank))
    V = np.zeros((m*p, min_rank))
    W = np.zeros((p*n, min_rank))
    
    for r, idx in enumerate(basis_indices):
        i, j, k = terms[idx]
        U[i*m + j, r] = 1.0
        V[j*p + k, r] = 1.0
        W[k*n + i, r] = 1.0
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

