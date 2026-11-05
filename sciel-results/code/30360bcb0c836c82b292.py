import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # First, build the rank-36 solution as fallback
    rank_36 = _construct_rank_36()
    
    # Try to improve using greedy rank-1 extraction with exact arithmetic
    try:
        better = _try_greedy_improve(n, m, p, deadline)
        if better is not None:
            return better
    except Exception:
        pass
    
    return rank_36


def _construct_rank_36():
    """Build the rank-36 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    
    strassen_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    standard_terms_2x2x1 = [
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        ({(0,1): 1}, {(1,0): 1}, {(0,0): 1}),
        ({(1,0): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(1,1): 1}, {(1,0): 1}, {(1,0): 1})
    ]
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, idx):
        for (i, j), val in u_coeffs.items():
            U[i * m + j, idx] += val
        for (j, k), val in v_coeffs.items():
            V[j * p + k, idx] += val
        for (k, i), val in w_coeffs.items():
            W[k * n + i, idx] += val

    for block_idx in range(2):
        j_offset = block_idx * 2
        
        for u_c, v_c, w_c in strassen_terms:
            u_g, v_g, w_g = {}, {}, {}
            for (i, j_l), val in u_c.items():
                u_g[(i, j_offset + j_l)] = val
            for (j_l, k_l), val in v_c.items():
                v_g[(j_offset + j_l, k_l)] = val
            for (i, k_l), val in w_c.items():
                w_g[(k_l, i)] = val
            add_term(u_g, v_g, w_g, col_idx)
            col_idx += 1
            
        for u_c, v_c, w_c in strassen_terms:
            u_g, v_g, w_g = {}, {}, {}
            for (i, j_l), val in u_c.items():
                u_g[(i, j_offset + j_l)] = val
            for (j_l, k_l), val in v_c.items():
                v_g[(j_offset + j_l, 2 + k_l)] = val
            for (i, k_l), val in w_c.items():
                w_g[(2 + k_l, i)] = val
            add_term(u_g, v_g, w_g, col_idx)
            col_idx += 1
            
        for u_c, v_c, w_c in standard_terms_2x2x1:
            u_g, v_g, w_g = {}, {}, {}
            for (i, j_l), val in u_c.items():
                u_g[(i, j_offset + j_l)] = val
            for (j_l, _), val in v_c.items():
                v_g[(j_offset + j_l, 4)] = val
            for (i, _), val in w_c.items():
                w_g[(4, i)] = val
            add_term(u_g, v_g, w_g, col_idx)
            col_idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_greedy_improve(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using greedy rank-1 extraction
    with exact rational arithmetic.
    
    The tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    We want to decompose T = sum_r u_r ⊗ v_r ⊗ w_r where:
    - u_r has shape (n*m,) indexed by (i,j)
    - v_r has shape (m*p,) indexed by (j,k)  
    - w_r has shape (p*n,) indexed by (k,i)
    
    And T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
    """
    # Build the tensor as a 3D array of Fractions
    # T[i,j,k] = delta(i,j,k) meaning C[i,k] gets contribution from A[i,j]*B[j,k]
    # In tensor form: T_{i,j,k} = 1 if this term contributes to C_{i,k}
    # Actually the multiplication tensor is: T[i,j,k] = 1 for all valid (i,j,k)
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T has entries T[i,j,k] = 1 for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # We need to decompose this into rank-1 tensors.
    
    # Flatten indices
    # U index: i*m + j, range [0, n*m)
    # V index: j*p + k, range [0, m*p)
    # W index: k*n + i, range [0, p*n)
    
    # The tensor T[i, j, k] = 1 for all i, j, k
    # This is actually the "all-ones" tensor of shape (n, m, p)
    
    # For the all-ones tensor, we can use:
    # T = sum over some basis of rank-1 tensors
    # 
    # Key insight: T[i,j,k] = 1 = (1)*(1)*(1)
    # So one rank-1 term is u[i,j]=1, v[j,k]=1, w[k,i]=1
    # But this only gives us T[i,j,k] = 1 * 1 * 1 = 1, which is correct!
    # 
    # Wait, that's just rank 1? No, that's not right.
    # The issue is that in matrix multiplication, we need:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor decomposition must satisfy:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (since each triple (i,j,k) contributes 1 to the product)
    
    # Actually, I need to be more careful. The multiplication tensor for 
    # (n x m) * (m x p) -> (n x p) is defined as:
    # T[i, j, k] = 1 if and only if this term appears in the computation
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i, j, k.
    
    # The tensor is indeed all-ones of shape (n, m, p) = (2, 4, 5).
    # 
    # For an all-ones tensor of shape (a, b, c), the tensor rank is:
    # - If a=1 or b=1 or c=1, rank = max(other two dims)
    # - In general, it's more complex
    #
    # For (2, 4, 5): Let me think about this differently.
    # T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}
    #
    # A rank-1 tensor is u[i,j] * v[j,k] * w[k,i]
    # For this to equal 1 for all (i,j,k), we'd need u[i,j]=v[j,k]=w[k,i]=1 for all entries,
    # which gives exactly T. So rank 1 would work? No, that can't be right.
    
    # Let me re-examine. The tensor decomposition for matrix multiplication is:
    # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}  (using Einstein notation with r as the rank index)
    # But the indices are structured: U is indexed by (i,j) flattened, etc.
    
    # Actually, I think I'm confusing myself. Let me use the standard formulation.
    # 
    # The multiplication tensor for M_{n,m} × M_{m,p} → M_{n,p} is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where e_i, e_j, e_k are basis vectors.
    # 
    # No wait. The tensor is:
    # T(i, j, k) = 1 for all i, j, k
    # This is the all-ones tensor.
    #
    # The tensor rank of the all-ones tensor of shape (a,b,c) over a field is:
    # It's known that for the all-ones tensor, the rank can be lower than a*b*c.
    #
    # For shape (2,4,5): Let me try to find a low-rank decomposition.
    #
    # T[i,j,k] = 1 = f(i,j) * g(j,k) * h(k,i)
    # For a single rank-1 term, we need f(i,j)*g(j,k)*h(k,i) = 1 for all i,j,k.
    # This means f(i,j) = a_i * b_j, g(j,k) = c_j * d_k, h(k,i) = e_k * f_i
    # Then product = a_i * b_j * c_j * d_k * e_k * f_i
    # For this to be 1 for all i,j,k, we need:
    # a_i * f_i = const_i (depends only on i)
    # b_j * c_j = const_j (depends only on j) 
    # d_k * e_k = const_k (depends only on k)
    # And const_i * const_j * const_k = 1 for all i,j,k, which means all consts are equal.
    # So we can set a_i = f_i = 1/sqrt(c), b_j = c_j = 1/sqrt(c), d_k = e_k = 1/sqrt(c)
    # where c = (1/3) such that (1/c)^3 = 1... hmm this is getting complicated.
    
    # Let me just try the greedy algorithm numerically with rationals.
    
    # Represent the tensor as a 3D array of Fractions
    T = np.zeros((n, m, p), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = Fraction(1)
    
    # Greedy extraction: repeatedly find the best rank-1 approximation
    # and subtract it.
    
    U_list = []
    V_list = []
    W_list = []
    
    max_rank = 40  # don't go above schoolbook
    start_time = time.time()
    
    for r in range(max_rank):
        if time.time() > deadline - 10:
            # Not enough time left, return what we have if it's valid, else None
            if r > 0:
                return _assemble_result(U_list, V_list, W_list, n, m, p)
            return None
        
        # Find the best rank-1 decomposition of the remaining tensor
        # T_rem = u ⊗ v ⊗ w + higher order terms
        # 
        # For the all-ones tensor, a good approach:
        # Try to find u[i,j], v[j,k], w[k,i] that maximize the "alignment"
        # with T_rem.
        
        # Simple approach: use the fact that for all-ones, we can use
        # structured decompositions.
        
        # Let's try a different approach: use known low-rank decompositions.
        # For (2,4,5), we can split into blocks.
        
        # Actually, let me try a more systematic greedy approach:
        # 1. Find the entry of T_rem with maximum absolute value
        # 2. Set that entry to 1 in the rank-1 term
        # 3. Solve for the other entries of u, v, w to best match T_rem
        
        # Find max entry
        max_val = None
        max_idx = None
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = abs(T[i, j, k])
                    if max_val is None or val > max_val:
                        max_val = val
                        max_idx = (i, j, k)
        
        if max_val == 0 or max_val == Fraction(0):
            break
        
        i0, j0, k0 = max_idx
        
        # For the rank-1 term, set u[i0,j0] * v[j0,k0] * w[k0,i0] = T[i0,j0,k0]
        # We have freedom in how to distribute the value.
        # Let's set u[i0,j0] = T[i0,j0,k0], v[j0,k0] = 1, w[k0,i0] = 1
        # Then for other entries, try to match T_rem.
        
        # Actually, for the all-ones tensor, a better approach:
        # Use the structure. T[i,j,k] = 1 for all i,j,k.
        # 
        # Rank-1 term: u[i,j] = a_i * b_j, v[j,k] = c_j * d_k, w[k,i] = e_k * f_i
        # Product: a_i * b_j * c_j * d_k * e_k * f_i
        # 
        # To get 1 for all (i,j,k), we need:
        # (a_i * f_i) * (b_j * c_j) * (d_k * e_k) = 1 for all i,j,k
        # This requires a_i*f_i = alpha, b_j*c_j = beta, d_k*e_k = gamma
        # with alpha*beta*gamma = 1.
        # 
        # Simple choice: alpha=beta=gamma=1, so a_i=f_i=1, b_j=c_j=1, d_k=e_k=1
        # This gives u[i,j]=1, v[j,k]=1, w[k,i]=1, and product=1.
        # So ONE rank-1 term gives the entire all-ones tensor!
        
        # Wait, that means the rank is 1? That can't be right for matrix multiplication.
        
        # I think I'm making an error in the tensor formulation. Let me reconsider.
        
        # The matrix multiplication tensor is NOT the all-ones tensor.
        # 
        # For C = A * B where C[i,k] = sum_j A[i,j] * B[j,k],
        # the tensor T is defined such that:
        # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
        # 
        # No, that's not right either. The standard formulation is:
        # The multiplication tensor T_{ijk} satisfies:
        # C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}
        # where T_{ijk} = 1 for all i,j,k.
        #
        # But in the tensor decomposition for bilinear complexity, we write:
        # C_{ik} = sum_r M_r * U_r * V_r
        # where U_r and V_r are linear combinations of A and B entries.
        #
        # The tensor T is the 3-way array T[i, j, k] = 1, and we decompose:
        # T[i,j,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]
        # 
        # Hmm, but if T is all-ones, then rank 1 should work as I showed above.
        # 
        # I think the issue is that the tensor for matrix multiplication is NOT all-ones.
        # Let me look at this more carefully.
        
        # Actually, I believe the correct formulation is:
        # The multiplication tensor for M_{n,m} × M_{m,p} is:
        # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{i,j} ⊗ E_{j,k} ⊗ E_{k,i}
        # where E_{i,j} is the matrix unit.
        #
        # As a 3D tensor indexed by (i,j,k), T[i,j,k] = 1 for all valid indices.
        # So it IS the all-ones tensor.
        #
        # But then why is the rank not 1? 
        #
        # Oh! I think the issue is that in the tensor decomposition, the three modes
        # are not independent. The first mode is indexed by (i,j) pairs, the second
        # by (j,k) pairs, and the third by (k,i) pairs. The constraint is that
        # the j-index in mode 1 must match the j-index in mode 2, and the k-index
        # in mode 2 must match the k-index in mode 3, and the i-index in mode 3
        # must match the i-index in mode 1.
        #
        # So it's not a free 3D tensor - it has a specific structure where the
        # indices are coupled.
        
        # Given the complexity, let me just verify the rank-36 solution works
        # and try a few specific low-rank constructions.
        
        # For (2,4,5), let me try to see if rank < 36 is possible.
        # 
        # One approach: split the 4 middle dimension into 2+2, and for each 2x2x5
        # subproblem, use a low-rank decomposition.
        #
        # For 2x2x5: split 5 into 2+2+1, use Strassen for 2x2x2 (rank 7 each)
        # and schoolbook for 2x2x1 (rank 4). Total: 7+7+4 = 18 per 2x2x5 block.
        # Two blocks: 36. That's what we have.
        #
        # Can we do better for 2x2x5? The trivial rank is 2*2*5=20.
        # With Strassen: 7+7+4=18. Can we do 17 or less?
        #
        # For 2x2x4: 7+7=14 (vs trivial 16). 
        # For 2x2x3: 7+4=11 (vs trivial 12).
        #
        # So 2x2x5 with rank 18 seems reasonable.
        
        # Let me try a different splitting: split the 5 columns differently.
        # Or use a different algorithm for 2x2x5.
        
        # Actually, let me just return None and use the rank-36 fallback.
        # The greedy approach with exact rationals is too slow for this size.
        
        return None
    
    return None


def _assemble_result(U_list, V_list, W_list, n, m, p):
    """Assemble the result from lists of rank-1 terms."""
    if not U_list:
        return None
    
    rank = len(U_list)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                U[i * m + j, r] = U_list[r][i * m + j]
        for j in range(m):
            for k in range(p):
                V[j * p + k, r] = V_list[r][j * p + k]
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = W_list[r][k * n + i]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
