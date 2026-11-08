import time
import numpy as np
from scipy.linalg import svd

DIMENSIONS = [3, 3, 8]

def _build_strassen_3x3x3x8():
    """Build the 68-rank Strassen-based decomposition for 3x3 x 3x8."""
    n, m, p = 3, 3, 8
    U_t = []
    V_t = []
    W_t = []
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b):
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k):
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    for c in range(0, 8, 2):
        strassen_decomp(0, 0, 0, c)
    
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k)
            
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k)
            
    for k in range(8):
        direct_term(2, 2, k)
    
    rank = len(W_t)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i, idx] = coeff
            
    return U, V, W

def _verify_decomp(U, V, W):
    """Verify decomposition is exact."""
    n, m, p = 3, 3, 8
    rank = U.shape[1]
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_mat = U[:, r].reshape(n, m)
            v_mat = V[:, r].reshape(m, p)
            w_mat = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[i, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i] * A[i, j] * B[j, k]
        if np.max(np.abs(C_computed - C_expected)) > 1e-9:
            return False
    return True

def _to_nice_list(mat):
    res = []
    for row in mat:
        nrow = []
        for v in row:
            if abs(v - round(v*2)/2) < 1e-9:
                nv = round(v*2)/2
                if abs(nv - round(nv)) < 1e-9:
                    nrow.append(int(round(nv)))
                else:
                    nrow.append(nv)
            else:
                nrow.append(v)
        res.append(nrow)
    return res

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 135.0

    def to_nice_list(mat):
        res = []
        for row in mat:
            nrow = []
            for v in row:
                if abs(v - round(v*2)/2) < 1e-9:
                    nv = round(v*2)/2
                    if abs(nv - round(nv)) < 1e-9:
                        nrow.append(int(round(nv)))
                    else:
                        nrow.append(nv)
                else:
                    nrow.append(v)
            res.append(nrow)
        return res

    def verify(U, V, W):
        """Verify that sum_r U[:,r] * V[:,r] * W[:,r] equals the matmul tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # Tensor T[a_idx, b_idx, c_idx] = sum_r U[a_idx,r]*V[b_idx,r]*W[c_idx,r]
        # a_idx = i*m + j, b_idx = j*p + k, c_idx = k*n + i
        # T should be 1 when indices are consistent (same j, same k, same i), 0 otherwise
        # Actually the multiplication tensor: T_{(i,j),(j',k),(k',i')} = 1 if i=i', j=j', k=k' else 0
        # So T is the identity-like tensor: T[a,b,c] = 1 iff a%m == b//p and b%p == c//n and c%n == a//m
        # Let's just check numerically: for random A, B, check C = A@B
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # Compute C from decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # Contribution: sum over all i,j,k of U[i,j,r]*V[j,k,r]*W[k,i,r] * A[i,j]*B[j,k]
            # = (A * u) @ (v * B) ... no
            # C_computed[i,k] += sum_j A[i,j]*B[j,k]*sum_r ... no
            # The decomposition says: C[i,k] = sum_{i',j',k'} T[(i',j'),(j',k'),(k',i')] * A[i',j'] * B[j',k']
            # where T[(i',j'),(j',k'),(k',i')] = sum_r U[i',j',r]*V[j',k',r]*W[k',i',r]
            # For this to equal C[i,k] = sum_j A[i,j]*B[j,k], we need:
            # sum_r U[i',j',r]*V[j',k',r]*W[k',i',r] = delta_{i',i} * delta_{j',k'} ... 
            # Actually it should be: sum_r U[i',j',r]*V[j',k',r]*W[k',i',r] = 1 if (i'=i, j'=j, k'=k) for the (i,j,k) term
            # No wait. The tensor T is defined so that <T, A, B> = <C, I> where C = A@B.
            # <T, A, B> = sum_{i,j,k} T[(i,j),(j,k),(k,i)] * A[i,j] * B[j,k]
            # <C, I> = sum_{i,k} C[i,k] * I[i,k] = sum_{i,k} C[i,k] (if I is identity... no)
            # Actually the convention is: the multiplication tensor satisfies
            # C[i,k] = sum_{i',j',k'} T[(i',j'),(j',k'),(k',i')] A[i',j'] B[j',k']
            # For standard matmul: C[i,k] = sum_j A[i,j] B[j,k]
            # So T[(i,j),(j,k),(k,i)] = 1 and T = 0 otherwise (in the sense that only matching indices contribute)
            # More precisely: T[(i',j'),(j',k'),(k',i')] = 1 if i'=i and j'=j and k'=k for some (i,j,k), else 0
            # But that's just T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k, and 0 for inconsistent index triples.
            
            # Simplest check: reconstruct the tensor and verify
            pass
        
        # Better: just check that the decomposition reproduces matmul for random inputs
        C_from_decomp = np.zeros((n, p))
        for r in range(rank):
            u_vec = U[:, r]  # size n*m
            v_vec = V[:, r]  # size m*p
            w_vec = W[:, r]  # size p*n
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            w_mat = w_vec.reshape(p, n)
            # The r-th term contributes: for each (i,j,k), U[i,j,r]*V[j,k,r]*W[k,i,r] * A[i,j]*B[j,k]
            # Sum over i,j,k: this equals sum_{i,j} A[i,j] * (sum_k V[j,k,r]*W[k,i,r]*B[j,k]) * U[i,j,r]
            # = sum_i (sum_j A[i,j]*U[i,j,r]) * (sum_k W[k,i,r]*B[j,k]) ... this is getting complex
            # Let me just do it directly for small sizes
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_from_decomp[i, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i] * A[i, j] * B[j, k]
        
        if np.max(np.abs(C_from_decomp - C_expected)) > 1e-9:
            return False
        return True

    # Step 1: Build 3x3 x 3x1 decomposition (rank 7: Strassen 7 for 2x2 part + 1 direct for 3rd row/col)
    # A is 3x3, B is 3x1, C is 3x1
    # Use Strassen for top-left 2x2 block: A[0:2,0:2] * B[0:2,0] -> C[0:2,0]
    # Direct: A[0:2,2]*B[2,0] -> C[0:2,0], A[2,0:2]*B[0:2,0] -> C[2,0], A[2,2]*B[2,0] -> C[2,0]
    # That's 7 + 2 + 2 + 1 = 12 terms. Too many.
    # Actually for 3x3 x 3x1, the direct method is 9 terms. Strassen doesn't help much for 3x1.
    # Let me reconsider.
    
    # Better approach: Use the Kronecker product structure.
    # 3x3 x 3x8 can be seen as: for each of the 8 columns of B, we do 3x3 x 3x1.
    # But we can share the A-factor across columns.
    # 
    # The key insight from the directive: treat as Kronecker product of (3x3 x 3x1) and (1x1 x 1x8).
    # 3x3 x 3x1: rank = 9 (direct) or we can try to do better.
    # Actually, the rank of 3x3 x 3x1 matmul is exactly 9 (it's just 9 scalar products).
    # 1x1 x 1x8: rank = 8 (each output is independent).
    # Kronecker product gives rank 9*8 = 72. Worse than current 68.
    #
    # Alternative: The current approach uses Strassen for 2x2 blocks which gives rank 7 per 2x2x2 block.
    # For 3x3 x 3x8: 
    #   - 4 blocks of 2x2 x 2x2 (using Strassen, rank 7 each) = 28
    #   - Remaining terms for the 3rd row/col
    #   Current: 28 + 16 + 16 + 8 = 68
    #
    # Let me try a different decomposition strategy.
    # 
    # Approach: Use the fact that 3x3 x 3x8 = (2x2 + 1) x (2x2 + 1) x (2x2 + 1) x 4
    # But the Kronecker product idea from the directive suggests:
    # Factor 1: 3x3 x 3x1 (the "row" part)
    # Factor 2: 1x1 x 1x8 (the "column" part)  
    # 
    # For 3x3 x 3x1, the minimal rank is 9 (direct).
    # For 1x1 x 1x8, the minimal rank is 8.
    # Kronecker: 9*8 = 72. Not better.
    #
    # But wait - maybe we can use a non-trivial decomposition of 3x3 x 3x1.
    # Actually, let me try a different factorization.
    # 
    # The current 68-rank decomposition:
    # - 4 Strassen blocks (2x2*2x2) for columns 0-1, 2-3, 4-5, 6-7 of B, using A[0:2,0:2]: 4*7=28
    # - A[0:2,2] * B[2,0:8]: 2*8=16
    # - A[2,0:2] * B[0:2,0:8]: 2*8=16  
    # - A[2,2] * B[2,0:8]: 8
    # Total: 68
    #
    # Can we do better? Let's try using Strassen for different block structures.
    # 
    # What if we use a 3x3 x 3x3 block (rank 7 via Strassen-like for 3x3? No, 3x3 min rank is 19 or so)
    # 
    # Actually, let me try the SVD-based reduction on the current 68-rank decomposition
    # to see if there are linear dependencies.
    
    def build_current_decomposition():
        U_t = []
        V_t = []
        W_t = []
        
        def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b):
            a = (row_off_a, col_off_a)
            b = (row_off_a, col_off_a + 1)
            c = (row_off_a + 1, col_off_a)
            d = (row_off_a + 1, col_off_a + 1)
            e = (row_off_b, col_off_b)
            f = (row_off_b, col_off_b + 1)
            g = (row_off_b + 1, col_off_b)
            h = (row_off_b + 1, col_off_b + 1)
            
            C00 = (row_off_a, col_off_b)
            C01 = (row_off_a, col_off_b + 1)
            C10 = (row_off_a + 1, col_off_b)
            C11 = (row_off_a + 1, col_off_b + 1)
            
            U_t.append([(a, 1), (d, 1)])
            V_t.append([(e, 1), (h, 1)])
            W_t.append([(C00, 1), (C11, 1)])
            
            U_t.append([(c, 1), (d, 1)])
            V_t.append([(e, 1)])
            W_t.append([(C10, 1), (C11, -1)])
            
            U_t.append([(a, 1)])
            V_t.append([(f, 1), (h, -1)])
            W_t.append([(C01, 1), (C11, 1)])
            
            U_t.append([(d, 1)])
            V_t.append([(g, 1), (e, -1)])
            W_t.append([(C00, 1), (C10, 1)])
            
            U_t.append([(a, 1), (b, 1)])
            V_t.append([(h, 1)])
            W_t.append([(C00, -1), (C01, 1)])
            
            U_t.append([(c, 1), (a, -1)])
            V_t.append([(e, 1), (f, 1)])
            W_t.append([(C11, 1)])
            
            U_t.append([(b, 1), (d, -1)])
            V_t.append([(g, 1), (h, 1)])
            W_t.append([(C00, 1)])
        
        def direct_term(i, j, k):
            U_t.append([((i, j), 1)])
            V_t.append([((j, k), 1)])
            W_t.append([((i, k), 1)])
        
        for c in range(0, 8, 2):
            strassen_decomp(0, 0, 0, c)
        
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k)
                
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k)
                
        for k in range(8):
            direct_term(2, 2, k)
            
        rank = len(W_t)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W[k * n + i, idx] = coeff
                
        return U, V, W

    def build_kronecker_decomposition():
        """Build decomposition using Kronecker product structure.
        3x3 x 3x8 = (3x3 x 3x1) kron (1x1 x 1x8)
        But we need to be careful about how the indices work.
        
        Actually, let me think about this differently.
        The multiplication tensor for 3x3 x 3x8 is:
        T[(i,j),(j,k),(k,i)] = 1
        
        We can write this as: T[(i,j),(j,k),(k,i)] = S[(i,j),(j,i)] * R[(i,k),(k,k)]
        where S is the 3x3 x 3x1 tensor and R is the 1x1 x 1x8 tensor.
        But this factorization doesn't work directly because the indices don't separate cleanly.
        
        Let me try a different approach: use the current 68-rank decomp and try to reduce via SVD.
        """
        return build_current_decomposition()

    # Build the initial decomposition
    U, V, W = build_current_decomposition()
    rank = W.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    # Try SVD-based rank reduction
    if time.time() < DEADLINE:
        dim = n * m * m * p * p * n  # 9 * 24 * 24 = 5184
        M = np.zeros((dim, rank))
        
        for r in range(rank):
            vec = np.kron(np.kron(U[:, r], V[:, r]), W[:, r])
            M[:, r] = vec
        
        # Compute SVD to find numerical rank
        U_svd, s, Vt = svd(M, full_matrices=False)
        
        # Find the numerical rank - look for a gap in singular values
        # Since entries are integers/half-integers, exact dependencies give singular value ~0
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There are linear dependencies. Try to find a minimal subset.
            # Use the fact that M = U_svd @ diag(s) @ Vt
            # The column space of M is spanned by U_svd[:, :num_rank]
            # We need to express original columns as combinations of basis vectors
            # and find a rank-1 decomposition of the resulting tensor.
            
            # Alternative: find num_rank linearly independent columns and express others
            # in terms of them, then absorb the coefficients.
            # This is complex for tensor decompositions.
            
            # Simpler: just use the first num_rank right singular vectors as new "terms"
            # But these won't be rank-1 tensors in general.
            
            # For now, if we find dependencies, we know the true rank is lower.
            # Let's try a greedy approach: remove columns one at a time if they're
            # in the span of the remaining columns.
            
            # Actually, let's try a different approach: use the SVD to get a
            # lower-rank approximation and check if it's exact.
            # If s[num_rank] is truly 0 (within tol), then the first num_rank
            # terms span the tensor exactly.
            
            # We can form a new decomposition with num_rank terms by taking
            # the SVD components. But they won't be rank-1 in the original sense.
            
            # Let me try: if the tensor has rank num_rank < rank, we can find
            # num_rank rank-1 tensors that sum to T. The SVD gives us the
            # "directions" but not necessarily rank-1 components.
            
            # For now, just note the reduced rank and try to construct explicitly.
            # This is a research problem, so let's just return the current best.
            pass

    # Try alternative decomposition structures
    if time.time() < DEADLINE:
        # Try: Use Strassen for 2x2 blocks but with different grouping
        # What if we group B columns as (0,1,2) and (3,4,5,6,7)?
        # Or use a 3x3 x 3x3 block with a known rank-19 decomposition?
        
        # Actually, let me try: 3x3 x 3x8 where we use the fact that
        # rank(3x3 x 3x8) <= rank(3x3 x 3x3) + rank(3x3 x 3x5)
        # This doesn't directly help.
        
        # Let me try a completely different structure:
        # Use 2x2 Strassen for A[0:2,0:2] and handle the rest with a
        # more efficient scheme.
        
        # Current: 4*7 + 16 + 16 + 8 = 68
        # What if we use 3x3 x 3x8 = (2x2 + 1) x (2x2 + 1) x (4x2)?
        # This is getting complicated. Let me just try the SVD reduction properly.
        
        # Proper SVD-based rank reduction:
        # If M has rank r < rank, we can find r linearly independent columns.
        # Say columns c_1, ..., c_r are independent.
        # Each other column c_j = sum_i alpha_{ji} c_i.
        # In the tensor decomposition: T = sum_j U_j V_j W_j
        # If c_j = sum_i alpha_{ji} c_i, then the j-th rank-1 tensor is a linear
        # combination of the first r rank-1 tensors. But a linear combination of
        # rank-1 tensors is not generally rank-1.
        # 
        # However, if we can find coefficients such that the combination can be
        # "absorbed" into the other terms, we can reduce the rank.
        # This is only possible if the linear dependency has a special structure.
        
        # For now, let's just verify the current decomposition is valid and return it.
        pass

    # Final verification
    if not verify(U, V, W):
        # Fallback to direct decomposition
        rank_direct = n * m * p
        U = np.zeros((n * m, rank_direct))
        V = np.zeros((m * p, rank_direct))
        W = np.zeros((p * n, rank_direct))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank_direct

    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
# EVOLVE-BLOCK-END
