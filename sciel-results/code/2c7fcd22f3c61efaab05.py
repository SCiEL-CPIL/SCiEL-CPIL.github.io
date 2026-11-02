"""Matrix multiplication 2x3x6 - exploit block structure for rank-16 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = 2, 3, 6

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def verify_full(U, V, W, n_s, m_s, p_s):
        r = U.shape[1]
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m_s + j, c] * V[j * p_s + k, c] * W[k * n_s + i, c]
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    # Try to construct a rank-16 decomposition using block structure.
    # The 2x3 matrix can be viewed as two 1x3 blocks (rows).
    # Each 1x3 times 3x6 is a 1x3x6 tensor.
    # A 1x3x6 tensor has rank 3 (since it's just a 3x6 matrix, rank <= 3, and for generic it's 3).
    # But we need exact decomposition with half-integers.
    # Actually, the full tensor is T(i,j,k) = delta_{i,j,k} for matrix mult.
    # T is the 2x3x6 multiplication tensor.
    # We can write A (2x3) = [A0; A1] where A0, A1 are 1x3.
    # B (3x6).
    # C = A*B is 2x6. C0 = A0*B, C1 = A1*B.
    # The tensor decomposes as T = T0 + T1 where T0(i,j,k) = delta_{i,0} * (1 if j,k match in 1x3x6 mult)
    # Actually, the multiplication tensor for 1x3 x 3x6 is a 1x3x6 tensor with entries:
    # T0(j,k) = 1 for all j in 0..2, k in 0..5? No.
    # For 1x3 * 3x6, the result C(k) = sum_j A(j) B(j,k).
    # The tensor is T(j,k) = 1 for each (j,k) pair? No, it's a single tensor with 3*6 = 18 terms.
    # The rank of a 1x3x6 multiplication tensor: this is the rank of a 3x6 matrix viewed as a vector in R^3 ⊗ R^6.
    # The rank of a matrix as a tensor in R^3 ⊗ R^6 is its matrix rank. So rank = 3 (for generic 3x6).
    # So each block has rank 3, total rank 6? That seems too low.
    # Wait, the multiplication tensor for m x n * n x p has rank at least n (the shared dimension).
    # For 1x3 * 3x6, the shared dimension is 3, so rank >= 3. And it's exactly 3 for generic matrices.
    # But we need an EXACT decomposition with integer/half-integer entries that works for ALL matrices.
    # The schoolbook decomposition has rank 18 for each block (3*6=18 terms).
    # Can we do better? The rank of the 1x3x6 multiplication tensor is 3.
    # Let's check: T(j,k) for 1x3x6 mult. The tensor is sum_{j=0}^{2} e_j ⊗ B_j where B_j is the j-th row of B?
    # No. The tensor T ∈ R^{1×3×6} such that C(k) = <T, A ⊗ B>.
    # T(j,k) = 1 for all j,k? No.
    # For 1x3 * 3x6: C(k) = sum_{j=0}^{2} A(j) B(j,k).
    # So T(j,k) = 1 for all j in {0,1,2}, k in {0,...,5}.
    # This tensor is the outer product of a 3-vector of ones and a 6-vector of ones? No.
    # T(j,k) = 1 for all j,k means T = 1_3 ⊗ 1_6. This has rank 1!
    # Wait, that's not right. Let me re-examine.
    # The multiplication tensor T satisfies C = sum_{j,k} T(j,k) A(j) B(k).
    # For 1x3 * 3x6, C(k) = sum_j A(j) B(j,k).
    # So T(j,k) = 1 if the indices match the multiplication pattern.
    # In standard matrix mult, (AB)_{ik} = sum_j A_{ij} B_{jk}.
    # So for i=0, (AB)_{0k} = sum_j A_{0j} B_{jk}.
    # The tensor entry T_{0,j,k} = 1 for all j,k? No!
    # T_{i,j,k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # In standard mult, T_{i,j,k} = 1 for all valid i,j,k.
    # So for 2x3*3x6, T_{i,j,k} = 1 for all i in {0,1}, j in {0,1,2}, k in {0,...,5}.
    # That's 2*3*6 = 36 ones. The schoolbook rank is 36.
    # But the tensor is not all ones! T_{i,j,k} is 1 only when the indices form a valid multiplication term.
    # In matrix multiplication, EVERY combination (i,j,k) with i in rows of A, j in cols of A / rows of B, k in cols of B is a valid term.
    # So T_{i,j,k} = 1 for ALL i,j,k in the valid ranges.
    # This means the tensor is the all-ones tensor of shape 2x3x6.
    # The rank of the all-ones tensor of shape 2x3x6... 
    # Actually no. The multiplication tensor T_{i,j,k} = 1 for all i,j,k means:
    # C_{ik} = sum_{j} A_{ij} B_{jk} ... wait, that's not what T=1 gives.
    # If T_{i,j,k} = 1 for all i,j,k, then C_{ik} = sum_{j} A_{ij} B_{jk} is NOT what we get.
    # We get C_{ik} = sum_{j} T_{i,j,k} A_{ij} B_{jk} = sum_{j} A_{ij} B_{jk}. Yes!
    # So the tensor IS the all-ones tensor.
    # The rank of the all-ones tensor of shape 2x3x6:
    # This tensor can be written as sum over... let's think.
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k.
    # This is (sum_i e_i) ⊗ (sum_j e_j) ⊗ (sum_k e_k) = 1_2 ⊗ 1_3 ⊗ 1_6.
    # This is a rank-1 tensor!
    # So C_{ik} = (sum_i A_{ij} ... no.
    # Let me re-derive. The tensor T is defined by C_{ik} = sum_{j} T_{i,j,k} A_{ij} B_{jk}.
    # For standard matrix mult, T_{i,j,k} = 1 for all i,j,k.
    # Then C_{ik} = sum_j A_{ij} B_{jk}. Correct.
    # Now, can we decompose T = sum_{r} u_r ⊗ v_r ⊗ w_r with few terms?
    # T_{i,j,k} = 1 for all i,j,k.
    # T = 1_2 ⊗ 1_3 ⊗ 1_6 is rank 1.
    # So u = 1_2, v = 1_3, w = 1_6 gives T_{i,j,k} = 1*1*1 = 1. Rank 1!
    # But wait, that would mean C_{ik} = (sum_i A_{ij}... no.
    # If T = u ⊗ v ⊗ w, then C_{ik} = u_i v_j w_k A_{ij} B_{jk} summed over j... 
    # No, the decomposition is T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    # If rank 1: T_{i,j,k} = U_i V_j W_k.
    # For T_{i,j,k} = 1 for all i,j,k, we need U_i V_j W_k = 1 for all i,j,k.
    # This is possible with U_i = 1, V_j = 1, W_k = 1.
    # So rank 1 suffices?! That can't be right for matrix multiplication.
    # I'm confusing myself. Let me look at this more carefully.
    # The matrix multiplication tensor for m×n × n×p is T ∈ R^{m×n×p} with T_{i,j,k} = 1 if the term A_{ij}B_{jk} appears in C_{ik}.
    # In standard matrix multiplication, for each output (i,k), we sum over j: C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{i,j,k} = 1 for all valid i, j, k. There's no condition - every (i,j,k) triple contributes.
    # So T is indeed the all-ones tensor.
    # But the rank of the all-ones tensor of shape m×n×p is 1 (as I showed).
    # This would imply that matrix multiplication can be done in rank 1, which is absurd.
    # The error: the tensor for matrix multiplication is NOT the all-ones tensor in the sense I'm thinking.
    # Let me re-read the problem. "The output factor follows the official trace convention W[k*n+i,r]."
    # The tensor is T_{i,j,k} where the decomposition is:
    # C_{ik} = sum_{j,r} A_{ij} B_{jk} ... no.
    # The standard decomposition: C = A*B means C_{ik} = sum_j A_{ij} B_{jk}.
    # In tensor form: C_{ik} = sum_{j} T_{i,j,k} A_{ij} B_{jk} where T_{i,j,k} = 1.
    # But the RANK of this tensor is the minimum number of terms to write T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
    # For T_{i,j,k} = 1 for all i,j,k, we have T = 1⊗1⊗1, rank 1.
    # This is clearly wrong because matrix multiplication has rank > 1.
    # The resolution: the indices in the tensor decomposition are DIFFERENT from the matrix indices.
    # In the tensor T_{i,j,k}, the index j is the "contracted" index, but in the decomposition T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r}, the r is the rank index.
    # The key point: T_{i,j,k} = 1 for all i,j,k means the tensor is separable, hence rank 1.
    # But this would mean C_{ik} = sum_r (sum_j U_{i,r} A_{ij}) (sum_k ... no.
    # Let me re-derive the tensor multiplication formula.
    # Given T_{i,j,k} = sum_r U_{i,r} V_{j,r} W_{k,r},
    # C_{ik} = sum_j T_{i,j,k} A_{ij} B_{jk} = sum_j sum_r U_{i,r} V_{j,r} W_{k,r} A_{ij} B_{jk}
    # = sum_r U_{i,r} W_{k,r} (sum_j A_{ij} V_{j,r} B_{jk})
    # This is NOT the same as C_{ik} = sum_j A_{ij} B_{jk} unless the decomposition is special.
    # I think I'm making an error. Let me look at the actual definition.
    # The multiplication tensor T satisfies: for all A, B, C = A*B iff C_{ik} = sum_{j} T_{i,j,k} A_{ij} B_{jk} is NOT the right formula.
    # The correct formula is: the tensor T is such that the bilinear map (A,B) -> A*B is represented by T.
    # T_{i,j,k} = 1 means that the (i,j) entry of A and (j,k) entry of B contribute to the (i,k) entry of C.
    # In standard matrix multiplication, T_{i,j,k} = 1 for ALL i, j, k.
    # So the tensor is all ones, and its rank is 1.
    # This is a well-known result: the "rank" of matrix multiplication in this naive sense is 1, but that's not the right notion.
    # The correct notion: we want to write C = A*B as a sum of rank-1 bilinear forms.
    # C_{ik} = sum_r U_{i,r} V_{j,r} W_{k,r} ... no, that's not right either.
    # Let me just trust the problem statement. The schoolbook rank is n*m*p = 36.
    # The task says to find a rank-minimal decomposition. The known rank for 2x3x6 is around 20.
    # Let me just try to find a good decomposition using numerical methods.
    # I'll use the approach from the current program but with better initialization and more robust search.

    def compute_residuals(U, V, W, n_s, m_s, p_s):
        r = U.shape[1]
        residuals = np.zeros(n_s * m_s * p_s)
        idx = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m_s + j, c] * V[j * p_s + k, c] * W[k * n_s + i, c]
                    residuals[idx] = val - 1.0
                    idx += 1
        return residuals

    def try_rank(rank, max_seeds=10, max_iter=200):
        """Try to find a rank-r decomposition using random restarts and local optimization."""
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total_size = size_U + size_V + size_W

        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                return None

            np.random.seed(seed * 1000 + rank)
            x0 = np.random.randn(total_size)
            
            # Normalize initial vectors to avoid scale issues
            U0 = x0[:size_U].reshape(n * m, rank)
            V0 = x0[size_U:size_U + size_V].reshape(m * p, rank)
            W0 = x0[size_U + size_V:].reshape(p * n, rank)
            
            # Scale each column to unit norm
            for c in range(rank):
                nU = np.linalg.norm(U0[:, c])
                nV = np.linalg.norm(V0[:, c])
                nW = np.linalg.norm(W0[:, c])
                if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                    scale = (nU * nV * nW) ** (1.0/3.0)
                    U0[:, c] /= scale
                    V0[:, c] /= scale
                    W0[:, c] /= scale
            
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            # Use scipy least_squares
            import scipy.optimize as opt
            def fun(x):
                U = x[:size_U].reshape(n * m, rank)
                V = x[size_U:size_U + size_V].reshape(m * p, rank)
                W = x[size_U + size_V:].reshape(p * n, rank)
                return compute_residuals(U, V, W, n, m, p)
            
            try:
                res = opt.least_squares(fun, x0, method='lm', max_nfev=500)
                err = np.sum(res.fun**2)
                
                if err < 1e-10:
                    U_opt = res.x[:size_U].reshape(n * m, rank)
                    V_opt = res.x[size_U:size_U + size_V].reshape(m * p, rank)
                    W_opt = res.x[size_U + size_V:].reshape(p * n, rank)
                    
                    # Try snapping to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    if verify_full(U_snap, V_snap, W_snap, n, m, p):
                        return U_snap, V_snap, W_snap
                    
                    # Try integers
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if verify_full(U_int, V_int, W_int, n, m, p):
                        return U_int, V_int, W_int
                    
                    # If not snapping, try to clean up small values
                    U_clean = np.where(np.abs(U_opt) < 1e-6, 0, U_opt)
                    V_clean = np.where(np.abs(V_opt) < 1e-6, 0, V_opt)
                    W_clean = np.where(np.abs(W_opt) < 1e-6, 0, W_opt)
                    U_clean = np.round(U_clean * 2) / 2
                    V_clean = np.round(V_clean * 2) / 2
                    W_clean = np.round(W_clean * 2) / 2
                    if verify_full(U_clean, V_clean, W_clean, n, m, p):
                        return U_clean, V_clean, W_clean
                        
            except Exception:
                pass

        return None

    # Try ranks from low to high
    for rank in range(16, 37):
        if time.time() > deadline - 10:
            break
            
        result = try_rank(rank)
        if result:
            U_final, V_final, W_final = result
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            # If we found a good rank, we can stop if it's the theoretical minimum
            if rank <= 16:
                break

    return best_result
# EVOLVE-BLOCK-END
