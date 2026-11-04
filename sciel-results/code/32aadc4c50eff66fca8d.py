"""Find rank-minimal exact tensor decomposition for 2x4 by 4x7 matrix multiplication."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    nm, mp, pn = n*m, m*p, p*n  # 8, 28, 14
    
    # Build the target multiplication tensor
    # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T_tensor = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_tensor[a, b, c] = 1.0
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        R = U.shape[1]
        T_approx = np.zeros((nm, mp, pn), dtype=np.float64)
        for r_idx in range(R):
            T_approx += np.outer(np.outer(U[:, r_idx], V[:, r_idx]), W[:, r_idx])
        return np.allclose(T_tensor, T_approx, atol=1e-9)
    
    def verify_fast(U, V, W):
        """Faster verification using tensor contraction."""
        R = U.shape[1]
        # T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Reshape: T_approx.reshape(nm, mp*p*n) ... no, let's use einsum
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_tensor, T_approx, atol=1e-9)
    
    # ========================================================================
    # STRATIFIED CONSTRUCTIVE APPROACH
    # ========================================================================
    # Key idea: Exploit the structure of 2x4 * 4x7.
    # 
    # The output C[2,7] = A[2,4] * B[4,7]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Strategy: Use Strassen for 2x2 blocks where possible, and direct 
    # rank-1 terms for the remaining 1x2 blocks.
    #
    # For the 4x7 matrix B, we can group columns into blocks:
    # - 3 groups of 2 columns (using Strassen for 2x2 output blocks)
    # - 1 group of 1 column (direct)
    #
    # For each 2-column group, the output is C[:, 2g:2g+2] = A * B[:, 2g:2g+2]
    # This is a 2x4 * 4x2 = 2x2 matrix multiplication.
    #
    # A 2x2 * 2x2 can be done with Strassen in rank 7.
    # But here it's 2x4 * 4x2. Let me think more carefully.
    #
    # Actually, let's think of it differently. We have:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    #
    # For a pair of output columns (k1, k2), we need:
    # C[:, k1:k2] = A * B[:, k1:k2]  (2x4 * 4x2 = 2x2)
    #
    # The 2x4 * 4x2 multiplication can be decomposed.
    # Let A be viewed as [A1 | A2] where A1 = A[:, 0:2], A2 = A[:, 2:4]
    # Let B[:, k1:k2] be viewed as [B1; B2] where B1 = B[0:2, k1:k2], B2 = B[2:4, k1:k2]
    #
    # Then C[:, k1:k2] = A1 * B1 + A2 * B2  (2x2 * 2x2 + 2x2 * 2x2)
    #
    # Each 2x2 * 2x2 can be done with Strassen (rank 7), so total rank 14 per column pair.
    # But we can do better by sharing.
    #
    # Actually, let's try a different approach. Let's directly construct a decomposition
    # using the algebraic structure.
    #
    # The key insight: for 2x4 * 4x7, we can use the following decomposition:
    # Treat the 4 intermediate dimension as 2+2, and use bilinear forms.
    #
    # Let me try to find a constructive decomposition with rank < 56.
    #
    # Approach: Use the fact that n=2 is small.
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # We can write this as a sum of rank-1 terms in the tensor decomposition sense.
    # Each term contributes to exactly one (a,b,c) entry.
    #
    # Let's try a hybrid: use Strassen-like tricks for groups of 2 columns.
    
    # Actually, let me try a more systematic approach.
    # The minimum rank for matrix multiplication M_{2,4,7} is at least 
    # max(matricization ranks). Let me compute these.
    
    # Matricization 1: T as (nm x (mp*pn)) matrix
    T1 = T_tensor.reshape(nm, mp * pn)
    r1 = np.linalg.matrix_rank(T1, tol=1e-10)
    
    # Matricization 2: T as (mp x (nm*pn)) matrix
    T2 = T_tensor.transpose(1, 0, 2).reshape(mp, nm * pn)
    r2 = np.linalg.matrix_rank(T2, tol=1e-10)
    
    # Matricization 3: T as (pn x (nm*mp)) matrix
    T3 = T_tensor.transpose(2, 0, 1).reshape(pn, nm * mp)
    r3 = np.linalg.matrix_rank(T3, tol=1e-10)
    
    lower_bound = max(r1, r2, r3)
    
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s kill
    
    def time_ok():
        return (time.time() - start_time) < deadline
    
    # ========================================================================
    # Try constructive decomposition using block structure
    # ========================================================================
    # For 2x4 * 4x7, partition the 4 intermediate rows into two groups of 2.
    # A = [A0 | A1] where A0 = A[:, 0:2] (2x2), A1 = A[:, 2:4] (2x2)
    # B = [B0; B1] where B0 = B[0:2, :] (2x7), B1 = B[2:4, :] (2x7)
    # C = A0 * B0 + A1 * B1
    #
    # A0 * B0 is 2x2 * 2x7 = 2x7
    # A1 * B1 is 2x2 * 2x7 = 2x7
    #
    # For 2x2 * 2x7: partition 7 columns into 3 pairs + 1 single.
    # Each 2x2 * 2x2 can use Strassen (rank 7).
    # Each 2x2 * 2x1 needs rank 2 (direct).
    #
    # So A0 * B0: 3*7 + 2 = 23 terms
    # A1 * B1: 3*7 + 2 = 23 terms
    # Total: 46 terms
    #
    # But we can potentially share between A0*B0 and A1*B1.
    # Let's try the direct construction first.
    
    def strassen_2x2_2x2(U, V, W, rank_offset, a_idx, b_idx, c_idx):
        """
        Add Strassen decomposition for 2x2 * 2x2.
        a_idx: indices for A[0,0], A[0,1], A[1,0], A[1,1] in U
        b_idx: indices for B[0,0], B[0,1], B[1,0], B[1,1] in V
        c_idx: indices for C[0,0], C[0,1], C[1,0], C[1,1] in W
        
        Strassen for 2x2*2x2 -> 2x2:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B01-B11)
        M4 = A11*(B10-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        
        C00 = M1+M4-M5+M7
        C01 = M3+M5
        C10 = M2+M4
        C11 = M1-M2+M3+M6
        """
        # We need 7 rank-1 terms for Strassen
        # But Strassen gives 7 bilinear forms, each of which is a product of 
        # linear combinations. In the tensor decomposition, each bilinear form
        # M_i = (linear_comb of A) * (linear_comb of B) contributes to 
        # multiple C entries.
        #
        # Actually, in the tensor decomposition T = sum u_r ⊗ v_r ⊗ w_r,
        # each term is a single product. Strassen's 7 terms are:
        # M1 = (A00+A11) * (B00+B11) -> this is a product of a linear form in A
        #      and a linear form in B. In tensor form, this corresponds to:
        #      u vector has 1s at positions A00 and A11
        #      v vector has 1s at positions B00 and B11
        #      w vector: this M1 contributes to C00 (+1) and C11 (+1)
        #
        # So each Strassen term gives one rank-1 term in the decomposition.
        
        terms = [
            # (A_coeffs, B_coeffs, C_coeffs)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # M2
            ([1, 1, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),  # M3
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),  # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),  # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),  # M7
        ]
        
        for a_c, b_c, c_c in terms:
            for i, val in enumerate(a_c):
                if val != 0:
                    U[a_idx[i], rank_offset] += val
            for i, val in enumerate(b_c):
                if val != 0:
                    V[b_idx[i], rank_offset] += val
            for i, val in enumerate(c_c):
                if val != 0:
                    W[c_idx[i], rank_offset] += val
            rank_offset += 1
        
        return rank_offset
    
    def direct_2x2_2x1(U, V, W, rank_offset, a_idx, b_idx, c_idx):
        """
        Direct decomposition for 2x2 * 2x1 -> 2x1.
        C0 = A00*B0 + A01*B1
        C1 = A10*B0 + A11*B1
        4 terms.
        a_idx: A[0,0], A[0,1], A[1,0], A[1,1]
        b_idx: B[0], B[1]
        c_idx: C[0], C[1]
        """
        terms = [
            ([1, 0, 0, 0], [1, 0], [1, 0]),  # A00*B0 -> C0
            ([0, 1, 0, 0], [0, 1], [1, 0]),  # A01*B1 -> C0
            ([0, 0, 1, 0], [1, 0], [0, 1]),  # A10*B0 -> C1
            ([0, 0, 0, 1], [0, 1], [0, 1]),  # A11*B1 -> C1
        ]
        for a_c, b_c, c_c in terms:
            for i, val in enumerate(a_c):
                if val != 0:
                    U[a_idx[i], rank_offset] += val
            for i, val in enumerate(b_c):
                if val != 0:
                    V[b_idx[i], rank_offset] += val
            for i, val in enumerate(c_c):
                if val != 0:
                    W[c_idx[i], rank_offset] += val
            rank_offset += 1
        return rank_offset
    
    def direct_2x2_2x2(U, V, W, rank_offset, a_idx, b_idx, c_idx):
        """
        Direct decomposition for 2x2 * 2x2 -> 2x2.
        8 terms (one for each (i,j,k) combination).
        """
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[a_idx[i*2+j], rank_offset] = 1.0
                    V[b_idx[j*2+k], rank_offset] = 1.0
                    W[c_idx[i*2+k], rank_offset] = 1.0
                    rank_offset += 1
        return rank_offset
    
    # Build the constructive decomposition
    U = np.zeros((nm, 60), dtype=np.float64)  # max possible rank
    V = np.zeros((mp, 60), dtype=np.float64)
    W = np.zeros((pn, 60), dtype=np.float64)
    rank_offset = 0
    
    # A indices: A[i,j] -> index i*m+j = i*4+j
    # B indices: B[j,k] -> index j*p+k = j*7+k
    # C indices: C[i,k] -> index k*n+i = k*2+i
    
    # Group 1: j=0,1 (A0, B0)
    # A0: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # B0: B[0,k]=k, B[1,k]=7+k for k=0..6
    a0_idx = [0, 1, 4, 5]
    
    for col_pair in range(3):
        # Columns 2*col_pair, 2*col_pair+1
        k1 = 2 * col_pair
        k2 = k1 + 1
        # B0 sub-block: B[0,k1], B[0,k2], B[1,k1], B[1,k2]
        b_idx = [k1, k2, 7 + k1, 7 + k2]
        # C sub-block: C[0,k1], C[0,k2], C[1,k1], C[1,k2]
        c_idx = [k1*2+0, k1*2+1, k2*2+0, k2*2+1]
        # Wait, C[i,k] index is k*n+i = k*2+i
        c_idx = [k1*2+0, k2*2+0, k1*2+1, k2*2+1]
        # C[0,k1]=k1*2+0, C[0,k2]=k2*2+0, C[1,k1]=k1*2+1, C[1,k2]=k2*2+1
        # But the order in c_idx should be C[0,0], C[0,1], C[1,0], C[1,1] relative
        # i.e., C[i_row, k_col] in the 2x2 sub-block
        # c_idx[0] = C[0,k1], c_idx[1] = C[0,k2], c_idx[2] = C[1,k1], c_idx[3] = C[1,k2]
        c_idx = [k1*2+0, k2*2+0, k1*2+1, k2*2+1]
        
        if not time_ok():
            break
        rank_offset = strassen_2x2_2x2(U, V, W, rank_offset, a0_idx, b_idx, c_idx)
    
    # Column 6 (single)
    k1 = 6
    b_idx_single = [k1, 7 + k1]
    c_idx_single = [k1*2+0, k1*2+1]
    # 2x2 * 2x1 direct
    if not time_ok():
        pass
    else:
        rank_offset = direct_2x2_2x1(U, V, W, rank_offset, a0_idx, b_idx_single, c_idx_single)
    
    # Group 2: j=2,3 (A1, B1)
    a1_idx = [2, 3, 6, 7]
    # B1: B[2,k]=14+k, B[3,k]=21+k for k=0..6
    
    for col_pair in range(3):
        k1 = 2 * col_pair
        k2 = k1 + 1
        b_idx = [14 + k1, 14 + k2, 21 + k1, 21 + k2]
        c_idx = [k1*2+0, k2*2+0, k1*2+1, k2*2+1]
        
        if not time_ok():
            break
        rank_offset = strassen_2x2_2x2(U, V, W, rank_offset, a1_idx, b_idx, c_idx)
    
    # Column 6 (single)
    k1 = 6
    b_idx_single = [14 + k1, 21 + k1]
    c_idx_single = [k1*2+0, k1*2+1]
    if not time_ok():
        pass
    else:
        rank_offset = direct_2x2_2x1(U, V, W, rank_offset, a1_idx, b_idx_single, c_idx_single)
    
    # Trim to actual rank used
    U = U[:, :rank_offset]
    V = V[:, :rank_offset]
    W = W[:, :rank_offset]
    
    # Verify the constructive decomposition
    if verify_fast(U, V, W):
        # Check if entries are integer/half-integer
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        if np.allclose(U, U_half, atol=1e-9) and np.allclose(V, V_half, atol=1e-9) and np.allclose(W, W_half, atol=1e-9):
            if verify_fast(U_half, V_half, W_half):
                return {
                    "U": U_half.tolist(),
                    "V": V_half.tolist(),
                    "W": W_half.tolist()
                }
        # Return as-is
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # If constructive failed, try ALS
    def try_als(rank, n_restarts=10, max_iters=100, seed=42):
        """Try ALS to find a rank-r decomposition."""
        for restart in range(n_restarts):
            if not time_ok():
                return None
            
            rng = np.random.RandomState(seed + restart)
            U = rng.randn(nm, rank) * 0.1
            V = rng.randn(mp, rank) * 0.1
            W = rng.randn(pn, rank) * 0.1
            
            for it in range(max_iters):
                if not time_ok():
                    return None
                
                # Update U
                B_u = np.zeros((mp * pn, rank), dtype=np.float64)
                for r_idx in range(rank):
                    B_u[:, r_idx] = np.kron(V[:, r_idx], W[:, r_idx])
                T_u = T_tensor.reshape(nm, mp * pn)
                try:
                    U_new = T_u @ np.linalg.pinv(B_u.T)
                except np.linalg.LinAlgError:
                    break
                
                # Update V
                B_v = np.zeros((nm * pn, rank), dtype=np.float64)
                for r_idx in range(rank):
                    B_v[:, r_idx] = np.kron(U[:, r_idx], W[:, r_idx])
                T_v = T_tensor.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_new = T_v @ np.linalg.pinv(B_v.T)
                except np.linalg.LinAlgError:
                    break
                
                # Update W
                B_w = np.zeros((nm * mp, rank), dtype=np.float64)
                for r_idx in range(rank):
                    B_w[:, r_idx] = np.kron(U[:, r_idx], V[:, r_idx])
                T_w = T_tensor.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_new = T_w @ np.linalg.pinv(B_w.T)
                except np.linalg.LinAlgError:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                if it % 10 == 0 or it == max_iters - 1:
                    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                    error = np.linalg.norm(T_tensor - T_approx) / max(np.linalg.norm(T_tensor), 1e-12)
                    if error < 1e-10:
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if np.allclose(U, U_r, atol=1e-6) and np.allclose(V, V_r, atol=1e-6) and np.allclose(W, W_r, atol=1e-6):
                            if verify_fast(U_r, V_r, W_r):
                                return (U_r, V_r, W_r)
                        if error < 1e-12:
                            return (U, V, W)
        
        return None
    
    # Try ALS from lower bound
    best_result = None
    for rank in range(lower_bound, min(lower_bound + 30, 60)):
        if not time_ok():
            break
        result = try_als(rank, n_restarts=5, max_iters=60)
        if result is not None:
            U, V, W = result
            if verify_fast(U, V, W):
                best_result = (rank, U, V, W)
                break
    
    if best_result is not None:
        rank, U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook decomposition with rank = n*m*p = 56
    rank = n * m * p
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
