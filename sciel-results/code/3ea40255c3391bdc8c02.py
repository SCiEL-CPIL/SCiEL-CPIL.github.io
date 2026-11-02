"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        """Verify tensor decomposition: sum_r U[i*m+j1,r]*V[j2*p+k,r]*W[k*n+i,r] = delta(j1,j2)"""
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    # Build rank-17 decomposition using Strassen for 2x2x2 core + schoolbook for cross terms
    # This is the known-good baseline
    rank_17 = 17
    U_17 = np.zeros((n * m, rank_17))
    V_17 = np.zeros((m * p, rank_17))
    W_17 = np.zeros((p * n, rank_17))
    
    # Strassen products for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # M1: (A00+A11)(B00+B11) -> C00, C11
    U_17[0, 0] = 1; U_17[4, 0] = 1
    V_17[0, 0] = 1; V_17[4, 0] = 1
    W_17[0, 0] = 1; W_17[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, C11
    U_17[3, 1] = 1; U_17[4, 1] = 1
    V_17[0, 1] = 1
    W_17[1, 1] = 1; W_17[3, 1] = -1
    
    # M3: A00(B01-B11) -> C01, C11
    U_17[0, 2] = 1
    V_17[1, 2] = 1; V_17[4, 2] = -1
    W_17[2, 2] = 1; W_17[3, 2] = 1
    
    # M4: A11(B10-B00) -> C00, C10
    U_17[4, 3] = 1
    V_17[3, 3] = 1; V_17[0, 3] = -1
    W_17[0, 3] = 1; W_17[1, 3] = 1
    
    # M5: (A00+A01)B11 -> C00, C01
    U_17[0, 4] = 1; U_17[1, 4] = 1
    V_17[4, 4] = 1
    W_17[0, 4] = -1; W_17[2, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C00, C10
    U_17[3, 5] = 1; U_17[0, 5] = -1
    V_17[0, 5] = 1; V_17[1, 5] = 1
    W_17[0, 5] = 1; W_17[1, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00, C11
    U_17[1, 6] = 1; U_17[4, 6] = -1
    V_17[3, 6] = 1; V_17[4, 6] = 1
    W_17[0, 6] = 1; W_17[3, 6] = 1
    
    # Cross terms: A[:, 2] * B[2, :] -> C[:, 0:2]
    # r=7: A02 * B20 -> C00
    U_17[2, 7] = 1
    V_17[6, 7] = 1
    W_17[0, 7] = 1
    
    # r=8: A02 * B21 -> C01
    U_17[2, 8] = 1
    V_17[7, 8] = 1
    W_17[2, 8] = 1
    
    # r=9: A12 * B20 -> C10
    U_17[5, 9] = 1
    V_17[6, 9] = 1
    W_17[1, 9] = 1
    
    # r=10: A12 * B21 -> C11
    U_17[5, 10] = 1
    V_17[7, 10] = 1
    W_17[3, 10] = 1
    
    # Cross terms: A[:, 0:3] * B[:, 2] -> C[:, 2]
    # r=11: A00 * B02 -> C02
    U_17[0, 11] = 1
    V_17[2, 11] = 1
    W_17[4, 11] = 1
    
    # r=12: A01 * B12 -> C02
    U_17[1, 12] = 1
    V_17[5, 12] = 1
    W_17[4, 12] = 1
    
    # r=13: A02 * B22 -> C02
    U_17[2, 13] = 1
    V_17[8, 13] = 1
    W_17[4, 13] = 1
    
    # r=14: A10 * B02 -> C12
    U_17[3, 14] = 1
    V_17[2, 14] = 1
    W_17[5, 14] = 1
    
    # r=15: A11 * B12 -> C12
    U_17[4, 15] = 1
    V_17[5, 15] = 1
    W_17[5, 15] = 1
    
    # r=16: A12 * B22 -> C12
    U_17[5, 16] = 1
    V_17[8, 16] = 1
    W_17[5, 16] = 1
    
    # Verify rank-17
    if not _verify(U_17, V_17, W_17, n, m, p, rank_17):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_fb = np.zeros((n * m, rank_sb))
        V_fb = np.zeros((m * p, rank_sb))
        W_fb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Now try to find a rank-16 decomposition by looking for linear dependencies
    # Strategy: try to express one of the 17 rank-1 tensors as a combination of the others
    # by solving for coefficients in the tensor space
    
    # The tensor space is R^{n*m} x R^{m*p} x R^{p*n} but we're looking for
    # linear dependencies among the 17 decomposable tensors T_r = U_r ⊗ V_r ⊗ W_r
    
    # Flatten each rank-1 tensor into a vector in R^{(n*m)*(m*p)*(p*n)} = R^{54}
    # Actually, the tensor has shape (n*m, m*p, p*n) = (6, 9, 6) = 324 dimensions
    # But due to the structure, we can work with the verification equations directly
    
    # Alternative: try specific merges. For each pair of products, check if their
    # difference can be absorbed into other products.
    
    # Try: can we merge M6 and M7 into a single product?
    # M6 = (A10-A00)(B00+B01) contributes to C00, C10
    # M7 = (A01-A11)(B10+B11) contributes to C00, C11
    # If we could find a single bilinear form that reproduces both contributions...
    
    # Actually, let's try a different approach: use the fact that in the full 2x3x3 problem,
    # the Strassen identities might allow cancellation with cross terms.
    
    # Try rank 16 by replacing M6 and M7 with a single product
    # and adjusting cross terms
    if time.time() < deadline - 30:
        # Attempt 1: Merge M7 into cross terms
        # M7 = (A01-A11)(B10+B11) -> C00 += 1, C11 += 1
        # Can we express this using existing cross-term products?
        # Cross terms involve A[:,2] and B[:,2], so no direct overlap.
        # This merge likely won't work.
        
        # Attempt 2: Use a different Strassen variant
        # Try the "Laderman" style or other 6-product schemes for embedded 2x2
        
        # Let's try: replace M6=(A10-A00)(B00+B01) and M7=(A01-A11)(B10+B11)
        # with a single product M_new that has the right contributions
        # M6 contributes: C00 += (A10-A00)(B00+B01), C10 += (A10-A00)(B00+B01)
        # M7 contributes: C00 += (A01-A11)(B10+B11), C11 += (A01-A11)(B10+B11)
        
        # For a single product to replace both, we'd need:
        # U_new * V_new to give the right coefficients for all 4 output entries
        # This is generally not possible with a single rank-1 tensor.
        
        # Attempt 3: Look for a linear dependency among all 17 tensors
        # Build the 324-dimensional representation and check rank
        # Tensor T_r has entries T_r[i,j,k] = U_r[i,j] * V_r[j,k] * W_r[k,i]
        # where i in 0..1, j in 0..2, k in 0..2
        
        # Actually, the structure tensor for matrix mult 2x3x3 has a specific form.
        # Let's build the matrix of all 17 rank-1 tensors flattened and check if rank < 17
        
        # Flatten each tensor into a vector of length n*m*p*p*n... no.
        # The tensor being decomposed is in R^{(n*m) x (m*p) x (p*n)} but with constraints.
        # Actually it's simpler: the tensor M has M[a,b,c] where a=(i,j), b=(j',k), c=(k',i')
        # and M[a,b,c] = 1 if j=j' and k=k' and i=i' else 0.
        # So M has exactly n*p = 6 non-zero entries (one for each (i,k) with j=j').
        # Wait no: for each i, k, and j, M[(i,j),(j,k),(k,i)] = 1. So there are n*m*p = 18 non-zero entries.
        
        # Let's index the full tensor space as R^{(n*m)*(m*p)*(p*n)} = R^{6*9*6} = R^{324}
        # Each rank-1 tensor T_r = u_r ⊗ v_r ⊗ w_r where u_r in R^6, v_r in R^9, w_r in R^6
        # Flatten: T_r_flat[i,j,k] = U_r[i] * V_r[j] * W_r[k] where i in 0..5, j in 0..8, k in 0..5
        
        # Build 324 x 17 matrix
        M_tensor = np.zeros((n*m * m*p * p*n, rank_17))
        for r in range(rank_17):
            u = U_17[:, r]
            v = V_17[:, r]
            w = W_17[:, r]
            # Outer product flattened
            outer = np.outer(np.outer(u, v), w).flatten()
            M_tensor[:, r] = outer
        
        # Check if the 17 columns are linearly dependent
        # Use SVD to find the rank
        try:
            U_svd, s_vals, Vt = np.linalg.svd(M_tensor, full_matrices=False)
            tensor_rank = np.sum(s_vals > 1e-10)
            if tensor_rank < rank_17:
                # There's a linear dependency! Find it
                # The null space of M_tensor gives the dependency
                # Vt has shape (rank_17, rank_17) and the last row(s) with small singular values span the null space
                # Find the null space vector
                if tensor_rank == rank_17 - 1:
                    # One dependency
                    null_vec = Vt[-1, :]  # last row of Vt
                    # null_vec[r] * T_r = 0 for all r
                    # We can eliminate one tensor: pick the one with largest |null_vec[r]|
                    elim_idx = np.argmax(np.abs(null_vec))
                    # New rank-16: remove tensor elim_idx and adjust others
                    # T_elim = -sum_{r != elim} null_vec[r]/null_vec[elim] * T_r
                    # But this doesn't directly give us a rank-16 decomposition because
                    # the sum of rank-1 tensors is not rank-1 in general.
                    # However, if the dependency means T_elim is in the span of the others,
                    # we can't simply remove it. We need a different approach.
                    pass
        except:
            pass
        
        # The SVD approach tells us if there's a linear dependency, but removing one tensor
        # from a dependent set doesn't give a valid decomposition of the same tensor.
        # We need the sum of all 17 tensors to equal the target, and if they're dependent,
        # the sum is still the same with or without the dependent one ONLY if the coefficient is 0.
        
        # Better approach: try to find an explicit rank-16 decomposition by parameter search.
        # Use the known rank-17 as a starting point and try to merge two products.
        
        # Try merging each pair and checking if the result is still valid
        best_result = None
        for r1 in range(rank_17):
            for r2 in range(r1 + 1, rank_17):
                if time.time() > deadline - 10:
                    break
                # Try to replace tensors r1 and r2 with a single tensor
                # The sum T_r1 + T_r2 should be expressible as a single rank-1 tensor
                # T_r1 + T_r2 = u1⊗v1⊗w1 + u2⊗v2⊗w2
                # This is rank-1 only if u1⊗v1⊗w1 and u2⊗v2⊗w2 are "parallel" in some sense
                # Generally, sum of two rank-1 tensors is rank-2, not rank-1.
                # So this approach won't work directly.
        
        # Alternative: try to find a different set of 16 products that works.
        # Use the structure of the problem: the 2x2x2 core needs at least 7 products
        # (Strassen is optimal for 2x2x2), and the cross terms need at least 10 products
        # (4 for A[:,2]*B[2,:] and 6 for A[:]*B[:,2]).
        # Total minimum: 7 + 10 = 17. So rank 16 might not be achievable with this block structure.
        
        # But maybe a non-block-structured decomposition can achieve rank 16?
        # Let's try a more systematic search: start with schoolbook (rank 18) and
        # try to find one linear dependency.
        
        # Build schoolbook rank-18
        U_18 = np.zeros((n * m, 18))
        V_18 = np.zeros((m * p, 18))
        W_18 = np.zeros((p * n, 18))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_18[i*m+j, col] = 1
                    V_18[j*p+k, col] = 1
                    W_18[k*n+i, col] = 1
                    col += 1
        
        # Check if the 18 schoolbook tensors are linearly dependent
        M_18 = np.zeros((n*m * m*p * p*n, 18))
        for r in range(18):
            u = U_18[:, r]
            v = V_18[:, r]
            w = W_18[:, r]
            M_18[:, r] = np.outer(np.outer(u, v), w).flatten()
        
        try:
            _, s18, Vt18 = np.linalg.svd(M_18, full_matrices=False)
            rank18 = np.sum(s18 > 1e-10)
            if rank18 < 18:
                # Schoolbook tensors are dependent! Find the dependency
                null_vec = Vt18[-1, :]
                # This means sum_r null_vec[r] * T_r = 0
                # We can eliminate one tensor
                elim_idx = np.argmax(np.abs(null_vec))
                # Remove tensor elim_idx and we have a valid rank-17 decomposition
                # (the sum of the remaining 17 still equals the target since the eliminated
                #  one has coefficient that makes the total sum unchanged)
                # Wait, that's not right. The sum of ALL 18 equals the target.
                # If sum_r null_vec[r] * T_r = 0, then the 18 tensors are dependent.
                # But the sum T_0 + T_1 + ... + T_17 = target.
                # Removing T_elim gives sum_{r != elim} T_r = target - T_elim, which is NOT the target.
                # So we can't just remove one.
                pass
        except:
            pass
        
        # The key insight: we need to find 16 rank-1 tensors whose SUM equals the target tensor.
        # This is a different problem from finding linear dependencies.
        
        # Let's just return the rank-17 result which we know is correct
        pass
    
    return {"U": U_17.tolist(), "V": V_17.tolist(), "W": W_17.tolist()}
# EVOLVE-BLOCK-END
