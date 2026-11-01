import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Primary goal: Find rank 20 by exploiting linear dependencies in the block-diagonal Strassen (rank 21) decomposition.
    Fallback: Return the standard rank 21 decomposition.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s/150s limit
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    # 1. Construct the base Rank 21 decomposition (Block-diagonal Strassen)
    rank_base = 21
    U_base = np.zeros((nm, rank_base))
    V_base = np.zeros((mp, rank_base))
    W_base = np.zeros((pn, rank_base))
    
    # Strassen coefficients for 2x2 multiplication
    # s=0..6
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Index mappings
    v_indices = []
    w_indices = []
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        # V: j*6 + k (j is local row of B, k is local col of B)
        v_indices.append([0 * 6 + k0, 0 * 6 + k1, 1 * 6 + k0, 1 * 6 + k1])
        # W: k*2 + i (k is local row of C, i is local col of C)
        w_indices.append([k0 * 2 + 0, k1 * 2 + 0, k0 * 2 + 1, k1 * 2 + 1])

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            u_list = step['u']
            v_list = step['v']
            w_list = step['w']
            v_idx_list = v_indices[g]
            w_idx_list = w_indices[g]
            
            for local_idx in range(4):
                if u_list[local_idx] != 0:
                    U_base[local_idx, r] = u_list[local_idx]
                if v_list[local_idx] != 0:
                    V_base[v_idx_list[local_idx], r] = v_list[local_idx]
                if w_list[local_idx] != 0:
                    W_base[w_idx_list[local_idx], r] = w_list[local_idx]

    # Verify base rank 21
    T_check = np.einsum('ar,br,cr->abc', U_base, V_base, W_base)
    T_exp = np.zeros((4, 12, 12))
    for i in range(2):
        for j in range(2):
            for k in range(6):
                a = i * 2 + j
                b = j * 6 + k
                c = k * 2 + i
                T_exp[a, b, c] = 1.0
                
    if not np.allclose(T_check, T_exp, atol=1e-10):
        # Fallback to schoolbook rank 24 if something is fundamentally wrong (shouldn't happen)
        rank_school = 24
        U_s = np.zeros((4, rank_school))
        V_s = np.zeros((12, rank_school))
        W_s = np.zeros((12, rank_school))
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(6):
                    a = i * 2 + j
                    b = j * 6 + k
                    c = k * 2 + i
                    U_s[a, col] = 1
                    V_s[b, col] = 1
                    W_s[c, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # 2. Attempt to find Rank 20
    # Strategy: The tensor is a direct sum of 3 rank-7 tensors.
    # A rank reduction implies a linear relation between the 21 rank-1 components.
    # We check if the vector space spanned by the "components" in the tensor product space
    # has dimension < 21. However, simple linearity check on U,V,W separately isn't enough.
    # We look for a non-trivial linear combination of the 21 terms that sums to zero tensor.
    # T = sum_r U_r \otimes V_r \otimes W_r.
    # If sum_r c_r (U_r \otimes V_r \otimes W_r) = 0, then we can eliminate one term.
    
    # This is equivalent to checking if the 21 vectors (flattened U_r \otimes V_r \otimes W_r) are linearly dependent.
    # However, this is a very high-dimensional space (4*12*12 = 576 dims).
    # A more efficient check for this specific structure (block diagonal) is to check if the 
    # "interaction" between blocks allows cancellation.
    
    # Given the time constraint and the high probability that rank 21 is minimal for <2,2,6>
    # (as <2,2,2> is 7 and blocks are disjoint in standard embedding), a full search is expensive.
    # We perform a limited randomized check for a linear dependency vector c.
    
    # If no dependency is found quickly, we return rank 21.
    # We use a small number of random projections to test for dependencies.
    
    rank_20_found = False
    c_vec = None
    removed_idx = None
    
    # Time check: if we already spent too much time constructing, just return base
    if time.time() > deadline - 5.0:
        return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}

    # Perform a few iterations of random linear dependency search
    # We construct the matrix M where columns are flattened (U_r \otimes V_r \otimes W_r)
    # This matrix is 576 x 21. We check its rank.
    
    flat_terms = []
    for r in range(rank_base):
        u_r = U_base[:, r]
        v_r = V_base[:, r]
        w_r = W_base[:, r]
        term = np.einsum('a,b,c->abc', u_r, v_r, w_r)
        flat_terms.append(term.flatten())
        
    M = np.column_stack(flat_terms)
    
    # Compute rank of M
    # If rank(M) < 21, there is a linear dependency.
    # We use SVD to find the left null space.
    
    try:
        U_svd, s, Vh = np.linalg.svd(M, full_matrices=False)
        # Rank is number of significant singular values
        tol = 1e-10 * s[0]
        rank_M = np.sum(s > tol)
        
        if rank_M < 21:
            # Find null space vectors
            # Vh is 21x21 (or 21 x rank_M)
            # The null space is spanned by rows of Vh corresponding to zero singular values.
            # Since s has length min(576, 21)=21, all singular values are in s.
            # Null space indices are where s < tol
            null_indices = np.where(s < tol)[0]
            if len(null_indices) > 0:
                # Take the first null vector
                c_vec = Vh[null_indices[0], :]
                # c_vec is a vector of length 21
                # We need to find a non-zero entry to remove
                # To maximize stability, pick the largest magnitude entry
                max_idx = np.argmax(np.abs(c_vec))
                removed_idx = max_idx
                
                # Construct the new rank 20 matrices
                # T = sum_{r != removed} (1/c_removed) * c_r * (U_r \otimes V_r \otimes W_r)  ??? 
                # No, the relation is sum_r c_r T_r = 0.
                # So T_removed = -1/c_removed * sum_{r != removed} c_r T_r.
                # This means the tensor T can be represented by the remaining 20 terms,
                # but with modified coefficients?
                # Wait. The decomposition is T = sum T_r.
                # If sum c_r T_r = 0, then T_{removed} = -1/c_removed sum_{r != removed} c_r T_r.
                # This implies the term T_removed is a linear combination of the others.
                # However, the rank-1 decomposition requires the terms to be rank-1 tensors.
                # The sum of rank-1 tensors is generally NOT rank-1.
                # Therefore, we CANNOT simply remove one term and keep the others as-is.
                # We would need to replace the removed term with a sum of 20 terms? No.
                # The existence of a linear dependency in the space of tensors 
                # sum c_r (u_r \otimes v_r \otimes w_r) = 0 
                # DOES NOT imply a lower rank decomposition exists.
                # It only implies that the specific 21 vectors are linearly dependent in the 576-dim space.
                # The rank of the tensor is the minimum number of rank-1 tensors.
                # Linear dependence of a specific set of 21 rank-1 tensors does NOT allow us to reduce the count,
                # because the replacement term would not be rank-1.
                
                # Therefore, this search is theoretically flawed for finding a lower rank,
                # UNLESS the dependency allows us to re-parameterize such that one term becomes zero
                # or can be merged. But generally, linear dependence of the vectors 
                # v_r = u_r \otimes v_r \otimes w_r in the ambient space is a necessary but not sufficient 
                # condition for rank < 21? No, it's just that the specific basis is not minimal.
                # It doesn't prove the tensor rank is < 21.
                
                # Conclusion: This search will likely find dependencies (since 21 vectors in 576D space 
                # are almost certainly independent? Wait. 21 vectors in 576D are generically independent.
                # So rank(M) should be 21.
                # If rank(M) < 21, it's a very special case.
                
                pass 
    except Exception:
        pass

    # Since the linear dependence check above doesn't directly yield a lower rank decomposition
    # (due to the non-rank-1 nature of the sum), and given the high probability that 21 is minimal,
    # we stick with the known optimal Rank 21.
    
    return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank 21 decomposition generated.")
