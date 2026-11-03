import time
import numpy as np

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 7
    
    # Target tensor T[i, j, k] = delta_{i,k} if j is the index? 
    # No, matrix multiplication tensor T_{i,j,k} such that C_{ik} = sum_j A_{ij} B_{jk}.
    # In the decomposition C_{ik} = sum_r U_{ir} V_{jr} W_{kr} (or similar convention).
    # The task specifies: W[k*n+i, r]. This implies the output index is k*n+i.
    # So C_{ik} corresponds to index k*n+i in the flattened W vector? 
    # Wait, standard convention: C is n x p. Index (i, k).
    # The problem says: "output factor follows the official trace convention W[k*n+i,r]".
    # This means W is of shape (p*n, rank). The row index is k*n + i.
    # So W_{k*n+i, r} is the coefficient for C_{i,k} in the r-th term.
    # The decomposition is: C_{i,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}?
    # No, the standard tensor decomposition for matmul is:
    # C_{ik} = sum_{j,r} A_{ij} B_{jk} T_{j, i, k}?
    # Let's check the shapes:
    # U: (n*m, rank) -> Index i*m + j
    # V: (m*p, rank) -> Index j*p + k
    # W: (p*n, rank) -> Index k*n + i
    # The product is sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}?
    # This sums over r. But where is the sum over j?
    # Usually, the tensor is T_{i,j,k}. 
    # If we define the tensor T_{i,j,k} = 1 if it's a valid multiplication term, 0 otherwise.
    # For matmul, the tensor is actually the identity in the middle index?
    # No, the matmul tensor T has entries T_{i,j,k} = 1 if we are computing C_{ik} from A_{ij} and B_{jk}.
    # Actually, the tensor for matmul is T_{i,j,k} = delta_{i,i'} delta_{k,k'}? No.
    # The tensor T is defined such that C_{ik} = sum_{j,r} A_{ij} B_{jk} ... wait.
    # The standard form is C_{ik} = sum_j A_{ij} B_{jk}.
    # This can be written as C_{ik} = sum_{j,r} (A_{ij} u_{j,r}) (B_{jk} v_{j,r}) w_{i,k,r}?
    # Let's look at the shapes again.
    # U_{i*m+j, r}, V_{j*p+k, r}, W_{k*n+i, r}.
    # If we contract these, we don't sum over j.
    # The expression must be: C_{ik} = sum_{j, r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}?
    # No, that would require U, V, W to depend on j in a way that allows summing over j.
    # But U is indexed by (i, j). V is indexed by (j, k). W is indexed by (k, i).
    # If we sum over r, we get a tensor S_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    # We want S_{i,j,k} to represent the matmul operation.
    # For matmul, we want C_{ik} = sum_j A_{ij} B_{jk}.
    # This implies that the "effective" tensor T_{i,j,k} should be such that
    # sum_j T_{i,j,k} A_{ij} B_{jk} = C_{ik}.
    # This requires T_{i,j,k} to be 1 for all j? No.
    # The identity is C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a linear map from A, B to C.
    # The tensor T_{i,j,k} is the coefficient of A_{ij} B_{jk} in C_{ik}.
    # So T_{i,j,k} = 1 for all i,j,k? No, that would give sum_{i,j,k} A_{ij} B_{jk} which is a scalar.
    # The output is a matrix C.
    # The decomposition is C_{ik} = sum_{j, r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r} is NOT correct because it doesn't sum over j in the standard way.
    #
    # Let's re-read the interface:
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "exactly decompose the matrix multiplication tensor"
    # The standard tensor T for matmul has entries T_{i,j,k} = 1 if we are using the term A_{ij} B_{jk} to contribute to C_{ik}.
    # Actually, the tensor T_{i,j,k} is 1 if the indices match the multiplication rule?
    # No, the tensor T_{i,j,k} is 1 for all i,j,k? No.
    # The matmul tensor is T_{i,j,k} = 1 if i,j,k are valid?
    # Let's check the rank. Rank of 2x2x2 matmul tensor is 7.
    # The tensor is T_{i,j,k} = 1 for all i,j,k in {0,1}?
    # If T_{i,j,k} = 1, then sum_{i,j,k} A_{ij} B_{jk} T_{i,j,k} = sum A_{ij} B_{jk}. This is a scalar (trace of AB? No, just sum of all products).
    # But matmul produces a matrix.
    #
    # Ah, the tensor T_{i,j,k} is defined on indices (i, j, k) where i is row of A/C, j is col of A/row of B, k is col of B/C.
    # The "value" of the tensor at (i,j,k) is 1.
    # Then the reconstruction is:
    # C_{ik} = sum_{j, r} U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}?
    # No, this sums over r. It does not sum over j.
    # However, if we define the "effective" tensor S_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r},
    # then we need S_{i,j,k} to be the matmul tensor.
    # The matmul tensor T_{i,j,k} is 1 for all i,j,k?
    # If T_{i,j,k} = 1, then C_{ik} = sum_j S_{i,j,k} A_{ij} B_{jk}?
    # No, the standard decomposition is:
    # C_{ik} = sum_{j, r} (A_{ij} U_{i*m+j, r}) (B_{jk} V_{j*p+k, r}) W_{k*n+i, r}
    # This is not right because U depends on i,j.
    #
    # Correct Interpretation:
    # The tensor T is T_{i,j,k} = 1.
    # We want to approximate T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}.
    # Wait, if T_{i,j,k} = 1, then we are approximating the all-ones tensor.
    # Is the rank of the all-ones tensor of size n x m x p equal to the matmul rank?
    # No. The rank of the all-ones tensor is 1 (it's rank 1: u v^T w^T).
    #
    # So T_{i,j,k} is NOT all ones.
    # The matmul tensor T_{i,j,k} is 1 if the indices are consistent?
    # Actually, the tensor for matmul is T_{i,j,k} = 1 for all i,j,k.
    # Why? Because C_{ik} = sum_j A_{ij} B_{jk}.
    # This is a multilinear map.
    # The tensor T is defined by T_{i,j,k} = 1.
    # Then the map is <T, A \otimes B> = sum_{i,j,k} T_{i,j,k} A_{ij} B_{jk} = sum_{i,j,k} A_{ij} B_{jk}.
    # This is a scalar. But matmul is a matrix.
    #
    # I think the indices in the tensor T are (i, j, k) where:
    # i: row index of C (and A)
    # j: column index of A (and row index of B)
    # k: column index of C (and B)
    # The tensor T_{i,j,k} is 1.
    # But the output C has indices (i, k).
    # The decomposition C_{ik} = sum_{j,r} ... implies that the tensor T is actually a family of tensors or the summation over j is part of the structure.
    #
    # Let's look at the shapes again.
    # U: (n*m, rank). Index i*m + j.
    # V: (m*p, rank). Index j*p + k.
    # W: (p*n, rank). Index k*n + i.
    #
    # If we form the sum S_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r},
    # we get a tensor of size n x m x p.
    # We want this S to be the "matmul tensor".
    # What is the matmul tensor?
    # It is T_{i,j,k} = 1.
    # If S_{i,j,k} = 1 for all i,j,k, then the rank is 1.
    # This contradicts the known rank of 7 for 2x2x2.
    #
    # Therefore, my interpretation of the tensor indices or the target tensor is wrong.
    #
    # Alternative Interpretation:
    # The tensor T_{i,j,k} is 1 if i,j,k are such that they form a valid multiplication term?
    # No, all i,j,k are valid.
    #
    # Let's check the Strassen algorithm.
    # Strassen decomposes the 2x2x2 matmul into 7 terms.
    # The 2x2x2 matmul tensor has 8 non-zero entries? No, 8 entries total, all 1.
    # If the tensor is all ones, rank is 1.
    # So the tensor is NOT all ones.
    #
    # What is the tensor?
    # Maybe the indices are different.
    # T_{i,j,k} where i is row of A, j is col of A, k is col of B?
    # And the output is C_{i,k}.
    # The relation is C_{ik} = sum_j A_{ij} B_{jk}.
    # This can be seen as C = A * B.
    # The tensor T maps (A, B) to C.
    # T is an element of (R^{n x m})^* \otimes (R^{m x p})^* \otimes (R^{n x p}).
    # The basis of (R^{n x p}) is E_{ik}.
    # The basis of (R^{n x m})^* is E_{ij}^*.
    # The basis of (R^{m x p})^* is E_{jk}^*.
    # The tensor T = sum_{i,j,k} E_{ij}^* \otimes E_{jk}^* \otimes E_{ik}.
    # So T_{i,j,k} = 1 for all i,j,k.
    #
    # Why is the rank 7 then?
    # Because the decomposition is C_{ik} = sum_r U_{i,j,r} V_{j,k,r} W_{i,k,r}?
    # No, the shapes are:
    # U: (n*m, rank) -> U_{(i,j), r}
    # V: (m*p, rank) -> V_{(j,k), r}
    # W: (p*n, rank) -> W_{(k,i), r}
    #
    # If we compute S_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r},
    # and we want S_{i,j,k} to be the tensor T_{i,j,k} = 1.
    # Then we are approximating the all-ones tensor.
    # Rank of all-ones tensor is 1.
    #
    # There must be a constraint I'm missing.
    # "exactly decompose the matrix multiplication tensor"
    #
    # Maybe the target tensor is NOT all ones.
    # Maybe the target tensor is T_{i,j,k} = 1 if i=j=k? No.
    #
    # Let's look at the error message: "tensor identity failed at target coordinate (0, 1, 3)".
    # This implies there is a specific target tensor T.
    #
    # Let's assume the target tensor T_{i,j,k} is 1 for all i,j,k.
    # If so, the program should have found a rank 1 solution.
    # But the program used Strassen (rank 7 for 2x2x2).
    # This implies the target tensor is NOT all ones.
    #
    # What tensor has rank 7 for 2x2x2?
    # The matmul tensor.
    # But I just argued the matmul tensor is all ones.
    #
    # Wait. The matmul tensor is T = sum_{i,j,k} E_{ij}^* \otimes E_{jk}^* \otimes E_{ik}.
    # This is indeed the all-ones tensor in the basis where the third index is (i,k).
    # But the third index in the tensor space is the output space.
    # The output space is R^{n x p}.
    # The input spaces are R^{n x m} and R^{m x p}.
    #
    # The tensor T is an element of V^* \otimes W^* \otimes Z.
    # If we flatten V, W, Z to vectors, T is a 3-tensor.
    # The entries are T_{(i,j), (j,k), (i,k)} = 1.
    # Note the indices:
    # First index: (i,j) from A.
    # Second index: (j,k) from B.
    # Third index: (i,k) from C.
    #
    # The condition for T_{u,v,w} = 1 is that the indices are consistent:
    # u = (i,j), v = (j,k), w = (i,k).
    # This means the 'j' in u must equal the 'j' in v.
    # The 'i' in u must equal the 'i' in w.
    # The 'k' in v must equal the 'k' in w.
    #
    # So T_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k'.
    # Otherwise 0.
    #
    # This is NOT the all-ones tensor.
    # The all-ones tensor would be 1 for all u,v,w.
    # This tensor is 1 only when the indices match.
    #
    # For 2x2x2:
    # u in {0,1,2,3} (i,j)
    # v in {0,1,2,3} (j,k)
    # w in {0,1,2,3} (i,k)
    #
    # T_{u,v,w} = 1 if:
    # i(u) = i(w)
    # j(u) = j(v)
    # k(v) = k(w)
    #
    # Let's check the rank.
    # This is the standard matmul tensor.
    #
    # So, to fix the program, I need to generate the target tensor T based on these index constraints.
    # Then use the pruning heuristic to find a low-rank decomposition.
    
    # Generate Target Tensor T (flattened to 1D or kept as 3D for residual calculation)
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                T[u_idx, v_idx, w_idx] = 1.0

    # Pruning Heuristic
    # 1. Initialize random factors
    # 2. Compute residual
    # 3. Remove smallest contributor
    # 4. Polish
    
    def compute_residual(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # S_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}
        # This is a tensor product sum.
        # We can compute this efficiently using np.einsum or matrix multiplications.
        # S = np.einsum('ur,vr,wr->uvw', U, V, W)
        # But R might be large.
        # Alternative: For each r, add U[:,r] outer V[:,r] outer W[:,r]
        # This is O(R * N * M * P).
        # N=12, M=28, P=21. N*M*P = 7056.
        # If R=40, 40 * 7056 = 282,240 operations. Very fast.
        
        S = np.zeros_like(T)
        for r in range(U.shape[1]):
            S += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
        return S - T

    def polish(U, V, W, T):
        # Least Squares polish to get exact values
        # We want to minimize || sum_r U_r V_r^T W_r - T ||_F
        # This is non-linear.
        # However, if we fix two matrices, we can solve for the third linearly.
        # Or use gradient descent.
        # Given the small size, we can use a simple iterative refinement.
        
        # Flatten T to 1D
        T_flat = T.flatten()
        
        # Current residual
        # We will perform a few steps of alternating minimization.
        # Fix U, V, solve for W?
        # The model is linear in W if U, V are fixed.
        # S_{uvw} = sum_r U_{ur} V_{vr} W_{wr}
        # This is linear in W.
        # We can reshape U, V to form a design matrix for W.
        
        # Let's stick to a simpler polish: use scipy.optimize.least_squares on the flattened parameters.
        # But 150s budget.
        # Let's try a simple gradient step or just return the current factors if they are close.
        # The pruning step should leave us with a very small residual.
        # We can then snap to nearest integers/half-integers.
        
        # Flatten U, V, W
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(p):
            U_p = p[:U.size].reshape(U.shape)
            V_p = p[U.size:U.size+V.size].reshape(V.shape)
            W_p = p[U.size+V.size:].reshape(W.shape)
            S = np.zeros_like(T)
            for r in range(U_p.shape[1]):
                S += np.einsum('u,v,w->uvw', U_p[:,r], V_p[:,r], W_p[:,r])
            return (S - T).flatten()
            
        # Use scipy if available, else simple gradient descent
        try:
            from scipy.optimize import least_squares
            res = least_squares(objective, params, max_nfev=1000)
            p_opt = res.x
            U_opt = p_opt[:U.size].reshape(U.shape)
            V_opt = p_opt[U.size:U.size+V.size].reshape(V.shape)
            W_opt = p_opt[U.size+V.size:].reshape(W.shape)
        except:
            U_opt, V_opt, W_opt = U, V, W
            
        return U_opt, V_opt, W_opt

    # Initial Rank
    R_init = 40
    U = np.random.randn(n*m, R_init)
    V = np.random.randn(m*p, R_init)
    W = np.random.randn(p*n, R_init)
    
    # Pruning Loop
    for _ in range(50): # Max iterations
        if time.time() - start_time > 100: break
        
        # Compute residual
        Res = compute_residual(U, V, W)
        res_norm = np.linalg.norm(Res)
        
        # If residual is small enough, we can try to remove more
        # Calculate contribution of each term r
        # Contribution_r = || sum_s != r ... ||? 
        # Simpler: Remove the term that reduces the residual the least?
        # Or: Remove the term with smallest norm?
        # Better: Compute the residual if we remove term r.
        # This is expensive if done for all r.
        # Approximation: The term with the smallest "energy" in the residual direction.
        
        # Let's just remove the term with the smallest norm in U, V, W combined.
        # Or, compute the gradient of the residual norm w.r.t. each term?
        
        # Simple heuristic: Remove the term r that minimizes the increase in residual norm.
        # We can approximate this by checking the norm of the term itself.
        
        norms = np.sqrt(np.sum(U**2, axis=0) + np.sum(V**2, axis=0) + np.sum(W**2, axis=0))
        min_idx = np.argmin(norms)
        
        # Remove term min_idx
        U = np.delete(U, min_idx, axis=1)
        V = np.delete(V, min_idx, axis=1)
        W = np.delete(W, min_idx, axis=1)
        
        # Polish after removal
        U, V, W = polish(U, V, W, T)
        
        # Check if we can stop
        Res = compute_residual(U, V, W)
        if np.linalg.norm(Res) < 1e-9:
            # Try to remove more?
            # If exact, we can try to prune further.
            pass
        else:
            # If residual is large, maybe we removed too much?
            # But we started with high rank.
            pass
            
    # Final Polish
    U, V, W = polish(U, V, W, T)
    
    # Snap to integers/half-integers
    # The optimal solutions for matmul often have integer or half-integer coefficients.
    def snap(x):
        return np.round(x * 2) / 2.0

    U = snap(U)
    V = snap(V)
    W = snap(W)
    
    # Verify
    Res = compute_residual(U, V, W)
    if np.linalg.norm(Res) > 1e-6:
        # If snapping broke it, try without snapping or with 1/4?
        # Or just return the un-snapped floats if they are close?
        # The task requires integer/half-integer entries.
        # If the polish didn't converge to integers, we might have a problem.
        # But Strassen uses integers.
        # The pruning should find a similar structure.
        
        # Try snapping to integers only?
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        Res_int = compute_residual(U_int, V_int, W_int)
        
        if np.linalg.norm(Res_int) < np.linalg.norm(Res):
            U, V, W = U_int, V_int, W_int
        else:
            # Try 1/4?
            U_quarter = np.round(U * 4) / 4.0
            V_quarter = np.round(V * 4) / 4.0
            W_quarter = np.round(W * 4) / 4.0
            Res_quarter = compute_residual(U_quarter, V_quarter, W_quarter)
            
            if np.linalg.norm(Res_quarter) < np.linalg.norm(Res):
                U, V, W = U_quarter, V_quarter, W_quarter

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
