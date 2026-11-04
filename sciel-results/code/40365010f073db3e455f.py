import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 4, 6
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for c_r in range(n):
        for b in range(m):
            for c_c in range(p):
                i = c_r * m + b
                k = b * p + c_c
                l = c_c * n + c_r
                target[i, k, l] = 1.0

    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        return np.allclose(R, target, atol=1e-9)

    def find_nullspace_dependence(U, V, W):
        """Find linear dependencies among rank-1 components using tensor flattening.
        
        The tensor T[i,k,l] = sum_c U[i,c]*V[k,c]*W[l,c].
        We can form a matrix M where each column c is the outer product u_c ⊗ v_c ⊗ w_c
        flattened. If these columns are linearly dependent, we can remove a component.
        
        More efficiently: fix W, then T[:, :, l] = sum_c W[l,c] * (u_c ⊗ v_c).
        For each l, we have a matrix equation. Alternatively, use the fact that
        if sum_c alpha_c * u_c ⊗ v_c ⊗ w_c = 0, then for the tensor to be preserved,
        we need a non-trivial null vector.
        
        We'll build the "structure matrix" by flattening the third mode and checking
        for linear dependence in the resulting 2D tensor space.
        """
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        
        # Build matrix: rows are (i,k) pairs, columns are l-indexed weighted sums
        # Actually, let's use a different approach: 
        # T[i,k,l] = sum_c U[i,c]*V[k,c]*W[l,c]
        # Reshape: T is (nm, mp, pn). 
        # For fixed l, T[:,:,l] = sum_c W[l,c] * outer(U[:,c], V[:,c])
        # So we have pn equations: for each l, sum_c W[l,c] * outer(U[:,c], V[:,c]) = T[:,:,l]
        # This is a linear system in the variables (U[:,c]*V[:,c]) but they're coupled.
        
        # Better approach: Build the "flattened tensor matrix" 
        # M where M[row, c] represents the contribution. 
        # Actually, the standard way: the rank-r decomposition gives T = sum_c u_c v_c^T w_c^T
        # Flattening along the third mode: T_3[l] (an nm x mp matrix) = sum_c W[l,c] * (U[:,c] V[:,c]^T)
        # So if we stack all T_3[l] into a vector, we get a linear system.
        
        # Let's build: for each l, T[:,:,l].reshape(-1) = sum_c W[l,c] * (U[:,c] * V[:,c]).reshape(-1)
        # This gives us: B @ x = y where B is (nm*mp) x rank, x[l] is... no.
        
        # Simpler: The columns of the "unfolding" matrix are linearly dependent iff 
        # there exists alpha != 0 such that sum_c alpha_c * (u_c ⊗ v_c ⊗ w_c) = 0.
        
        # Let's build a matrix where column c is vec(u_c ⊗ v_c ⊗ w_c) but that's too large.
        # Instead, use: if sum_c alpha_c u_c = 0 AND sum_c alpha_c v_c = 0 AND sum_c alpha_c w_c = 0, 
        # then the tensor contribution is zero. But that's sufficient not necessary.
        
        # The correct necessary and sufficient condition: 
        # The rank-1 tensors u_c ⊗ v_c ⊗ w_c are linearly dependent iff 
        # there exists alpha with sum_c alpha_c u_c ⊗ v_c ⊗ w_c = 0.
        
        # We can check this by building a matrix. Let's use the approach:
        # Fix the W matrix. Then T[:,:,l] = sum_c W[l,c] * (U[:,c] V[:,c]^T)
        # So: for each l, vec(T[:,:,l]) = sum_c W[l,c] * vec(U[:,c] V[:,c]^T)
        # Let A = [vec(U[:,0] V[:,0]^T), ..., vec(U[:,r-1] V[:,r-1]^T)]  (shape nm*mp x r)
        # Let Y = [vec(T[:,:,0]), ..., vec(T[:,:,pn-1])]  (shape nm*mp x pn)
        # Then: Y = A @ W^T
        # So: A @ W^T = Y. If W has a null space, and the corresponding combination of A's columns
        # is also in the null space of the mapping... 
        
        # Actually, the decomposition is exact iff A @ W^T = Y where Y is the target.
        # If there exists alpha such that W^T @ alpha = 0 AND A @ alpha = 0, then we can remove that component.
        # More precisely, if there's a null vector alpha of W^T, check if A @ alpha is in the null space 
        # of the "residual" mapping. Since we need A @ W^T = Y exactly, if W^T @ alpha = 0, then
        # the new W' = W - w_alpha * alpha^T would still satisfy W'^T = W^T - alpha * (alpha^T @ W^T) = W^T.
        # Wait, that's not right. Let me think again.
        
        # If we remove column c from U, V, W, the new tensor is T' = T - u_c ⊗ v_c ⊗ w_c.
        # We need T' = T, so we need u_c ⊗ v_c ⊗ w_c = 0, which means one of them is zero.
        # That's not useful.
        
        # The correct approach: if there's a linear dependence among the rank-1 components,
        # i.e., sum_c alpha_c * (u_c ⊗ v_c ⊗ w_c) = 0, then we can remove one component
        # and adjust the others. But this is complex.
        
        # Simpler practical approach: check if any column of U, V, or W is in the span of others
        # in a way that allows removal. Let's just try the greedy approach but with a smarter 
        # initial ordering based on column norms.
        
        return None  # Return None to indicate no simple nullspace found

    def greedy_prune_random(U, V, W, max_iter=200, seed=None):
        """Aggressive greedy pruning with random column ordering."""
        if seed is not None:
            random.seed(seed)
            
        current_U = np.array(U, dtype=float)
        current_V = np.array(V, dtype=float)
        current_W = np.array(W, dtype=float)
        current_rank = current_U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 0:
                break
            
            removed_any = False
            # Try removing each column in random order
            cols_to_try = list(range(current_rank))
            random.shuffle(cols_to_try)
            
            for col in cols_to_try:
                if time.time() > deadline - 5.0:
                    break
                
                # Create new arrays without this column
                mask = [i for i in range(current_rank) if i != col]
                U_new = current_U[:, mask]
                V_new = current_V[:, mask]
                W_new = current_W[:, mask]
                
                # Quick check using numpy
                R = np.einsum('ic,kc,lc->ikl', U_new, V_new, W_new)
                
                if np.allclose(R, target, atol=1e-9):
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank -= 1
                    removed_any = True
                    break  # Restart scanning with new random order
            
            if not removed_any:
                break
        
        # Convert back to lists for return
        return current_U.tolist(), current_V.tolist(), current_W.tolist(), current_rank

    def nullspace_prune(U, V, W, tol=1e-8):
        """Try to find and remove linearly dependent components via nullspace analysis.
        
        For the tensor T[i,k,l] = sum_c U[i,c]*V[k,c]*W[l,c], if there exists a non-zero
        vector alpha such that sum_c alpha_c * (U[:,c] ⊗ V[:,c] ⊗ W[:,c]) = 0, then
        the components are linearly dependent and we can reduce rank.
        
        We check this by forming the matrix A where A[i*k, c] = U[i,c]*V[k,c] and checking
        if W^T has a nullspace that also annihilates A in the appropriate sense.
        """
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        
        # Build matrix A of shape (n*m * m*p) x rank where A[i*m*p + k, c] = U[i,c] * V[k,c]
        # Then T[:,:,l] flattened = A @ W[l,:]
        # Actually: T[i,k,l] = sum_c U[i,c]*V[k,c]*W[l,c]
        # So for fixed (i,k): T[i,k,:] = (U[i,:] * V[k,:]) @ W^T
        # Let B[i,k,c] = U[i,c]*V[k,c]. Then T[i,k,l] = sum_c B[i,k,c]*W[l,c]
        # Reshape: T_flat[i*k + k_idx, l] = sum_c B_flat[i*k + k_idx, c] * W[l, c]
        # So T_flat = B_flat @ W^T where T_flat is (n*m*m*p) x (p*n)
        
        nm = n * m
        mp = m * p
        pn = p * n
        
        # B_flat has shape (nm * mp) x rank
        B_flat = np.zeros((nm * mp, rank), dtype=float)
        for i in range(nm):
            for k in range(mp):
                B_flat[i * mp + k, :] = Ua[i, :] * Va[k, :]
        
        # T_flat has shape (nm * mp) x pn
        T_flat = np.zeros((nm * mp, pn), dtype=float)
        for l in range(pn):
            T_flat[:, l] = target.reshape(nm * mp, pn)[:, l]
        
        # We need B_flat @ W^T = T_flat
        # If there's a null vector alpha of W^T (i.e., W^T @ alpha = 0), 
        # then we need B_flat @ alpha to be in the null space of the residual.
        # Actually, since B_flat @ W^T = T_flat exactly, if W^T @ alpha = 0,
        # then the column combination alpha of B_flat must satisfy:
        # (B_flat @ W^T - T_flat) + B_flat @ (W^T @ alpha) = 0 + B_flat @ 0 = 0
        # That doesn't help directly.
        
        # Better: if we can find alpha such that B_flat @ alpha = 0 AND W^T @ alpha = 0,
        # then removing that linear combination doesn't change the tensor.
        # But we want to remove a single column, not a combination.
        
        # Practical approach: check if any column of [B_flat; W^T] (stacked) is in the span
        # of the others. If column c is in the span, we can potentially remove it.
        
        # Let's just try: for each column c, check if removing it still gives exact decomposition.
        # This is what greedy_prune does, but let's do it with a smarter initial order.
        
        # Actually, let's just return the input and let greedy_prune handle it.
        return Ua.tolist(), Va.tolist(), Wa.tolist(), rank

    def get_schoolbook():
        # Rank n*m*p = 72
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        
        col = 0
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    U[i][col] = 1
                    V[k][col] = 1
                    W[l][col] = 1
                    col += 1
        return U, V, W

    def get_strassen_66():
        # Reconstruct the 66 rank solution from the seed
        rank = 66
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)

        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)
        
        return U, V, W

    best_U, best_V, best_W, best_rank = None, None, None, float('inf')
    
    # Generate candidates
    candidates = []
    
    # 1. The 66-rank solution
    U66, V66, W66 = get_strassen_66()
    candidates.append((U66, V66, W66))
    
    # 2. The 72-rank schoolbook solution
    U72, V72, W72 = get_schoolbook()
    candidates.append((U72, V72, W72))
    
    # Prune each candidate using random restarts
    num_restarts = 15
    
    for U, V, W in candidates:
        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            
            seed_val = restart * 12345 + 67890
            U_p, V_p, W_p, r_p = greedy_prune_random(U, V, W, max_iter=300, seed=seed_val)
            
            # Verify the pruned decomposition
            if verify(U_p, V_p, W_p):
                if r_p < best_rank:
                    best_rank = r_p
                    best_U, best_V, best_W = U_p, V_p, W_p
            else:
                # If pruning broke it, fall back to original if valid
                if verify(U, V, W):
                    r_orig = len(U[0])
                    if r_orig < best_rank:
                        best_rank = r_orig
                        best_U, best_V, best_W = U, V, W
    
    # If we didn't find anything better than 66, return 66
    if best_U is None:
        best_U, best_V, best_W = get_strassen_66()
        best_rank = 66

    return {"U": best_U, "V": best_V, "W": best_W}
