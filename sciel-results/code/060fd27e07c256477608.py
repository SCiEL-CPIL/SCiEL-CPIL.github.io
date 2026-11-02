import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    def verify_decomp(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)

    def build_rank32():
        """Build the known rank-32 decomposition."""
        terms = []
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
            terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
            terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
            terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
            terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
            terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
            terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
        
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
        
        rank = len(terms)
        U = np.zeros((4, rank))
        V = np.zeros((18, rank))
        W = np.zeros((18, rank))
        
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx, t] = val
            for idx, val in Vd.items():
                V[idx, t] = val
            for idx, val in Wd.items():
                W[idx, t] = val
        
        return U, V, W

    def try_pair_term_removal(U, V, W):
        """Try removing pairs of terms that are linearly dependent."""
        R = U.shape[1]
        n, m, p = 2, 2, 9
        
        # Build the full tensor
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        
        # For each pair (a, b), check if we can eliminate both terms
        # by adjusting the remaining terms
        # A pair can be removed if the two terms are "redundant"
        # i.e., their contributions can be absorbed by others
        
        # Strategy: for each pair (a,b), set U[:,a]=U[:,b]=0 and 
        # try to find corrections to remaining terms
        for a in range(R):
            if time.time() > deadline:
                return None
            for b in range(a+1, R):
                # Create modified system where terms a and b are zeroed
                # and try to solve for corrections to other terms
                U_mod = U.copy()
                V_mod = V.copy()
                W_mod = W.copy()
                U_mod[:, a] = 0
                U_mod[:, b] = 0
                V_mod[:, a] = 0
                V_mod[:, b] = 0
                W_mod[:, a] = 0
                W_mod[:, b] = 0
                
                # Compute current tensor with terms a,b zeroed
                U_r_mod = U_mod.reshape(n, m, R)
                V_r_mod = V_mod.reshape(m, p, R)
                W_r_mod = W_mod.reshape(p, n, R)
                T_mod = np.einsum('ikr,kjr,jir->ikj', U_r_mod, V_r_mod, W_r_mod)
                
                # Residual: what's missing
                residual = np.ones((n, m, p)) - T_mod
                
                # Check if residual can be expressed as combination of 
                # rank-1 tensors formed by pairs of columns from U,V,W
                # Actually, we need to check if the residual is in the span
                # of the "cross product" space of the remaining terms
                
                # Simpler check: is the residual zero? (unlikely for pair removal)
                # Better: try a small optimization to see if we can fix it
                # with small adjustments to remaining terms
                
                # For efficiency, only try if residual norm is small
                if np.linalg.norm(residual) > 0.5:
                    continue
                
                # Try least squares to find corrections
                # We have (R-2) terms to adjust
                remaining = [i for i in range(R) if i != a and i != b]
                R_rem = len(remaining)
                
                # Parameterize corrections as small deltas
                # U_corr[i, r] = delta_u[i, r, k] * U[i, r] + ... 
                # This is getting complex. Let's use a simpler approach:
                # Just check if the two terms are exactly linearly dependent
                # in the tensor product space
                
                # Term a contributes: U[:,a] ⊗ V[:,a] ⊗ W[:,a] (outer product structure)
                # Actually the contribution is more complex due to the einsum structure
                # The contribution of term r to T[i,k,j] is U[i,k,r]*V[k,j,r]*W[j,i,r]
                
                # For pair removal, we need:
                # sum_{r not in {a,b}} U[i,k,r]*V[k,j,r]*W[j,i,r] = 1 for all i,k,j
                # This is a system of 36 equations in (R-2)*3*... variables
                # Too complex for direct check. Use the residual norm as proxy.
                
                if np.linalg.norm(residual, ord=np.inf) < 1e-10:
                    # Perfect removal
                    U_new = U_mod[:, remaining]
                    V_new = V_mod[:, remaining]
                    W_new = W_mod[:, remaining]
                    if verify_decomp(U_new, V_new, W_new, n, m, p):
                        return U_new, V_new, W_new
        
        return None

    def try_single_term_removal(U, V, W):
        """Try removing single terms that are redundant."""
        R = U.shape[1]
        n, m, p = 2, 2, 9
        
        for a in range(R):
            if time.time() > deadline:
                return None
            
            U_mod = U.copy()
            V_mod = V.copy()
            W_mod = W.copy()
            U_mod[:, a] = 0
            V_mod[:, a] = 0
            W_mod[:, a] = 0
            
            U_r_mod = U_mod.reshape(n, m, R)
            V_r_mod = V_mod.reshape(m, p, R)
            W_r_mod = W_mod.reshape(p, n, R)
            T_mod = np.einsum('ikr,kjr,jir->ikj', U_r_mod, V_r_mod, W_r_mod)
            
            residual = np.ones((n, m, p)) - T_mod
            
            if np.linalg.norm(residual, ord=np.inf) < 1e-10:
                remaining = [i for i in range(R) if i != a]
                U_new = U_mod[:, remaining]
                V_new = V_mod[:, remaining]
                W_new = W_mod[:, remaining]
                if verify_decomp(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
        
        return None

    def epigraph_local_search(U, V, W, max_iter=50):
        """
        Smooth epigraph-based local search to try to reduce rank.
        Uses soft constraints to allow some terms to have near-zero weight.
        """
        R = U.shape[1]
        n, m, p = 2, 2, 9
        U_r = U.reshape(n, m, R).copy()
        V_r = V.reshape(m, p, R).copy()
        W_r = W.reshape(p, n, R).copy()
        
        # Try to find a direction that makes some term vanish
        # while maintaining the tensor equation
        # Use a penalty method: minimize sum of squares of T[i,k,j]-1
        # with a penalty on the norm of term r
        
        best_result = None
        best_score = float('inf')
        
        for target_term in range(R):
            if time.time() > deadline:
                break
            
            # Try to make term target_term vanish
            # Parameterize: scale term target_term by s, adjust others
            # Use least squares to find optimal adjustment
            
            # Create design: for each remaining term r, we can scale it
            # and also add small corrections
            # Simplified: just scale all terms and see if we can drop one
            
            # Actually, use a more direct approach:
            # Set up a linear system where we want to find coefficients
            # c_r such that sum_r c_r * T_r = 1 (the target tensor)
            # where T_r is the tensor contribution of term r
            # If we can find c with some c_r = 0, we've reduced rank
            
            # Build the linear system
            # Variables: c_r for r in 0..R-1
            # Equations: for each (i,k,j), sum_r c_r * U[i,k,r]*V[k,j,r]*W[j,i,r] = 1
            
            # This is a linear system! A * c = 1
            # where A[i,k,j, r] = U[i,k,r]*V[k,j,r]*W[j,i,r]
            
            A = np.zeros((n*m*p, R))
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        for r in range(R):
                            A[i*m*p + k*p + j, r] = U_r[i,k,r] * V_r[k,j,r] * W_r[j,i,r]
            
            # Solve A * c = ones
            # If the system has a solution with some c_r = 0, we can reduce rank
            # Use least squares to find the minimum-norm solution
            try:
                c, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n*m*p), rcond=None)
            except:
                continue
            
            # Check if any c_r is close to zero
            # Also check if the solution is valid
            T_check = np.einsum('ikr,kjr,jir,r->ikj', U_r, V_r, W_r, c)
            if not np.allclose(T_check, np.ones((n,m,p)), atol=1e-8):
                continue
            
            # Find the smallest |c_r|
            min_idx = np.argmin(np.abs(c))
            min_val = np.abs(c[min_idx])
            
            if min_val < 1e-6:
                # We can potentially remove this term
                remaining = [i for i in range(R) if i != min_idx]
                # Rescale remaining terms by c_r / 1 (since c_r * T_r = 1)
                # Actually, we need: sum_{r remaining} c_r * T_r = 1
                # So the new decomposition uses c_r as scaling
                U_new = U_r[:, :, remaining].copy()
                V_new = V_r[:, :, remaining].copy()
                W_new = W_r[:, :, remaining].copy()
                
                # Scale by c_r
                c_rem = c[remaining]
                # Distribute scaling: U_new * c^{1/3}, V_new * c^{1/3}, W_new * c^{1/3}
                # Or just put all scaling in one factor
                scale = c_rem
                U_new *= scale[None, None, :]
                # Verify
                U_new_flat = U_new.reshape(n*m, len(remaining))
                V_new_flat = V_new.reshape(m*p, len(remaining))
                W_new_flat = W_new.reshape(p*n, len(remaining))
                
                if verify_decomp(U_new_flat, V_new_flat, W_new_flat, n, m, p):
                    if len(remaining) < best_score:
                        best_score = len(remaining)
                        best_result = (U_new_flat, V_new_flat, W_new_flat)
            
            # Also try: can we find a solution where TWO terms have c_r = 0?
            # This requires the system to be underdetermined in a specific way
            # For now, just do single removal
            
            if best_result is not None and best_score <= R - 2:
                break
        
        if best_result is not None:
            return best_result
        return None

    # Build the rank-32 base
    U, V, W = build_rank32()
    current_rank = U.shape[1]
    
    # Try single term removal
    if time.time() < deadline - 5:
        result = try_single_term_removal(U, V, W)
        if result is not None:
            U, V, W = result
            current_rank = U.shape[1]
    
    # Try pair term removal
    if time.time() < deadline - 5:
        result = try_pair_term_removal(U, V, W)
        if result is not None:
            U, V, W = result
            current_rank = U.shape[1]
    
    # Try epigraph-based search
    if time.time() < deadline - 10:
        result = epigraph_local_search(U, V, W)
        if result is not None:
            U, V, W = result
            current_rank = U.shape[1]
    
    # Final verification
    if not verify_decomp(U, V, W, n, m, p):
        # Fall back to rank-32
        U, V, W = build_rank32()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
