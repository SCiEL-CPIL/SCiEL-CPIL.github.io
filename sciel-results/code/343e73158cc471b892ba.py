"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import combinations

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    U_s, V_s, W_s = schoolbook()
    best_rank = U_s.shape[1]
    best_U, best_V, best_W = U_s, V_s, W_s

    deadline = time.time() + 140  # safety margin

    def verify(U, V, W, tol=1e-10):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(S, T, atol=tol)

    def round_and_verify(U, V, W):
        """Try rounding to integers or half-integers and verify."""
        # Try integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int):
            return U_int, V_int, W_int
        
        # Try half-integers
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        if verify(U_half, V_half, W_half):
            return U_half, V_half, W_half
            
        # Try quarters
        U_q = np.round(U * 4) / 4.0
        V_q = np.round(V * 4) / 4.0
        W_q = np.round(W * 4) / 4.0
        if verify(U_q, V_q, W_q):
            return U_q, V_q, W_q

        return None

    def laser_method(W_fixed, max_rank=None):
        """
        Laser method: given W's column space (pn x R matrix),
        solve for U and V such that T = U @ V^T @ W^T (in tensor form).
        
        For each column r of W, we need:
        sum_r U[:,r] outer V[:,r] outer W[:,r] = T
        
        This is equivalent to: for each (a,b) in (nm, mp),
        sum_r U[a,r] * V[b,r] * W[:,r] = T[a,b,:]
        
        So: U[a,:] @ (V[b,:] * W) = T[a,b,:]
        
        This means: for each a, U[a,:] must satisfy:
        U[a,:] @ M_a = T[a,:,:] where M_a = V * W (element-wise, shape mp x pn)
        
        Actually, let's think differently. Fix W (pn x R). Then we need to find
        U (nm x R) and V (mp x R) such that:
        T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        For fixed W, this is a bilinear problem in U and V.
        
        Alternative: Fix the column space of W. Let W be pn x R.
        Then for each a, we need: sum_r U[a,r] (V[:,r] * W[:,r]) = T[a,:,:]
        Let M_r = V[:,r] * W[:,r] (mp x pn outer product... no, element-wise)
        
        Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        For fixed a: T[a,:,:] = sum_r U[a,r] (V[:,r] outer W[:,r])
        
        So if we fix W and V, then U can be solved linearly.
        If we fix W, we need to find V and U such that:
        For each a: T[a,:,:] = sum_r U[a,r] (V[:,r] * W[:,r])  [element-wise product, then outer? No]
        
        Wait: V[:,r] is mp-vector, W[:,r] is pn-vector.
        V[:,r] * W[:,r] doesn't make sense dimensionally unless we mean outer product.
        
        T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        
        For fixed a and c: T[a,:,c] = sum_r U[a,r] W[c,r] V[:,r]
        
        So T[:, :, c] = U @ diag(W[:,c]) @ V^T  (nm x mp matrix)
        
        Yes! That's the key. For each c:
        T[:,:,c] = U @ diag(W[:,c]) @ V^T
        
        So if we fix W, we have pn equations:
        T[:,:,c] = U @ diag(W[:,c]) @ V^T  for c = 0,...,pn-1
        
        This is a system of bilinear equations in U and V.
        
        Alternative approach: Fix W's columns to be standard basis vectors.
        If W[:,c] = e_c (standard basis), then T[:,:,c] = U @ diag(e_c) @ V^T = u_c v_c^T
        where u_c = U[:,c] and v_c = V[:,c].
        
        But T[:,:,c] is generally not rank-1, so this won't work directly.
        
        Better: Use the fact that we can choose W's column space. Let's try small R
        and search over possible W matrices with entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}.
        """
        pass

    def try_fixed_W(W, R):
        """
        Given W (pn x R), find U (nm x R) and V (mp x R) such that
        T = sum_r U[:,r] outer V[:,r] outer W[:,r].
        
        For each c: T[:,:,c] = U @ diag(W[:,c]) @ V^T
        
        This is a bilinear system. We can use ALS or solve iteratively.
        But let's try a different approach: assume V has a specific structure.
        
        Actually, let's use the following: for fixed W, the problem becomes:
        Find U and V such that for all c: T[:,:,c] = U @ D_c @ V^T where D_c = diag(W[:,c])
        
        This is equivalent to: for all c, (T[:,:,c]) = U @ D_c @ V^T
        
        If W[:,c] has only one nonzero entry at position d_c, then D_c = e_{d_c} e_{d_c}^T,
        and T[:,:,c] = U[:,d_c] V[:,d_c]^T, so T[:,:,c] must be rank-1.
        
        More generally, if W[:,c] has support S_c, then T[:,:,c] must have rank <= |S_c|.
        
        Let's try: fix W to have columns that are standard basis vectors.
        Then each T[:,:,c] must be rank-1. Check if this is possible.
        """
        # For simplicity, try W with entries in {-1, 0, 1}
        # Use ALS to solve for U, V given W
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        
        for it in range(50):
            if time.time() > deadline:
                return None
            
            # Update U: for each a, T[a,:,:] = sum_r U[a,r] (V[:,r] * W[:,r])
            # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed a: T[a,:,:] = sum_r U[a,r] (V[:,r] outer W[:,r])
            # Let M_r = V[:,r] outer W[:,r] (mp x pn)
            # Then T[a,:,:] = sum_r U[a,r] M_r
            # So: T[a,:,:] = U[a,:] @ M where M = sum... no, M is a list of matrices
            
            # Reshape: T[a] is (mp, pn). We need U[a,:] such that:
            # sum_r U[a,r] * (V[:,r] outer W[:,r]) = T[a]
            # This is: U[a,:] @ [V[:,0]*W[:,0], V[:,1]*W[:,1], ...] = T[a] reshaped?
            
            # Better: vectorize. T_flat[a] = T[a,:,:].flatten() (mp*pn,)
            # M_r_flat = (V[:,r] * W[:,r]).flatten()? No, outer product.
            # M_r = np.outer(V[:,r], W[:,r]) (mp x pn)
            # M_r_flat = M_r.flatten() (mp*pn,)
            # T[a].flatten() = sum_r U[a,r] M_r_flat
            
            # So: T_flat = U @ M_flat where M_flat is (R, mp*pn) with M_flat[r] = M_r.flatten()
            # Wait, dimensions: U is (nm, R), M_flat should be (R, mp*pn)
            # T_flat is (nm, mp*pn)
            # T_flat = U @ M_flat^T
            
            # Compute M_flat
            M_flat = np.zeros((R, mp * pn))
            for r in range(R):
                M_flat[r] = np.outer(V[:, r], W[:, r]).flatten()
            
            # Solve U = T_flat @ M_flat^T @ (M_flat @ M_flat^T)^{-1}
            # Or use least squares
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_flat.T, T.reshape(nm, mp*pn).T, rcond=None)
                U = U_new.T
            except:
                pass
            
            # Update V similarly
            # T[b,:] = sum_r V[b,r] (U[:,r] outer W[:,r])
            # M_r = U[:,r] outer W[:,r] (nm x pn)
            # T[:,b,:] = sum_r V[b,r] M_r
            # T_flat2 = T.transpose(1,0,2).reshape(mp, nm*pn)
            # M_flat2[r] = M_r.flatten() (nm*pn,)
            # T_flat2 = V @ M_flat2^T
            
            M_flat2 = np.zeros((R, nm * pn))
            for r in range(R):
                M_flat2[r] = np.outer(U[:, r], W[:, r]).flatten()
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_flat2.T, T.transpose(1,0,2).reshape(mp, nm*pn).T, rcond=None)
                V = V_new.T
            except:
                pass
        
        # Verify
        if verify(U, V, W):
            result = round_and_verify(U, V, W)
            if result:
                return result
        
        return None

    # Strategy: Try small ranks with structured W matrices
    # W entries in {-1, 0, 1} or {-0.5, 0, 0.5, 1}
    
    def generate_W_candidates(R, entries):
        """Generate random W matrices with entries from given set."""
        candidates = []
        for _ in range(20):
            W = np.random.choice(entries, size=(pn, R))
            # Ensure W is not all zeros
            if np.any(W != 0):
                candidates.append(W)
        return candidates

    # Try ranks from lower bound upwards
    # Lower bound for matrix multiplication rank is max(nm, mp, pn) = 28
    lower_bound = max(nm, mp, pn)
    
    # Try a few ranks with structured W
    for target_rank in range(lower_bound, min(lower_bound + 8, best_rank)):
        if time.time() > deadline:
            break
        
        # Try W with entries {-1, 0, 1}
        for W in generate_W_candidates(target_rank, [-1, 0, 1]):
            if time.time() > deadline:
                break
            result = try_fixed_W(W, target_rank)
            if result is not None:
                best_rank = target_rank
                best_U, best_V, best_W = result
                break
        
        if time.time() > deadline:
            break
        
        # Try W with entries {-0.5, 0, 0.5, 1}
        for W in generate_W_candidates(target_rank, [-0.5, 0, 0.5, 1.0]):
            if time.time() > deadline:
                break
            result = try_fixed_W(W, target_rank)
            if result is not None:
                best_rank = target_rank
                best_U, best_V, best_W = result
                break
        
        if best_rank == target_rank:
            break

    # If we found something better, also try to improve with random W
    if best_rank < n * m * p:
        for target_rank in range(best_rank, min(best_rank + 3, n * m * p)):
            if time.time() > deadline:
                break
            for _ in range(10):
                if time.time() > deadline:
                    break
                W = np.random.randn(pn, target_rank)
                result = try_fixed_W(W, target_rank)
                if result is not None:
                    best_rank = target_rank
                    best_U, best_V, best_W = result
                    break

    def to_list(M):
        L = []
        for row in M:
            L.append([float(x) for x in row])
        return L
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
