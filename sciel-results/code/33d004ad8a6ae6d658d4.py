"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # T[i, j, k, l] = 1 if i==k and j==l else 0
    # Output index: k*n + i (trace convention W[k*n+i, r])
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # Build target tensor as flat array
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Let's define indices:
    # U index: i*m + j  (i in 0..n-1, j in 0..m-1)
    # V index: j*p + k  (j in 0..m-1, k in 0..p-1)  
    # W index: k*n + i  (k in 0..p-1, i in 0..n-1)
    
    # Target: for each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[(i*m+j, j*p+k, k*n+i)] = 1.0
    
    def make_target_array():
        arr = np.zeros((n*m, m*p, p*n))
        for (a, b, c), val in target.items():
            arr[a, b, c] = val
        return arr
    
    T = make_target_array()
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose T. Returns True if exact."""
        r = U.shape[1]
        # Reconstruct: sum over c of U[:,c] outer V[:,c] outer W[:,c]
        # Check all target entries
        for (a, b, c), val in target.items():
            s = 0.0
            for idx in range(r):
                s += U[a, idx] * V[b, idx] * W[c, idx]
            if abs(s - val) > 1e-9:
                return False
        # Also check that non-target entries are zero
        # This is expensive; for now just check target entries
        # Actually we need to verify ALL entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    expected = target.get((a, b, c), 0.0)
                    s = 0.0
                    for idx in range(r):
                        s += U[a, idx] * V[b, idx] * W[c, idx]
                    if abs(s - expected) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using matrix operations."""
        r = U.shape[1]
        # For each pair (a,b), compute the vector over c: sum_c U[a,c]*V[b,c]*W[c,:]
        # This is (U[a,:] * V[b,:]) @ W
        # Build full tensor and compare
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # = (U[:,r] * V[:,r]) @ W[c,r] for each r
        # More efficiently: for each r, outer product
        T_recon = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        # Compare with target
        max_err = 0.0
        for (a, b, c), val in target.items():
            err = abs(T_recon[a, b, c] - val)
            if err > max_err:
                max_err = err
        # Check zeros
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    if (a, b, c) not in target:
                        err = abs(T_recon[a, b, c])
                        if err > max_err:
                            max_err = err
        return max_err < 1e-9, max_err
    
    def try_rank(r, num_restarts=50, time_limit=10.0):
        """Try to find a rank-r decomposition. Returns (U,V,W) or None."""
        start_time = time.time()
        best_err = float('inf')
        best_UVW = None
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Iterative refinement: multiplicative update style
            # For tensor T = sum_r U_r ⊗ V_r ⊗ W_r
            # Update U: U_new[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] / sum_{b,c} V[b,r]^2 * W[c,r]^2
            # This is like ALS for CP decomposition
            
            for it in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U
                # U_new[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r] / (sum_{b,c} V[b,r]^2 * W[c,r]^2 + eps)
                V_sq = V ** 2
                W_sq = W ** 2
                # denom[a, r] = sum_{b,c} V[b,r]^2 * W[c,r]^2 = sum_b V[b,r]^2 * sum_c W[c,r]^2
                denom = np.outer(np.sum(V_sq, axis=0), np.sum(W_sq, axis=0))
                # numerator[a, r] = sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                # = sum_b (sum_c T[a,b,c] * W[c,r]) * V[b,r]
                # Let's compute: for each a, r: sum_{b,c} T[a,b,c] * V[b,r] * W[c,r]
                # This is like: (T[a,:,:] * (V[:,r] * W[:,r])) summed over b,c
                # More efficiently: 
                # num_U[a, r] = sum_b V[b,r] * sum_c T[a,b,c] * W[c,r]
                # Let M_W[a, b, r] = sum_c T[a,b,c] * W[c,r]
                # Then num_U[a, r] = sum_b V[b,r] * M_W[a, b, r]
                
                # Compute M_W: shape (n*m, m*p, r)
                # M_W[a, b, r] = sum_c T[a,b,c] * W[c,r]
                M_W = np.einsum('abc,cr->abr', T, W)  # (n*m, m*p, r)
                num_U = np.einsum('br,abr->ar', V, M_W)  # (n*m, r)
                
                U_new = num_U / (denom + 1e-12)
                
                # Update V
                # num_V[b, r] = sum_{a,c} T[a,b,c] * U[a,r] * W[c,r]
                M_U = np.einsum('abc,ar->bcr', T, U_new)  # (m*p, p*n, r)
                num_V = np.einsum('cr,bcr->br', W, M_U)  # (m*p, r)
                
                # denom_V[b, r] = sum_{a,c} U[a,r]^2 * W[c,r]^2
                denom_V = np.outer(np.sum(U_new ** 2, axis=0), np.sum(W ** 2, axis=0))
                V_new = num_V / (denom_V + 1e-12)
                
                # Update W
                # num_W[c, r] = sum_{a,b} T[a,b,c] * U[a,r] * V[b,r]
                M_UV = np.einsum('abc,ar,br->cr', T, U_new, V_new)  # (p*n, r)
                num_W = M_UV
                # denom_W[c, r] = sum_{a,b} U[a,r]^2 * V[b,r]^2
                denom_W = np.outer(np.sum(U_new ** 2, axis=0), np.sum(V_new ** 2, axis=0))
                W_new = num_W / (denom_W + 1e-12)
                
                U, V, W = U_new, V_new, W_new
            
            # Check error
            ok, err = verify_fast(U, V, W)
            if err < best_err:
                best_err = err
                best_UVW = (U.copy(), V.copy(), W.copy())
            
            if ok:
                # Found exact decomposition! Try to round to integers/half-integers
                return _try_round(U, V, W)
        
        return best_UVW if best_err < 1e-6 else None
    
    def _try_round(U, V, W):
        """Try to round the decomposition to integers or half-integers."""
        # Scale to make entries closer to integers
        # Try different scalings
        best = None
        for scale in [1.0, 0.5, 2.0, 0.25, 4.0]:
            Us = U * scale
            Vs = V * scale
            Ws = W / (scale * scale)
            # Round
            Ur = np.round(Us * 2) / 2
            Vr = np.round(Vs * 2) / 2
            Wr = np.round(Ws * 2) / 2
            ok, err = verify_fast(Ur, Vr, Wr)
            if ok:
                return (Ur, Vr, Wr)
            if best is None or err < best[0]:
                best = (err, Ur, Vr, Wr)
        
        # Try integer rounding
        for scale in [1.0, 0.5, 2.0]:
            Us = U * scale
            Vs = V * scale
            Ws = W / (scale * scale)
            Ur = np.round(Us)
            Vr = np.round(Vs)
            Wr = np.round(Ws)
            ok, err = verify_fast(Ur, Vr, Wr)
            if ok:
                return (Ur, Vr, Wr)
        
        # If no exact rounding, return the real-valued solution (shouldn't happen if verify passed)
        return (U, V, W)
    
    # Try ranks from low to high
    # Known: schoolbook = 45
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But for matrix multiplication, better lower bounds exist
    # For 3x3x3, rank is 23 (Strassen-like). For 3x3x5, let's try from ~20 up
    
    deadline = time.time() + 130  # Leave 20s margin
    
    # First, let's try to see if we can do better than 45
    # Start from a reasonable lower bound
    # The tensor has 45 non-zero entries, so rank >= 15 (since each rank-1 tensor has at most 9*15*15=2025 non-zeros, but more carefully)
    # Actually, a rank-1 tensor U⊗V⊗W has non-zeros at positions where all three are non-zero
    # For our target, non-zeros are at (i*m+j, j*p+k, k*n+i) for i,j,k
    # A rank-1 term can cover at most min(n*m, m*p, p*n) = 9 positions if it's sparse
    
    # Let's try ranks from 20 to 45
    for r in range(20, 46):
        if time.time() > deadline:
            break
        
        result = try_rank(r, num_restarts=30, time_limit=min(8.0, max(1.0, deadline - time.time() - 5)))
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(U[i, j]) for j in range(r)] for i in range(n*m)]
            V_list = [[float(V[i, j]) for j in range(r)] for i in range(m*p)]
            W_list = [[float(W[i, j]) for j in range(r)] for i in range(p*n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
