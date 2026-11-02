"""Matrix multiplication tensor decomposition for 2x3x8."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    # Schoolbook rank = n*m*p = 48
    # Lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # The tensor T[i,j,k]=1 for all i,j,k has rank at least max(n*m, m*p, p*n)
    # For 2x3x8, we need to find the minimal rank.
    
    # Key structural insight: 
    # For fixed (i,k), we need: sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c]*x_j) = 1 for all x
    # This means: for each (i,k), the vector v_{i,k} = sum_c W[k*n+i,c] * u_i_c ⊗ v_k_c 
    # where u_i_c = (U[i*m,c], U[i*m+1,c], U[i*m+2,c]) and v_k_c = (V[0*p+k,c], V[1*p+k,c], V[2*p+k,c])
    # must satisfy: sum_j (v_{i,k})_j = 1 for all j... 
    # Actually more precisely: for each (i,k), and for each j:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let me define for each c:
    # a_c = (U[0,c], U[1,c], U[2,c], U[3,c], U[4,c], U[5,c])  -- 6 values
    # b_c = (V[0,c], ..., V[23,c])  -- 24 values  
    # d_c = (W[0,c], ..., W[15,c])  -- 16 values
    
    # Constraint for (i,j,k): a_c[i*m+j] * b_c[j*p+k] * d_c[k*n+i] summed over c = 1
    
    # This is a 3-tensor decomposition problem. The tensor is all-ones of shape (6,24,16)
    # but with the specific index mapping.
    
    # Actually, let me reconsider. The tensor M has indices (u_idx, v_idx, w_idx) where:
    # u_idx = i*m+j (0..5), v_idx = j*p+k (0..23), w_idx = k*n+i (0..15)
    # M[u_idx, v_idx, w_idx] = 1 if there exist i,j,k such that u_idx=i*m+j, v_idx=j*p+k, w_idx=k*n+i
    # Otherwise M = 0.
    
    # For n=2, m=3, p=8:
    # i in {0,1}, j in {0,1,2}, k in {0,...,7}
    # u_idx = i*3+j: i=0 -> 0,1,2; i=1 -> 3,4,5
    # v_idx = j*8+k: j=0 -> 0..7; j=1 -> 8..15; j=2 -> 16..23
    # w_idx = k*2+i: k=0 -> 0,1; k=1 -> 2,3; ... k=7 -> 14,15
    
    # For a given (i,j,k): u_idx = i*3+j, v_idx = j*8+k, w_idx = k*2+i
    # These are uniquely determined. So M[u_idx, v_idx, w_idx] = 1 for valid (i,j,k) triples.
    # But are there conflicts? Can two different (i,j,k) give the same (u_idx, v_idx, w_idx)?
    # u_idx determines i (first 3 = i=0, last 3 = i=1) and j.
    # v_idx determines j (first 8 = j=0, next 8 = j=1, last 8 = j=2) and k.
    # w_idx determines k (even = k/2*2, odd = k/2*2+1... wait w_idx = k*2+i)
    # So w_idx even -> i=0, w_idx odd -> i=1. And k = w_idx // 2.
    
    # Check consistency: from u_idx we get i, j. From v_idx we get j, k. From w_idx we get k, i.
    # So we need: i from u_idx == i from w_idx, j from u_idx == j from v_idx, k from v_idx == k from w_idx.
    # If these are consistent, M=1, else M=0.
    
    # Let me build the target tensor:
    M = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                M[u_idx, v_idx, w_idx] = 1.0
    
    # Now we need: sum_c U[:,c] * V[:,c] * W[:,c] (outer product) = M
    # i.e., the tensor M = sum_c u_c ⊗ v_c ⊗ w_c where u_c = U[:,c], etc.
    
    # This is a standard 3-tensor decomposition. The rank of M is what we want.
    
    # For small tensors, we can try to find the exact rank.
    # Let's try ranks from lower bound to schoolbook and use optimization.
    
    lower_bound = max(n*m, m*p, p*n)  # 24
    schoolbook = n * m * p  # 48
    
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook
    
    # First, let's try a structured approach.
    # For the all-ones-like tensor with this specific structure, 
    # maybe we can find a low-rank decomposition.
    
    # Let me try: can we express M as sum of rank-1 tensors?
    # A rank-1 tensor is u ⊗ v ⊗ w where u is (6,), v is (24,), w is (16,).
    # (u ⊗ v ⊗ w)[a,b,c] = u[a]*v[b]*w[c]
    
    # We need sum_r u_r[a] * v_r[b] * w_r[c] = M[a,b,c]
    
    # Let me try to solve this with a systematic approach.
    # Since the dimensions are small (6, 24, 16), and the tensor is sparse 
    # (only 48 non-zero entries out of 6*24*16 = 2304), maybe we can find
    # a compact decomposition.
    
    # Approach: use the fact that M[a,b,c] = 1 iff (a,b,c) corresponds to valid (i,j,k).
    # Let me enumerate the valid (a,b,c) tuples:
    valid_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_entries.append((a, b, c))
    
    # There are n*m*p = 48 valid entries.
    # We need sum_r u_r[a]*v_r[b]*w_r[c] = 1 for each (a,b,c) in valid_entries
    # and = 0 for all other (a,b,c).
    
    # This is a system of 2304 equations in 48*r unknowns (if we use r rank-1 terms).
    # But it's nonlinear.
    
    # Let me try a different approach: fix W and solve for U,V, or use ALS.
    
    # ALS: 
    # Given V and W, solve for U:
    # For each a: sum_r U[a,r] * (sum_b' v_r[b']*w_r[c']) ... no.
    # Actually: sum_r U[a,r] * V[b,r] * W[c,r] = M[a,b,c] for all b,c.
    # For fixed a: sum_r U[a,r] * (V[:,r] * W[:,r])[b,c] = M[a,b,c]
    # Let P_r = V[:,r] * W[:,r] (element-wise product, shape (24,16))
    # Then: sum_r U[a,r] * P_r[b,c] = M[a,b,c] for all b,c
    # This is: for each (b,c): sum_r U[a,r] * P_r[b,c] = M[a,b,c]
    # Vector form: U[a,:] @ (P_r[b,c] for all r) = M[a,b,c]
    # Or: U[a,:] is a vector of length r, and we have 24*16=384 equations.
    # Let Q_a be a (384, r) matrix where Q_a[(b,c), r] = V[b,r]*W[c,r]
    # Then U[a,:] @ Q_a = M[a,:,:].flatten()
    # This is a linear system: Q_a.T @ U[a,:].T = M[a,:,:].flatten()
    # If Q_a.T is (r, 384), we can solve least squares.
    
    # Similarly for V and W.
    
    def _als_solve_U(U, V, W, M, n, m, p, tol=1e-10):
        """Solve for U given V, W."""
        nm = n * m
        mp = m * p
        pn = p * n
        r = U.shape[1]
        
        U_new = np.zeros((nm, r))
        for a in range(nm):
            # Build Q: (mp*pn, r) where Q[(b,c), r] = V[b,r]*W[c,r]
            Q = np.zeros((mp * pn, r))
            for b in range(mp):
                for c in range(pn):
                    Q[b * pn + c, :] = V[b, :] * W[c, :]
            # Target: M[a, :, :].flatten()
            target = M[a, :, :].flatten()
            # Solve Q.T @ x = target, where x = U[a, :]
            # Q.T is (r, mp*pn)
            # Use least squares
            try:
                x, residuals, rank_q, sv = np.linalg.lstsq(Q.T, target, rcond=tol)
                U_new[a, :] = x
            except np.linalg.LinAlgError:
                U_new[a, :] = U[a, :]
        return U_new
    
    def _als_solve_V(U, V, W, M, n, m, p, tol=1e-10):
        """Solve for V given U, W."""
        nm = n * m
        mp = m * p
        pn = p * n
        r = V.shape[1]
        
        V_new = np.zeros((mp, r))
        for b in range(mp):
            Q = np.zeros((nm * pn, r))
            for a in range(nm):
                for c in range(pn):
                    Q[a * pn + c, :] = U[a, :] * W[c, :]
            target = M[:, b, :].flatten()
            try:
                x, residuals, rank_q, sv = np.linalg.lstsq(Q.T, target, rcond=tol)
                V_new[b, :] = x
            except np.linalg.LinAlgError:
                V_new[b, :] = V[b, :]
        return V_new
    
    def _als_solve_W(U, V, W, M, n, m, p, tol=1e-10):
        """Solve for W given U, V."""
        nm = n * m
        mp = m * p
        pn = p * n
        r = W.shape[1]
        
        W_new = np.zeros((pn, r))
        for c in range(pn):
            Q = np.zeros((nm * mp, r))
            for a in range(nm):
                for b in range(mp):
                    Q[a * mp + b, :] = U[a, :] * V[b, :]
            target = M[:, :, c].flatten()
            try:
                x, residuals, rank_q, sv = np.linalg.lstsq(Q.T, target, rcond=tol)
                W_new[c, :] = x
            except np.linalg.LinAlgError:
                W_new[c, :] = W[c, :]
        return W_new
    
    def _verify_decomp(U, V, W, M, tol=1e-8):
        """Check if U,V,W decompose M."""
        # Compute sum_c U[:,c] * V[:,c] * W[:,c]
        result = np.zeros(M.shape)
        for c in range(U.shape[1]):
            result += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        return np.allclose(result, M, atol=tol)
    
    # Try ALS for various ranks
    for target_rank in range(lower_bound, schoolbook + 1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(target_rank * 1000 + attempt)
            U = rng.randn(n*m, target_rank) * 0.1
            V = rng.randn(m*p, target_rank) * 0.1
            W = rng.randn(p*n, target_rank) * 0.1
            
            # Run ALS iterations
            for it in range(50):
                if time.time() > deadline:
                    break
                U = _als_solve_U(U, V, W, M, n, m, p)
                V = _als_solve_V(U, V, W, M, n, m, p)
                W = _als_solve_W(U, V, W, M, n, m, p)
                
                if _verify_decomp(U, V, W, M, tol=1e-6):
                    found = True
                    break
            
            if found:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if found:
            break
    
    # If we found a solution, return it; otherwise return schoolbook
    if best_U is None:
        # Fall back to schoolbook
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
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
