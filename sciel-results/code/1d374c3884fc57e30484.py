"""Rewritten program: structured search for low-rank decomposition of 4x4*4x9 matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 by 4x9 matrix.
    
    The tensor T has indices (a, b, c) where:
      a = i*m + j  (i in [0,n), j in [0,m))  -> row of A, col of A
      b = j'*p + k (j' in [0,m), k in [0,p)) -> row of B, col of B  
      c = k'*n + i'(k' in [0,p), i' in [0,n)) -> col of C, row of C
    
    T[a,b,c] = 1 iff j == j' AND i == i'
    
    Decomposition: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    start_time = time.time()
    deadline = start_time + 105  # 105s deadline (15s safety margin for 120s limit)
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify(U, V, W):
        """Full verification of the decomposition."""
        rank = U.shape[1]
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        # Compute all tensor entries: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use einsum: result[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is a 3-tensor contraction
        # For memory efficiency, check in chunks
        for a in range(nm):
            for b in range(mp):
                # Compute vector over c: sum_r U[a,r]*V[b,r]*W[c,r]
                uv = U[a] * V[b]  # shape (rank,)
                vec = W.T @ uv    # shape (pn,)
                
                i = a // m
                j = a % m
                j2 = b // p
                
                for c in range(pn):
                    i2 = c % n
                    if j == j2:
                        target = 1.0 if i2 == i else 0.0
                    else:
                        target = 0.0
                    if abs(vec[c] - target) > 1e-8:
                        return False
        return True
    
    def compute_error(U, V, W):
        """Compute Frobenius error of the decomposition."""
        rank = U.shape[1]
        # For each (a,b), the vector over c is W.T @ (U[a]*V[b])
        # Target: T[a,b,c] = 1 if (a%m == b//p) and (a//m == c%n)
        
        total_err = 0.0
        for a in range(nm):
            i = a // m
            j = a % m
            for b in range(mp):
                j2 = b // p
                uv = U[a] * V[b]
                vec = W.T @ uv  # shape (pn,)
                
                for c in range(pn):
                    i2 = c % n
                    if j == j2:
                        target = 1.0 if i2 == i else 0.0
                    else:
                        target = 0.0
                    total_err += (vec[c] - target) ** 2
        return np.sqrt(total_err)
    
    def compute_error_fast(U, V, W):
        """Faster error computation using vectorization."""
        rank = U.shape[1]
        # Precompute: for each a, U[a] is (rank,)
        # For each (a,b): vec_c = W.T @ (U[a]*V[b])
        # We can vectorize over b for each a
        
        total_err = 0.0
        for a in range(nm):
            i = a // m
            j = a % m
            # For all b: U[a]*V[b] for each b
            # UV[b] = U[a] * V[b] -> shape (mp, rank)
            UV = V * U[a]  # broadcasting: (mp, 1) * (rank,) -> (mp, rank)
            # vecs = W.T @ UV -> shape (pn, mp)
            vecs = W.T @ UV  # (pn, rank) @ (rank, mp) -> (pn, mp)
            
            for b in range(mp):
                j2 = b // p
                vec = vecs[:, b]  # shape (pn,)
                for c in range(pn):
                    i2 = c % n
                    if j == j2:
                        target = 1.0 if i2 == i else 0.0
                    else:
                        target = 0.0
                    total_err += (vec[c] - target) ** 2
        return np.sqrt(total_err)
    
    def compute_error_vectorized(U, V, W):
        """Fully vectorized error computation."""
        rank = U.shape[1]
        
        # Build target tensor T as a 3D array
        T = np.zeros((nm, mp, pn))
        for a in range(nm):
            i = a // m
            j = a % m
            for b in range(mp):
                j2 = b // p
                if j == j2:
                    for c in range(pn):
                        i2 = c % n
                        if i2 == i:
                            T[a, b, c] = 1.0
        
        # Compute decomposition tensor: D[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # D = einsum('ar,br,cr->abc', U, V, W)
        # But this is nm*mp*pn*rank operations which is 16*36*36*rank
        # For rank ~100: 16*36*36*100 = 2,073,600 - manageable
        
        D = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sqrt(np.sum((D - T) ** 2))
        return err
    
    def solve_W_given_UV(U, V, T):
        """Solve for W given U, V, and target tensor T.
        For each c: sum_r W[c,r] * U[a,r] * V[b,r] = T[a,b,c] for all (a,b)
        This is a linear system: B @ W[c,:] = t_c
        where B[a,b,r] = U[a,r]*V[b,r] reshaped, and t_c[a,b] = T[a,b,c]
        """
        rank = U.shape[1]
        # B has shape (nm*mp, rank) with B[idx, r] = U[a,r]*V[b,r] where idx = a*mp+b
        B = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm * mp, rank)
        
        W = np.zeros((pn, rank))
        for c in range(pn):
            t_c = T[:, :, c].flatten()  # shape (nm*mp,)
            # Solve B @ W[c,:] = t_c (least squares)
            W[c] = np.linalg.lstsq(B, t_c, rcond=None)[0]
        return W
    
    def solve_U_given_VW(V, W, T):
        """Solve for U given V, W, and target tensor T.
        For each a: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all (b,c)
        This is: B @ U[a,:] = t_a where B[b,c,r] = V[b,r]*W[c,r]
        """
        rank = V.shape[1]
        # B has shape (mp*pn, rank) with B[idx, r] = V[b,r]*W[c,r] where idx = b*pn+c
        B = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(mp * pn, rank)
        
        U = np.zeros((nm, rank))
        for a in range(nm):
            t_a = T[a, :, :].flatten()  # shape (mp*pn,)
            U[a] = np.linalg.lstsq(B, t_a, rcond=None)[0]
        return U
    
    def solve_V_given_UW(U, W, T):
        """Solve for V given U, W, and target tensor T.
        For each b: sum_r V[b,r] * U[a,r] * W[c,r] = T[a,b,c] for all (a,c)
        This is: B @ V[b,:] = t_b where B[a,c,r] = U[a,r]*W[c,r]
        """
        rank = U.shape[1]
        # B has shape (nm*pn, rank) with B[idx, r] = U[a,r]*W[c,r] where idx = a*pn+c
        B = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(nm * pn, rank)
        
        V = np.zeros((mp, rank))
        for b in range(mp):
            t_b = T[:, b, :].flatten()  # shape (nm*pn,)
            V[b] = np.linalg.lstsq(B, t_b, rcond=None)[0]
        return V
    
    # Build target tensor T
    T = np.zeros((nm, mp, pn))
    for a in range(nm):
        i = a // m
        j = a % m
        for b in range(mp):
            j2 = b // p
            if j == j2:
                for c in range(pn):
                    i2 = c % n
                    if i2 == i:
                        T[a, b, c] = 1.0
    
    best_rank = n * m * p  # schoolbook rank = 144
    best_result = None
    
    # Try various target ranks
    # Known bounds: for 4x4*4x4, best known is 22 (from literature)
    # For 4x4*4x9, we can try ranks from ~80 up to 144
    target_ranks = [80, 90, 100, 108, 116, 120, 128, 132, 136, 140]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(1000 + attempt * 137 + target_rank)
            
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # Alternating Least Squares
            prev_err = np.inf
            max_iters = 30
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Solve for W given U, V
                W = solve_W_given_UV(U, V, T)
                
                # Solve for U given V, W
                U = solve_U_given_VW(V, W, T)
                
                # Solve for V given U, W
                V = solve_V_given_UW(U, W, T)
                
                # Compute error
                err = compute_error_vectorized(U, V, W)
                
                if err < 1e-10:
                    # Verify fully
                    if verify(U, V, W):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {
                                "U": U.tolist(),
                                "V": V.tolist(),
                                "W": W.tolist()
                            }
                        found = True
                        break
                    else:
                        # Numerical issue, try to polish
                        break
                
                if abs(prev_err - err) < 1e-15:
                    break
                prev_err = err
            
            if found:
                break
            
            # Try a different initialization: Kronecker-based
            if attempt == max_attempts - 1:
                # Use a structured initialization
                # Split the problem into blocks
                pass
    
    if best_result is not None:
        return best_result
    
    # Fallback: try a few more ranks with different strategies
    for target_rank in [144, 140, 136, 132]:
        if time.time() > deadline - 5:
            break
        
        np.random.seed(9999 + target_rank)
        U = np.random.randn(nm, target_rank) * 0.3
        V = np.random.randn(mp, target_rank) * 0.3
        W = np.random.randn(pn, target_rank) * 0.3
        
        prev_err = np.inf
        for it in range(20):
            if time.time() > deadline - 3:
                break
            
            W = solve_W_given_UV(U, V, T)
            U = solve_U_given_VW(V, W, T)
            V = solve_V_given_UW(U, W, T)
            
            err = compute_error_vectorized(U, V, W)
            
            if err < 1e-10:
                if verify(U, V, W):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                break
            
            if abs(prev_err - err) < 1e-15:
                break
            prev_err = err
    
    if best_result is not None:
        return best_result
    
    # Final fallback: schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
