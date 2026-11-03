"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

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

        # Try eighths
        U_e = np.round(U * 8) / 8.0
        V_e = np.round(V * 8) / 8.0
        W_e = np.round(W * 8) / 8.0
        if verify(U_e, V_e, W_e):
            return U_e, V_e, W_e

        return None

    def polish(U, V, W, max_iters=50):
        """High-precision polish pass with tighter tolerance."""
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            M = np.zeros((W.shape[1], mp * pn))
            for r in range(W.shape[1]):
                M[r] = np.outer(V[:, r], W[:, r]).flatten()
            try:
                U_new, _, _, _ = np.linalg.lstsq(M.T, T.reshape(nm, mp*pn).T, rcond=None)
                U = U_new.T
            except:
                break
            
            # Update V
            M2 = np.zeros((W.shape[1], nm * pn))
            for r in range(W.shape[1]):
                M2[r] = np.outer(U[:, r], W[:, r]).flatten()
            try:
                V_new, _, _, _ = np.linalg.lstsq(M2.T, T.transpose(1,0,2).reshape(mp, nm*pn).T, rcond=None)
                V = V_new.T
            except:
                break
            
            # Check convergence with tighter tolerance
            S = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(S, T, atol=1e-12):
                break
        
        return U, V

    def als_solve(W, R, max_iters=80):
        """
        Alternating Least Squares to find U and V given W.
        T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        """
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update U: for each a, T[a,:,:] = sum_r U[a,r] (V[:,r] outer W[:,r])
            # Vectorized: T_flat = U @ M^T where M[r] = (V[:,r] outer W[:,r]).flatten()
            M = np.zeros((R, mp * pn))
            for r in range(R):
                M[r] = np.outer(V[:, r], W[:, r]).flatten()
            
            try:
                # T reshaped to (nm, mp*pn), solve U @ M^T = T_reshaped
                U_new, _, _, _ = np.linalg.lstsq(M.T, T.reshape(nm, mp*pn).T, rcond=None)
                U = U_new.T
            except:
                return None
            
            # Update V: for each b, T[:,b,:] = sum_r V[b,r] (U[:,r] outer W[:,r])
            # Vectorized: T_flat2 = V @ M2^T where M2[r] = (U[:,r] outer W[:,r]).flatten()
            M2 = np.zeros((R, nm * pn))
            for r in range(R):
                M2[r] = np.outer(U[:, r], W[:, r]).flatten()
            
            try:
                V_new, _, _, _ = np.linalg.lstsq(M2.T, T.transpose(1,0,2).reshape(mp, nm*pn).T, rcond=None)
                V = V_new.T
            except:
                return None
            
            # Check convergence
            if it > 10:
                S = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(S, T, atol=1e-8):
                    # Polish pass
                    U, V = polish(U, V, W)
                    break
        
        # Final verification with polish
        if verify(U, V, W):
            U, V = polish(U, V, W)
            if verify(U, V, W):
                result = round_and_verify(U, V, W)
                if result:
                    return result
        
        return None

    def generate_W_sparse(R, nnz_min=2, nnz_max=4):
        """Generate sparse W with entries in {-1, 0, 1}."""
        W = np.zeros((pn, R))
        for r in range(R):
            num_nonzeros = random.randint(nnz_min, nnz_max)
            positions = random.sample(range(pn), num_nonzeros)
            for pos in positions:
                W[pos, r] = random.choice([-1, 1])
        return W

    def generate_W_dense(R):
        """Generate dense random W."""
        return np.random.randn(pn, R)

    def generate_W_structured(R):
        """Generate W with structure based on tensor geometry."""
        W = np.zeros((pn, R))
        # Try to exploit the structure: W[k*n+i, r]
        for r in range(R):
            # Random subset of (k,i) pairs
            num_pairs = random.randint(2, min(5, n*p))
            pairs = random.sample(range(n*p), num_pairs)
            for idx in pairs:
                k, i = divmod(idx, n)
                W[k*n+i, r] = random.choice([-1, 1])
        return W

    def try_rank(target_rank):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline:
            return None
        
        # Try different W initialization strategies
        num_attempts = 25 if target_rank < 40 else 12
        
        # Strategy 1: Sparse W
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None
            W = generate_W_sparse(target_rank)
            result = als_solve(W, target_rank)
            if result is not None:
                return result
        
        # Strategy 2: Dense random W
        for attempt in range(5):
            if time.time() > deadline:
                return None
            W = generate_W_dense(target_rank)
            result = als_solve(W, target_rank)
            if result is not None:
                return result
        
        # Strategy 3: Structured W
        for attempt in range(5):
            if time.time() > deadline:
                return None
            W = generate_W_structured(target_rank)
            result = als_solve(W, target_rank)
            if result is not None:
                return result
        
        return None

    # Lower bound for matrix multiplication rank
    lower_bound = max(nm, mp, pn)  # = 28
    
    # Try ranks from lower bound upwards
    for target_rank in range(lower_bound, min(lower_bound + 25, best_rank)):
        if time.time() > deadline:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            break
    
    # If we found something better, try to improve further
    if best_rank < n * m * p and time.time() < deadline:
        for target_rank in range(best_rank - 1, max(lower_bound - 1, 0), -1):
            if time.time() > deadline:
                break
            result = try_rank(target_rank)
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
