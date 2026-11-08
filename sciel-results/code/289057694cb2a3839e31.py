"""Rewritten program: search for rank-minimal exact tensor decomposition of 3x5 by 5x6 matmul."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 5, 6]
DEADLINE_SECONDS = 145  # safety margin before 150s wall-clock kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 90

    # Precompute the target tensor T as a 3D array T[a,b,d]
    # a in [0, n*m), b in [0, m*p), d in [0, p*n)
    # T[a,b,d] = 1 if (a%m == b//p) and (b%p == d//n) and (a//m == d%n), else 0
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0

    # Schoolbook solution as fallback
    def schoolbook():
        rank = rank_schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, c] = 1.0
                    V[j*p + k, c] = 1.0
                    W[k*n + i, c] = 1.0
                    c += 1
        return U, V, W

    best_U, best_V, best_W = schoolbook()
    best_rank = rank_schoolbook

    # Try to find a lower rank decomposition using ALS + rounding + verification
    # Lower bound on rank for matrix multiplication tensor is roughly n*m*p / (n+m+p-2) or similar,
    # but for exact decomposition we need to search.
    
    def verify(U, V, W):
        """Check if U,V,W exactly decompose T (within tight tolerance)."""
        # Compute sum over c of U[:,c] * V[:,c] * W[:,c] as tensor
        # For each (a,b,d): sum_c U[a,c]*V[b,c]*W[d,c]
        rank = U.shape[1]
        reconstructed = np.zeros_like(T)
        for c in range(rank):
            reconstructed += U[:, c] * V[:, c] * W[:, c]
        # Check max error
        err = np.max(np.abs(reconstructed - T))
        return err < 1e-9

    def round_to_half(U, V, W):
        """Round entries to nearest half-integer."""
        return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2

    def als_decompose(T, rank, n_iters=50, seed=42):
        """Alternating Least Squares for CP decomposition."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        
        # Normalize to avoid divergence
        U /= np.linalg.norm(U) + 1e-10
        V /= np.linalg.norm(V) + 1e-10
        W /= np.linalg.norm(W) + 1e-10
        
        for it in range(n_iters):
            # Update U: for each column c, minimize ||T - sum_c u_c v_c w_c||
            # Fix V and W, update U[:, c]
            for c in range(rank):
                # Build matrix: for each c, the contribution is outer(U[:,c], V[:,c]*W[:,c] reshaped)
                # Actually: T[a,b,d] = sum_c U[a,c] * V[b,c] * W[d,c]
                # Fix V,W, for each a: T[a,:, :] = sum_c U[a,c] * outer(V[:,c], W[:,c])
                # So for fixed c, U[a,c] is coefficient.
                # Reshape T to (n*m, m*p*p*n)
                T2 = T.reshape(n*m, m*p * p*n)
                # Build A where A[a, (b,d)] = sum_{c' != c} U[a,c'] * V[b,c'] * W[d,c']
                # This is expensive. Use the standard ALS approach:
                # For each a, we want to find U[a,c] such that
                # T2[a,:] - sum_{c' != c} U[a,c'] * (V[:,c'] * W[:,c']).reshape(-1) is projected
                # Actually, standard ALS for CP:
                # Fix V and W. Then for each a, define B_a = [outer(V[:,c], W[:,c]) for c]
                # This is a (m*p*p*n) x rank matrix.
                # Then U[a, :] = least squares solution to T2[a,:] = B_a @ U[a,:]
                
                # This is too expensive for large tensors. Let's use a different approach.
                # Alternative: use the unfolding approach.
                # T_{(1)} is (n*m) x (m*p*p*n)
                # T_{(1)} = U * (V * W) where * is Khatri-Rao product
                # So U = T_{(1)} * pinv(V * W)
                
            # This approach is complex. Let's use a simpler iterative method.
            # Actually, for small sizes, let's just use scipy's optimization or a simple gradient method.
            pass
        
        return U, V, W

    # Given the complexity, let's try a more direct approach:
    # For small ranks, we can try to solve the system directly.
    # The number of unknowns for rank r is r*(n*m + m*p + p*n) = r*(15 + 30 + 18) = 63r
    # The number of constraints is n*m*p = 90 (since T has 90 ones and the rest are 0, but actually all entries must match)
    # Wait, T has size 15*30*18 = 8100 entries, but only 90 are 1, rest are 0.
    # So we have 8100 equations.
    # For rank r, we have 63r unknowns. We need 63r >= 8100 => r >= 129. That's not right.
    # Actually, the tensor is sparse in a structured way. The matmul tensor has rank at most n*m*p = 90.
    # Known results: rank of matrix multiplication tensor for n=m=p=2 is 7. For 3x3x3, it's 23.
    # For 3x5x6, the rank is likely between 30 and 90.
    
    # Let's try a heuristic search with random initializations and a simple optimization.
    # Since time is limited, we'll try a few ranks and see if we can find a solution.
    
    def try_rank(rank, n_trials=5, n_als_iters=100):
        """Try to find a decomposition of given rank."""
        for trial in range(n_trials):
            if time.time() - start_time > DEADLINE_SECONDS:
                return False, None, None, None
            seed = trial * 1000 + 42
            rng = np.random.RandomState(seed)
            
            # Initialize
            U = rng.randn(n*m, rank)
            V = rng.randn(m*p, rank)
            W = rng.randn(p*n, rank)
            
            # Normalize
            for c in range(rank):
                norm_u = np.linalg.norm(U[:, c])
                norm_v = np.linalg.norm(V[:, c])
                norm_w = np.linalg.norm(W[:, c])
                if norm_u > 1e-10:
                    U[:, c] /= norm_u
                if norm_v > 1e-10:
                    V[:, c] /= norm_v
                if norm_w > 1e-10:
                    W[:, c] /= norm_w
            
            # Simple ALS: fix two, update the third
            for it in range(n_als_iters):
                if time.time() - start_time > DEADLINE_SECONDS:
                    return False, None, None, None
                
                # Update U: for each a, U[a, :] = solve( (V*W)^T (V*W), (V*W)^T T2[a,:] )
                # Where V*W is Khatri-Rao product: (V[:,c] * W[:,c]) for each c
                # Shape: (m*p*p*n, rank)
                # This is 540 x rank, which is manageable for small rank.
                
                T2 = T.reshape(n*m, m*p * p*n)
                # Build Khatri-Rao matrix
                # KR[:, c] = (V[:,c] * W[:,c]).reshape(-1)
                KR = np.zeros((m*p * p*n, rank))
                for c in range(rank):
                    KR[:, c] = (V[:, c] * W[:, c]).reshape(-1)
                
                # U = T2 @ pinv(KR)
                try:
                    U_new = T2 @ np.linalg.pinv(KR)
                    U = U_new
                except:
                    pass
                
                # Similarly for V and W
                # T3 = T.reshape(m*p, n*m * p*n)
                # KR2 = np.zeros((n*m * p*n, rank))
                # for c in range(rank):
                #     KR2[:, c] = (U[:, c] * W[:, c]).reshape(-1)
                # V = T3 @ np.linalg.pinv(KR2)
                
                # T4 = T.reshape(p*n, n*m * m*p)
                # KR3 = np.zeros((n*m * m*p, rank))
                # for c in range(rank):
                #     KR3[:, c] = (U[:, c] * V[:, c]).reshape(-1)
                # W = T4 @ np.linalg.pinv(KR3)
                
                T3 = T.reshape(m*p, n*m * p*n)
                KR2 = np.zeros((n*m * p*n, rank))
                for c in range(rank):
                    KR2[:, c] = (U[:, c] * W[:, c]).reshape(-1)
                try:
                    V = T3 @ np.linalg.pinv(KR2)
                except:
                    pass
                
                T4 = T.reshape(p*n, n*m * m*p)
                KR3 = np.zeros((n*m * m*p, rank))
                for c in range(rank):
                    KR3[:, c] = (U[:, c] * V[:, c]).reshape(-1)
                try:
                    W = T4 @ np.linalg.pinv(KR3)
                except:
                    pass
            
            # Check if we got a good solution
            err = 0
            reconstructed = np.zeros_like(T)
            for c in range(rank):
                reconstructed += U[:, c] * V[:, c] * W[:, c]
            err = np.max(np.abs(reconstructed - T))
            
            if err < 1e-6:
                # Try rounding to half-integers
                Ur, Vr, Wr = round_to_half(U, V, W)
                # Verify rounded solution
                reconstructed_r = np.zeros_like(T)
                for c in range(rank):
                    reconstructed_r += Ur[:, c] * Vr[:, c] * Wr[:, c]
                err_r = np.max(np.abs(reconstructed_r - T))
                if err_r < 1e-9:
                    return True, Ur, Vr, Wr, err_r
                else:
                    # Return unrounded if close enough? No, must be exact.
                    # Try a local polish
                    pass
        
        return False, None, None, None

    # Search ranks from a reasonable lower bound up to schoolbook
    # Known: rank(2x2x2) = 7, rank(3x3x3) = 23
    # For 3x5x6, let's try from 30 to 89
    for rank in range(30, rank_schoolbook):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        success, Ur, Vr, Wr, err = try_rank(rank, n_trials=3, n_als_iters=50)
        if success:
            best_U, best_V, best_W = Ur, Vr, Wr
            best_rank = rank
            break

    # Convert to lists for the return format
    U_list = [[float(val) for val in row] for row in best_U.tolist()]
    V_list = [[float(val) for val in row] for row in best_V.tolist()]
    W_list = [[float(val) for val in row] for row in best_W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
