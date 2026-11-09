"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    target_rank = 125  # schoolbook
    
    # Precompute tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = target_rank
    
    # Always have a fallback: schoolbook
    def make_schoolbook():
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
            U[i*m+j, idx] = 1.0
            V[j*p+k, idx] = 1.0
            W[k*n+i, idx] = 1.0
        return U, V, W
    
    U_school, V_school, W_school = make_schoolbook()
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Attempt to find a rank < 125 decomposition using iterative tensor power method
    # This is a heuristic, so it may not succeed, but if it does, we get a better score.
    # We try several random initializations.
    
    def verify(U, V, W):
        # Check if U @ V.T @ W.T approximates T
        # T is (n*m, m*p, p*n)
        # U is (n*m, r), V is (m*p, r), W is (p*n, r)
        # Product: sum_r U[:,r] * V[:,r] * W[:,r]
        # In numpy: np.einsum('ir,jr,kr->ijk', U, V, W)
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_rec, T, atol=1e-10)
    
    for rank in range(target_rank - 1, 50, -1):
        if time.time() > deadline:
            break
            
        # Try a few random starts for this rank
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Random init
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Power method iterations
            for _ in range(50):
                if time.time() > deadline:
                    break
                    
                # Update U
                for r in range(rank):
                    # U[:,r] should approximate sum_{j,k} V[j,r] W[k,r] T[:,j,k]
                    # T[:,j,k] is a vector of length n*m
                    # Let's do this efficiently
                    # U_new[:,r] = sum_j sum_k V[j,r] W[k,r] T[:,j,k]
                    # = sum_j sum_k V[j,r] W[k,r] * T[:,j,k]
                    # This is a weighted sum of slices of T
                    # Let's just do the einsum for the whole rank update at once? No, rank is the new dimension.
                    
                    # Actually, let's use the standard tensor power method:
                    # U_new = T * V * W  (where * denotes contraction on the last two dims)
                    # U_new[i, r] = sum_j,k T[i,j,k] V[j,r] W[k,r]
                    # This is U_new = np.einsum('ijk,jr,kr->ir', T, V, W)
                    
                    pass # placeholder
                
                # Update all U at once
                U_new = np.einsum('ijk,jr,kr->ir', T, V, W)
                # Update V
                V_new = np.einsum('ijk,ir,kr->jr', T, U, W)
                # Update W
                W_new = np.einsum('ijk,ir,jr->kr', T, U, V)
                
                # Normalize to prevent explosion/vanishing
                for r in range(rank):
                    norm_u = np.linalg.norm(U_new[:,r])
                    norm_v = np.linalg.norm(V_new[:,r])
                    norm_w = np.linalg.norm(W_new[:,r])
                    if norm_u > 0 and norm_v > 0 and norm_w > 0:
                        # We need to preserve the product U*V*W
                        # Scale such that product of norms remains constant or 1
                        # Simple: normalize each to 1? No, that changes the scale.
                        # Better: scale U by 1/norm_u, V by 1/sqrt(norm_v), W by 1/sqrt(norm_w)?
                        # Or just normalize each to unit norm and multiply by a common scale?
                        # Let's just normalize each to unit norm for now. The rank minimization might need specific scales, but power method usually converges to a subspace.
                        # Actually, for the rank to be exact, we need the specific values.
                        # Let's try to keep the norm of the outer product constant.
                        scale = 1.0 / (norm_u * norm_v * norm_w) ** (1.0/3.0)
                        U_new[:,r] = U_new[:,r] * scale
                        V_new[:,r] = V_new[:,r] * scale
                        W_new[:,r] = W_new[:,r] * scale
                
                U = U_new
                V = V_new
                W = W_new
                
            # Verify
            if verify(U, V, W):
                # Found a valid decomposition with rank
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank < target_rank:
            break
    
    if best_rank < target_rank:
        # Convert to lists of floats/ints if possible
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U_list = [[float(x) for x in row] for row in U_school]
        V_list = [[float(x) for x in row] for row in V_school]
        W_list = [[float(x) for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
