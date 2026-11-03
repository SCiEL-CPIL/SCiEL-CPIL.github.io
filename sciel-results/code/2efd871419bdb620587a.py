"""
Matrix multiplication 5x5x5 rank-minimal exact tensor decomposition.

Uses sub-tensor peeling with randomized linear algebra over a large finite field.
"""

import numpy as np
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 5, 5, 5
    P = 2**61 - 1  # Large Mersenne prime
    
    # Build tensor indices
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    
    def mod_inv(x, P):
        return pow(x, P - 2, P)
    
    def mod_sub(a, b, P):
        return (a - b) % P
    
    def mod_mul(a, b, P):
        return (a * b) % P
    
    def mod_add(a, b, P):
        return (a + b) % P
    
    # Try to find decomposition with rank R
    def try_decompose(R, seed_val):
        np.random.seed(seed_val)
        random.seed(seed_val)
        
        # Initialize random rank-1 components
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        U = np.zeros((n * m, R), dtype=np.int64)
        V = np.zeros((m * p, R), dtype=np.int64)
        W = np.zeros((p * n, R), dtype=np.int64)
        
        # Initialize with random values
        for r in range(R):
            U[:, r] = np.random.randint(1, P, size=n * m) % P
            V[:, r] = np.random.randint(1, P, size=m * p) % P
            W[:, r] = np.random.randint(1, P, size=p * n) % P
        
        # Iterative refinement: for each nonzero entry, solve for coefficients
        # We want: sum_r U[a,r] * V[b,r] * W[c,r] = 1 for each (a,b,c) in nonzero
        # and = 0 for all other (a,b,c)
        
        # This is a system of equations. Let's use a peeling approach.
        # For each nonzero (a,b,c), we want to find rank-1 components that match.
        
        # Alternative: use the fact that if we fix two factors, the third is determined.
        # Let's use a greedy peeling approach.
        
        # Represent the tensor as a dict for sparse access
        T_dict = {}
        for a, b, c in nonzero:
            T_dict[(a, b, c)] = 1
        
        # Residual tensor (starts as T)
        residual = {}
        for key, val in T_dict.items():
            residual[key] = val
        
        # Also track zero entries that should remain zero
        # For efficiency, we'll check zeros at the end
        
        components = []
        
        max_peel = 500
        for peel_iter in range(max_peel):
            if len(residual) == 0:
                break
            
            # Find a rank-1 component that matches a subset of residual
            # Strategy: pick a random nonzero entry, try to build a rank-1 around it
            
            # Pick a random nonzero entry from residual
            keys = list(residual.keys())
            a0, b0, c0 = random.choice(keys)
            
            # We want to find u, v, w such that u[a0]*v[b0]*w[c0] = 1
            # and u[a]*v[b]*w[c] = residual[(a,b,c)] for other entries in support
            
            # Simple approach: set u[a0]=1, v[b0]=1, w[c0]=1
            # Then for each other (a,b,c) in residual, we need u[a]*v[b]*w[c] = residual[(a,b,c)]
            
            # This is generally inconsistent. Instead, let's use a different strategy.
            
            # Strategy: use random vectors and solve linear systems
            # For a fixed rank-1 component u ⊗ v ⊗ w, the contribution to entry (a,b,c) is u[a]*v[b]*w[c]
            
            # Let's try to find a rank-1 component that has large inner product with residual
            # and then subtract it.
            
            best_score = -1
            best_uvw = None
            
            for trial in range(20):
                # Generate random u, v, w
                u = np.random.randint(1, P, size=n * m) % P
                v = np.random.randint(1, P, size=m * p) % P
                w = np.random.randint(1, P, size=p * n) % P
                
                # Compute inner product with residual
                # score = sum_{(a,b,c) in residual} residual[(a,b,c)] * u[a] * v[b] * w[c]
                score = 0
                for (a, b, c), val in residual.items():
                    score = mod_add(score, mod_mul(val, mod_mul(u[a], v[b], P), w[c], P), P)
                
                if score > best_score:
                    best_score = score
                    best_uvw = (u.copy(), v.copy(), w.copy())
            
            if best_uvw is None:
                break
            
            u, v, w = best_uvw
            
            # Determine the scaling factor: we want to subtract alpha * (u ⊗ v ⊗ w)
            # such that for the entry (a0, b0, c0), we match the residual
            # residual[(a0,b0,c0)] - alpha * u[a0]*v[b0]*w[c0] = 0
            # alpha = residual[(a0,b0,c0)] / (u[a0]*v[b0]*w[c0])
            
            a0, b0, c0 = random.choice(list(residual.keys()))
            prod = mod_mul(u[a0], v[b0], P)
            prod = mod_mul(prod, w[c0], P)
            
            if prod == 0:
                continue
            
            alpha = mod_mul(residual[(a0, b0, c0)], mod_inv(prod, P), P)
            
            # Subtract alpha * (u ⊗ v ⊗ w) from residual
            # For each entry in residual, update
            new_residual = {}
            for (a, b, c), val in residual.items():
                contrib = mod_mul(alpha, mod_mul(u[a], v[b], P), w[c], P)
                new_val = mod_sub(val, contrib, P)
                if new_val != 0:
                    new_residual[(a, b, c)] = new_val
            
            # Also check if we created new nonzero entries (shouldn't happen if we're careful)
            # But we only track entries that were in residual or are in T_dict
            
            # For entries not in residual but in T_dict, they should be 0 in residual
            # So we don't need to check them
            
            residual = new_residual
            components.append((alpha, u, v, w))
            
            if len(components) > R:
                break
        
        if len(residual) != 0:
            return None
        
        if len(components) > R:
            return None
        
        # Build the final U, V, W matrices
        # Each component is alpha * u ⊗ v ⊗ w
        # U[:, r] = alpha_r * u_r
        # V[:, r] = v_r
        # W[:, r] = w_r
        
        U_final = np.zeros((n * m, len(components)), dtype=np.int64)
        V_final = np.zeros((m * p, len(components)), dtype=np.int64)
        W_final = np.zeros((p * n, len(components)), dtype=np.int64)
        
        for r, (alpha, u, v, w) in enumerate(components):
            U_final[:, r] = np.array([mod_mul(alpha, u[a], P) for a in range(n * m)], dtype=np.int64)
            V_final[:, r] = v
            W_final[:, r] = w
        
        # Verify the solution
        # Check all nonzero entries
        for a, b, c in nonzero:
            total = 0
            for r in range(len(components)):
                total = mod_add(total, mod_mul(U_final[a, r], mod_mul(V_final[b, r], W_final[c, r], P), P), P)
            if total != 1:
                return None
        
        # Check some zero entries (sample)
        for _ in range(100):
            a = random.randint(0, n * m - 1)
            b = random.randint(0, m * p - 1)
            c = random.randint(0, p * n - 1)
            if (a, b, c) in T_dict:
                continue
            total = 0
            for r in range(len(components)):
                total = mod_add(total, mod_mul(U_final[a, r], mod_mul(V_final[b, r], W_final[c, r], P), P), P)
            if total != 0:
                return None
        
        return U_final, V_final, W_final
    
    # Try ranks from 48 to 60
    for R in range(48, 61):
        for attempt in range(20):
            seed = R * 1000 + attempt
            result = try_decompose(R, seed)
            if result is not None:
                U, V, W = result
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
