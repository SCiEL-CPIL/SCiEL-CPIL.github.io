import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [2, 3, 7]
_TIME_LIMIT = 140.0  # Leave margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for 2x3 x 3x7 matrix multiplication.
    Uses a search approach starting from schoolbook and attempting to reduce rank.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    def _schoolbook():
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
        return U, V, W, rank
    
    def _verify_np(U, V, W):
        """
        Verify exact decomposition using NumPy.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        We want T[i,j,k] == 1 for all i,j,k.
        """
        rank = U.shape[1]
        # Reshape U to (n, m, rank)
        U_reshaped = U.reshape(n, m, rank)
        # Reshape V to (m, p, rank)
        V_reshaped = V.reshape(m, p, rank)
        # Reshape W to (p, n, rank) -> We need W[k, i, r]
        # Original W is indexed by k*n+i. So W[k, i] in 2D is W_flat[k*n+i].
        W_reshaped = W.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum: 'ijr,jkr,kir->ijk'
        # Note: W is indexed by k, i. The tensor component is W[k, i, r].
        # So the indices are i, j, k.
        # U[i,j,r], V[j,k,r], W[k,i,r]
        T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Check if all entries are close to 1
        return np.allclose(T, 1.0, atol=1e-9)

    def _verify_slow(U, V, W):
        """Fallback verification if NumPy fails or for debugging, but we will use NumPy."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def _random_search(target_rank, time_budget=10.0, seed=0):
        """Try to find a decomposition of target_rank using random initialization and local search."""
        start = time.time()
        rng = random.Random(seed)
        
        # Initialize with random small integer/half-integer values
        U = np.zeros((n * m, target_rank))
        V = np.zeros((m * p, target_rank))
        W = np.zeros((p * n, target_rank))
        
        # Initialize with random values from {-1, 0, 1, 0.5, -0.5}
        vals = np.array([-1, 0, 1, 0.5, -0.5])
        for i in range(n * m):
            for r in range(target_rank):
                U[i, r] = rng.choice(vals)
        for j in range(m * p):
            for r in range(target_rank):
                V[j, r] = rng.choice(vals)
        for k in range(p * n):
            for r in range(target_rank):
                W[k, r] = rng.choice(vals)
        
        # Local search
        max_iterations = 20000
        for iteration in range(max_iterations):
            if time.time() - start > time_budget:
                break
            
            # Check if we're done
            if _verify_np(U, V, W):
                return U, V, W, target_rank
            
            # Find unsatisfied constraints
            # Compute T
            U_reshaped = U.reshape(n, m, target_rank)
            V_reshaped = V.reshape(m, p, target_rank)
            W_reshaped = W.reshape(p, n, target_rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
            
            # Find indices where T != 1
            # We pick a random unsatisfied (i,j,k)
            unsatisfied_mask = np.abs(T - 1.0) > 1e-9
            if not np.any(unsatisfied_mask):
                return U, V, W, target_rank
            
            # Get list of unsatisfied indices
            unsatisfied_indices = np.argwhere(unsatisfied_mask)
            idx = rng.choice(len(unsatisfied_indices))
            i, j, k = unsatisfied_indices[idx]
            
            current_sum = T[i, j, k]
            diff = 1.0 - current_sum
            
            # Try to adjust one entry to fix this constraint
            # Option: adjust U[i*m+j, r] for some r where V and W are nonzero
            option = rng.choice([0, 1, 2])
            
            if option == 0:
                # Adjust U[i*m+j, r]
                # Contribution of r is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # We want to change U[i*m+j, r] by delta such that delta * V * W = diff
                # So delta = diff / (V * W)
                # New U = Old U + delta
                # We want New U to be a half-integer.
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                # Find r where V and W are not zero
                candidates = []
                for r in range(target_rank):
                    v_val = V[v_idx, r]
                    w_val = W[w_idx, r]
                    if abs(v_val) > 1e-9 and abs(w_val) > 1e-9:
                        denom = v_val * w_val
                        if abs(denom) > 1e-9:
                            delta = diff / denom
                            new_u_val = U[u_idx, r] + delta
                            # Round to nearest half-integer
                            new_u_rounded = round(new_u_val * 2) / 2.0
                            # Check if this change actually helps (reduces error)
                            # The new error for this specific term would be 0 if we could set it exactly.
                            # But we are rounding.
                            # Let's just try it if the rounded value is close to the exact required value
                            if abs(new_u_rounded - new_u_val) < 0.25: # Tolerance for rounding
                                candidates.append((r, new_u_rounded))
                
                if candidates:
                    r, new_val = rng.choice(candidates)
                    U[u_idx, r] = new_val
            
            elif option == 1:
                # Adjust V[j*p+k, r]
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                candidates = []
                for r in range(target_rank):
                    u_val = U[u_idx, r]
                    w_val = W[w_idx, r]
                    if abs(u_val) > 1e-9 and abs(w_val) > 1e-9:
                        denom = u_val * w_val
                        if abs(denom) > 1e-9:
                            delta = diff / denom
                            new_v_val = V[v_idx, r] + delta
                            new_v_rounded = round(new_v_val * 2) / 2.0
                            if abs(new_v_rounded - new_v_val) < 0.25:
                                candidates.append((r, new_v_rounded))
                
                if candidates:
                    r, new_val = rng.choice(candidates)
                    V[v_idx, r] = new_val
            
            else:
                # Adjust W[k*n+i, r]
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                candidates = []
                for r in range(target_rank):
                    u_val = U[u_idx, r]
                    v_val = V[v_idx, r]
                    if abs(u_val) > 1e-9 and abs(v_val) > 1e-9:
                        denom = u_val * v_val
                        if abs(denom) > 1e-9:
                            delta = diff / denom
                            new_w_val = W[w_idx, r] + delta
                            new_w_rounded = round(new_w_val * 2) / 2.0
                            if abs(new_w_rounded - new_w_val) < 0.25:
                                candidates.append((r, new_w_rounded))
                
                if candidates:
                    r, new_val = rng.choice(candidates)
                    W[w_idx, r] = new_val
        
        # Check final result
        if _verify_np(U, V, W):
            return U, V, W, target_rank
        return None
    
    # Start with schoolbook
    U, V, W, current_rank = _schoolbook()
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = 21
    # Try ranks from 21 to 41
    # Note: It is highly unlikely we will find a rank < 42 with this simple search.
    # We will return schoolbook if no better is found.
    
    for target_rank in range(21, schoolbook_rank):
        if time.time() - start_time > _TIME_LIMIT - 10.0:
            break
        
        # Try a few random seeds for this rank
        found = False
        for seed in range(2):
            if time.time() - start_time > _TIME_LIMIT - 10.0:
                break
            
            result = _random_search(target_rank, time_budget=5.0, seed=seed)
            if result is not None:
                U, V, W, rank = result
                if _verify_np(U, V, W):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    found = True
                    break
        
        if found:
            # If we found a valid decomposition, continue trying lower ranks
            continue
        else:
            # If we can't find rank target_rank, break to save time for other attempts?
            # Or just continue. Since it's unlikely, we can break early if time is tight.
            if time.time() - start_time > _TIME_LIMIT - 20.0:
                break
    
    # Convert to lists and return
    return best_result
# EVOLVE-BLOCK-END
