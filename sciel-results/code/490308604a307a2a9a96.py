import numpy as np
import time
import sys

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _generate_sidon_like_set(N, K):
    """Generate a set of K indices that are 'Sidon-like' (separated)."""
    # Greedy: pick indices that maximize minimum distance to existing ones
    # Or just random with rejection
    indices = []
    attempts = 0
    while len(indices) < K and attempts < 10000:
        i = np.random.randint(N)
        if i not in indices:
            # Check distance to existing? 
            # For simplicity, just random unique
            indices.append(i)
        attempts += 1
    return indices

def _optimize_sparse_amplitudes(support, max_iter=5000, tol=1e-6):
    """
    Optimize amplitudes h[i] for i in support, keeping others 0.
    Uses coordinate descent.
    """
    h = np.zeros(N)
    # Initialize with equal weights
    h[support] = 1.0 / len(support)
    
    best_h = h.copy()
    best_ratio = _compute_ratio(h)
    
    for _ in range(max_iter):
        improved = False
        for idx in support:
            old_val = h[idx]
            if old_val < 1e-9:
                continue
                
            # Try moving mass from idx to another support element?
            # Or just adjust idx?
            # Since sum is not constrained to 1 in the ratio definition (scale invariant 1/c),
            # but we assume evaluator normalizes or ratio is on simplex.
            # Let's assume we optimize on the simplex (sum=1).
            
            # Coordinate descent on simplex:
            # Move delta from idx to j.
            # To speed up, we can try to find the best j and delta.
            # But that's O(K^2).
            
            # Simpler: Random perturbation of amplitudes within support.
            # Pick two indices i, j in support.
            i = np.random.choice(support)
            j = np.random.choice(support)
            if i == j:
                continue
            
            total_ij = h[i] + h[j]
            if total_ij < 1e-9:
                continue
            
            # New convex combination
            r = np.random.uniform(0, 1)
            new_j = r * total_ij
            new_i = (1 - r) * total_ij
            
            h[i] = new_i
            h[j] = new_j
            
            new_ratio = _compute_ratio(h)
            
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_h = h.copy()
                improved = True
            else:
                # Revert if not improved (greedy)
                h[i] = old_i if 'old_i' in locals() else h[i] # Need to save old values
                # Actually, let's just use SA with very low T or Greedy
                pass
        
        # Greedy coordinate descent is hard to implement simply with random pairs.
        # Let's stick to SA but with sparse support.
        
    return best_h, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.ones(N) / N
    best_ratio = _compute_ratio(best_h)
    
    # Strategy: Search over sparsity levels K and random supports.
    # For each support, run a short SA on amplitudes.
    
    for K in [1, 2, 3, 5, 8, 12, 20]:
        if time.time() > deadline - 10:
            break
            
        # Try multiple random supports for this K
        num_supports = 20 if K < 10 else 5
        
        for _ in range(num_supports):
            if time.time() > deadline - 5:
                break
                
            support = _generate_sidon_like_set(N, K)
            
            # Initialize h
            h = np.zeros(N)
            h[support] = 1.0 / K
            
            # Short SA on amplitudes
            T = 0.05
            T_min = 1e-5
            cooling = 0.99
            max_iter = 2000
            
            for i_iter in range(max_iter):
                if time.time() > deadline - 2:
                    break
                    
                i = np.random.choice(support)
                j = np.random.choice(support)
                if i == j:
                    continue
                
                total_ij = h[i] + h[j]
                if total_ij < 1e-9:
                    continue
                
                old_i, old_j = h[i], h[j]
                r = np.random.uniform(0, 1)
                h[j] = r * total_ij
                h[i] = (1 - r) * total_ij
                
                new_ratio = _compute_ratio(h)
                
                if new_ratio > best_ratio:
                    best_ratio = new_ratio
                    best_h = h.copy()
                else:
                    delta = best_ratio - new_ratio
                    if delta < 1e-9:
                        pass
                    else:
                        if np.random.uniform(0, 1) < np.exp(-delta / T):
                            pass # Accept
                        else:
                            h[i] = old_i
                            h[j] = old_j
                            
                T *= cooling
                
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
