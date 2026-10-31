import numpy as np
import time
import random

N = 50

def _autoconv_peak(h):
    """Compute the maximum value of the autoconvolution of h."""
    h = np.asarray(h, dtype=np.float64)
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||autoconv||_inf)"""
    h = np.asarray(h, dtype=np.float64)
    if np.max(h) < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if l1 * peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def _is_sidon_set(indices):
    """Check if a set of indices forms a Sidon set (all pairwise sums are distinct)."""
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _generate_sidon_sets(n_points, max_size, rng):
    """Generate random Sidon sets of various sizes."""
    sidon_sets = []
    
    for size in range(1, min(max_size + 1, n_points + 1)):
        for _ in range(50):
            if len(sidon_sets) >= 200:
                break
            candidate = set()
            attempts = 0
            while len(candidate) < size and attempts < 1000:
                idx = rng.randint(0, n_points - 1)
                # Check if adding idx maintains Sidon property
                valid = True
                for existing in candidate:
                    s = idx + existing
                    # Check if this sum already exists
                    for other in candidate:
                        if other != idx and other > existing:
                            if other + existing == s:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    candidate.add(idx)
                attempts += 1
            if len(candidate) == size:
                sidon_sets.append(list(candidate))
    return sidon_sets

def _optimize_weights_for_support(support_indices, deadline, rng):
    """Optimize weights on a fixed support to maximize the ratio."""
    n = len(support_indices)
    if n == 0:
        return np.zeros(N), 0.0
    
    # Start with uniform weights
    weights = np.ones(n, dtype=np.float64)
    
    def _eval_weights(w):
        h = np.zeros(N, dtype=np.float64)
        for i, idx in enumerate(support_indices):
            h[idx] = w[i]
        return _score(h)
    
    best_w = weights.copy()
    best_s = _eval_weights(weights)
    
    # Coordinate ascent on weights
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(n):
            old_w = weights[i]
            # Try different weight values
            candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
            if old_w > 0:
                candidates.extend([old_w * 0.5, old_w * 1.5, old_w * 2.0, old_w * 0.75])
            
            best_cand_w = old_w
            best_cand_s = best_s
            
            for val in candidates:
                if val < 0:
                    continue
                weights[i] = val
                s = _eval_weights(weights)
                if s > best_cand_s + 1e-12:
                    best_cand_s = s
                    best_cand_w = val
            
            if best_cand_w != old_w:
                weights[i] = best_cand_w
                best_s = best_cand_s
                improved = True
            else:
                weights[i] = old_w
        
        if not improved:
            break
    
    h = np.zeros(N, dtype=np.float64)
    for i, idx in enumerate(support_indices):
        h[idx] = weights[i]
    
    return h, best_s

def construct():
    deadline = time.time() + 140.0  # safety margin
    
    best_h = np.ones(N, dtype=np.float64)
    best_score = _score(best_h)
    
    rng = random.Random(42)
    
    # Strategy 1: Try Sidon set supports with optimized weights
    # Sidon sets minimize autoconvolution collisions
    
    sidon_sets = _generate_sidon_sets(N, 12, rng)
    
    for support in sidon_sets:
        if time.time() > deadline - 10:
            break
        h, s = _optimize_weights_for_support(support, deadline, rng)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Strategy 2: Try some structured patterns as fallback
    patterns = [np.ones(N)]
    
    # Sparse random patterns
    for k in range(1, 15):
        for _ in range(10):
            idx = set(rng.sample(range(N), k))
            h = np.zeros(N)
            h[list(idx)] = 1.0
            patterns.append(h)
    
    for h in patterns:
        if time.time() > deadline - 10:
            break
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Strategy 3: Local optimization from best solution
    current_h = best_h.copy()
    current_score = best_score
    
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5:
                break
            
            old_val = current_h[i]
            candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
            if old_val > 0:
                candidates.append(old_val * 0.5)
                candidates.append(old_val * 2.0)
            
            best_candidate = None
            best_candidate_score = current_score
            
            for val in candidates:
                if val < 0:
                    continue
                current_h[i] = val
                s = _score(current_h)
                if s > best_candidate_score + 1e-12:
                    best_candidate_score = s
                    best_candidate = val
            
            if best_candidate is not None:
                current_h[i] = best_candidate
                current_score = best_candidate_score
                improved = True
            else:
                current_h[i] = old_val
        
        if not improved:
            break
        
        if current_score > best_score:
            best_score = current_score
            best_h = current_h.copy()
    
    # Strategy 4: Random restarts with Sidon-inspired initialization
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        
        # Generate a random sparse support
        n_nonzero = rng.randint(3, 12)
        idx = rng.sample(range(N), n_nonzero)
        h = np.zeros(N)
        for i in idx:
            h[i] = rng.uniform(0.1, 2.0)
        
        current_h = h.copy()
        current_score = _score(current_h)
        
        # Quick local optimization
        for iteration in range(50):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                old_val = current_h[i]
                candidates = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
                if old_val > 0:
                    candidates.append(old_val * 0.5)
                    candidates.append(old_val * 2.0)
                
                best_candidate = None
                best_candidate_score = current_score
                
                for val in candidates:
                    if val < 0:
                        continue
                    current_h[i] = val
                    s = _score(current_h)
                    if s > best_candidate_score + 1e-12:
                        best_candidate_score = s
                        best_candidate = val
                
                if best_candidate is not None:
                    current_h[i] = best_candidate
                    current_score = best_candidate_score
                    improved = True
                else:
                    current_h[i] = old_val
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_h = current_h.copy()
    
    return list(best_h)
