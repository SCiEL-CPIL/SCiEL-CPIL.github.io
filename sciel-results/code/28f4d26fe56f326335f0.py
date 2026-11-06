import time
import random
import numpy as np
from itertools import product

def _score_vectorized(A):
    """
    Compute max(|A+A|, |A*A|) for set A using vectorized operations.
    A is a numpy array of integers.
    """
    # A+A
    sums = np.add.outer(A, A)
    # A*A
    prods = np.multiply.outer(A, A)
    
    # Count unique values
    # For small n (20), n^2=400. Unique counting is fast.
    # Using set() on flattened arrays is usually faster than np.unique for small arrays in Python context
    # but let's try np.unique for consistency.
    # Actually, Python sets are very fast for small integers.
    # Let's benchmark mentally: np.unique involves sorting or hashing in C.
    # set() involves hashing in C.
    # For 400 elements, both are negligible.
    
    # To be safe and fast:
    num_sums = len(np.unique(sums))
    num_prods = len(np.unique(prods))
    
    return max(num_sums, num_prods)

def _generate_lattice_set(b, c, n=20, max_val=10000):
    """
    Generate a set from a logarithmic lattice: a_k = floor(c * b^k)
    b > 1, c > 0
    """
    A = []
    current = c
    for k in range(n):
        val = int(round(current))
        if val <= 0:
            val = 1
        if val > max_val:
            break
        if val not in A:
            A.append(val)
        else:
            # If duplicate, try to nudge it up slightly to maintain distinctness
            # This breaks the perfect lattice structure slightly
            new_val = val + 1
            while new_val in A and new_val <= max_val:
                new_val += 1
            if new_val <= max_val:
                A.append(new_val)
            else:
                # Can't fit, break
                break
        current *= b
        
    # If we didn't get n elements, fill with larger values
    if len(A) < n:
        existing = set(A)
        last_val = A[-1] if A else 1
        # Try to continue the pattern or fill randomly from above
        # For lattice, continuing the pattern is better for structure
        for _ in range(n - len(A)):
            # Estimate next step
            if len(A) >= 2:
                # Approximate ratio
                ratio = A[-1] / A[-2]
                next_est = int(A[-1] * ratio)
            else:
                next_est = last_val + 100
            
            # Search for a valid spot near next_est
            found = False
            for delta in range(0, 1000):
                for sign in [1, -1]:
                    if delta == 0:
                        candidates = [next_est]
                    else:
                        candidates = [next_est + sign * delta]
                    for cand in candidates:
                        if cand > last_val and cand <= max_val and cand not in existing:
                            A.append(cand)
                            existing.add(cand)
                            last_val = cand
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                # Fallback: random large number
                for _ in range(1000):
                    val = random.randint(last_val + 1, max_val)
                    if val not in existing:
                        A.append(val)
                        existing.add(val)
                        last_val = val
                        break
                else:
                    # If still not found, break
                    break
                    
    A = sorted(A[:n])
    
    if len(A) != n or len(set(A)) != n or any(x < 1 or x > max_val for x in A):
        return None
        
    return np.array(A, dtype=np.int64)

def _local_search(A, rng, time_limit):
    """
    Vectorized local search.
    Tries to swap or adjust elements to reduce max(|A+A|, |A*A|).
    """
    A = A.copy()
    n = len(A)
    current_score = _score_vectorized(A)
    
    # Precompute current sets for quick delta check? 
    # No, re-computing is O(n^2) = 400 ops. Very fast.
    # But we can optimize by only checking affected sums/prods.
    # However, for simplicity and robustness, full re-score is fine if fast.
    
    start_time = time.time()
    no_improve_count = 0
    T = 50.0
    
    while time.time() < time_limit:
        improved = False
        # Shuffle indices
        indices = list(range(n))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > time_limit:
                break
            
            old_val = A[i]
            
            # Generate neighbors
            # 1. Small perturbations
            # 2. Swaps with other elements (if we allow reordering? No, set is unordered, but we fix indices)
            # Let's try changing value.
            
            # Determine search range for new value
            # To minimize product set, we want values that are "multiplicatively close" to others.
            # To minimize sum set, we want values that are "additively close" to others.
            # A good heuristic: try values that are factors or multiples of existing values, 
            # or values that create sum collisions.
            
            # Simple approach: random sample in a range, plus some structured candidates
            candidates = set()
            
            # Structured: factors/multiples of other elements
            for j in range(n):
                if j == i: continue
                v_j = A[j]
                # Multiples
                for k in range(2, 5):
                    if v_j * k <= 10000:
                        candidates.add(v_j * k)
                # Factors
                if v_j > 1:
                    for k in range(2, min(5, int(np.sqrt(v_j)) + 1)):
                        if v_j % k == 0:
                            candidates.add(v_j // k)
            
            # Random perturbations
            # Range: +/- 10% of value, or +/- 50
            delta_range = max(10, int(old_val * 0.1))
            for _ in range(10):
                delta = rng.randint(-delta_range, delta_range)
                new_val = old_val + delta
                if 1 <= new_val <= 10000:
                    candidates.add(new_val)
            
            # Also try random large/small values to escape local minima
            if rng.random() < 0.1:
                candidates.add(rng.randint(1, 10000))
            
            # Evaluate candidates
            best_candidate_score = current_score
            best_candidate_val = old_val
            
            for new_val in candidates:
                if new_val == old_val:
                    continue
                if new_val in A and new_val != old_val:
                    continue
                
                # Check validity
                if new_val < 1 or new_val > 10000:
                    continue
                
                new_A = A.copy()
                new_A[i] = new_val
                # Sort not strictly needed for score, but keeps A canonical
                # np.sort is fast
                new_A = np.sort(new_A)
                
                s = _score_vectorized(new_A)
                
                if s < best_candidate_score:
                    best_candidate_score = s
                    best_candidate_val = new_val
            
            # Accept best candidate if improved or with SA probability
            if best_candidate_val != old_val:
                delta = best_candidate_score - current_score
                if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                    A[i] = best_candidate_val
                    A = np.sort(A)
                    current_score = best_candidate_score
                    improved = True
                    no_improve_count = 0
                    T *= 0.99
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count > 50:
                T *= 0.5
                no_improve_count = 0
                
    return A, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    rng = random.Random(42)
    
    best_A = None
    best_score = float('inf')
    
    # Phase 1: Deterministic Grid Search over Log-Lattice Parameters
    # We search for base b and offset c.
    # b determines the "spread". 
    # To fit 20 terms in [1, 10000], b^19 * c <= 10000.
    # If c=1, b <= 10000^(1/19) ~ 1.85.
    # If c is larger, b must be smaller.
    
    # We will scan b in log space and c in log space.
    
    # Log b range: [log(1.01), log(1.85)]
    log_b_min = np.log(1.01)
    log_b_max = np.log(1.85)
    
    # Log c range: [log(1), log(10000)]
    log_c_min = np.log(1)
    log_c_max = np.log(10000)
    
    # Number of samples
    # We have ~80s. 
    # Let's do a coarse grid first, then refine.
    
    # Coarse Grid
    num_b_samples = 50
    num_c_samples = 20
    
    b_samples = np.exp(np.linspace(log_b_min, log_b_max, num_b_samples))
    c_samples = np.exp(np.linspace(log_c_min, log_c_max, num_c_samples))
    
    # We can prune c_samples based on b to ensure feasibility
    # c * b^19 <= 10000 => c <= 10000 / b^19
    
    candidates = []
    
    for b in b_samples:
        if time.time() > deadline - 30:
            break
        max_c = 10000.0 / (b ** 19)
        if max_c < 1:
            continue
            
        # Filter c samples
        valid_c = c_samples[c_samples <= max_c]
        if len(valid_c) == 0:
            continue
            
        for c in valid_c:
            if time.time() > deadline - 30:
                break
            
            A = _generate_lattice_set(b, c)
            if A is None:
                continue
                
            score = _score_vectorized(A)
            
            if score < best_score:
                best_score = score
                best_A = A.copy()
                
            # Keep top candidates for local search
            candidates.append((score, A.copy()))
            
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Local Search on Top Candidates
    # We have ~50s left.
    # Pick top 10 candidates.
    top_candidates = [c[1] for c in candidates[:10]]
    
    # If no candidates were generated (should not happen), fallback
    if not top_candidates:
        # Fallback to random
        A = np.array(sorted(rng.sample(range(1, 10001), 20)), dtype=np.int64)
        top_candidates = [A]
        best_score = _score_vectorized(A)
        best_A = A.copy()
        
    for A_init in top_candidates:
        if time.time() > deadline - 5:
            break
            
        # Allocate time per candidate
        # If 10 candidates, ~5s each.
        time_per_candidate = min(5.0, (deadline - time.time()) / len(top_candidates))
        
        A_local, score_local = _local_search(A_init, rng, time.time() + time_per_candidate)
        
        if score_local < best_score:
            best_score = score_local
            best_A = A_local.copy()
            
    # Phase 3: Final Polish on the best found
    if time.time() < deadline - 2:
        # Do a short high-temperature SA or more iterations on the best
        A_polished, score_polished = _local_search(best_A, rng, deadline - 1)
        if score_polished < best_score:
            best_score = score_polished
            best_A = A_polished.copy()
            
    # Fallback
    if best_A is None:
        best_A = np.array(list(range(1, 21)), dtype=np.int64)
        
    return sorted(best_A.tolist())

if __name__ == "__main__":
    res = construct()
    print(res)
