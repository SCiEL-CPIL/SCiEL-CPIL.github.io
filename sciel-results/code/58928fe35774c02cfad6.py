"""Rewritten program for kissing number in dimension 11."""
import numpy as np
import time
import random
import itertools

D = 11
DEADLINE = 140  # seconds, safety margin before 150s kill


def _generate_weight4_vectors():
    """Generate all nonzero weight-4 integer vectors in R^11."""
    vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for i, s in zip(support, signs):
                v[i] = s
            vectors.append(v)
    return vectors


def _greedy_select(candidates, existing, time_limit):
    """Greedy selection: add candidate if compatible with all existing."""
    # Precompute norms squared
    norms_sq = [np.dot(v, v) for v in existing]
    max_norm = max(norms_sq) if norms_sq else 0
    max_norm_sq = max_norm * max_norm  # squared max norm for comparison
    
    for v in candidates:
        if time.time() > time_limit:
            break
        v_norm_sq = np.dot(v, v)
        # New max norm squared
        new_max_norm_sq = max(max_norm_sq, v_norm_sq * v_norm_sq)
        
        # Check compatibility with all existing vectors
        compatible = True
        for e in existing:
            dot = np.dot(v, e)
            dist_sq = v_norm_sq + np.dot(e, e) - 2 * dot
            if dist_sq < new_max_norm_sq:
                compatible = False
                break
        
        if compatible:
            existing.append(v)
            norms_sq.append(v_norm_sq)
            max_norm_sq = new_max_norm_sq
    
    return existing


def _try_mixed_weights(existing, time_limit):
    """Try adding weight-2 and weight-3 vectors to improve the set."""
    # Generate weight-2 vectors
    w2_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                w2_vectors.append(v)
    
    # Generate weight-3 vectors
    w3_vectors = []
    for support in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for i, s in zip(support, signs):
                v[i] = s
            w3_vectors.append(v)
    
    # Try adding weight-2 vectors first (they have smaller norm)
    for v in w2_vectors:
        if time.time() > time_limit:
            break
        v_norm_sq = np.dot(v, v)
        # Check compatibility
        compatible = True
        for e in existing:
            dot = np.dot(v, e)
            dist_sq = v_norm_sq + np.dot(e, e) - 2 * dot
            # Max norm after adding v
            max_norm_sq_candidate = max([np.dot(x, x) for x in existing] + [v_norm_sq])
            max_norm_sq_candidate = max(max_norm_sq_candidate, v_norm_sq)
            if dist_sq < max_norm_sq_candidate:
                compatible = False
                break
        
        if compatible:
            existing.append(v)
    
    # Try adding weight-3 vectors
    for v in w3_vectors:
        if time.time() > time_limit:
            break
        v_norm_sq = np.dot(v, v)
        compatible = True
        for e in existing:
            dot = np.dot(v, e)
            dist_sq = v_norm_sq + np.dot(e, e) - 2 * dot
            max_norm_sq_candidate = max([np.dot(x, x) for x in existing] + [v_norm_sq])
            max_norm_sq_candidate = max(max_norm_sq_candidate, v_norm_sq)
            if dist_sq < max_norm_sq_candidate:
                compatible = False
                break
        
        if compatible:
            existing.append(v)
    
    return existing


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Generate all weight-4 vectors
    w4_vectors = _generate_weight4_vectors()
    print(f"Generated {len(w4_vectors)} weight-4 vectors")
    
    best_result = []
    
    # Multiple random restarts with greedy selection
    random.seed(42)
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Randomly shuffle candidates
        candidates = w4_vectors[:]
        random.shuffle(candidates)
        
        # Greedy selection
        current = _greedy_select(candidates, [], deadline - 15)
        
        # Try to improve with mixed weights
        current = _try_mixed_weights(current, deadline - 10)
        
        if len(current) > len(best_result):
            best_result = current
            print(f"Restart {restart}: found {len(current)} vectors")
        
        if time.time() > deadline - 20:
            break
    
    # If we have time, try more restarts with different strategies
    if time.time() < deadline - 30:
        # Try starting with specific patterns
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            # Start with a random subset
            start_size = random.randint(5, 15)
            initial = random.sample(w4_vectors, start_size)
            
            # Filter for mutual compatibility
            filtered = []
            for v in initial:
                compatible = True
                for e in filtered:
                    dot = np.dot(v, e)
                    dist_sq = np.dot(v, v) + np.dot(e, e) - 2 * dot
                    max_norm_sq = max(np.dot(x, x) for x in filtered)
                    max_norm_sq = max(max_norm_sq, np.dot(v, v))
                    if dist_sq < max_norm_sq:
                        compatible = False
                        break
                if compatible:
                    filtered.append(v)
            
            candidates = [v for v in w4_vectors if v not in filtered]
            random.shuffle(candidates)
            
            current = _greedy_select(candidates, filtered, deadline - 10)
            current = _try_mixed_weights(current, deadline - 5)
            
            if len(current) > len(best_result):
                best_result = current
                print(f"Pattern restart {restart}: found {len(current)} vectors")
    
    # Convert to numpy array
    if not best_result:
        # Fallback: return weight-2 vectors
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    return np.array(best_result, dtype=np.int64)
