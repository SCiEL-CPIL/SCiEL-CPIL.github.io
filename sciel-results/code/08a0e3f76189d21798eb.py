import math
import time
import random

def _score_set(U):
    """Compute the score for a set U."""
    if not U:
        return 1.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    diffs = set()
    sums = set()
    n = len(U)
    for i in range(n):
        for j in range(n):
            diffs.add(U[i] - U[j])
            sums.add(U[i] + U[j])
    size_diff = len(diffs)
    size_sum = len(sums)
    if size_sum == 0:
        return 1.0
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _greedy_build(seed_val, max_elements=20, deadline=None):
    """
    Greedy packing: start with {0}, repeatedly add the candidate x that
    maximizes marginal difference yield per unit increase in log span.
    """
    random.seed(seed_val)
    U = [0]
    diffs = {0}
    sums = {0}
    
    # Candidate pool: generate candidates dynamically
    # We'll consider candidates up to some max value
    max_candidate = 500
    
    for step in range(max_elements):
        if deadline and time.time() > deadline:
            break
            
        best_x = None
        best_marginal = -float('inf')
        best_new_diffs = None
        best_new_sums = None
        best_new_max = 0
        
        # Sample candidates from a range, biased toward values that create new differences
        # Try a mix of small and large candidates
        candidates_to_try = set()
        
        # Small candidates near existing elements
        for u in U:
            for delta in range(1, 30):
                c = u + delta
                if 0 < c <= max_candidate:
                    candidates_to_try.add(c)
                c2 = u - delta
                if c2 > 0:
                    candidates_to_try.add(c2)
        
        # Some random candidates
        for _ in range(50):
            candidates_to_try.add(random.randint(1, max_candidate))
        
        # Also try specific structured candidates
        for u in U:
            for v in U:
                c = u + v
                if 0 < c <= max_candidate:
                    candidates_to_try.add(c)
                c2 = abs(u - v)
                if 0 < c2 <= max_candidate:
                    candidates_to_try.add(c2)
        
        for x in candidates_to_try:
            if x in U:
                continue
            
            # Compute new differences and sums
            new_diffs = set()
            new_sums = set()
            for u in U:
                new_diffs.add(x - u)
                new_diffs.add(u - x)
                new_sums.add(x + u)
            
            added_diffs = len(new_diffs - diffs)
            added_sums = len(new_sums - sums)
            
            new_max = max(max(U), x)
            old_log_span = math.log(2 * max(U) + 1) if max(U) > 0 else 0
            new_log_span = math.log(2 * new_max + 1)
            
            if new_log_span == old_log_span:
                # No increase in span, marginal yield is infinite if we add diffs
                if added_diffs > 0:
                    marginal = float('inf')
                else:
                    marginal = -float('inf')
            else:
                span_increase = new_log_span - old_log_span
                if span_increase <= 0:
                    marginal = -float('inf')
                else:
                    # Marginal yield: new distinct differences / log span increase
                    # We want to maximize this
                    marginal = added_diffs / span_increase
            
            if marginal > best_marginal:
                best_marginal = marginal
                best_x = x
                best_new_diffs = new_diffs
                best_new_sums = new_sums
                best_new_max = new_max
        
        if best_x is None or best_marginal <= 0:
            break
        
        # Add the best candidate
        U.append(best_x)
        U.sort()
        diffs.update(best_new_diffs)
        sums.update(best_new_sums)
        
        # Periodically check score and keep track
        # But we need to be careful about time
    
    return U

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    # Known good candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 10],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 36, 40],
        [0, 1, 4, 10, 18, 23, 29, 36, 45, 52],
    ]
    
    best_set = candidates[0]
    best_score = _score_set(best_set)
    
    # Try greedy builds with different seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        
        U = _greedy_build(seed, max_elements=15, deadline=deadline)
        if not U or U[0] != 0:
            continue
        
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Local search: try swapping or removing elements from best_set
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            U = best_set[:]
            n = len(U)
            
            # Try removing each element
            for i in range(n):
                if time.time() > deadline:
                    break
                U_test = U[:i] + U[i+1:]
                if not U_test or U_test[0] != 0:
                    continue
                s = _score_set(U_test)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = U_test
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try replacing each element with a nearby value
            U = best_set[:]
            for i in range(1, len(U)):  # Don't change 0
                if time.time() > deadline:
                    break
                u_val = U[i]
                for delta in range(1, 10):
                    for new_val in [u_val - delta, u_val + delta]:
                        if new_val <= 0 or new_val == u_val:
                            continue
                        if new_val in U:
                            continue
                        U_test = U[:]
                        U_test[i] = new_val
                        U_test.sort()
                        s = _score_set(U_test)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = U_test
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best_set
