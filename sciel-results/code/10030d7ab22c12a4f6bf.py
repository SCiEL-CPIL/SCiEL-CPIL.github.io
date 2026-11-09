import time
import numpy as np
from itertools import product

# Global state
_init_done = False
_best_strategy = "geo_hash"
_start_time = 0.0

def _get_priority_func(strategy_name):
    """
    Returns a function that computes priority for a given vector and n.
    """
    if strategy_name == "geo_hash":
        def prio(vector, n):
            # 1. Primary: Balance (vectors with sum close to n are 'central')
            #    In {0,1,2}^n, sum=n is the most 'balanced'.
            s = sum(vector)
            balance_score = -abs(s - n)
            
            # 2. Secondary: Geometric Dispersion via Hash
            #    We want to break ties in a way that spreads points out.
            #    A simple hash is deterministic but arbitrary.
            #    We can use a hash of the tuple to get a pseudo-random float in [0, 1).
            #    By adding this to the balance score with a small weight, 
            #    we ensure that among all 'balanced' vectors, we pick a diverse set.
            #    However, standard hash might be too random.
            #    Let's try a hash based on the vector's coordinates interpreted as a number base 3.
            #    This maps the vector to a unique integer.
            #    Then we hash that integer.
            
            # Fast hash: treat vector as base-3 digits
            h = 0
            for v in vector:
                h = h * 3 + v
            
            # Use Python's hash or a simple modulo for distribution
            # We want higher priority to be picked first.
            # So we want the hash to be high for "good" dispersion?
            # Actually, since balance_score is the dominant term, the hash just breaks ties.
            # Let's use a normalized hash value.
            tie_break = (h % 100000) / 100000.0
            
            return balance_score + 0.01 * tie_break
        return prio

    elif strategy_name == "geo_hash_v2":
        def prio(vector, n):
            # Alternative: Prioritize vectors that are 'far' from the origin in a weighted sense?
            # Or prioritize vectors with specific coordinate patterns?
            # Let's try: Maximize the number of 1s? No, that's not balanced.
            # Let's try: Minimize the variance of coordinates?
            # Variance = mean(x^2) - mean(x)^2
            # For {0,1,2}, x^2 = x for 0,1 and 4 for 2.
            # This is computationally more expensive.
            
            # Let's stick to a slightly different tie-breaker:
            # XOR of coordinates?
            xor_val = 0
            for v in vector:
                xor_val ^= v
            # xor_val is in {0,1,2,3}. Not very distinct.
            
            # Let's use a linear combination of coordinates with irrational weights?
            # w1=1, w2=sqrt(2), w3=sqrt(3)?
            # This ensures that vectors with different coordinate counts have distinct priorities.
            w1 = 1.0
            w2 = 1.4142
            w3 = 1.7320
            
            score = 0.0
            for v in vector:
                if v == 1:
                    score += w1
                elif v == 2:
                    score += w2
                # v==0 adds 0
            
            # We want to maximize 'balance', so we want sum close to n?
            # Actually, if we just maximize this weighted sum, we favor 2s and 1s over 0s.
            # This might bias the set towards higher sums.
            # Let's combine with balance.
            s = sum(vector)
            balance_score = -abs(s - n)
            
            return balance_score + 0.01 * (score / n)
        return prio

    else:
        # Fallback to simple sum
        def prio(vector, n):
            return sum(vector)
        return prio

def run_greedy(n, priority_func):
    """
    Runs the greedy algorithm using the provided priority function.
    """
    vectors = list(product([0, 1, 2], repeat=n))
    
    # Calculate priorities
    # For n > 8, this might be slow if priority_func is complex, but our funcs are O(n).
    # 3^8 = 6561, 3^9 = 19683. 19683 * 9 ops is ~177k ops. Very fast.
    
    priorities = [priority_func(v, n) for v in vectors]
    
    # Sort indices by descending priority
    indexed = sorted(range(len(vectors)), key=lambda i: -priorities[i])
    
    S = set()
    
    for idx in indexed:
        v = vectors[idx]
        v_tuple = tuple(v)
        
        can_add = True
        # Check against all existing points in S
        for u in S:
            # Check if v, u, w form an AP
            # w = 2v - u (mod 3)
            w_candidate = tuple((2*vi - ui) % 3 for vi, ui in zip(v_tuple, u))
            if w_candidate in S:
                can_add = False
                break
        
        if can_add:
            S.add(v_tuple)
            
    return len(S)

def priority(vector, n):
    global _init_done, _best_strategy, _start_time
    
    if not _init_done:
        _start_time = time.time()
        deadline = _start_time + 10.0  # 10s for tuning
        
        # We will test a few strategies on a smaller n to see which yields a larger cap set.
        # The choice of strategy might depend on n.
        # For small n, we can test more.
        
        test_n = min(n, 7) # Test on n=7 to be representative but fast
        
        strategies = {
            "geo_hash": _get_priority_func("geo_hash"),
            "geo_hash_v2": _get_priority_func("geo_hash_v2"),
            "simple_sum": lambda v, n: sum(v)
        }
        
        best_score = -1
        best_name = "geo_hash"
        
        for name, func in strategies.items():
            if time.time() > deadline:
                break
            try:
                score = run_greedy(test_n, func)
                if score > best_score:
                    best_score = score
                    best_name = name
            except Exception:
                pass
        
        _best_strategy = best_name
        _init_done = True
        
    # Use the best strategy found
    if _best_strategy == "geo_hash":
        return _get_priority_func("geo_hash")(vector, n)
    elif _best_strategy == "geo_hash_v2":
        return _get_priority_func("geo_hash_v2")(vector, n)
    else:
        return sum(vector)

# Ensure the function is exposed correctly
# The evaluator calls priority(vector, n)
