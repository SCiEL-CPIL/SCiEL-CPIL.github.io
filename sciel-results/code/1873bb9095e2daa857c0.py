import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    N = 512
    MAX_A = 511
    
    # Precompute prime factors for each number 1..511
    # factors[i] is a set of prime factors of i
    factors = [set() for _ in range(MAX_A + 1)]
    factors[1] = set()
    for i in range(2, MAX_A + 1):
        if not factors[i]:
            # i is prime
            for j in range(i, MAX_A + 1, i):
                factors[j].add(i)
    
    # Weights
    weights = [0.0] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        weights[a] = 1.0 / (N - a)
    
    def get_score(S):
        return sum(weights[a] for a in S)
    
    def is_coprime(a, b):
        # Returns True if a and b are coprime
        if a == 1 or b == 1:
            return True
        # Check if they share any prime factor
        # Since we precomputed factors, we can check intersection
        # For small numbers, set intersection is fast
        return len(factors[a] & factors[b]) == 0

    def is_valid_addition(candidate, current_set):
        for x in current_set:
            if not is_coprime(candidate, x):
                return False
        return True

    def local_search(initial_set):
        S = set(initial_set)
        current_score = get_score(S)
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            best_gain = 0
            best_swap = None
            
            # Try all swaps (x in S, y not in S)
            # To optimize, we can iterate x in S
            for x in list(S):
                # For each x, we want to find y not in S such that 
                # y is coprime to all z in S \ {x}
                # and weights[y] > weights[x]
                
                # We can iterate y from MAX_A down to 1
                # Since weights are increasing, larger y is better
                # But we need to check validity.
                
                # Optimization: 
                # If we remove x, the remaining set R = S \ {x} must be coprime to y.
                
                # Let's iterate y in descending order of weight (i.e., descending y)
                # and pick the first valid one? 
                # No, we need the BEST swap globally.
                # However, since weights[y] is monotonic, the best y for a fixed x is the largest valid y.
                # And the best x to remove is the one that allows the largest y?
                # Not necessarily. Removing a small x might allow a large y, 
                # while removing a large x might allow a slightly smaller y but the gain 
                # (w[y] - w[x]) might be smaller if w[x] is large.
                
                # Let's just find the max gain over all pairs.
                
                for y in range(MAX_A, 0, -1):
                    if y in S:
                        continue
                    if weights[y] <= weights[x]:
                        # Since we iterate y descending, and weights increasing, 
                        # if weights[y] <= weights[x], all smaller y will also have <= weight.
                        # But wait, we are iterating y descending. 
                        # If y is small, weight is small. 
                        # We want y > x roughly.
                        # If y <= x, weight[y] <= weight[x], gain <= 0. 
                        # So we can skip y <= x.
                        continue
                    
                    # Check if y is coprime to all z in S \ {x}
                    valid = True
                    for z in S:
                        if z == x:
                            continue
                        if not is_coprime(y, z):
                            valid = False
                            break
                    
                    if valid:
                        gain = weights[y] - weights[x]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (x, y)
            
            if best_swap:
                x, y = best_swap
                S.remove(x)
                S.add(y)
                current_score += best_gain
                improved = True
        
        return S, current_score

    best_set = set()
    best_score = 0.0
    
    # Strategy 1: Standard Greedy (Largest to Smallest)
    S_greedy = set()
    for a in range(MAX_A, 0, -1):
        if is_valid_addition(a, S_greedy):
            S_greedy.add(a)
    
    S_final, score = local_search(S_greedy)
    if score > best_score:
        best_score = score
        best_set = S_final
    
    # Strategy 2: Randomized Greedy with multiple restarts
    # We bias the random choice towards larger numbers
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Create a list of candidates and shuffle with bias
        # Or simply pick random order but sort by weight with some noise?
        # Let's try: Shuffle the list of numbers, but keep a bias.
        # A simple effective heuristic: Randomly pick a starting point, then greedily add largest coprime neighbors?
        # Or just random order greedy.
        
        candidates = list(range(1, MAX_A + 1))
        random.shuffle(candidates)
        
        S_rand = set()
        # To ensure we get good solutions, we should still prefer larger numbers.
        # Let's do a "noisy" greedy: 
        # Sort by weight, but swap adjacent elements with small probability?
        # Or just run greedy on a randomly permuted list? That might be weak.
        
        # Better: 
        # Start with empty set.
        # Repeatedly pick the largest available number that is coprime to current set.
        # But to diversify, we can randomly remove one element from the set and re-optimize?
        
        # Let's try a different initialization:
        # Pick a random large number, then greedily fill.
        # Or, pick a random subset of "large" numbers and try to extend?
        
        # Let's stick to a simple diversification:
        # Randomly shuffle the order of addition in the greedy step, 
        # but weight the selection by 1/weight? No.
        
        # Let's try: 
        # 1. Greedy from largest to smallest.
        # 2. Randomly remove 1-3 elements.
        # 3. Run local search.
        
        S_curr = set(S_greedy)
        # Randomly remove 1 to 3 elements
        num_removals = random.randint(1, 3)
        if len(S_curr) > num_removals:
            to_remove = random.sample(list(S_curr), num_removals)
            for r in to_remove:
                S_curr.remove(r)
        
        S_final, score = local_search(S_curr)
        if score > best_score:
            best_score = score
            best_set = S_final
            
    # Additional restarts if time permits
    # Try random order greedy?
    # It's unlikely to beat the greedy + local search, but let's try a few.
    for i in range(5):
        if time.time() > deadline - 5:
            break
        
        candidates = list(range(1, MAX_A + 1))
        random.shuffle(candidates)
        S_rand = set()
        for a in candidates:
            if is_valid_addition(a, S_rand):
                S_rand.add(a)
        
        S_final, score = local_search(S_rand)
        if score > best_score:
            best_score = score
            best_set = S_final
            
    return list(best_set)
