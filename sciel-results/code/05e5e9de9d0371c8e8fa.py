import time
import random
from math import gcd

def construct():
    start_time = time.time()
    timeout = 140  # seconds
    
    N = 512
    # Precompute weights
    weights = {a: 1.0 / (N - a) for a in range(1, N)}
    
    # Precompute prime factors or just use gcd
    # Using gcd is fast enough for < 100 elements
    
    def get_score(S):
        return sum(weights[a] for a in S)
        
    def is_coprime_to_all(a, S):
        for b in S:
            if gcd(a, b) != 1:
                return False
        return True
        
    def is_valid_set(S):
        # Check pairwise coprime
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if gcd(S_list[i], S_list[j]) != 1:
                    return False
        return True
        
    # 1. Greedy Initial Solution
    def greedy_construct():
        S = set()
        # Sort by weight descending (which is a ascending)
        candidates = sorted(range(1, N), key=lambda x: -weights[x])
        for a in candidates:
            if is_coprime_to_all(a, S):
                S.add(a)
        return S
        
    best_S = greedy_construct()
    best_score = get_score(best_S)
    
    # 2. Local Search with Restarts
    def local_search(S):
        S = set(S)
        current_score = get_score(S)
        improved = True
        while improved:
            improved = False
            # Try to add
            # We want to add the best candidate that fits
            # Candidates are those not in S
            # To speed up, we can iterate all a in 1..511 not in S
            # But checking all 511 every time is O(511 * |S|). |S| ~ 100. 51100 ops.
            # This is fast.
            
            best_add = None
            best_add_gain = 0
            for a in range(1, N):
                if a in S:
                    continue
                if is_coprime_to_all(a, S):
                    gain = weights[a]
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = a
            
            if best_add is not None:
                S.add(best_add)
                current_score += best_add_gain
                improved = True
                continue # Restart loop to check more additions or swaps
                
            # Try to swap
            # Remove b from S, add a not in S
            # Gain = weight[a] - weight[b]
            # Must be coprime to S \ {b}
            
            best_swap = None
            best_swap_gain = 0
            S_list = list(S)
            for b in S_list:
                S_minus_b = S - {b}
                # Check all a not in S
                # Optimization: Only consider a with weight[a] > weight[b]
                for a in range(1, N):
                    if a in S:
                        continue
                    if weights[a] <= weights[b]:
                        continue
                    if is_coprime_to_all(a, S_minus_b):
                        gain = weights[a] - weights[b]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (a, b)
                            
            if best_swap is not None:
                a, b = best_swap
                S.remove(b)
                S.add(a)
                current_score += best_swap_gain
                improved = True
                continue # Restart loop
                
        return S, current_score
        
    # Run local search on initial
    best_S, best_score = local_search(best_S)
    
    # Random Restarts
    # We need to ensure we don't just loop.
    # We will perform several restarts.
    # Each restart: Start with a subset of the best solution or random large numbers.
    
    # Strategy: 
    # 1. Keep the best solution found so far.
    # 2. Perturb the best solution by removing some random elements.
    # 3. Run local search on the perturbed set.
    
    num_restarts = 0
    while time.time() - start_time < timeout:
        # Perturb
        S_perturbed = set(best_S)
        # Remove 5-10 random elements
        removal_count = random.randint(5, 10)
        to_remove = random.sample(list(S_perturbed), min(removal_count, len(S_perturbed)))
        for a in to_remove:
            S_perturbed.discard(a)
            
        # Also, maybe add some random large numbers to break local maxima?
        # Local search will add them if beneficial.
        
        S_new, score_new = local_search(S_perturbed)
        
        if score_new > best_score:
            best_score = score_new
            best_S = S_new
            num_restarts = 0
        else:
            num_restarts += 1
            # If we've tried many times without improvement, maybe change strategy?
            # For now, just keep going.
            
    return list(best_S)

# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd

    start_time = time.time()
    timeout = 140  # seconds
    
    N = 512
    # Precompute weights
    weights = {a: 1.0 / (N - a) for a in range(1, N)}
    
    def get_score(S):
        return sum(weights[a] for a in S)
        
    def is_coprime_to_all(a, S):
        for b in S:
            if gcd(a, b) != 1:
                return False
        return True
        
    # 1. Greedy Initial Solution
    def greedy_construct():
        S = set()
        # Sort by weight descending (which is a ascending)
        candidates = sorted(range(1, N), key=lambda x: -weights[x])
        for a in candidates:
            if is_coprime_to_all(a, S):
                S.add(a)
        return S
        
    best_S = greedy_construct()
    best_score = get_score(best_S)
    
    # 2. Local Search
    def local_search(S):
        S = set(S)
        current_score = get_score(S)
        improved = True
        while improved:
            improved = False
            # Try to add
            best_add = None
            best_add_gain = 0
            for a in range(1, N):
                if a in S:
                    continue
                if is_coprime_to_all(a, S):
                    gain = weights[a]
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = a
            
            if best_add is not None:
                S.add(best_add)
                current_score += best_add_gain
                improved = True
                continue
                
            # Try to swap
            best_swap = None
            best_swap_gain = 0
            S_list = list(S)
            for b in S_list:
                S_minus_b = S - {b}
                for a in range(1, N):
                    if a in S:
                        continue
                    if weights[a] <= weights[b]:
                        continue
                    if is_coprime_to_all(a, S_minus_b):
                        gain = weights[a] - weights[b]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (a, b)
                            
            if best_swap is not None:
                a, b = best_swap
                S.remove(b)
                S.add(a)
                current_score += best_swap_gain
                improved = True
                continue
                
        return S, current_score
        
    # Run local search on initial
    best_S, best_score = local_search(best_S)
    
    # Random Restarts
    while time.time() - start_time < timeout:
        # Perturb
        S_perturbed = set(best_S)
        removal_count = random.randint(5, 10)
        if len(S_perturbed) > 0:
            to_remove = random.sample(list(S_perturbed), min(removal_count, len(S_perturbed)))
            for a in to_remove:
                S_perturbed.discard(a)
        
        S_new, score_new = local_search(S_perturbed)
        
        if score_new > best_score:
            best_score = score_new
            best_S = S_new
            
    return list(best_S)
# EVOLVE-BLOCK-END
