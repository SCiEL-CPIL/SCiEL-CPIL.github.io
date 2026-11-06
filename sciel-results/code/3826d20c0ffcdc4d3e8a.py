import time
import random

def construct():
    start_time = time.time()
    limit = 55.0 # Safety margin
    
    n = 256
    # Weights
    weights = [0.0] + [1.0/i for i in range(1, n+1)]
    
    # Initial Greedy Solution
    # Add numbers in increasing order of index (decreasing weight)
    S = set()
    for x in range(1, n+1):
        # Check if x can be added
        valid = True
        for a in S:
            # Check if a, x, 2x-a is an AP (x is middle or end)
            # Case 1: x is middle. a, x, c -> c = 2x - a
            c = 2*x - a
            if c != x and c in S:
                valid = False
                break
            # Case 2: x is end. a, b, x or x, b, a.
            # If x is the larger end: a, b, x -> 2b = a+x -> b = (a+x)/2
            # If x is the smaller end: x, b, a -> 2b = x+a -> b = (x+a)/2
            # So check if (a+x) is even and (a+x)//2 in S
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b != x and b in S:
                    valid = False
                    break
        if valid:
            S.add(x)
            
    current_score = sum(weights[x] for x in S)
    best_score = current_score
    best_S = set(S)
    
    # Local Search
    # Precompute list of all numbers
    all_nums = list(range(1, n+1))
    
    def is_valid_add(x, S):
        # Check if adding x to S creates a 3-AP
        for a in S:
            c = 2*x - a
            if c != x and c in S:
                return False
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b != x and b in S:
                    return False
        return True

    def calculate_score(S):
        return sum(weights[x] for x in S)

    # Hill Climbing with 1-swaps
    # We will try to improve by removing one and adding one, or just adding one.
    
    while time.time() - start_time < limit:
        improved = False
        
        # Try to add any element
        # To avoid bias, shuffle candidates
        candidates_add = [x for x in all_nums if x not in S]
        random.shuffle(candidates_add)
        
        for x in candidates_add:
            if is_valid_add(x, S):
                S.add(x)
                new_score = calculate_score(S)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                    improved = True
                    # Continue checking other adds in this pass? 
                    # Usually adding one might block another, but we re-evaluate in next loop.
                    # Breaking here might be safer to ensure we don't add conflicting things in same batch?
                    # But is_valid_add checks against current S.
                    # So if we add x, subsequent checks use S with x.
                    # This is fine.
        
        # Try 1-swaps
        # Remove x in S, add y not in S
        # To be efficient, we can iterate y (high weight) and see if removing some x makes it valid.
        # Or iterate x (low weight) and see if adding some y becomes valid.
        
        # Let's try removing low weight elements first
        S_list = list(S)
        S_list.sort(key=lambda x: weights[x]) # Lowest weight first
        
        for x in S_list:
            if time.time() - start_time > limit:
                break
            S.remove(x)
            score_without_x = current_score - weights[x]
            
            # Try to add any y not in S (and y != x)
            # We want to maximize score_without_x + weights[y]
            # So we should try high weight y first.
            candidates_add = [y for y in all_nums if y not in S and y != x]
            # Sort by weight descending
            candidates_add.sort(key=lambda y: weights[y], reverse=True)
            
            found_swap = False
            for y in candidates_add:
                if is_valid_add(y, S):
                    # Check if swap improves score
                    new_score = score_without_x + weights[y]
                    if new_score > current_score:
                        S.add(y)
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
                        improved = True
                        found_swap = True
                        break # Only add one y per x removal attempt to keep it simple?
                             # Or we could try to add multiple? 
                             # Adding multiple might be complex. Let's stick to 1-swap for now.
            if not found_swap:
                S.add(x) # Restore x
        
        if not improved:
            # Stuck. Try random perturbation or 2-swaps?
            # Let's try a random restart or random swap even if it decreases score slightly (Simulated Annealing lite)
            # But for now, let's just break or do a random 1-swap.
            # Given the time, maybe just break if no improvement for a while?
            # Let's try a random 1-swap to escape local minima.
            S_list = list(S)
            if len(S_list) > 0:
                x = random.choice(S_list)
                S.remove(x)
                candidates_add = [y for y in all_nums if y not in S and y != x]
                if candidates_add:
                    y = random.choice(candidates_add)
                    if is_valid_add(y, S):
                        S.add(y)
                        current_score = calculate_score(S)
                        if current_score > best_score:
                            best_score = current_score
                            best_S = set(S)
                    else:
                        S.add(x)
                else:
                    S.add(x)
            else:
                break

    return list(best_S)
