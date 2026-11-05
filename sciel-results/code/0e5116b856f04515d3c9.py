"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit
    
    N = 512
    
    def get_score(chosen):
        return sum(1.0 / (N - a) for a in chosen)
    
    def is_coprime_to_all(a, chosen):
        return all(gcd(a, b) == 1 for b in chosen)
    
    # Greedy initialization: pick from 511 down
    chosen = []
    for a in range(N - 1, 0, -1):
        if is_coprime_to_all(a, chosen):
            chosen.append(a)
    
    best_score = get_score(chosen)
    best_set = list(chosen)
    chosen_set = set(chosen)
    
    # Precompute weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    def try_1_swap(chosen, chosen_set, best_score):
        """Try single-element replacement swaps."""
        improved = False
        for i in range(len(chosen)):
            if time.time() - start_time > DEADLINE:
                return chosen, chosen_set, best_score, True
            removed = chosen[i]
            others = [chosen[j] for j in range(len(chosen)) if j != i]
            
            # Try adding a better number (larger weight)
            for a in range(N - 1, 0, -1):
                if a in chosen_set or a == removed:
                    continue
                if not is_coprime_to_all(a, others):
                    continue
                new_score = best_score - weights[removed] + weights[a]
                if new_score > best_score + 1e-12:
                    chosen[i] = a
                    chosen_set.remove(removed)
                    chosen_set.add(a)
                    best_score = new_score
                    improved = True
                    break  # restart with new set
            if improved:
                break
        return chosen, chosen_set, best_score, improved
    
    def try_2_for_1_swap(chosen, chosen_set, best_score):
        """Try replacing two elements with one larger element."""
        improved = False
        n = len(chosen)
        for i in range(n):
            if time.time() - start_time > DEADLINE:
                return chosen, chosen_set, best_score, True
            for j in range(i + 1, n):
                if time.time() - start_time > DEADLINE:
                    return chosen, chosen_set, best_score, True
                removed1 = chosen[i]
                removed2 = chosen[j]
                others = [chosen[k] for k in range(n) if k != i and k != j]
                
                # Try adding a single element that is coprime to all others
                # and has higher combined weight than removed1 + removed2
                combined_weight_removed = weights[removed1] + weights[removed2]
                
                for a in range(N - 1, 0, -1):
                    if a in chosen_set or a == removed1 or a == removed2:
                        continue
                    if not is_coprime_to_all(a, others):
                        continue
                    if weights[a] > combined_weight_removed + 1e-12:
                        # Replace both with a
                        new_chosen = [chosen[k] for k in range(n) if k != i and k != j]
                        new_chosen.append(a)
                        chosen_set.remove(removed1)
                        chosen_set.remove(removed2)
                        chosen_set.add(a)
                        chosen[:] = new_chosen
                        best_score = get_score(chosen)
                        improved = True
                        return chosen, chosen_set, best_score, improved
        return chosen, chosen_set, best_score, improved
    
    def try_1_for_2_swap(chosen, chosen_set, best_score):
        """Try replacing one element with two smaller elements."""
        improved = False
        n = len(chosen)
        for i in range(n):
            if time.time() - start_time > DEADLINE:
                return chosen, chosen_set, best_score, True
            removed = chosen[i]
            others = [chosen[j] for j in range(n) if j != i]
            
            # Try adding two elements that are coprime to all others and to each other
            # and have higher combined weight than removed
            for a1 in range(N - 1, 0, -1):
                if time.time() - start_time > DEADLINE:
                    return chosen, chosen_set, best_score, True
                if a1 in chosen_set or a1 == removed:
                    continue
                if not is_coprime_to_all(a1, others):
                    continue
                for a2 in range(a1 - 1, 0, -1):
                    if a2 in chosen_set or a2 == removed or a2 == a1:
                        continue
                    if not is_coprime_to_all(a2, others + [a1]):
                        continue
                    if weights[a1] + weights[a2] > weights[removed] + 1e-12:
                        # Replace removed with a1 and a2
                        new_chosen = [chosen[j] for j in range(n) if j != i]
                        new_chosen.extend([a1, a2])
                        chosen_set.remove(removed)
                        chosen_set.add(a1)
                        chosen_set.add(a2)
                        chosen[:] = new_chosen
                        best_score = get_score(chosen)
                        improved = True
                        return chosen, chosen_set, best_score, improved
        return chosen, chosen_set, best_score, improved
    
    # Main search loop
    max_iterations = 1000
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try 1-swap
        chosen, chosen_set, best_score, improved = try_1_swap(chosen, chosen_set, best_score)
        if improved:
            continue
        
        # Try 2-for-1 swap
        chosen, chosen_set, best_score, improved = try_2_for_1_swap(chosen, chosen_set, best_score)
        if improved:
            continue
        
        # Try 1-for-2 swap
        chosen, chosen_set, best_score, improved = try_1_for_2_swap(chosen, chosen_set, best_score)
        if improved:
            continue
        
        # No improvement, stop
        break
    
    # Return the best set found
    return list(chosen)
# EVOLVE-BLOCK-END
