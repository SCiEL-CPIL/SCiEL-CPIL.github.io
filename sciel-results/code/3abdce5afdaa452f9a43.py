import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal fractions
    recips = [Fraction(1, d) for d in denominators]
    
    # Greedy initialization: sort by reciprocal descending, add if sum < 1
    # This gives a good starting point close to 1
    order = sorted(range(n), key=lambda i: -recips[i])
    greedy_set = set()
    greedy_sum = Fraction(0, 1)
    for i in order:
        if greedy_sum + recips[i] < 1:
            greedy_sum += recips[i]
            greedy_set.add(i)
    
    # Local search: try to improve by adding, removing, or swapping
    best_sum = greedy_sum
    best_set = set(greedy_set)
    
    def try_improve(current_set, current_sum):
        """Try single-element moves to improve (increase sum while staying < 1)"""
        improved = False
        # Try adding elements not in set
        for i in range(n):
            if i not in current_set:
                new_sum = current_sum + recips[i]
                if new_sum < 1 and new_sum > best_sum:
                    best_set = set(current_set)
                    best_set.add(i)
                    best_sum = new_sum
                    improved = True
        # Try removing and re-adding (swap)
        for i in list(current_set):
            for j in range(n):
                if j in current_set:
                    continue
                new_sum = current_sum - recips[i] + recips[j]
                if new_sum < 1 and new_sum > best_sum:
                    best_set = set(current_set)
                    best_set.remove(i)
                    best_set.add(j)
                    best_sum = new_sum
                    improved = True
        return improved, best_set, best_sum
    
    # Run local search until no improvement or time runs out
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved, best_set, best_sum = try_improve(best_set, best_sum)
    
    # Try multiple random restarts with perturbation
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb: start from best, remove 1-3 random elements, re-greedy
        perturbed_set = set(best_set)
        remove_count = random.randint(1, 3)
        to_remove = random.sample(list(perturbed_set), min(remove_count, len(perturbed_set)))
        for i in to_remove:
            perturbed_set.discard(i)
        
        # Re-greedy from perturbed state
        perturbed_sum = Fraction(0, 1)
        for i in perturbed_set:
            perturbed_sum += recips[i]
        
        # Try adding more
        for i in range(n):
            if i not in perturbed_set:
                if perturbed_sum + recips[i] < 1:
                    perturbed_sum += recips[i]
                    perturbed_set.add(i)
        
        if perturbed_sum > best_sum:
            best_sum = perturbed_sum
            best_set = perturbed_set
        
        # Local search from perturbed state
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved, perturbed_set, perturbed_sum = try_improve(perturbed_set, perturbed_sum)
            if perturbed_sum > best_sum:
                best_sum = perturbed_sum
                best_set = perturbed_set
    
    # Final verification
    result = sorted(best_set)
    if not result:
        # Fallback: just take {2,3,7,43} which sums to 1/2+1/3+1/7+1/43 = 0.999...
        # Actually let's use a known good subset
        result = [0, 1, 6, 42]  # denominators 2,3,7,43
        total = sum(recips[i] for i in result)
        if total >= 1:
            result = [0]  # Just 1/2
    
    # Verify strictly below 1
    final_sum = sum(recips[i] for i in result)
    if final_sum >= 1:
        # Remove largest element until valid
        while final_sum >= 1 and len(result) > 1:
            max_idx = max(result, key=lambda i: recips[i])
            result.remove(max_idx)
            final_sum = sum(recips[i] for i in result)
    
    if not result:
        result = [0]
    
    return result

if __name__ == "__main__":
    res = construct()
    pass
