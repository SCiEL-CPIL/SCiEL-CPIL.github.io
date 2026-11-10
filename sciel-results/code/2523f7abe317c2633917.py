"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s/90s kill
    n = 192
    max_k = 48
    
    # Precompute all sums of 3 distinct integers in [1, n]
    # For each integer a, we need to check if a divides any such sum
    # where the 3 summands are distinct from a and from each other.
    
    # Strategy: Greedy with backtracking, trying to maximize cardinality.
    # We iterate candidates in a smart order.
    
    # First, let's try a simple greedy approach but with better ordering.
    # The key insight: larger numbers are less likely to divide sums of smaller numbers?
    # Actually, smaller numbers are more likely to divide sums.
    # So we want to include large numbers and avoid small numbers that divide many sums.
    
    # Let's precompute for each candidate x, how "conflicting" it is.
    # A candidate x conflicts with a set S if x divides the sum of any 3 distinct elements of S\{x}.
    
    # Better approach: Use a greedy algorithm that picks elements one by one,
    # always choosing the element that causes the fewest future conflicts,
    # or simply try to build a large set using backtracking with pruning.
    
    # Given the time limit, let's use a randomized greedy with restarts.
    
    import random
    
    best_set = []
    best_size = 0
    
    def is_valid_set(s):
        """Check if set s is valid: no element divides sum of any 3 distinct others."""
        if len(s) < 4:
            return True
        s_list = list(s)
        for i, a in enumerate(s_list):
            others = [v for j, v in enumerate(s_list) if j != i]
            for combo in combinations(others, 3):
                if sum(combo) % a == 0:
                    return False
        return True
    
    def can_add(x, current_set):
        """Check if x can be added to current_set without violating constraints."""
        # x must not divide sum of any 3 distinct elements of current_set
        # And each element in current_set must not divide sum of x + any 2 distinct elements of current_set
        # Wait, the condition is: for every a in final_set, a does not divide sum of any 3 distinct others in final_set.
        
        # When we add x to current_set, new violations can only involve x or existing elements with x in the sum.
        
        # 1. Check if x divides sum of any 3 distinct elements of current_set
        if len(current_set) >= 3:
            for combo in combinations(current_set, 3):
                if sum(combo) % x == 0:
                    return False
        
        # 2. For each a in current_set, check if a divides sum of any 3 distinct elements from (current_set + [x]) \ {a}
        # The new triples involving x are: {x, u, v} where u, v are distinct elements of current_set \ {a}
        for a in current_set:
            others = [v for v in current_set if v != a]
            # Triples with x: {x, u, v} where u, v in others, u != v
            if len(others) >= 2:
                for combo in combinations(others, 2):
                    s = x + combo[0] + combo[1]
                    if s % a == 0:
                        return False
        
        return True
    
    # Greedy with randomization and multiple restarts
    for restart in range(200):
        if time.time() > deadline:
            break
        
        current_set = []
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        # Sort candidates by some heuristic? Maybe larger numbers first?
        # Let's try different orderings
        if restart % 3 == 0:
            candidates.sort(reverse=True)
        elif restart % 3 == 1:
            candidates.sort()
        else:
            pass  # keep shuffled
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, current_set):
                current_set.append(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        if best_size >= max_k:
            break
    
    # Try to improve with local search
    if time.time() < deadline:
        # Try to add more elements
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x not in best_set and can_add(x, best_set):
                best_set.append(x)
                best_size = len(best_set)
                if best_size >= max_k:
                    break
    
    # Try to swap elements to increase size?
    # This is complex. Let's stick with the greedy result.
    
    return best_set
# EVOLVE-BLOCK-END
