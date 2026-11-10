# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    # Precompute all forbidden triples
    # Triple (a,b,c) is forbidden if lcm(a,b)==lcm(a,c)==lcm(b,c)
    # For speed, precompute pairwise LCMs
    lcm_table = {}
    for i in range(n):
        for j in range(i+1, n):
            lcm_table[(i, j)] = lcm(elements[i], elements[j])
    
    # Find all forbidden triples
    forbidden = set()
    triple_count = {}  # element index -> count of forbidden triples it's in
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                l1 = lcm_table[(i, j)]
                l2 = lcm_table[(i, k)]
                l3 = lcm_table[(j, k)]
                if l1 == l2 == l3:
                    forbidden.add((i, j, k))
                    triple_count[i] = triple_count.get(i, 0) + 1
                    triple_count[j] = triple_count.get(j, 0) + 1
                    triple_count[k] = triple_count.get(k, 0) + 1
    
    # Check if a set of indices contains any forbidden triple
    def has_forbidden_triple(idx_set):
        """Check if the set of indices contains any forbidden triple."""
        idx_list = sorted(idx_set)
        for i in range(len(idx_list)):
            for j in range(i+1, len(idx_list)):
                for k in range(j+1, len(idx_list)):
                    if (idx_list[i], idx_list[j], idx_list[k]) in forbidden:
                        return True
        return False
    
    # Faster check: for a given set, check if adding element x creates a forbidden triple
    def creates_forbidden_triple(existing_set, x):
        """Check if adding x to existing_set creates any forbidden triple."""
        for a, b in combinations(sorted(existing_set), 2):
            if (a, b, x) in forbidden or (a, x, b) in forbidden or (b, a, x) in forbidden:
                return True
        return False
    
    def is_valid_set(idx_set):
        """Check if a set of indices is valid (no forbidden triples)."""
        idx_list = sorted(idx_set)
        for i in range(len(idx_list)):
            for j in range(i+1, len(idx_list)):
                for k in range(j+1, len(idx_list)):
                    if (idx_list[i], idx_list[j], idx_list[k]) in forbidden:
                        return False
        return True
    
    # Greedy removal: start with all elements, remove those in most triples
    def greedy_remove():
        remaining = set(range(n))
        # Iteratively remove the element that's in the most forbidden triples
        while True:
            if time.time() > deadline:
                break
            # Count how many forbidden triples each remaining element is in
            counts = {}
            for (a, b, c) in forbidden:
                if a in remaining and b in remaining and c in remaining:
                    counts[a] = counts.get(a, 0) + 1
                    counts[b] = counts.get(b, 0) + 1
                    counts[c] = counts.get(c, 0) + 1
            if not counts:
                break
            # Remove the element with highest count (tie-break by index)
            max_count = max(counts.values())
            candidates = [e for e, c in counts.items() if c == max_count]
            to_remove = min(candidates)
            remaining.remove(to_remove)
        return remaining
    
    # Local search: try 1-swaps (remove one, add one) and 2-swaps
    def local_search(initial_set, max_iterations=5000):
        current = set(initial_set)
        current_size = len(current)
        best = set(current)
        best_size = current_size
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try 1-swaps: remove one element, add another
            improved = False
            for rem in list(current):
                for add in range(n):
                    if add in current:
                        continue
                    new_set = (current - {rem}) | {add}
                    if len(new_set) != current_size:
                        continue
                    # Check validity
                    if is_valid_set(new_set):
                        # Try to extend: after swap, try to add more elements
                        temp_set = set(new_set)
                        # Greedily try to add elements
                        for candidate in range(n):
                            if candidate not in temp_set:
                                if is_valid_set(temp_set | {candidate}):
                                    temp_set.add(candidate)
                        if len(temp_set) > best_size:
                            best = set(temp_set)
                            best_size = len(temp_set)
                            current = set(temp_set)
                            current_size = len(current)
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        
        return best
    
    # Run greedy removal multiple times with different tie-breaking
    best_result = set()
    best_size = 0
    
    # First, try standard greedy removal
    result = greedy_remove()
    best_result = set(result)
    best_size = len(result)
    
    # Now try local search from this result
    if time.time() < deadline:
        improved = local_search(result, max_iterations=2000)
        if len(improved) > best_size:
            best_result = set(improved)
            best_size = len(improved)
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Randomized greedy removal
        remaining = set(range(n))
        while True:
            if time.time() > deadline - 5:
                break
            counts = {}
            for (a, b, c) in forbidden:
                if a in remaining and b in remaining and c in remaining:
                    counts[a] = counts.get(a, 0) + 1
                    counts[b] = counts.get(b, 0) + 1
                    counts[c] = counts.get(c, 0) + 1
            if not counts:
                break
            max_count = max(counts.values())
            candidates = [e for e, c in counts.items() if c == max_count]
            # Random selection among top candidates
            to_remove = random.choice(candidates)
            remaining.remove(to_remove)
        
        if len(remaining) > best_size:
            best_result = set(remaining)
            best_size = len(remaining)
        
        # Local search from this result
        if time.time() < deadline - 10:
            improved = local_search(remaining, max_iterations=1000)
            if len(improved) > best_size:
                best_result = set(improved)
                best_size = len(improved)
    
    # Convert back to values
    result_values = [elements[i] for i in sorted(best_result)]
    
    # Ensure nonempty
    if not result_values:
        result_values = [1]
    
    return result_values

# EVOLVE-BLOCK-END
