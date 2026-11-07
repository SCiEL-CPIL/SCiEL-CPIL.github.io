import itertools
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 12
    m = 40
    
    # Precompute all subsets as bitmasks
    all_subsets = []
    for mask in range(1 << n):
        all_subsets.append(mask)
    
    # Precompute subset relationships
    # is_subset[a][b] = True if a is subset of b (a <= b in terms of set inclusion)
    # We need strict subset: a < b means a is proper subset of b
    # For efficiency, precompute for all pairs? That's 4096^2 = 16M entries, might be too much.
    # Instead, compute on the fly with bit operations.
    
    def is_strict_subset(a, b):
        """Check if a is a strict subset of b."""
        return (a & b) == a and a != b
    
    def is_comparable(a, b):
        """Check if a and b are comparable (one is strict subset of the other)."""
        if a == b:
            return False
        return (a & b) == a or (a & b) == b
    
    def count_comparable(family):
        """Count number of unordered pairs that are comparable."""
        count = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if is_comparable(family[i], family[j]):
                    count += 1
        return count
    
    # Greedy initialization: build a chain and then add sets that maximize comparability
    def greedy_init():
        # Start with empty set and full set
        family = [0, (1 << n) - 1]
        remaining = set(all_subsets) - set(family)
        
        while len(family) < m:
            best_set = None
            best_gain = -1
            for s in remaining:
                gain = sum(1 for t in family if is_comparable(s, t))
                if gain > best_gain:
                    best_gain = gain
                    best_set = s
            if best_set is None:
                break
            family.append(best_set)
            remaining.discard(best_set)
        
        # If we still need more, just add arbitrary sets
        while len(family) < m:
            for s in remaining:
                family.append(s)
                remaining.discard(s)
                if len(family) == m:
                    break
        
        return family
    
    # Simulated annealing
    def simulated_annealing(initial_family, max_iters=500000, t_start=1.0, t_end=0.01, alpha=0.99999):
        current = list(initial_family)
        current_score = count_comparable(current)
        best = list(current)
        best_score = current_score
        
        t = t_start
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick two random indices to swap or replace
            i = random.randint(0, m - 1)
            j = random.randint(0, m - 1)
            if i == j:
                continue
            
            # Try swapping or replacing
            if random.random() < 0.5:
                # Swap
                new_family = list(current)
                new_family[i], new_family[j] = new_family[j], new_family[i]
            else:
                # Replace one element
                new_family = list(current)
                new_set = random.choice(all_subsets)
                new_family[i] = new_set
            
            # Check for duplicates
            if len(set(new_family)) != m:
                continue
            
            new_score = count_comparable(new_family)
            delta = new_score - current_score
            
            if delta >= 0 or random.random() < pow(2.718281828, delta / t):
                current = new_family
                current_score = new_score
                if current_score > best_score:
                    best = list(current)
                    best_score = current_score
            
            t = max(t * alpha, t_end)
        
        return best, best_score
    
    # Multi-restart approach
    best_family = None
    best_score = -1
    
    # First, greedy initialization
    greedy_family = greedy_init()
    greedy_score = count_comparable(greedy_family)
    best_family = greedy_family
    best_score = greedy_score
    
    # Try several SA restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Perturb the best solution
        perturbed = list(best_family)
        for _ in range(random.randint(1, 5)):
            idx = random.randint(0, m - 1)
            perturbed[idx] = random.choice(all_subsets)
        
        result_family, result_score = simulated_annealing(perturbed, max_iters=200000)
        
        if result_score > best_score:
            best_family = result_family
            best_score = result_score
    
    # Local search polish: try all possible replacements
    def local_search(family, max_iters=100000):
        current = list(family)
        current_score = count_comparable(current)
        best = list(current)
        best_score = current_score
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            for i in range(m):
                if time.time() > deadline:
                    break
                for new_set in all_subsets:
                    if new_set in current:
                        continue
                    new_family = list(current)
                    new_family[i] = new_set
                    if len(set(new_family)) != m:
                        continue
                    new_score = count_comparable(new_family)
                    if new_score > best_score:
                        best = new_family
                        best_score = new_score
                        current = new_family
                        current_score = new_score
                        improved = True
                        break
        
        return best, best_score
    
    # Final polish
    if time.time() < deadline:
        best_family, best_score = local_search(best_family, max_iters=50000)
    
    # Convert to list of lists of integers
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
