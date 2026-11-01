import time
import sys
import random
from itertools import combinations

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115
    
    # Precompute all valid rows (Sidon sets) up to a reasonable maximum scope
    # For k=5, we have 6 elements. The differences are C(6,2)=15 per row.
    # Total differences = n_rows * 15.
    # We need to find n_rows disjoint sets of 15 differences.
    
    # Let's estimate the max scope. 
    # For (7,5), best known is 112. 
    # Let's search up to 150 to be safe, but we can start lower and expand.
    # Actually, let's just generate all valid rows up to some max M.
    # A valid row is a set {0, a1, ..., ak} such that all differences are distinct.
    
    max_scope = 150
    valid_rows = []
    row_diffs = []  # List of frozensets of differences for each valid row
    
    # Generate all valid rows of length k+1 starting with 0, max element <= max_scope
    # We can do this by iterating through all combinations of k elements from 1..max_scope
    # and checking if {0, a1, ..., ak} is a Sidon set.
    
    # Optimization: 
    # Instead of all combinations, we can generate recursively.
    # But C(150, 5) is about 150^5/120 ~ 1.5 * 10^8, which is too many.
    # We need a smarter way to generate valid rows.
    
    # Alternative: 
    # We don't need ALL valid rows. We just need enough to pick n_rows disjoint ones.
    # We can use a randomized approach to pick rows, or a greedy approach.
    
    # Let's try a hybrid:
    # 1. Generate a large pool of valid rows (randomized).
    # 2. Use a matching algorithm (or greedy) to select n_rows disjoint rows.
    
    # Actually, the "Exact Cover" idea suggests we want to model this as selecting rows such that no difference is shared.
    # This is equivalent to finding a matching in a hypergraph? Or just a set packing problem.
    
    # Let's try a different structural change:
    # Randomized construction with strong local search, but using a different representation.
    # Or, use the fact that for small n_rows, we can just try many random valid rows and pick a compatible set.
    
    # Step 1: Generate a pool of candidate rows.
    # We want rows that are "compact" (small max element) to help minimize scope.
    
    pool = []
    pool_diffs = []
    
    # Generate candidates
    # We'll generate random Sidon sets.
    # How to generate a random Sidon set of size k+1?
    # Start with 0. Add elements one by one, ensuring no differences clash.
    
    def generate_random_sidon_set(k, max_val):
        row = [0]
        used_diffs = set()
        for _ in range(k):
            found = False
            # Try random positions, biased towards smaller values
            # Or just iterate from small to large with random skipping?
            # Let's try a range of values.
            # To get compact sets, we should prefer smaller values.
            # Let's try values from 1 to max_val.
            # Shuffle the order? No, we want compact.
            # Let's pick a random valid next element from a range [current_max+1, current_max + window]
            # If none, expand window.
            
            window = 10
            while window < max_val:
                candidates = []
                start = row[-1] + 1
                end = min(row[-1] + window, max_val)
                
                # Randomize the order of checking in this window?
                # Or just check sequentially?
                # Sequential is faster for finding the smallest valid.
                # But we want diversity.
                # Let's pick a random candidate from the valid ones in the window.
                
                for x in range(start, end + 1):
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                
                if candidates:
                    x = random.choice(candidates)
                    for a in row:
                        used_diffs.add(x - a)
                    row.append(x)
                    found = True
                    break
                else:
                    window *= 2
            
            if not found:
                return None
        
        return row
    
    # Generate pool
    # How many? Maybe 1000-5000.
    # We need n_rows disjoint ones.
    
    pool_size = 2000
    for _ in range(pool_size):
        if time.time() > deadline - 20:
            break
        row = generate_random_sidon_set(k, max_scope)
        if row:
            diffs = frozenset(row[j] - row[i] for i in range(k+1) for j in range(i+1, k+1))
            pool.append(row)
            pool_diffs.append(diffs)
    
    # Now, select n_rows rows from the pool such that their difference sets are disjoint.
    # This is a Set Packing problem.
    # We can use a greedy approach with backtracking, or just random selection.
    
    # Greedy:
    # Sort rows by max element (scope) to prefer compact ones.
    # Then pick one by one, ensuring no overlap in diffs.
    
    # To improve, we can try multiple random orderings or use a better selection.
    
    best_selection = None
    best_max_scope = float('inf')
    
    # Try several times with random shuffling of the pool
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
            
        # Shuffle pool indices
        indices = list(range(len(pool)))
        random.shuffle(indices)
        
        # Sort by scope? Or just pick in random order?
        # If we sort by scope, we might get stuck.
        # Let's try sorting by scope first, then randomizing ties.
        # Or just random order.
        
        selected_rows = []
        selected_diffs = set()
        
        for idx in indices:
            if len(selected_rows) == n_rows:
                break
            
            row = pool[idx]
            diffs = pool_diffs[idx]
            
            if diffs.isdisjoint(selected_diffs):
                selected_rows.append(row)
                selected_diffs.update(diffs)
        
        if len(selected_rows) == n_rows:
            max_scope = max(max(r) for r in selected_rows)
            if max_scope < best_max_scope:
                best_max_scope = max_scope
                best_selection = [r[:] for r in selected_rows]
    
    if best_selection:
        return best_selection
    
    # Fallback: Greedy construction from scratch
    return _greedy_construct(n_rows, k)

def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Search for next element
            # To ensure we find a valid one, we might need to search far.
            # But for small k, it should be close.
            for x in range(current_max + 1, current_max + 500):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows
