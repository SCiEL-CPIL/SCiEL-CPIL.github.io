import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    n = 12
    m = 40
    all_masks = 1 << n
    
    def masks_to_sets(masks):
        return [[b for b in range(n) if (mask >> b) & 1] for mask in masks]
    
    def count_comparable(masks):
        """Count unordered pairs where one is strict subset of other."""
        count = 0
        for i in range(m):
            for j in range(i + 1, m):
                a, b = masks[i], masks[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (a & b) == b and a != b:
                    count += 1
        return count
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    # Strategy: Build a family that approximates a chain structure
    # We'll use sets of various sizes, trying to make them nested
    
    # Initial construction: create a "lattice" structure
    # Pick a base chain of masks, then add variations
    
    def initial_family():
        """Create initial family using a structured approach."""
        family = []
        used = set()
        
        # Create a main chain: empty, {0}, {0,1}, {0,1,2}, ..., {0..11}
        # That's 13 sets. We need 40, so we need to add more.
        
        # Strategy: For each size k, have multiple sets that are "compatible"
        # with sets of size k-1 and k+1
        
        # Let's build by size levels
        # Size 0: {} (1 set)
        # Size 1: {0}, {1}, {2} (3 sets) 
        # Size 2: {0,1}, {0,2}, {1,2}, {0,3} (4 sets)
        # etc.
        
        # Better: use a greedy approach to build a dense comparable family
        # Start with random distinct sets, then optimize
        
        # Initial: pick 40 random distinct subsets
        while len(family) < m:
            mask = random.randint(0, all_masks - 1)
            if mask not in used:
                family.append(mask)
                used.add(mask)
        
        return family
    
    # Local search: try swapping one set for another to improve count
    def local_search(family, time_limit):
        best = family[:]
        best_score = count_comparable(best)
        
        # Precompute comparability matrix for speed
        def compute_comp_matrix(fam):
            comp = [[False] * m for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    if is_comparable(fam[i], fam[j]):
                        comp[i][j] = True
                        comp[j][i] = True
            return comp
        
        comp = compute_comp_matrix(best)
        
        # For each position, try replacing with a different mask
        # Delta = new_comparables - old_comparables
        # When replacing position i with new_mask:
        #   Remove: count of j where comp[i][j] is True
        #   Add: count of j where is_comparable(new_mask, best[j]) and j != i
        
        while time.time() < time_limit:
            improved = False
            for i in range(m):
                if time.time() > time_limit:
                    break
                old_comp_count = sum(1 for j in range(m) if j != i and comp[i][j])
                
                # Try a few candidate replacements
                for _ in range(5):
                    new_mask = random.randint(0, all_masks - 1)
                    if new_mask == best[i]:
                        continue
                    # Check if new_mask is already in family (at another position)
                    if any(best[j] == new_mask for j in range(m) if j != i):
                        continue
                    
                    # Compute new comparables for new_mask with others
                    new_comp_count = 0
                    for j in range(m):
                        if j != i and is_comparable(new_mask, best[j]):
                            new_comp_count += 1
                    
                    delta = new_comp_count - old_comp_count
                    if delta > 0:
                        # Accept the move
                        best[i] = new_mask
                        # Update comp matrix
                        for j in range(m):
                            if j != i:
                                comp[i][j] = comp[j][i] = is_comparable(new_mask, best[j])
                        best_score += delta
                        improved = True
                        break  # Move to next i
        
            if not improved:
                # No improvement found, try perturbation
                # Swap two random positions
                i, j = random.sample(range(m), 2)
                best[i], best[j] = best[j], best[i]
                # Update comp matrix
                for k in range(m):
                    if k != i and k != j:
                        comp[i][k] = comp[k][i] = is_comparable(best[i], best[k])
                        comp[j][k] = comp[k][j] = is_comparable(best[j], best[k])
                comp[i][j] = comp[j][i] = is_comparable(best[i], best[j])
        
        return best, best_score
    
    # Run local search with time budget
    family = initial_family()
    family, score = local_search(family, deadline - 5.0)
    
    # Convert to list of lists
    result = masks_to_sets(family)
    
    # Ensure types are correct (Python int, list)
    for i in range(m):
        result[i] = [int(x) for x in result[i]]
    
    return result
