import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Precompute subset relationships using bit manipulation
    # For each mask, precompute which masks are strict subsets/supersets
    # is_subset[a][b] = True if a is strict subset of b
    # We'll use a more efficient approach: for a given family, count comparable pairs
    
    def is_strict_subset(a, b):
        """Check if a is a strict subset of b using bit masks"""
        return (a & b) == a and a != b
    
    def count_comparable(family):
        """Count unordered pairs where one is strict subset of the other"""
        count = 0
        k = len(family)
        for i in range(k):
            for j in range(i + 1, k):
                a, b = family[i], family[j]
                if is_strict_subset(a, b) or is_strict_subset(b, a):
                    count += 1
        return count
    
    # Strategy: Start with a chain-like structure and improve via local search
    # First, create a good initial solution
    
    # Initial: pick a chain of sizes 0-12, then add more sets
    # Better: use a "layered" approach with nested sets
    
    # Let's try: pick 40 sets that maximize comparability
    # Key insight: sets that differ by only 1 element in a controlled way
    
    # Generate all masks by size
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(total_masks):
        size = bin(mask).count('1')
        masks_by_size[size].append(mask)
    
    # Initial solution: greedy chain construction
    # Start with empty set, then add elements one by one to form a chain
    # Then add more sets that are comparable with many existing sets
    
    def greedy_initial():
        family = [0]  # Start with empty set
        used = {0}
        
        # Build a chain: add one element at a time
        current = 0
        for i in range(1, n + 1):
            # Add element i-1 to current
            new_mask = current | (1 << (i - 1))
            if new_mask not in used:
                family.append(new_mask)
                used.add(new_mask)
                current = new_mask
        
        # Now we have 13 sets (chain). Need 27 more.
        # Add sets that are comparable with many existing sets
        # Priority: sets that are subsets/supersets of chain elements
        
        # For each possible mask, score it by how many family members it's comparable with
        remaining = [mask for mask in range(total_masks) if mask not in used]
        
        while len(family) < m:
            best_mask = -1
            best_score = -1
            for mask in remaining:
                if mask in used:
                    continue
                score = 0
                for f in family:
                    if is_strict_subset(mask, f) or is_strict_subset(f, mask):
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask
            
            if best_mask == -1 or best_score == 0:
                # Pick a random unused mask
                for mask in remaining:
                    if mask not in used:
                        best_mask = mask
                        break
            
            family.append(best_mask)
            used.add(best_mask)
        
        return family
    
    # Local search: swap elements to improve
    def local_search(family, max_iters=5000):
        family = list(family)
        current_score = count_comparable(family)
        best_family = list(family)
        best_score = current_score
        used = set(family)
        
        # Precompute comparability for faster evaluation
        # For a swap: remove set A, add set B
        # Change in score = (comparability of B with family\{A}) - (comparability of A with family\{A})
        
        iters = 0
        while iters < max_iters:
            if time.time() > deadline - 5.0:
                break
            iters += 1
            
            improved = False
            # Try swapping each family member with a random candidate
            for i in range(m):
                if time.time() > deadline - 5.0:
                    break
                a = family[i]
                # Try several random candidates
                for _ in range(20):
                    b = random.randint(0, total_masks - 1)
                    if b == a or b in used:
                        continue
                    
                    # Calculate change in comparable pairs
                    # Remove a: lose comparability with all other family members
                    # Add b: gain comparability with all other family members
                    delta = 0
                    for j in range(m):
                        if j == i:
                            continue
                        f = family[j]
                        # Was a comparable with f?
                        was_comp = is_strict_subset(a, f) or is_strict_subset(f, a)
                        # Is b comparable with f?
                        is_comp = is_strict_subset(b, f) or is_strict_subset(f, b)
                        if was_comp and not is_comp:
                            delta -= 1
                        elif is_comp and not was_comp:
                            delta += 1
                    
                    if delta > 0:
                        # Accept the swap
                        family[i] = b
                        used.remove(a)
                        used.add(b)
                        current_score += delta
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_family = list(family)
                        break  # Restart the outer loop
            
            if not improved:
                # Try a random perturbation
                i = random.randint(0, m - 1)
                a = family[i]
                b = random.randint(0, total_masks - 1)
                while b == a or b in used:
                    b = random.randint(0, total_masks - 1)
                family[i] = b
                used.remove(a)
                used.add(b)
                current_score = count_comparable(family)
        
        return best_family, best_score
    
    # Generate initial solution
    family = greedy_initial()
    initial_score = count_comparable(family)
    
    best_family = list(family)
    best_score = initial_score
    
    # Run local search with multiple restarts
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 10.0:
        restarts += 1
        if restarts == 1:
            # First iteration: use the greedy initial
            fam = list(family)
        else:
            # Create a perturbed version
            fam = list(best_family)
            # Randomly swap 5-15 elements
            num_swaps = random.randint(5, 15)
            used = set(fam)
            for _ in range(num_swaps):
                i = random.randint(0, m - 1)
                b = random.randint(0, total_masks - 1)
                while b == fam[i] or b in used:
                    b = random.randint(0, total_masks - 1)
                old = fam[i]
                fam[i] = b
                used.remove(old)
                used.add(b)
        
        improved_fam, improved_score = local_search(fam, max_iters=3000)
        
        if improved_score > best_score:
            best_score = improved_score
            best_family = improved_fam
    
    # Final verification and conversion to required format
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    # Ensure exactly 40 distinct lists
    assert len(result) == 40
    assert len(set(tuple(sorted(s)) for s in result)) == 40
    
    return result
