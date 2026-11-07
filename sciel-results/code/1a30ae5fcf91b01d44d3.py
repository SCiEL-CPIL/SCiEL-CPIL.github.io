import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Precompute: for each pair of masks, whether one is strict subset of other
    # comp[i][j] = True if mask i is strict subset of mask j
    # We'll use a 2D list of bytes for speed
    # Actually, let's precompute the "comparable" relation
    # comp[i][j] = 1 if i is strict subset of j, -1 if j is strict subset of i, 0 otherwise
    
    # For speed, precompute subset relation
    # is_subset[i][j] = 1 if i is subset of j (not necessarily strict)
    # But 4096x4096 is 16M entries, too large for Python lists
    # Instead, use bit operations on the fly, but precompute popcount
    
    # Precompute popcount for all masks
    popcount = [0] * total_masks
    for i in range(1, total_masks):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Helper: check if a is strict subset of b
    def is_strict_subset(a, b):
        return (a & b) == a and a != b
    
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b
    
    # Count comparable pairs in a family (list of masks)
    def count_comparable(family):
        count = 0
        k = len(family)
        for i in range(k):
            ai = family[i]
            for j in range(i + 1, k):
                aj = family[j]
                if is_comparable(ai, aj):
                    count += 1
        return count
    
    # Greedy construction: build family by adding sets that maximize new comparable pairs
    def greedy_construct():
        family = []
        used = [False] * total_masks
        
        # Start with some seed sets
        # Try to build a structure with many comparable pairs
        # Start with empty set and full set, then add intermediate
        
        # Strategy: pick sets that are comparable to many existing sets
        # Initialize with a chain-like structure
        # Add sets of increasing size
        
        # First, add the empty set
        family.append(0)
        used[0] = True
        
        # Then add sets greedily
        for _ in range(m - 1):
            best_mask = -1
            best_gain = -1
            # Try all unused masks
            for mask in range(total_masks):
                if used[mask]:
                    continue
                # Count how many in family are comparable to mask
                gain = 0
                for f in family:
                    if is_comparable(mask, f):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            family.append(best_mask)
            used[best_mask] = True
        
        return family
    
    # Local search: try to improve by replacing one set
    def local_search(family, max_iters=10000):
        family = list(family)
        current_score = count_comparable(family)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            # Try replacing each element
            for i in range(m):
                old_mask = family[i]
                # Try a few random replacements
                for attempt in range(20):
                    new_mask = random.randint(0, total_masks - 1)
                    if new_mask == old_mask:
                        continue
                    # Check if new_mask is already in family (except at position i)
                    dup = False
                    for j in range(m):
                        if j != i and family[j] == new_mask:
                            dup = True
                            break
                    if dup:
                        continue
                    
                    # Compute score change
                    # Remove contributions of old_mask, add contributions of new_mask
                    old_contrib = 0
                    new_contrib = 0
                    for j in range(m):
                        if j == i:
                            continue
                        fj = family[j]
                        if is_comparable(old_mask, fj):
                            old_contrib += 1
                        if is_comparable(new_mask, fj):
                            new_contrib += 1
                    
                    delta = new_contrib - old_contrib
                    if delta > 0:
                        family[i] = new_mask
                        current_score += delta
                        improved = True
                        break  # Restart scan from beginning
        
        return family, current_score
    
    # Run greedy to get initial family
    family = greedy_construct()
    best_family = list(family)
    best_score = count_comparable(best_family)
    
    # Run local search with multiple restarts
    n_restarts = 0
    while time.time() < deadline - 5:
        # Perturb the best family
        perturbed = list(best_family)
        # Randomly replace a few elements
        n_perturb = random.randint(1, 5)
        for _ in range(n_perturb):
            idx = random.randint(0, m - 1)
            new_mask = random.randint(0, total_masks - 1)
            # Ensure uniqueness
            attempts = 0
            while new_mask in perturbed and attempts < 50:
                new_mask = random.randint(0, total_masks - 1)
                attempts += 1
            if new_mask not in perturbed:
                perturbed[idx] = new_mask
        
        # Local search
        result_family, result_score = local_search(perturbed, max_iters=20000)
        
        if result_score > best_score:
            best_score = result_score
            best_family = result_family
        
        n_restarts += 1
    
    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
