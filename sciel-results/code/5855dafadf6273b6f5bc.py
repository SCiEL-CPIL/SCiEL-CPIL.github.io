"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute subset relation table for fast scoring
    # is_subset[a][b] = True if a is a strict subset of b
    # Using a 2D list of bytes or a set of pairs
    # For 4096x4096, that's 16M entries. Let's use a more compact approach.
    # Actually, we can precompute for each mask, the list of masks it's a subset of
    # But that might be too large. Let's just use the bitwise check which is fast enough.
    
    # Better: precompute a lookup table where for each pair (a, b) we know if a < b in subset order
    # is_strict_subset[a][b] = 1 if a is strict subset of b
    # This is N x N = 4096 * 4096 = 16,777,216 booleans
    # In Python, this would be slow to build and use. Let's use the direct check.
    
    def is_strict_subset(a, b):
        """Check if a is a strict subset of b using bit operations."""
        return (a & b) == a and a != b
    
    def score_family(masks):
        """Count comparable pairs in the family."""
        count = 0
        for i in range(m):
            for j in range(i + 1, m):
                a, b = masks[i], masks[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (a & b) == b and a != b:
                    count += 1
        return count
    
    # Generate initial family using chain-based approach
    # Create a chain of subsets and add variations
    def generate_chain_family():
        """Generate a family with many subset relations using chains."""
        family = []
        
        # Create a main chain: empty set, {0}, {0,1}, {0,1,2}, ..., {0..11}
        chain = []
        for i in range(n + 1):
            mask = (1 << i) - 1  # {0, 1, ..., i-1}
            chain.append(mask)
        
        family.extend(chain)
        
        # Add sets that are subsets of chain elements
        # For each chain element, add some of its subsets
        random.seed(42)
        used = set(family)
        
        # Add random subsets of the full set
        while len(family) < m:
            # Pick a random size
            size = random.randint(0, n)
            # Pick random elements
            elems = random.sample(range(n), size)
            mask = 0
            for e in elems:
                mask |= (1 << e)
            if mask not in used:
                family.append(mask)
                used.add(mask)
        
        return family[:m]
    
    # Initial family
    current_family = generate_chain_family()
    current_score = score_family(current_family)
    
    # Hill climbing with precomputed delta computation
    # For each element i in the family, try replacing it with another mask
    # and compute the change in score efficiently
    
    best_family = list(current_family)
    best_score = current_score
    
    # Precompute for each mask in the family, which other masks it's comparable with
    # This allows fast delta computation when swapping
    
    def compute_comparability(masks):
        """For each position i, compute the set of positions j where masks[i] and masks[j] are comparable."""
        comp = [set() for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                a, b = masks[i], masks[j]
                if (a & b) == a and a != b:
                    comp[i].add(j)
                    comp[j].add(i)
                elif (a & b) == b and a != b:
                    comp[i].add(j)
                    comp[j].add(i)
        return comp
    
    comp_table = compute_comparability(current_family)
    
    # Candidate masks to try
    all_masks = list(range(N))
    random.shuffle(all_masks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try swapping each element with a candidate
        for i in range(m):
            if time.time() > deadline - 5:
                break
            old_mask = current_family[i]
            # Try a few candidates
            for candidate in all_masks[:200]:
                if candidate == old_mask:
                    continue
                # Compute delta: change in score when replacing element i with candidate
                # Remove contributions involving old_mask at position i
                # Add contributions involving candidate at position i
                
                # Old contributions: pairs (i, j) for j != i where old_mask is comparable with current_family[j]
                old_contrib = 0
                for j in comp_table[i]:
                    # j is in comp_table[i] means they were comparable
                    # But we need to be careful: comp_table is based on current_family
                    # Since we're only changing position i, the comparability with j depends on old_mask vs current_family[j]
                    # We already counted these in comp_table
                    pass
                
                # Actually, let's just compute the delta directly
                # Delta = (new comparable pairs involving candidate at pos i) - (old comparable pairs involving old_mask at pos i)
                
                new_contrib = 0
                old_contrib = 0
                for j in range(m):
                    if j == i:
                        continue
                    other_mask = current_family[j]
                    # Check if old_mask is comparable with other_mask
                    a, b = old_mask, other_mask
                    if (a & b) == a and a != b:
                        old_contrib += 1
                    elif (a & b) == b and a != b:
                        old_contrib += 1
                    # Check if candidate is comparable with other_mask
                    a, b = candidate, other_mask
                    if (a & b) == a and a != b:
                        new_contrib += 1
                    elif (a & b) == b and a != b:
                        new_contrib += 1
                
                delta = new_contrib - old_contrib
                if delta > 0:
                    # Accept the move
                    current_family[i] = candidate
                    current_score += delta
                    # Update comp_table
                    comp_table[i] = set()
                    for j in range(m):
                        if j == i:
                            continue
                        other_mask = current_family[j]
                        a, b = candidate, other_mask
                        if (a & b) == a and a != b:
                            comp_table[i].add(j)
                            comp_table[j].add(i)
                        elif (a & b) == b and a != b:
                            comp_table[i].add(j)
                            comp_table[j].add(i)
                    # Remove old_mask from j's comp sets
                    for j in range(m):
                        if j != i:
                            comp_table[j].discard(i)
                    # Re-add properly
                    for j in range(m):
                        if j == i:
                            continue
                        other_mask = current_family[j]
                        a, b = candidate, other_mask
                        if (a & b) == a and a != b:
                            comp_table[j].add(i)
                        elif (a & b) == b and a != b:
                            comp_table[j].add(i)
                    
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_family = list(current_family)
                    break  # Move on to next i
    
    # Final polish: try more candidates
    if time.time() < deadline - 2:
        # Do another pass with more candidates
        for i in range(m):
            if time.time() > deadline - 2:
                break
            old_mask = current_family[i]
            for candidate in all_masks[:500]:
                if candidate == old_mask:
                    continue
                new_contrib = 0
                old_contrib = 0
                for j in range(m):
                    if j == i:
                        continue
                    other_mask = current_family[j]
                    a, b = old_mask, other_mask
                    if (a & b) == a and a != b:
                        old_contrib += 1
                    elif (a & b) == b and a != b:
                        old_contrib += 1
                    a, b = candidate, other_mask
                    if (a & b) == a and a != b:
                        new_contrib += 1
                    elif (a & b) == b and a != b:
                        new_contrib += 1
                
                delta = new_contrib - old_contrib
                if delta > 0:
                    current_family[i] = candidate
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_family = list(current_family)
                    break
    
    # Convert best_family to list of lists
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
