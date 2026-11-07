"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute bit count for all masks (0..4095)
    # Using a simple DP or built-in
    bit_counts = [0] * N
    for i in range(1, N):
        bit_counts[i] = bit_counts[i >> 1] + (i & 1)
    
    # Precompute subset relation: subset[a][b] = True if a is strict subset of b
    # But N^2 = 16M booleans, which is ~16MB. That's fine.
    # However, checking (a & b) == a and a != b is fast.
    # Let's just use the bitwise check directly in the scoring function.
    
    def score(family_masks):
        """Count comparable pairs in family_masks (list of 40 ints)."""
        count = 0
        for i in range(m):
            a = family_masks[i]
            for j in range(i + 1, m):
                b = family_masks[j]
                if a == b:
                    continue
                if (a & b) == a:
                    count += 1
                elif (a & b) == b:
                    count += 1
        return count
    
    # Initial family: try to construct a dense chain-like structure.
    # A chain of length 13 (empty to full) has 13 sets.
    # We need 40 sets.
    # Strategy: Start with a chain, then add sets that are comparable to many existing sets.
    # Or: Random restarts with local search.
    
    best_family = None
    best_score = -1
    
    # Deterministic initial guess:
    # Take all subsets of size 0..11? No, that's 4096.
    # Take a specific structure.
    # Let's try: All subsets of {0..11} of size 0, 1, ..., 11?
    # Actually, let's just do random search with local improvement.
    
    def random_family():
        # Generate 40 distinct random masks
        masks = random.sample(range(N), m)
        return masks
    
    def local_search(family, deadline):
        family = list(family)
        current_score = score(family)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomly pick an index to modify
            idx = random.randint(0, m - 1)
            old_mask = family[idx]
            # Try a few random new masks
            for _ in range(50):
                new_mask = random.randint(0, N - 1)
                if new_mask == old_mask:
                    continue
                # Check distinctness
                if new_mask in family:
                    continue
                
                # Calculate score change
                # Remove old_mask contributions, add new_mask contributions
                # This is O(m) per check
                delta = 0
                for j in range(m):
                    if j == idx:
                        continue
                    other = family[j]
                    # Old contribution
                    if old_mask != other:
                        if (old_mask & other) == old_mask or (old_mask & other) == other:
                            delta -= 1
                    # New contribution
                    if new_mask != other:
                        if (new_mask & other) == new_mask or (new_mask & other) == other:
                            delta += 1
                
                if delta > 0:
                    family[idx] = new_mask
                    current_score += delta
                    improved = True
                    break
        
        return family, current_score
    
    # Run multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Initial family: mix of random and chain-based
        if restart == 0:
            # Chain-based initial: take a random chain
            # Start with empty set, add one element at a time
            chain = [0]
            current = 0
            for _ in range(n):
                # Pick a random unused bit
                unused = [b for b in range(n) if not (current >> b) & 1]
                if not unused:
                    break
                b = random.choice(unused)
                current |= (1 << b)
                chain.append(current)
            # We have 13 sets. Fill the rest with random sets
            family = list(chain)
            while len(family) < m:
                new_mask = random.randint(0, N - 1)
                if new_mask not in family:
                    family.append(new_mask)
        else:
            family = random_family()
        
        # Local search
        family, cur_score = local_search(family, deadline)
        
        if cur_score > best_score:
            best_score = cur_score
            best_family = family
    
    # Convert best_family masks to lists of integers
    result = []
    for mask in best_family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure types are correct
    return result
# EVOLVE-BLOCK-END
