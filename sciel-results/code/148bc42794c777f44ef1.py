"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit (killed at 90s, but evaluator says 60s)
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    
    # Precompute subset relationships: subset_mask[i] is a bitmask of all masks that are strict supersets of i
    # Actually, let's precompute for each mask, which other masks are its strict subsets
    # We need: for a set of m masks, count pairs (i,j) where one is strict subset of other
    
    # Precompute: for each mask a, the list of masks b such that a is strict subset of b
    # That's: (a & b) == a and a != b
    # Using numpy for speed
    
    all_masks = np.arange(total_masks, dtype=np.uint16)
    
    # Precompute subset relation matrix (4096 x 4096) - too large (16M entries, ~32MB as bool)
    # Let's use a different approach: for each mask, precompute its strict supersets as a sorted list or bitset
    
    # Alternative: precompute for each mask a, a numpy array of length total_masks indicating if a is strict subset of each other mask
    # This is 4096 * 4096 = 16.7M booleans = ~16.7 MB. Acceptable.
    
    # Actually, let's precompute a 2D array: is_subset[i, j] = True if mask i is strict subset of mask j
    # Memory: 4096*4096 bytes = 16.7 MB for uint8, fine.
    
    # More efficient: for each mask i, compute which masks j satisfy (i & j) == i and i != j
    # We can vectorize this.
    
    # Let's precompute a 4096x4096 uint8 array
    # is_subset[i,j] = 1 if i is strict subset of j
    
    # For speed, let's compute this once
    # i is strict subset of j iff (i & j) == i and i != j
    
    # Vectorized computation:
    # For each i, the condition on j is: (i & j) == i, i.e., all bits in i must be in j
    # This means j must be a superset of i.
    
    # Let's build this efficiently:
    # For each i, the supersets of i are all j where j has all bits of i set.
    # We can enumerate them or use the property directly.
    
    # Actually, for the greedy approach, at each step we pick a new mask to add.
    # We need to compute, for a candidate mask c, how many existing masks are comparable with c.
    # That is: count of existing masks e such that e is strict subset of c, OR c is strict subset of e.
    
    # Let's precompute for each mask c, two lists:
    #   strict_subsets[c] = list of masks that are strict subsets of c
    #   strict_supersets[c] = list of masks that are strict supersets of c
    # But these lists can be large. For c=0, strict_supersets has 4095 elements.
    
    # Better: precompute a 4096x4096 bitset as a numpy array of uint8
    # is_subset[i, j] = 1 if i is strict subset of j
    
    # Let me compute this:
    # For i from 0 to 4095:
    #   For j from 0 to 4095:
    #     is_subset[i,j] = 1 if (i & j) == i and i != j
    
    # This is 4096*4096 = 16.7M operations. In numpy, we can do this in chunks.
    
    # Actually, let's just precompute it using broadcasting in a memory-efficient way.
    # 4096*4096 = 16,777,216 elements. As uint8, that's 16.7 MB. Fine.
    
    # But computing it with full broadcasting might use too much memory temporarily.
    # Let's do it in a loop over i, vectorized over j.
    
    is_subset = np.zeros((total_masks, total_masks), dtype=np.uint8)
    
    for i in range(total_masks):
        # j must satisfy: (i & j) == i and i != j
        # (i & j) == i means j has all bits of i
        # So j = i | k for some k, where k has no bits in common with ~i... 
        # Actually j must be a superset of i.
        # The supersets of i are: i | t for t in range(0, 1<<(n-popcount(i)))
        # But easier: just check all j with vectorized operation.
        
        # (i & all_masks) == i gives us all supersets of i (including i itself)
        # Then exclude i itself.
        supersets_mask = (all_masks & i) == i  # This is wrong: (i & j) == i means i is subset of j
        # Wait: (i & j) == i means every bit in i is also in j, so i ⊆ j.
        # So supersets_mask[j] = True if i ⊆ j.
        # Strict subset: i ⊂ j means i ⊆ j and i != j.
        
        # Let me re-check: (i & j) == i iff i is a subset of j. Yes.
        # So is_subset[i, j] = 1 if (i & j) == i and i != j.
        
        is_subset[i, :] = np.where(((all_masks & i) == i) & (all_masks != i), 1, 0)
    
    # Now is_subset[i, j] = 1 if i is strict subset of j.
    # For a family of masks, the number of comparable pairs is:
    # sum over i<j in family of (is_subset[family[i], family[j]] + is_subset[family[j], family[i]])
    
    # For greedy: at each step, we have a current set S of masks.
    # For a candidate c not in S, the gain is:
    #   number of s in S such that s is strict subset of c (is_subset[s, c] == 1)
    #   + number of s in S such that c is strict subset of s (is_subset[c, s] == 1)
    
    # Let's implement greedy with multiple restarts.
    
    best_score = 0
    best_family = None
    
    def evaluate_family(family):
        """Count comparable pairs in family (list of masks)."""
        k = len(family)
        count = 0
        for i in range(k):
            for j in range(i + 1, k):
                if is_subset[family[i], family[j]] or is_subset[family[j], family[i]]:
                    count += 1
        return count
    
    def greedy_build(initial_masks=None, rng=None):
        """Greedy construction of a family of m masks."""
        if rng is None:
            rng = random.Random()
        
        used = set()
        family = []
        
        # Start with some initial masks
        if initial_masks:
            for mask in initial_masks:
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
        
        while len(family) < m:
            # For each candidate not in used, compute the gain
            # gain(c) = sum_{s in family} (is_subset[s, c] + is_subset[c, s])
            
            best_gain = -1
            best_candidate = None
            
            # Vectorized: for all candidates not in used
            candidates = [c for c in range(total_masks) if c not in used]
            if not candidates:
                break
            
            cand_array = np.array(candidates, dtype=np.uint16)
            
            # For each candidate c, gain = sum over s in family of (is_subset[s,c] + is_subset[c,s])
            # = sum_s is_subset[s, c] + sum_s is_subset[c, s]
            
            # sum_s is_subset[s, c] for s in family:
            # This is: for each s in family, is_subset[s, c]. Sum over s.
            # = np.sum([is_subset[s, c] for s in family])
            
            # Vectorized over candidates:
            # For a fixed s, is_subset[s, :] gives a vector over all masks.
            # We want, for each candidate c, the sum over s in family of is_subset[s, c].
            # That's: sum_{s in family} is_subset[s, cand_array] -> shape (len(family), len(cand_array))
            # Then sum along axis 0.
            
            if family:
                family_array = np.array(family, dtype=np.uint16)
                # is_subset[family_array[:, None], cand_array[None, :]] would be too large
                # Instead, sum over s in family:
                # gain_down[c] = sum_{s in family} is_subset[s, c]  (s is subset of c)
                # gain_up[c] = sum_{s in family} is_subset[c, s]  (c is subset of s)
                
                # For gain_down: for each s, is_subset[s, cand_array] is a vector. Sum over s.
                gain_down = np.zeros(len(cand_array), dtype=np.int32)
                gain_up = np.zeros(len(cand_array), dtype=np.int32)
                
                for s in family:
                    gain_down += is_subset[s, cand_array]
                    gain_up += is_subset[cand_array, s]
                
                gains = gain_down + gain_up
            else:
                gains = np.zeros(len(cand_array), dtype=np.int32)
            
            # Find the candidate with max gain
            if len(gains) > 0:
                max_idx = np.argmax(gains)
                best_gain = gains[max_idx]
                best_candidate = candidates[max_idx]
            
            if best_candidate is None:
                break
            
            family.append(best_candidate)
            used.add(best_candidate)
        
        return family
    
    # Multiple restarts
    num_restarts = 50
    rng = random.Random(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Different strategies for initial masks
        strategy = restart % 5
        
        initial = []
        if strategy == 0:
            # Start with empty set and full set
            initial = [0, total_masks - 1]
        elif strategy == 1:
            # Start with a random small mask
            initial = [rng.randint(0, total_masks - 1)]
        elif strategy == 2:
            # Start with two random masks
            initial = [rng.randint(0, total_masks - 1) for _ in range(2)]
        elif strategy == 3:
            # Start with masks of size 1
            initial = [1 << i for i in rng.sample(range(n), 3)]
        else:
            # Start with a chain
            initial = [0]
            current = 0
            for _ in range(3):
                bit = 1 << rng.randint(0, n - 1)
                if not (current & bit):
                    current |= bit
                    initial.append(current)
        
        family = greedy_build(initial, rng)
        
        if len(family) < m:
            # Fill with random masks
            while len(family) < m:
                c = rng.randint(0, total_masks - 1)
                if c not in family:
                    family.append(c)
        
        score = evaluate_family(family)
        
        if score > best_score:
            best_score = score
            best_family = family[:]
        
        # Local search: try swapping one mask with another to improve
        if time.time() > deadline - 5.0:
            break
        
        # Simple local search: for a few iterations, try replacing a random mask with a better one
        for ls_iter in range(20):
            if time.time() > deadline - 3.0:
                break
            
            improved = False
            idx = rng.randint(0, len(family) - 1)
            old_mask = family[idx]
            
            # Try replacing old_mask with a new mask
            best_new_mask = old_mask
            best_new_score = score
            
            # Sample some candidates
            for _ in range(50):
                new_mask = rng.randint(0, total_masks - 1)
                if new_mask in family:
                    continue
                
                # Compute new score efficiently:
                # Remove old_mask, add new_mask
                # Delta = - (comparability of old_mask with others) + (comparability of new_mask with others)
                # where "others" = family \ {old_mask}
                
                others = [f for j, f in enumerate(family) if j != idx]
                
                # Comparability of old_mask with others
                old_comp = 0
                for o in others:
                    if is_subset[old_mask, o] or is_subset[o, old_mask]:
                        old_comp += 1
                
                # Comparability of new_mask with others
                new_comp = 0
                for o in others:
                    if is_subset[new_mask, o] or is_subset[o, new_mask]:
                        new_comp += 1
                
                new_score = score - old_comp + new_comp
                
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_new_mask = new_mask
            
            if best_new_mask != old_mask:
                family[idx] = best_new_mask
                score = best_new_score
                improved = True
                
                if score > best_score:
                    best_score = score
                    best_family = family[:]
        
        if not improved:
            continue
    
    # Convert best_family (list of masks) to list of lists
    if best_family is None:
        # Fallback: use the seed program's output
        best_family = list(range(m))
    
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
