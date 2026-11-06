import random
from itertools import combinations

def construct():
    # All 4-element subsets of [1, 16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    N = len(all_sets)  # 1820
    
    # Precompute pairwise intersections for fast sunflower check
    # inter[i][j] = all_sets[i] & all_sets[j]
    # Using a list of dicts or a 2D list. 1820^2 = ~3.3M entries, manageable.
    # Actually, let's use a more memory-efficient approach: store as frozensets in a dict keyed by (i,j) with i<j
    # Or just a 2D list of frozensets. 3.3M * ~100 bytes = 330 MB. Might be tight.
    # Let's use a dict instead.
    
    # Actually, for speed, let's precompute the sunflower conflict structure differently.
    # For each pair (i, j), the sunflower partners k are those where:
    #   I = S_i ∩ S_j
    #   S_k contains I
    #   S_k is disjoint from (S_i \ I) ∪ (S_j \ I)
    #   S_k ≠ S_i, S_k ≠ S_j
    #   |S_k| = 4
    
    # Let me precompute conflicts[i][j] = list of k (k > j, k != i, k != j) such that (i,j,k) is a sunflower
    # But storing all of this might be large. Let's think about memory.
    # For each pair (i,j), the number of valid k is at most C(16 - |S_i ∪ S_j|, 4 - |I|)
    # |S_i ∪ S_j| can range from 4 to 8. If |I|=0, |union|=8, free=8, C(8,4)=70.
    # If |I|=1, |union|=7, free=9, C(9,3)=84.
    # If |I|=2, |union|=6, free=10, C(10,2)=45.
    # If |I|=3, |union|=5, free=11, C(11,1)=11.
    # If |I|=4, then S_i=S_j, no valid k.
    # So at most ~84 per pair. Number of pairs ~1.65M. Total entries ~1.65M * 40 avg = 66M.
    # That's a lot but maybe manageable with compact storage.
    
    # Alternative: on-the-fly computation with good caching.
    # Let me try a different approach: precompute for each pair (i,j) the set of forbidden elements
    # and the required core, then quickly check candidates.
    
    # Let's just do the on-the-fly approach but with precomputed bitmasks for fast intersection checks.
    # Represent each set as a 16-bit integer.
    
    set_bits = []
    for s in all_sets:
        b = 0
        for x in s:
            b |= (1 << (x-1))
        set_bits.append(b)
    
    # Precompute popcount for 16-bit numbers? Or use bin(x).count('1')
    
    def popcount(x):
        return bin(x).count('1')
    
    # For the greedy algorithm, we need to check if adding set k to the chosen family
    # creates a sunflower with any two already-chosen sets.
    # For each pair (i, j) in chosen, check if (i, j, k) is a sunflower.
    
    # To speed this up, for each pair (i,j) in chosen, precompute:
    #   core = S_i & S_j
    #   forbidden = (S_i | S_j) & ~core  = (S_i \ core) | (S_j \ core)
    #   Then k is a sunflower partner iff:
    #     (S_k & core) == core  (core ⊆ S_k)
    #     (S_k & forbidden) == 0  (S_k disjoint from forbidden)
    #     S_k != S_i, S_k != S_j
    
    # Let's precompute for each pair (i,j) these values.
    # Store in a dict: pair_info[(i,j)] = (core_bits, forbidden_bits)
    
    pair_info = {}
    for i in range(N):
        for j in range(i+1, N):
            core = set_bits[i] & set_bits[j]
            union = set_bits[i] | set_bits[j]
            forbidden = union & ~core
            pair_info[(i,j)] = (core, forbidden)
    
    # Now, the greedy algorithm:
    # Maintain a list of chosen indices.
    # For each candidate k, count how many pairs (i,j) in chosen would form a sunflower with k.
    # Pick the candidate with minimum conflicts. If min conflicts > 0, stop.
    
    # To make this faster, maintain for each candidate k a count of how many pairs in chosen
    # would conflict with k. When we add a new set m to chosen, for each existing chosen i,
    # update the conflict count of all k that form a sunflower with (i, m).
    
    # Let's implement this incremental approach.
    
    # conflict_count[k] = number of pairs (i,j) in chosen such that (i,j,k) is a sunflower
    # When we add m to chosen, for each i in chosen (i != m), we need to find all k
    # such that (i, m, k) is a sunflower, and increment conflict_count[k].
    
    # To find all k such that (i, m, k) is a sunflower:
    #   core = S_i & S_m
    #   forbidden = (S_i | S_m) & ~core
    #   k must satisfy: (S_k & core) == core, (S_k & forbidden) == 0, S_k != S_i, S_k != S_m
    
    # We can iterate over all k and check, but that's O(N) per pair.
    # Better: precompute for each (core_bits, forbidden_bits) the list of k that satisfy the conditions.
    # But (core_bits, forbidden_bits) can take many values.
    
    # Alternative: just iterate over all k for each new pair. N=1820, and we do this for each new addition.
    # Total work: sum over each addition of (|chosen| * N). If we add K sets, total ~ K^2 * N / 2.
    # If K ~ 100, that's 100^2 * 1820 / 2 = 9.1M checks. Each check is a few bit operations. Should be fast.
    
    # Let's implement this.
    
    best_result = []
    
    for trial in range(20):
        random.seed(trial)
        
        chosen = []
        chosen_set = set()
        conflict_count = [0] * N
        available = list(range(N))
        random.shuffle(available)
        
        while available:
            # Find the candidate with minimum conflict count
            min_conf = float('inf')
            best_idx = -1
            for idx in available:
                c = conflict_count[idx]
                if c < min_conf:
                    min_conf = c
                    best_idx = idx
                    if c == 0:
                        break
            
            if min_conf > 0:
                break
            
            # Add best_idx to chosen
            chosen.append(best_idx)
            chosen_set.add(best_idx)
            available.remove(best_idx)
            
            # Update conflict counts for all pairs involving best_idx and existing chosen
            for i in chosen[:-1]:  # all previously chosen
                core = set_bits[i] & set_bits[best_idx]
                union = set_bits[i] | set_bits[best_idx]
                forbidden = union & ~core
                
                # Find all k in available that form a sunflower with (i, best_idx)
                for k in available:
                    if k == i or k == best_idx:
                        continue
                    sk = set_bits[k]
                    if (sk & core) == core and (sk & forbidden) == 0:
                        conflict_count[k] += 1
        
        if len(chosen) > len(best_result):
            best_result = chosen[:]
    
    return [sorted(all_sets[i]) for i in best_result]
