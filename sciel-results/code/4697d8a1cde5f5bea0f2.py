import random
from itertools import combinations

def construct():
    # Precompute all 4-subsets of [16] as bitmasks
    all_sets = []
    mask_to_idx = {}
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
        mask_to_idx[mask] = len(all_sets) - 1
    
    n = len(all_sets)  # 1820
    
    # Precompute conflicts
    # conflicts[i][j] is a set of indices k such that {i, j, k} is a sunflower
    # We only need to store this for i < j? No, we will look up conflicts[C][A] where C and A are in chosen.
    # Since we add C, we look at pairs (C, A) for A in chosen.
    # So we need conflicts[C][A].
    
    # To save memory and time, we can compute on the fly?
    # No, 1820^2 is manageable if we store as lists of ints.
    # But computing 1.6M pairs * ~50 items each is 80M items. Memory ~ 300MB? Maybe too much.
    # Let's check memory. 1820 * 1820 * 4 bytes (if int) = 13MB for pointers.
    # The lists themselves: avg 50 ints. 1.6M * 50 * 4 bytes = 320MB.
    # This might be tight.
    
    # Alternative: Compute conflicts on the fly but cache them?
    # Or, just compute the set of forbidden masks for the current state?
    
    # Let's try to optimize the "on the fly" check with the bucketing method but faster?
    # The previous method was slow.
    
    # Let's try a hybrid:
    # Precompute for each pair (A, B) the "forbidden set" as a bitmask of 1820 bits?
    # 1820 bits = 228 bytes.
    # 1.6M pairs * 228 bytes = 360MB. Too much.
    
    # How about we don't precompute all pairs, but compute the conflicts for a specific pair (C, A) quickly?
    # Given C and A, find K = C & A.
    # Compute V = (~ ( (C ^ K) | (A ^ K) )) & ((1<<16)-1)
    # V is the set of available elements outside C\K and A\K.
    # We need to check if there exists a 4-subset D such that:
    # D contains K, D subset of K | V, and D is in all_sets.
    # Actually, we just need to add all such D to Forbidden.
    # The number of such D is small (< 100).
    # So, inside the loop:
    # if C not in Forbidden:
    #   chosen.append(C)
    #   for A in chosen[:-1]:
    #      K = C & A
    #      # Generate all sunflowers with C and A
    #      # ...
    #      Forbidden.update(...)
    
    # Generating the sunflowers:
    # K = C & A
    # k = bit_count(K)
    # # Elements in C\K and A\K are forbidden for the "rest" of the sunflower C'.
    # # Let Rest_C = C ^ K, Rest_A = A ^ K.
    # # Forbidden elements for the rest: Rest_C | Rest_A.
    # # Available elements: FullMask ^ (Rest_C | Rest_A) ^ K ? No.
    # # The new set C' must contain K.
    # # C' \ K must be chosen from [16] \ (Rest_C | Rest_A | K).
    # # Let Pool = FullMask & ~(Rest_C | Rest_A | K)
    # # We need to pick 4-k elements from Pool.
    # # For each combination of 4-k from Pool, form C' = K | combo.
    # # Check if C' is in mask_to_idx (it always is, since it's a 4-subset of 16).
    # # Get its index and add to Forbidden.
    
    # This is fast.
    # Inner loop cost: |F| * (cost to generate combos).
    # Cost to generate combos: iterate over combinations of Pool of size 4-k.
    # Pool size is approx 16 - 2(4-k) - k = 16 - 8 + 2k - k = 8+k.
    # Max size 11.
    # Combinations of 11 choose 1 is 11.
    # Combinations of 10 choose 2 is 45.
    # Combinations of 9 choose 3 is 84.
    # Combinations of 8 choose 4 is 70.
    # So max 84 lookups per pair.
    # 84 * |F| operations per new element.
    # |F| ~ 200. 16,800 operations.
    # 1820 * 16,800 = 30M operations per trial.
    # 100 trials = 3B operations? No, 30M * 100 = 3B.
    # Python is slow. 30M ops might take 1-2 seconds.
    # 100 trials -> 100-200 seconds. Too slow.
    
    # We need to reduce the number of trials or speed up the inner loop.
    
    # Optimization:
    # Instead of updating Forbidden by iterating all A in chosen,
    # we can check conflict directly?
    # "if C not in Forbidden" is O(1).
    # The update is the bottleneck.
    
    # Can we avoid updating Forbidden for all A?
    # If we don't update Forbidden, we can't use O(1) check.
    
    # Maybe we can use a different greedy strategy?
    # "Local Search": Start with a random set, then try to swap elements?
    
    # Let's stick to greedy but optimize the update.
    # We can precompute the list of "partner" indices for each pair?
    # If we precompute `sunflower_partners[i][j]` as a list of indices,
    # then `Forbidden.update(indices)` is fast (set update with list).
    # Memory: 1820 * 1820 lists.
    # If we store as array of arrays of uint16?
    # Let's try to precompute just the lists, not the sets, to save overhead.
    # And only for i < j?
    # When adding C, we look at A in chosen.
    # We need `sunflower_partners[C][A]`.
    # We can store a 2D list `partners[i][j]`.
    # 1820 * 1820 * (avg 50 * 2 bytes) = 33MB.
    # This is acceptable.
    
    # Precomputation time:
    # 1.6M pairs.
    # For each, generate list.
    # This will take a few seconds.
    
    # Let's implement this.

    FullMask = (1 << 16) - 1
    partners = [[None] * n for _ in range(n)]
    
    # Symmetry: partners[i][j] == partners[j][i]
    for i in range(n):
        Ai = all_sets[i]
        for j in range(i + 1, n):
            Aj = all_sets[j]
            K = Ai & Aj
            k = bin(K).count('1')
            Rest_Ai = Ai ^ K
            Rest_Aj = Aj ^ K
            Forbidden_Elems = Rest_Ai | Rest_Aj | K
            Pool = FullMask & ~Forbidden_Elems
            
            # Generate combinations of Pool of size 4-k
            pool_list = [idx for idx in range(16) if Pool & (1 << idx)]
            comb_size = 4 - k
            conflict_indices = []
            for combo in combinations(pool_list, comb_size):
                C_mask = K
                for idx in combo:
                    C_mask |= (1 << idx)
                # C_mask is a valid 4-subset
                conflict_indices.append(mask_to_idx[C_mask])
            
            partners[i][j] = conflict_indices
            partners[j][i] = conflict_indices
            
    # Now run greedy trials
    best_size = 0
    best_family = []
    
    # Time limit 60s.
    # Let's estimate time per trial.
    # Inner loop: for each A in chosen, do Forbidden.update(partners[C][A]).
    # partners[C][A] is a list.
    # Set update with list is fast.
    
    num_trials = 500 # Try more trials
    
    for trial in range(num_trials):
        random.seed(trial)
        indices = list(range(n))
        random.shuffle(indices)
        
        chosen = []
        Forbidden = set()
        
        for idx in indices:
            c = all_sets[idx]
            c_idx = idx # idx is the index in all_sets
            
            if c_idx not in Forbidden:
                chosen.append(c_idx)
                # Update Forbidden
                for a_idx in chosen[:-1]:
                    Forbidden.update(partners[c_idx][a_idx])
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_family = chosen[:]
            
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = [i + 1 for i in range(16) if mask & (1 << i)]
        result.append(elems)
        
    return result
