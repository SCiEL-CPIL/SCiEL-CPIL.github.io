import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    MAX_MOD = 128
    
    moduli = list(range(2, MAX_MOD + 1))
    n_moduli = len(moduli)
    
    # Precompute compatibility masks
    # compat[m1][m2][r1] is a bitmask of r2 in [0, m2) disjoint from (r1, m1)
    # Dimensions: [MAX_MOD+1][MAX_MOD+1][m1]
    # This might be large. 128 * 128 * 128 ints ~ 2M ints. In Python, ints are objects, so this is huge memory.
    # Better to compute on the fly or store more compactly?
    # Or store as bytes? 
    # Actually, m1 is at most 128. So the list for a fixed (m1, m2) has 128 entries.
    # Total entries: sum_{m1=2}^{128} sum_{m2=2}^{128} m1 
    # = (sum m1) * (127) approx 8000 * 127 = 1M entries.
    # 1M Python integers is ~ 28MB. Acceptable.
    
    compat = [[[0 for _ in range(m1)] for m2 in range(MAX_MOD + 1)] for m1 in range(MAX_MOD + 1)]
    
    for m1 in range(2, MAX_MOD + 1):
        for m2 in range(2, MAX_MOD + 1):
            g = math.gcd(m1, m2)
            for r1 in range(m1):
                mask = 0
                # r2 is compatible if (r1 - r2) % g != 0
                # i.e., r2 % g != r1 % g
                r1_mod_g = r1 % g
                for r2 in range(m2):
                    if r2 % g != r1_mod_g:
                        mask |= (1 << r2)
                compat[m1][m2][r1] = mask

    # Initial valid masks for each modulus
    # valid_masks[m] is a bitmask of valid residues for modulus m
    initial_valid_masks = [0] * (MAX_MOD + 1)
    for m in range(2, MAX_MOD + 1):
        initial_valid_masks[m] = (1 << m) - 1

    best_count = 0
    best_solution = []
    
    # used_mask is a bitmask of moduli used. 
    # Since moduli are 2..128, we can use a set or a bitmask of size 129.
    # Let's use a set of moduli for simplicity in checking, but for speed, a bitmask is better.
    # Let's use a list `used` of booleans or a set.
    
    def count_bits(n):
        return bin(n).count('1')

    def dfs(current_count, valid_masks, used_moduli):
        nonlocal best_count, best_solution
        
        if time.time() > deadline:
            return

        # Pruning: If we can't beat best_count, return
        # Max possible additional classes is the number of moduli not used.
        # A tighter bound: number of moduli not used that have at least 1 valid residue.
        remaining_potential = 0
        for m in range(2, MAX_MOD + 1):
            if m not in used_moduli:
                if valid_masks[m] != 0:
                    remaining_potential += 1
        
        if current_count + remaining_potential <= best_count:
            return

        if current_count > best_count:
            best_count = current_count
            # Reconstruct solution? 
            # We need to track the path. 
            # Passing a list of chosen (m, r) is easier.
            # Let's pass `path` to dfs.
            
        # Find modulus with minimum valid residues (MRV heuristic)
        min_count = float('inf')
        best_m = -1
        for m in range(2, MAX_MOD + 1):
            if m not in used_moduli:
                c = count_bits(valid_masks[m])
                if c == 0:
                    continue
                if c < min_count:
                    min_count = c
                    best_m = m
        
        if best_m == -1:
            return
            
        # Branch on best_m
        vm = valid_masks[best_m]
        # Iterate through set bits in vm
        # To optimize, we can try residues in a specific order?
        # Randomization might help avoid local minima in search, but MRV is deterministic.
        # We can shuffle the residues if we want diversity, but standard B&B is fine.
        
        # Get list of residues
        residues = []
        temp = vm
        r = 0
        while temp:
            if temp & 1:
                residues.append(r)
            temp >>= 1
            r += 1
            
        # Optional: Shuffle to break symmetry? 
        # import random
        # random.shuffle(residues)
        
        for r in residues:
            # Apply selection (best_m, r)
            new_used = used_moduli | {best_m}
            # Update valid masks
            old_masks = {}
            new_valid_masks = valid_masks[:] # Shallow copy of list of ints (ints are immutable)
            
            for m in range(2, MAX_MOD + 1):
                if m == best_m:
                    continue
                if m in used_moduli:
                    continue
                # Update mask for m
                new_valid_masks[m] = valid_masks[m] & compat[best_m][m][r]
                # If it becomes 0, we can prune early?
                # The dfs will handle it.
            
            dfs(current_count + 1, new_valid_masks, new_used)

    # We need to track the path to return the solution.
    # Let's modify dfs to take path.
    
    best_path = []
    
    def dfs_with_path(current_count, valid_masks, used_moduli, path):
        nonlocal best_count, best_path
        
        if time.time() > deadline:
            return

        # Pruning
        remaining_potential = 0
        for m in range(2, MAX_MOD + 1):
            if m not in used_moduli:
                if valid_masks[m] != 0:
                    remaining_potential += 1
        
        if current_count + remaining_potential <= best_count:
            return

        if current_count > best_count:
            best_count = current_count
            best_path = path[:]
            
        # Find modulus with minimum valid residues (MRV heuristic)
        min_count = float('inf')
        best_m = -1
        for m in range(2, MAX_MOD + 1):
            if m not in used_moduli:
                c = count_bits(valid_masks[m])
                if c == 0:
                    continue
                if c < min_count:
                    min_count = c
                    best_m = m
        
        if best_m == -1:
            return
            
        vm = valid_masks[best_m]
        
        # Iterate through valid residues
        temp = vm
        r = 0
        while temp:
            if temp & 1:
                # Select (best_m, r)
                new_used = used_moduli | {best_m}
                new_valid_masks = valid_masks[:]
                
                for m in range(2, MAX_MOD + 1):
                    if m == best_m:
                        continue
                    if m in used_moduli:
                        continue
                    new_valid_masks[m] = valid_masks[m] & compat[best_m][m][r]
                    
                dfs_with_path(current_count + 1, new_valid_masks, new_used, path + [(best_m, r)])
            temp >>= 1
            r += 1

    # Initial call
    # used_moduli is a set
    dfs_with_path(0, initial_valid_masks, set(), [])
    
    if not best_path:
        # Fallback: Return a trivial valid set
        # e.g. (1, 2) is valid? 
        # Any single class is valid.
        return [[1, 2]]
        
    return [[r, m] for m, r in best_path]
