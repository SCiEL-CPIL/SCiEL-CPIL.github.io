import time

_START = time.time()
_DEADLINE = 60.0  # Safety margin, though computation is trivial

MOD = 101
N = 10
_FULL = 1 << N

def construct():
    # 1. Compute inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # 2. Precompute sum and popcount for all masks
    # sums[mask] = sum of inverses of set bits in mask, mod MOD
    # pop[mask] = number of set bits in mask
    sums = [0] * _FULL
    pop = [0] * _FULL
    
    for mask in range(1, _FULL):
        lsb = mask & (-mask)
        idx = lsb.bit_length()  # 1-based index
        prev = mask ^ lsb
        sums[mask] = (sums[prev] + inv[idx]) % MOD
        pop[mask] = pop[prev] + 1

    # 3. Determine minimum max row size K
    # We need to cover all 101 residues.
    # Let's check if K=1 works, then K=2, etc.
    # Since 1023 subsets cover 101 residues, and subsets are small, 
    # K is likely small (probably 2 or 3).
    
    # For each residue, store the smallest popcount mask that achieves it.
    # Actually, we just need to know if it's achievable with popcount <= K.
    # Let's find the minimal popcount required for each residue.
    min_pop_for_residue = [100] * MOD  # Initialize with large number
    
    for mask in range(1, _FULL):
        r = sums[mask]
        p = pop[mask]
        if p < min_pop_for_residue[r]:
            min_pop_for_residue[r] = p

    # The optimal maximum row length is the max of min_pop_for_residue
    optimal_K = max(min_pop_for_residue)
    
    # 4. Construct the rows
    # For each residue, pick a mask that achieves it with popcount <= optimal_K.
    # We can just reuse the mask that gave the min_pop, or any valid one.
    # To be safe and simple, we can just pick the first mask found that satisfies the condition,
    # but since we already tracked min_pop, we can just store the mask that achieved that min_pop.
    
    best_mask_for_residue = [0] * MOD
    # Re-scan to find a specific mask for each residue that has popcount == min_pop_for_residue[r]
    # Or simpler: just store the mask that updated min_pop_for_residue.
    
    # Let's re-do the tracking to keep the mask
    min_pop_for_residue = [100] * MOD
    best_mask_for_residue = [0] * MOD
    
    for mask in range(1, _FULL):
        r = sums[mask]
        p = pop[mask]
        if p < min_pop_for_residue[r]:
            min_pop_for_residue[r] = p
            best_mask_for_residue[r] = mask

    rows = []
    for r in range(MOD):
        mask = best_mask_for_residue[r]
        # Extract denominators from mask
        # Bit i corresponds to denominator i+1
        denoms = []
        for i in range(N):
            if mask & (1 << i):
                denoms.append(i + 1)
        rows.append(denoms)
        
    return rows
