import math
import sys

def construct():
    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)
    
    # Precompute masks for each modulus m2, divisor g, and remainder rem
    # mask[m2][g][rem] is a bitset of residues r2 in 0..m2-1 such that r2 % g == rem
    precomputed_masks = {}
    for m2 in moduli:
        precomputed_masks[m2] = {}
        # g must be a divisor of m2? No, g = gcd(m, m2) so g divides m2.
        # But we can just compute for all g in 1..m2 that divide m2.
        divisors = []
        for g in range(1, m2 + 1):
            if m2 % g == 0:
                divisors.append(g)
        for g in divisors:
            precomputed_masks[m2][g] = {}
            for rem in range(g):
                mask = 0
                for r2 in range(m2):
                    if r2 % g == rem:
                        mask |= (1 << r2)
                precomputed_masks[m2][g][rem] = mask

    # Precompute gcd table
    gcd_table = {}
    for i in range(n):
        for j in range(n):
            gcd_table[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])

    # Precompute conflict masks: conflict[m_idx][r][m2_idx]
    # This is a bitset of residues of moduli[m2_idx] that conflict with (moduli[m_idx], r)
    # We can compute this on the fly using precomputed_masks and gcd_table
    # But precomputing might be faster if memory allows.
    # Memory: n * max_m * n * 8 bytes (for int bitset) -> 127 * 128 * 127 * 8 ~ 16 MB. OK.
    
    conflict_masks = [[[0]*n for _ in range(moduli[i])] for i in range(n)]
    for i in range(n):
        m = moduli[i]
        for r in range(m):
            for j in range(n):
                m2 = moduli[j]
                if i == j:
                    continue
                g = gcd_table[(m, m2)]
                rem = r % g
                # The conflicting residues in m2 are those with r2 % g == rem
                # We need precomputed_masks[m2][g][rem]
                conflict_masks[i][r][j] = precomputed_masks[m2][g][rem]

    best_solution = []
    best_size = 0
    
    # B&B
    # masks is a list of bitsets, one for each modulus index
    # masks[i] is the bitset of available residues for moduli[i]
    
    def bb(idx, current_count, masks, assignment):
        nonlocal best_solution, best_size
        
        if idx >= n:
            if current_count > best_size:
                best_size = current_count
                best_solution = list(assignment)
            return
            
        # Upper bound: current_count + number of moduli from idx to n-1 that have non-zero masks
        # We can compute this quickly if we maintain a count, but let's just sum for now
        # Optimization: if current_count + (n - idx) <= best_size, return
        if current_count + (n - idx) <= best_size:
            return
            
        # More precise UB: count non-zero masks
        ub = current_count
        for k in range(idx, n):
            if masks[k] != 0:
                ub += 1
        if ub <= best_size:
            return

        m_idx = idx
        m = moduli[m_idx]
        mask = masks[m_idx]
        
        # Option 1: Skip this modulus
        # We don't change masks for future, just recurse
        bb(idx + 1, current_count, masks, assignment)
        
        # Option 2: Pick a residue
        # Iterate through set bits in mask
        # We need to update masks for future moduli
        # To avoid modifying the original masks list, we can create a copy or use backtracking
        # Since n is small, copying a list of 127 ints is cheap
        
        # Heuristic: try residues that leave the most options?
        # For now, just iterate all
        
        r = 0
        while mask:
            lsb = mask & (-mask)
            r = lsb.bit_length() - 1
            mask_without = mask & (mask - 1)
            
            # Create new masks
            new_masks = masks[:]
            new_masks[m_idx] = 0 # This modulus is used
            
            # Update future moduli
            feasible = True
            for j in range(idx + 1, n):
                if new_masks[j] == 0:
                    continue
                # Remove conflicting residues
                # conflict_masks[m_idx][r][j] is the mask of residues in j that conflict with (m_idx, r)
                new_masks[j] &= ~conflict_masks[m_idx][r][j]
                if new_masks[j] == 0:
                    # It's not infeasible for the whole problem, just that this modulus j cannot be picked
                    # But it reduces the upper bound
                    pass
            
            assignment.append((m, r))
            bb(idx + 1, current_count + 1, new_masks, assignment)
            assignment.pop()
            
            mask = mask_without

    init_masks = [(1 << moduli[i]) - 1 for i in range(n)]
    bb(0, 0, init_masks, [])
    
    # Convert best_solution to list of [residue, modulus]
    # best_solution contains (modulus, residue) pairs
    result = [[r, m] for m, r in best_solution]
    return result
