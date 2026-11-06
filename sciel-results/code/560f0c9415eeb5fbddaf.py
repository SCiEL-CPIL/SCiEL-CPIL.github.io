"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction
import sys

def construct():
    """
    Find a maximum weight independent set on the conflict graph of candidate (m, r) pairs.
    Uses Branch-and-Bound with fractional knapsack (sum of weights) upper bound.
    """
    
    # Increase recursion limit just in case, though we use iterative
    sys.setrecursionlimit(10000)

    # Generate all candidate (m, r) pairs
    candidates = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))
    
    n = len(candidates)
    
    # Sort candidates by weight (1/m) descending, then by m ascending, then r
    # This helps B&B find good solutions early
    candidates.sort(key=lambda x: (-1.0/x[0], x[0], x[1]))
    
    # Precompute weights
    weights = [Fraction(1, m) for (m, r) in candidates]
    
    # Precompute conflicts as bitmasks
    conflicts = [0] * n
    
    # Map modulus to list of indices
    mod_to_indices = {}
    for idx, (m, r) in enumerate(candidates):
        if m not in mod_to_indices:
            mod_to_indices[m] = []
        mod_to_indices[m].append(idx)
        
    moduli = list(mod_to_indices.keys())
    
    # Iterate over pairs of moduli to build conflicts
    for i_m in range(len(moduli)):
        m1 = moduli[i_m]
        for j_m in range(i_m, len(moduli)):
            m2 = moduli[j_m]
            g = math.gcd(m1, m2)
            
            # Group indices by residue mod g
            groups1 = {c: [] for c in range(g)}
            for idx in mod_to_indices[m1]:
                r = candidates[idx][1]
                c = r % g
                groups1[c].append(idx)
                
            groups2 = {c: [] for c in range(g)}
            for idx in mod_to_indices[m2]:
                r = candidates[idx][1]
                c = r % g
                groups2[c].append(idx)
                
            for c in range(g):
                idxs1 = groups1[c]
                idxs2 = groups2[c]
                
                if m1 == m2:
                    # All pairs in idxs1 conflict
                    for u in idxs1:
                        mask = 0
                        for v in idxs1:
                            if u != v:
                                mask |= (1 << v)
                        conflicts[u] |= mask
                else:
                    for u in idxs1:
                        mask = 0
                        for v in idxs2:
                            mask |= (1 << v)
                        conflicts[u] |= mask
                        # Symmetry
                        for v in idxs2:
                            conflicts[v] |= (1 << u)
                            
    # B&B Stack
    init_mask = (1 << n) - 1
    init_sum = sum(weights)
    
    stack = [(init_mask, Fraction(0), [], init_sum)]
    best_score = Fraction(0)
    best_indices = []
    
    while stack:
        avail, score, chosen, sum_avail = stack.pop()
        
        # Prune
        if score + sum_avail <= best_score:
            continue
            
        if avail == 0:
            if score > best_score:
                best_score = score
                best_indices = chosen
            continue
            
        # Select node u: highest weight in avail (lowest index)
        lsb = avail & -avail
        u = lsb.bit_length() - 1
        
        # Branch 1: Take u
        new_avail_take = avail & ~conflicts[u] & ~(1 << u)
        
        removed_mask = conflicts[u] & avail
        w_removed = 0
        temp = removed_mask
        while temp:
            lsb_t = temp & -temp
            idx_t = lsb_t.bit_length() - 1
            w_removed += weights[idx_t]
            temp ^= lsb_t
            
        new_sum_take = sum_avail - weights[u] - w_removed
        new_score_take = score + weights[u]
        new_chosen_take = chosen + [u]
        
        stack.append((new_avail_take, new_score_take, new_chosen_take, new_sum_take))
        
        # Branch 2: Skip u
        new_avail_skip = avail & ~(1 << u)
        new_sum_skip = sum_avail - weights[u]
        new_score_skip = score
        new_chosen_skip = chosen
        
        stack.append((new_avail_skip, new_score_skip, new_chosen_skip, new_sum_skip))
        
    result = []
    for idx in best_indices:
        m, r = candidates[idx]
        result.append([m, r])
        
    return result

# EVOLVE-BLOCK-END
