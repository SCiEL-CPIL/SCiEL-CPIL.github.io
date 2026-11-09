"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations_with_replacement

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # Safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80

    def check_addition(selected, x, triple_sums):
        """Check if x can be added to selected without creating duplicate triple sums.
        Returns (True, new_sums) if valid, (False, None) otherwise."""
        # New triple sums involving x:
        # (x, x, x), (x, x, y) for y in selected, (x, y, y) for y in selected, (x, y, z) for y<=z in selected
        new_sums = []
        # (x,x,x)
        s = 3 * x
        if s in triple_sums:
            return False, None
        new_sums.append(s)
        
        for y in selected:
            # (x, x, y)
            s = 2 * x + y
            if s in triple_sums or s in new_sums:
                return False, None
            new_sums.append(s)
            # (x, y, y)
            s = x + 2 * y
            if s in triple_sums or s in new_sums:
                return False, None
            new_sums.append(s)
        
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                y = selected[i]
                z = selected[j]
                s = x + y + z
                if s in triple_sums or s in new_sums:
                    return False, None
                new_sums.append(s)
        
        return True, new_sums

    def build_greedy_base3():
        """Build a B_3 set using base-3 inspired ordering."""
        selected = []
        triple_sums = set()
        
        # Generate candidates in a smart order: prefer numbers with sparse ternary representations
        # Powers of 3 and their sums give good B_3 structure
        candidates = []
        for mask in range(1, 256):  # 2^8 combinations of powers 3^0..3^7
            val = 0
            for bit in range(8):
                if mask & (1 << bit):
                    val += 3**bit
            if val <= N:
                candidates.append(val)
        
        # Sort candidates - try smallest first for dense packing
        candidates.sort()
        
        for x in candidates:
            if len(selected) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 25:
                break
            
            valid, new_sums = check_addition(selected, x, triple_sums)
            if valid:
                selected.append(x)
                triple_sums.update(new_sums)
        
        return selected, triple_sums

    def local_improve(selected, triple_sums):
        """Try to improve the set by swapping elements to allow more additions."""
        if time.time() > DEADLINE - 20:
            return selected, triple_sums
        
        selected = list(selected)
        triple_sums = set(triple_sums)
        improved = True
        max_iters = 50
        
        while improved and len(selected) < MAX_SIZE and time.time() < DEADLINE - 20:
            improved = False
            for _ in range(max_iters):
                if time.time() > DEADLINE - 20:
                    break
                # Try removing one element and adding a different one
                for rem_idx in range(len(selected) - 1, -1, -1):
                    if time.time() > DEADLINE - 20:
                        break
                    rem_val = selected[rem_idx]
                    # Remove rem_val
                    selected_no_rem = selected[:rem_idx] + selected[rem_idx+1:]
                    # Recompute triple_sums without rem_val - expensive, so try a few candidates
                    # Try adding some larger values
                    for add_val in range(rem_val + 1, min(rem_val + 200, N + 1)):
                        if add_val in selected_no_rem:
                            continue
                        # Quick check: can we add add_val to selected_no_rem?
                        # This is expensive; do a limited check
                        if time.time() > DEADLINE - 20:
                            break
                        # Rebuild triple sums for selected_no_rem is too expensive
                        # Instead, just try the full check
                        test_selected = selected_no_rem + [add_val]
                        test_sums = set()
                        valid = True
                        for a in test_selected:
                            for b in test_selected:
                                for c in test_selected:
                                    if a <= b <= c:
                                        s = a + b + c
                                        if s in test_sums:
                                            valid = False
                                            break
                                        test_sums.add(s)
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        if valid:
                            # Accept the swap
                            selected = test_selected
                            triple_sums = test_sums
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return selected, triple_sums

    def full_greedy_with_local():
        """Main construction: greedy base-3 then local improvement."""
        selected, triple_sums = build_greedy_base3()
        
        if time.time() < DEADLINE - 15:
            selected, triple_sums = local_improve(selected, triple_sums)
        
        return selected

    # Try the main approach
    result = full_greedy_with_local()
    
    # If we have time, try a few random restarts with different orderings
    if time.time() < DEADLINE - 10 and len(result) < MAX_SIZE:
        rng = np.random.default_rng(42)
        best = result
        
        for restart in range(5):
            if time.time() > DEADLINE - 5:
                break
            
            # Random perturbation: shuffle some candidates
            selected = []
            triple_sums = set()
            
            # Start with a random subset of the base-3 candidates
            base3_cands = []
            for mask in range(1, 256):
                val = 0
                for bit in range(8):
                    if mask & (1 << bit):
                        val += 3**bit
                if val <= N:
                    base3_cands.append(val)
            
            # Randomly permute and take first portion
            rng.shuffle(base3_cands)
            
            for x in base3_cands:
                if len(selected) >= MAX_SIZE:
                    break
                if time.time() > DEADLINE - 5:
                    break
                valid, new_sums = check_addition(selected, x, triple_sums)
                if valid:
                    selected.append(x)
                    triple_sums.update(new_sums)
            
            if len(selected) > len(best):
                best = selected
        
        result = best
    
    return result
# EVOLVE-BLOCK-END
