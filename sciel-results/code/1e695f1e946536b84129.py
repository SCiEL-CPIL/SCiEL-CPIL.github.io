"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 512
    # Vertices 0..N-1 represent numbers 1..N
    # Build compatibility bitmasks: compat[i] has bit j set if lcm(i+1, j+1) <= N
    compat = [0] * N
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            b = j + 1
            # lcm(a,b) = a*b // gcd(a,b)
            g = gcd(a, b)
            if a * b // g <= N:
                mask |= (1 << j)
        compat[i] = mask
    
    # Greedy initialization: pick vertices in order of descending degree
    degrees = [bin(c).count('1') for c in compat]
    order = sorted(range(N), key=lambda i: -degrees[i])
    
    # Greedy: add vertex if compatible with all in current set
    def greedy_init():
        selected = []
        selected_mask = 0
        for v in order:
            # Check if v is compatible with all selected
            if (compat[v] & selected_mask) == selected_mask:
                selected.append(v)
                selected_mask |= (1 << v)
        return selected, selected_mask
    
    best_set, best_mask = greedy_init()
    best_size = len(best_set)
    
    # Local search: try to swap out one element and add one or two compatible elements
    def try_improve(selected, selected_mask, deadline):
        """Try to improve the set via swaps."""
        n = len(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() >= deadline:
                    break
                v_remove = selected[i]
                # Remove v_remove from mask
                mask_without = selected_mask & ~(1 << v_remove)
                # Find candidates not in set that are compatible with mask_without
                # and also compatible with v_remove's absence
                # A candidate c is valid if (compat[c] & mask_without) == mask_without
                # and c not in selected
                candidates = []
                for c in range(N):
                    if not (selected_mask >> c) & 1:
                        if (compat[c] & mask_without) == mask_without:
                            candidates.append(c)
                
                if len(candidates) >= 1:
                    # Try adding one candidate
                    for c in candidates:
                        new_mask = mask_without | (1 << c)
                        # Check all pairs in new set are compatible
                        # Since c is compatible with mask_without, this is valid
                        new_selected = [v for v in selected if v != v_remove] + [c]
                        if len(new_selected) > len(selected):
                            selected = new_selected
                            selected_mask = new_mask
                            improved = True
                            break
                    if improved:
                        n = len(selected)
                        continue
                
                # Try adding two candidates
                if len(candidates) >= 2:
                    for ci in range(len(candidates)):
                        if time.time() >= deadline:
                            break
                        c1 = candidates[ci]
                        for cj in range(ci + 1, len(candidates)):
                            c2 = candidates[cj]
                            # Check c1 and c2 are compatible
                            if (compat[c1] >> c2) & 1:
                                new_mask = mask_without | (1 << c1) | (1 << c2)
                                new_selected = [v for v in selected if v != v_remove] + [c1, c2]
                                if len(new_selected) > len(selected):
                                    selected = new_selected
                                    selected_mask = new_mask
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        n = len(selected)
                        continue
        
        return selected, selected_mask
    
    # Run local search
    best_set, best_mask = try_improve(best_set, best_mask, deadline)
    best_size = len(best_set)
    
    # Try multiple random restarts
    import random
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() >= deadline:
            break
        
        # Randomized greedy
        shuffled_order = order[:]
        rng.shuffle(shuffled_order)
        selected = []
        selected_mask = 0
        for v in shuffled_order:
            if (compat[v] & selected_mask) == selected_mask:
                selected.append(v)
                selected_mask |= (1 << v)
        
        # Local search
        selected, selected_mask = try_improve(selected, selected_mask, deadline)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
            best_mask = selected_mask
    
    # Convert to actual numbers (1-indexed)
    result = [v + 1 for v in best_set]
    return sorted(result)
# EVOLVE-BLOCK-END
