"""Maximum independent set on square-sum conflict graph using bitmask branch-and-bound."""
import math
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # Filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict graph using bitmasks
    # conflict_mask[i] has bit j set if valid_numbers[i] conflicts with valid_numbers[j]
    conflict_mask = [0] * num_valid
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                j = val_to_idx[b]
                conflict_mask[i] |= (1 << j)
    
    # Precompute degrees
    degrees = [conflict_mask[i].bit_count() for i in range(num_valid)]
    
    # Greedy initialization
    def greedy_initial():
        active_mask = 0
        # Sort by degree ascending
        order = sorted(range(num_valid), key=lambda x: degrees[x])
        for v in order:
            if (conflict_mask[v] & active_mask) == 0:
                active_mask |= (1 << v)
        return active_mask
    
    # Branch and bound maximum independent set
    def branch_and_bound(candidates, incumbent_size, incumbent_mask, time_limit):
        """
        Find maximum independent set from candidates bitmask.
        Uses branch and bound with greedy upper bound.
        """
        best_size = incumbent_size
        best_mask = incumbent_mask
        t_start = time.time()
        
        # Greedy upper bound: repeatedly pick vertex with min degree in remaining graph
        def greedy_bound(cand):
            size = 0
            remaining = cand
            while remaining:
                size += 1
                # Pick vertex with minimum degree in remaining subgraph
                best_v = -1
                best_deg = float('inf')
                r = remaining
                while r:
                    lsb = r & (-r)
                    v = lsb.bit_length() - 1
                    deg = (conflict_mask[v] & remaining).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    r ^= lsb
                # Remove best_v and its neighbors from remaining
                remaining &= ~(conflict_mask[best_v] | (1 << best_v))
            return size
        
        # Recursive search
        def search(cand, current_mask, current_size):
            nonlocal best_size, best_mask
            
            if time.time() - t_start > time_limit:
                return
            
            if cand == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                return
            
            # Upper bound
            ub = current_size + greedy_bound(cand)
            if ub <= best_size:
                return
            
            # Pick pivot vertex
            # Heuristic: pick vertex with max degree in candidates
            best_v = -1
            best_deg = -1
            r = cand
            while r:
                lsb = r & (-r)
                v = lsb.bit_length() - 1
                deg = (conflict_mask[v] & cand).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                r ^= lsb
            
            if best_v < 0:
                return
            
            # Branch 1: include best_v
            new_mask = current_mask | (1 << best_v)
            # New candidates: remove best_v and all its neighbors
            new_cand = cand & ~(conflict_mask[best_v] | (1 << best_v))
            search(new_cand, new_mask, current_size + 1)
            
            # Branch 2: exclude best_v
            new_cand2 = cand & ~(1 << best_v)
            search(new_cand2, current_mask, current_size)
        
        search(candidates, 0, 0)
        
        return best_size, best_mask
    
    # Local search refinement
    def local_search(mask, time_limit):
        """Improve mask via add/remove/swap moves."""
        active = mask
        best_size = active.bit_count()
        best_mask = active
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            improved = False
            
            # Try to add vertices
            r = (~active) & ((1 << num_valid) - 1)
            while r:
                lsb = r & (-r)
                v = lsb.bit_length() - 1
                if (conflict_mask[v] & active) == 0:
                    active |= lsb
                    improved = True
                    if active.bit_count() > best_size:
                        best_size = active.bit_count()
                        best_mask = active
                r ^= lsb
            
            if not improved:
                # Try swaps: remove one, try to add different
                r = active
                swapped = False
                while r and time.time() - t_start < time_limit:
                    lsb = r & (-r)
                    v = lsb.bit_length() - 1
                    active_without_v = active & ~lsb
                    
                    # Try to add a vertex that conflicts with v
                    # But first, try to add any available vertex
                    r2 = (~active_without_v) & ((1 << num_valid) - 1)
                    while r2:
                        lsb2 = r2 & (-r2)
                        w = lsb2.bit_length() - 1
                        if w != v and (conflict_mask[w] & active_without_v) == 0:
                            new_active = active_without_v | lsb2
                            if new_active.bit_count() > best_size:
                                best_size = new_active.bit_count()
                                best_mask = new_active
                                active = new_active
                                swapped = True
                                break
                        r2 ^= lsb2
                    
                    if swapped:
                        break
                    r ^= lsb
            
            if not improved and not swapped:
                # No improvement possible, break
                break
        
        return best_mask
    
    # Main loop
    best_global_mask = 0
    best_global_size = 0
    
    # Phase 1: Greedy initialization + local search
    greedy_mask = greedy_initial()
    ls_mask = local_search(greedy_mask, time_limit=10.0)
    best_global_mask = ls_mask
    best_global_size = ls_mask.bit_count()
    
    # Phase 2: Branch and bound from current best
    # Use the complement of current solution as initial candidates
    # Actually, B&B works on the full graph. Let's run it with a time budget.
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        bb_size, bb_mask = branch_and_bound(
            (1 << num_valid) - 1,  # All vertices as candidates
            best_global_size,
            best_global_mask,
            time_limit=remaining_time - 3
        )
        if bb_size > best_global_size:
            best_global_size = bb_size
            best_global_mask = bb_mask
    
    # Phase 3: Final local search refinement
    remaining_time = deadline - time.time()
    if remaining_time > 1:
        final_mask = local_search(best_global_mask, time_limit=min(remaining_time - 1, 5.0))
        if final_mask.bit_count() > best_global_size:
            best_global_size = final_mask.bit_count()
            best_global_mask = final_mask
    
    # Convert mask to list of numbers
    result = []
    for v in range(num_valid):
        if (best_global_mask >> v) & 1:
            result.append(valid_numbers[v])
    
    if not result:
        result = [1]
    
    return sorted(result)

# EVOLVE-BLOCK-END
