import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)
    
    # 2. Build Adjacency Bitsets
    # neighbors[i] is a bitset where bit j is 1 if i and j are adjacent (conflict)
    neighbors = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        neighbors[i] = mask

    # Precompute complement masks? Not directly, but we use bit operations.
    
    # 3. Branch and Bound Logic
    best_set = []
    best_size = 0
    
    def update_best(current_set):
        nonlocal best_set, best_size
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

    # Improved Upper Bound:
    # We can use a greedy coloring of the complement graph? No, that's for the graph itself.
    # For MIS, an upper bound is the number of colors in a greedy coloring of the *induced subgraph*?
    # No, MIS size <= number of colors in a coloring of the *complement*? No.
    # Standard bound: If we can color the graph with k colors, then the independence number is at least n/k (lower bound).
    # For an UPPER bound on the MIS, we can use the fact that the MIS of a graph G is the clique number of G^c.
    # Or, we can use a greedy matching bound:
    # Let M be a maximal matching in the induced subgraph.
    # Then |IS| <= |V| - |M|.
    # This is a valid upper bound because each edge in the matching requires removing at least one vertex.
    # Since the matching is maximal, the remaining vertices form an independent set? No.
    # But the vertices not covered by the matching form an independent set? No.
    # However, any independent set can contain at most one endpoint of each edge in a matching.
    # So |IS| <= |V| - |M|.
    # This bound is cheap to compute if we have a fast greedy matching.

    def greedy_matching_bound(candidates_mask):
        # Returns a maximal matching size
        remaining = candidates_mask
        matching_size = 0
        while remaining:
            # Pick a vertex with minimum degree? Or just the first one?
            # To be fast, just pick the first set bit.
            # But a better matching gives a tighter bound.
            # Let's pick a vertex with min degree in the remaining subgraph.
            min_deg = m + 1
            pivot = -1
            # Iterate through bits
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                deg = (neighbors[idx] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    pivot = idx
                if deg == 0:
                    break
                temp ^= lsb
            
            if pivot == -1:
                break
            
            if min_deg == 0:
                # Isolated vertex, cannot be in a matching edge
                remaining &= ~(1 << pivot)
                continue

            # Find a neighbor
            neigh_mask = neighbors[pivot] & remaining
            lsb = neigh_mask & (-neigh_mask)
            neighbor_idx = lsb.bit_length() - 1
            
            matching_size += 1
            remaining &= ~(1 << pivot)
            remaining &= ~(1 << neighbor_idx)
            
        return matching_size

    def upper_bound(candidates_mask):
        # Combine bounds:
        # 1. |V|
        # 2. |V| - |Matching|
        # 3. Maybe a simple clique bound?
        # For now, use the matching bound.
        # Note: If the graph is dense, the matching size will be large, giving a tighter bound.
        num_v = candidates_mask.bit_count()
        if num_v == 0:
            return 0
        match_size = greedy_matching_bound(candidates_mask)
        return num_v - match_size

    # Heuristic: Pick a vertex to branch on.
    # Strategy: Pick vertex with maximum degree in the induced subgraph?
    # Or random? 
    # Let's try picking the vertex with the highest degree among candidates to prune faster.
    
    def bnb(candidates_mask, current_set, depth):
        nonlocal best_size, best_set
        
        # Pruning: if current size + max possible from candidates <= best_size, return
        # Use the tighter upper bound
        ub = upper_bound(candidates_mask)
        if len(current_set) + ub <= best_size:
            return
            
        if candidates_mask == 0:
            update_best(current_set)
            return

        # Check time
        if time.time() > deadline:
            return

        # Get list of candidates
        cands = [i for i in range(m) if (candidates_mask >> i) & 1]
        
        if not cands:
            update_best(current_set)
            return

        # Calculate degrees within candidate set
        # deg[i] = number of neighbors of i that are also in candidates
        max_deg = -1
        pivot = -1
        for i in cands:
            # Count bits in (neighbors[i] & candidates_mask)
            deg = (neighbors[i] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = i
        
        if pivot == -1:
            pivot = cands[0]

        # Branch 1: Include pivot
        # New candidates = candidates - {pivot} - neighbors(pivot)
        new_cands_1 = candidates_mask & ~(1 << pivot) & ~neighbors[pivot]
        current_set.append(pivot)
        
        # Recursive call
        # Use the tighter bound here too?
        # The check inside bnb will do it.
        bnb(new_cands_1, current_set, depth + 1)
        
        current_set.pop()

        # Branch 2: Exclude pivot
        new_cands_2 = candidates_mask & ~(1 << pivot)
        
        # Recursive call
        bnb(new_cands_2, current_set, depth + 1)

    # Initial setup
    # To improve B&B, we can order the initial candidates.
    # However, standard B&B with max-degree pivot is usually robust for N=205.
    # We will run multiple rounds with different random pivots or initial greedy solutions 
    # to set a high lower bound (best_size) early.

    # Phase 1: Greedy to get a good initial lower bound
    def greedy_random():
        remaining_mask = (1 << m) - 1
        selected = []
        while remaining_mask:
            cands = [i for i in range(m) if (remaining_mask >> i) & 1]
            if not cands:
                break
            # Pick random candidate
            pivot = random.choice(cands)
            selected.append(pivot)
            remaining_mask &= ~(1 << pivot)
            remaining_mask &= ~neighbors[pivot]
        return selected

    # Run several greedy iterations to find a good starting point
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        gs = greedy_random()
        if len(gs) > best_size:
            best_size = len(gs)
            best_set = gs

    # Phase 2: Branch and Bound
    # We need to ensure we don't run out of time.
    # Since N=205, the state space is large, but the graph is dense (forbidden set is large).
    # Density: Number of forbidden diffs?
    # Squares mod 205 (5*41). 
    # Legendre symbol properties.
    # If graph is dense, MIS is small, B&B should be fast.
    
    # Let's run B&B starting from empty set.
    # To make it deterministic and robust, we'll use a specific pivot strategy.
    # The max-degree pivot is good.
    
    # We'll run B&B in a loop, allowing it to run for the remaining time.
    # Since B&B is recursive, we can't easily "pause" it without a global flag.
    # The `time.time() > deadline` check inside `bnb` handles this.
    
    # To allow multiple independent B&B runs with different orderings/heuristics if time permits:
    # We can randomize the tie-breaking in pivot selection.
    
    random.seed(42)
    
    for run_id in range(10):
        if time.time() > deadline - 5:
            break
        
        # Modify pivot selection slightly for diversity?
        # Or just rely on the deterministic max-degree.
        # If max-degree is tied, pick random among ties.
        
        def bnb_randomized(candidates_mask, current_set, depth):
            nonlocal best_size, best_set
            
            # Use the tighter upper bound
            ub = upper_bound(candidates_mask)
            if len(current_set) + ub <= best_size:
                return
            
            if candidates_mask == 0:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                return

            if time.time() > deadline:
                return

            cands = [i for i in range(m) if (candidates_mask >> i) & 1]
            if not cands:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                return

            # Find max degree
            max_deg = -1
            pivots = []
            for i in cands:
                deg = (neighbors[i] & candidates_mask).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivots = [i]
                elif deg == max_deg:
                    pivots.append(i)
            
            pivot = random.choice(pivots)

            # Branch 1: Include pivot
            new_cands_1 = candidates_mask & ~(1 << pivot) & ~neighbors[pivot]
            current_set.append(pivot)
            
            # Check if worth recursing
            ub1 = upper_bound(new_cands_1)
            if len(current_set) + ub1 > best_size:
                bnb_randomized(new_cands_1, current_set, depth + 1)
            
            current_set.pop()

            # Branch 2: Exclude pivot
            new_cands_2 = candidates_mask & ~(1 << pivot)
            
            # Check if worth recursing
            ub2 = upper_bound(new_cands_2)
            if len(current_set) + ub2 > best_size:
                bnb_randomized(new_cands_2, current_set, depth + 1)

        full_mask = (1 << m) - 1
        bnb_randomized(full_mask, [], 0)

    # Final verification
    result = sorted(set(best_set))
    
    # Double check validity
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = (result[i] - result[j]) % m
            if d in forbidden:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a simple greedy if something went wrong
        result = greedy_random()

    return np.array(result, dtype=np.int64)

if __name__ == "__main__":
    res = furstenberg_sarkozy_205()
    print(f"Size: {len(res)}")
    print(res)
