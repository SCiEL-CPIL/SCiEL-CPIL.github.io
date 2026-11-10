# EVOLVE-BLOCK-START
import numpy as np
import time


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero
    quadratic residue modulo 205. Uses exact Branch-and-Bound with bitsets
    to maximize the set size.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 580  # 580s safety margin for 630s limit

    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Precompute conflict masks
    # conflict_mask[i] is a bitmask of vertices j such that (i - j) % m is in forbidden
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask

    best_size = 0
    best_set = []
    
    # Helper to update best
    def update_best(current_set):
        nonlocal best_size, best_set
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

    # Branch and Bound
    # cands: bitmask of available candidates
    # current_set: list of vertices in current independent set
    def search(cands, current_set):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return

        # Update best if current is better
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

        if cands == 0:
            return

        # Upper bound: current size + number of candidates
        # If even taking all candidates doesn't beat best, prune
        if len(current_set) + cands.bit_count() <= best_size:
            return

        # Select a vertex to branch on
        # Heuristic: Choose vertex with highest degree in the candidate subgraph
        # This tends to reduce the search space quickly when including it.
        best_v = -1
        max_deg = -1
        temp_cands = cands
        while temp_cands:
            # Get least significant set bit
            lsb = temp_cands & -temp_cands
            v = lsb.bit_length() - 1
            temp_cands ^= lsb
            
            # Calculate degree of v within cands
            # Neighbors of v in cands are cands & conflict_mask[v]
            deg = (cands & conflict_mask[v]).bit_count()
            
            if deg > max_deg:
                max_deg = deg
                best_v = v

        if best_v == -1:
            return

        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # New candidates: remove best_v and all its neighbors
        new_cands_include = cands & ~conflict_mask[best_v] & ~v_bit
        search(new_cands_include, current_set + [best_v])

        # Branch 2: Exclude best_v
        # New candidates: remove best_v
        new_cands_exclude = cands & ~v_bit
        search(new_cands_exclude, current_set)

    # Initial candidates: all vertices 0..204
    initial_cands = (1 << m) - 1
    search(initial_cands, [])

    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
