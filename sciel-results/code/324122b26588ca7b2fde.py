import itertools
import random
import time
import sys

sys.setrecursionlimit(10000)

def _is_non_2_colorable_fast(n, edges):
    """
    Fast check for non-2-colorability using bitmask backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable.
    """
    if not edges:
        return False

    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    full_mask = (1 << n) - 1

    # Precompute vertex degrees
    vertex_degree = [0] * n
    for mask in edge_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertex_degree[idx] += 1
            m ^= lsb

    # Precompute vertex to edges mapping
    vertex_to_edges = [[] for _ in range(n)]
    for i, mask in enumerate(edge_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertex_to_edges[idx].append(i)
            m ^= lsb

    def propagate(color1, unassigned):
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                # Check if all assigned
                if (mask & unassigned) == 0:
                    # All assigned. Check monochromatic
                    if (mask & ~color1 & full_mask) == 0: # All 1
                        return None, None, True
                    if (mask & color1) == 0: # All 0
                        return None, None, True
                    continue

                # Check for forcing
                unassigned_in_edge = mask & unassigned
                # If exactly 1 unassigned
                if unassigned_in_edge and (unassigned_in_edge & (unassigned_in_edge - 1)) == 0:
                    other_4 = mask ^ unassigned_in_edge
                    # Are all 4 assigned? Yes, because unassigned_in_edge is the only unassigned.
                    # Check if other 4 are all color 1
                    if (other_4 & ~color1 & full_mask) == 0 and (other_4 & color1) == other_4:
                        # Force to 0
                        unassigned ^= unassigned_in_edge
                        changed = True
                    # Are all 4 color 0?
                    elif (other_4 & color1) == 0:
                        # Force to 1
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True

        return color1, unassigned, False

    def backtrack(color1, unassigned):
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False

        if unassigned == 0:
            # All assigned, verify no monochromatic edge (should be caught by propagate, but safe)
            for mask in edge_masks:
                if (mask & ~color1 & full_mask) == 0:
                    return False
                if (mask & color1) == 0:
                    return False
            return True

        # Pick vertex with highest degree among unassigned
        best_v_bit = 0
        best_deg = -1
        m = unassigned
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if vertex_degree[idx] > best_deg:
                best_deg = vertex_degree[idx]
                best_v_bit = lsb
            m ^= lsb

        v_bit = best_v_bit

        # Try color 1
        if backtrack(color1 | v_bit, unassigned ^ v_bit):
            return True

        # Try color 0
        if backtrack(color1, unassigned ^ v_bit):
            return True

        return False

    # Symmetry breaking: Fix vertex 0 to color 0
    initial_color1 = 0
    initial_unassigned = full_mask ^ 1

    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True

    if initial_unassigned == 0:
        for mask in edge_masks:
            if (mask & ~initial_color1 & full_mask) == 0:
                return False
            if (mask & initial_color1) == 0:
                return False
        return False

    if backtrack(initial_color1, initial_unassigned):
        return False

    return True


def _local_search(n, edges, deadline, rng):
    """
    Local search with removal and swap moves to minimize edges.
    """
    current_edges = edges[:]
    rng.shuffle(current_edges)
    
    # Pre-generate a pool of random edges for swaps
    all_possible_edges = [list(e) for e in itertools.combinations(range(n), 5)]
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        rng.shuffle(current_edges)
        
        # Phase 1: Try to remove edges
        for i in range(len(current_edges) - 1, -1, -1):
            if time.time() > deadline:
                return current_edges
            
            e = current_edges[i]
            new_edges = current_edges[:i] + current_edges[i+1:]
            
            if _is_non_2_colorable_fast(n, new_edges):
                current_edges = new_edges
                improved = True
                break # Restart loop with new state
        
        # Phase 2: If no removal, try swaps
        if not improved:
            # Try a few swaps
            for _ in range(10):
                if time.time() > deadline:
                    return current_edges
                
                if len(current_edges) == 0:
                    break
                
                # Pick a random edge to remove
                idx = rng.randint(0, len(current_edges) - 1)
                e_to_remove = current_edges[idx]
                
                # Pick a random edge to add (not currently in the set)
                # To speed up, pick from all_possible_edges
                e_to_add = rng.choice(all_possible_edges)
                
                if e_to_add in current_edges:
                    continue
                
                new_edges = current_edges[:idx] + current_edges[idx+1:]
                new_edges.append(e_to_add)
                
                if _is_non_2_colorable_fast(n, new_edges):
                    current_edges = new_edges
                    improved = True
                    break

    return current_edges


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Try starting with K8^(5) first (56 edges)
    # If it's non-2-colorable, we have a much better starting point
    n_candidate = 8
    edges_8 = [list(e) for e in itertools.combinations(range(n_candidate), 5)]
    
    if _is_non_2_colorable_fast(n_candidate, edges_8):
        n = n_candidate
        initial_edges = edges_8
    else:
        # Fall back to K9^(5) (126 edges)
        n_candidate = 9
        n = n_candidate
        initial_edges = [list(e) for e in itertools.combinations(range(n_candidate), 5)]
    
    best_edges = None
    best_len = len(initial_edges)
    
    # Multiple randomized restarts
    seeds = [42, 123, 7, 999, 2024, 555, 888, 1337, 2718, 3141, 1, 2, 3]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        reduced = _local_search(n, initial_edges, deadline, rng)
        
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_edges = reduced[:]
            
        # Early stop if we hit the theoretical lower bound or known best
        if best_len <= 51:
            break

    if best_edges is None:
        best_edges = initial_edges[:]

    # Final verification
    if not _is_non_2_colorable_fast(n, best_edges):
        # Fallback to initial
        best_edges = initial_edges[:]

    return {'n': n, 'edges': best_edges}
