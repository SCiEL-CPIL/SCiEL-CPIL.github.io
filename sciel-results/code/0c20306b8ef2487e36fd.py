import itertools
import random
import time
import sys

sys.setrecursionlimit(10000)

def _is_non_2_colorable_fast(n, edges):
    """
    Fast check for non-2-colorability using bitmask backtracking with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable (i.e., no valid coloring exists).
    """
    if not edges:
        return False

    # Precompute edge bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    full_mask = (1 << n) - 1

    # Precompute vertex degrees for heuristic
    vertex_degree = [0] * n
    for mask in edge_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertex_degree[idx] += 1
            m ^= lsb

    # Precompute for each vertex, list of edge indices containing it
    vertex_to_edges = [[] for _ in range(n)]
    for i, mask in enumerate(edge_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            vertex_to_edges[idx].append(i)
            m ^= lsb

    def propagate(color1, unassigned):
        """
        Apply unit propagation. Returns (new_color1, new_unassigned, conflict_flag).
        If conflict, returns (None, None, True).
        """
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                # Intersect with assigned vertices
                assigned_in_edge = mask & ~unassigned

                # Check if all 5 assigned
                if assigned_in_edge == mask:
                    # If all color 1 or all color 0, conflict
                    if (mask & ~color1 & full_mask) == 0:  # All color 1
                        return None, None, True
                    if (mask & color1) == 0:  # All color 0
                        return None, None, True
                    continue

                # Check for forcing
                unassigned_in_edge = mask & unassigned
                # If exactly 1 unassigned
                if unassigned_in_edge and (unassigned_in_edge & (unassigned_in_edge - 1)) == 0:
                    other_4 = mask ^ unassigned_in_edge
                    # Are all 4 color 1?
                    if (other_4 & ~color1 & full_mask) == 0 and (other_4 & color1) == other_4:
                        # Force to color 0
                        unassigned ^= unassigned_in_edge
                        changed = True
                    # Are all 4 color 0?
                    elif (other_4 & color1) == 0:
                        # Force to color 1
                        color1 |= unassigned_in_edge
                        unassigned ^= unassigned_in_edge
                        changed = True

        return color1, unassigned, False

    def backtrack(color1, unassigned):
        # Check for immediate conflict via propagation
        color1, unassigned, conflict = propagate(color1, unassigned)
        if conflict:
            return False

        if unassigned == 0:
            # All assigned. Check if any edge is monochromatic.
            for mask in edge_masks:
                if (mask & ~color1 & full_mask) == 0:  # All color 1
                    return False
                if (mask & color1) == 0:  # All color 0
                    return False
            return True  # Valid coloring found

        # Pick a vertex to assign: highest degree among unassigned
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
        new_color1 = color1 | v_bit
        new_unassigned = unassigned ^ v_bit
        if backtrack(new_color1, new_unassigned):
            return True

        # Try color 0
        new_color1 = color1
        new_unassigned = unassigned ^ v_bit
        if backtrack(new_color1, new_unassigned):
            return True

        return False

    # Symmetry breaking: Fix vertex 0 to color 0.
    initial_color1 = 0
    initial_unassigned = full_mask ^ 1  # Exclude vertex 0

    initial_color1, initial_unassigned, conflict = propagate(initial_color1, initial_unassigned)
    if conflict:
        return True

    if initial_unassigned == 0:
        for mask in edge_masks:
            if (mask & ~initial_color1 & full_mask) == 0:
                return False
            if (mask & initial_color1) == 0:
                return False
        return False  # Coloring found

    if backtrack(initial_color1, initial_unassigned):
        return False  # Coloring found

    return True  # No coloring found


def _find_minimal_non2col(n, all_edges, deadline, rng):
    """
    Randomized local search to find a small non-2-colorable subgraph.
    Strategy: 
    1. Start with a random subset of edges
    2. If it's non-2-colorable, try to remove edges greedily
    3. If it's 2-colorable, try to add edges from a pool
    """
    m_all = len(all_edges)
    best_edges = None
    best_len = m_all
    
    # First, try greedy removal from full set with random ordering
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        current = all_edges[:]
        rng.shuffle(current)
        
        # Greedy removal
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            rng.shuffle(current)
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    break
                new_edges = current[:i] + current[i+1:]
                if _is_non_2_colorable_fast(n, new_edges):
                    current = new_edges
                    removed_any = True
        
        if len(current) < best_len:
            best_len = len(current)
            best_edges = current[:]
        
        if best_len <= 35:
            break
    
    return best_edges, best_len


def _random_local_search(n, all_edges, deadline, rng):
    """
    Random restart local search: start from random subsets, try to minimize.
    """
    best_edges = None
    best_len = len(all_edges)
    m_all = len(all_edges)
    
    # Try different initial sizes
    for target_size in [60, 55, 50, 45, 40]:
        if time.time() > deadline:
            break
        
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Start with a random subset of target_size
            current = rng.sample(all_edges, min(target_size, m_all))
            current_set = set(map(tuple, current))
            
            # Check if non-2-colorable
            if not _is_non_2_colorable_fast(n, current):
                # It's 2-colorable, try adding edges
                remaining = [e for e in all_edges if tuple(e) not in current_set]
                rng.shuffle(remaining)
                
                added = 0
                for e in remaining:
                    if time.time() > deadline:
                        break
                    if added >= 10:
                        break
                    new_current = current + [e]
                    if _is_non_2_colorable_fast(n, new_current):
                        current = new_current
                        current_set.add(tuple(e))
                        added += 1
            
            # Now try to minimize
            if _is_non_2_colorable_fast(n, current):
                # Greedy removal
                removed_any = True
                while removed_any and time.time() < deadline:
                    removed_any = False
                    rng.shuffle(current)
                    for i in range(len(current) - 1, -1, -1):
                        if time.time() > deadline:
                            break
                        new_edges = current[:i] + current[i+1:]
                        if _is_non_2_colorable_fast(n, new_edges):
                            current = new_edges
                            removed_any = True
                
                if len(current) < best_len:
                    best_len = len(current)
                    best_edges = current[:]
                
                if best_len <= 35:
                    return best_edges, best_len
    
    return best_edges, best_len


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Start with K9^(5)
    n = 9
    all_edges = [list(e) for e in itertools.combinations(range(n), 5)]

    # Method 1: Randomized greedy removal
    rng = random.Random(42)
    edges1, len1 = _find_minimal_non2col(n, all_edges, deadline, rng)

    # Method 2: Random local search
    if time.time() < deadline:
        rng2 = random.Random(123)
        edges2, len2 = _random_local_search(n, all_edges, deadline, rng2)
        
        if edges2 is not None and len2 < len1:
            edges1 = edges2
            len1 = len2

    # Final verification
    if edges1 is None:
        edges1 = all_edges[:]

    if not _is_non_2_colorable_fast(n, edges1):
        # Fallback to full K9 if something went wrong
        return {'n': n, 'edges': all_edges}

    return {'n': n, 'edges': edges1}
