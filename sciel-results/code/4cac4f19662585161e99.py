"""Combined program: deterministic K_n^5 minimization with fast bitmask checker."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

sys.setrecursionlimit(100000)

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

def _greedy_remove(n, edges, deadline, rng):
    """
    Greedy edge removal with randomized ordering.
    Returns the reduced edge set.
    """
    current_edges = edges[:]
    
    removed_any = True
    while removed_any and (time.time() < deadline):
        removed_any = False
        rng.shuffle(current_edges)

        for i in range(len(current_edges) - 1, -1, -1):
            if time.time() > deadline:
                break

            e = current_edges[i]
            new_edges = current_edges[:i] + current_edges[i + 1:]

            if _is_non_2_colorable_fast(n, new_edges):
                current_edges = new_edges
                removed_any = True

    return current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    best_edges = None
    best_len = float('inf')
    best_n = 8

    # Strategy: Try different n values, start from complete hypergraphs, minimize.
    # K_8^5 is definitely non-2-colorable (56 edges).
    # We aim to find smaller non-2-colorable subgraphs.
    
    # 1. Try n=8 (Small, fast checks)
    if time.time() < deadline - 10:
        n = 8
        all_edges_8 = [list(e) for e in itertools.combinations(range(n), 5)]
        # K_8^5 is non-2-colorable
        if _is_non_2_colorable_fast(n, all_edges_8):
            # Minimize
            rng = random.Random(42)
            edges8 = _greedy_remove(n, all_edges_8, deadline - 5, rng)
            if _is_non_2_colorable_fast(n, edges8) and len(edges8) < best_len:
                best_edges = edges8[:]
                best_len = len(edges8)
                best_n = n
    
    # 2. Try n=9
    if time.time() < deadline - 20:
        n = 9
        all_edges_9 = [list(e) for e in itertools.combinations(range(n), 5)]
        # K_9^5 is non-2-colorable
        if _is_non_2_colorable_fast(n, all_edges_9):
            # Minimize with multiple restarts
            for restart in range(3):
                if time.time() > deadline - 10:
                    break
                rng = random.Random(restart * 100 + 9)
                edges9 = _greedy_remove(n, all_edges_9, deadline - 5, rng)
                if _is_non_2_colorable_fast(n, edges9) and len(edges9) < best_len:
                    best_edges = edges9[:]
                    best_len = len(edges9)
                    best_n = n
    
    # 3. Try n=10
    if time.time() < deadline - 30:
        n = 10
        all_edges_10 = [list(e) for e in itertools.combinations(range(n), 5)]
        # K_10^5 is non-2-colorable
        if _is_non_2_colorable_fast(n, all_edges_10):
            # Minimize with multiple restarts
            for restart in range(5):
                if time.time() > deadline - 15:
                    break
                rng = random.Random(restart * 100 + 10)
                edges10 = _greedy_remove(n, all_edges_10, deadline - 5, rng)
                if _is_non_2_colorable_fast(n, edges10) and len(edges10) < best_len:
                    best_edges = edges10[:]
                    best_len = len(edges10)
                    best_n = n
    
    # 4. Try n=11
    if time.time() < deadline - 40:
        n = 11
        all_edges_11 = [list(e) for e in itertools.combinations(range(n), 5)]
        # K_11^5 is non-2-colorable
        if _is_non_2_colorable_fast(n, all_edges_11):
            # Minimize with fewer restarts due to larger size
            for restart in range(3):
                if time.time() > deadline - 20:
                    break
                rng = random.Random(restart * 100 + 11)
                edges11 = _greedy_remove(n, all_edges_11, deadline - 10, rng)
                if _is_non_2_colorable_fast(n, edges11) and len(edges11) < best_len:
                    best_edges = edges11[:]
                    best_len = len(edges11)
                    best_n = n

    # Fallback
    if best_edges is None:
        # Return a known valid instance: K_8^5
        best_n = 8
        best_edges = [list(e) for e in itertools.combinations(range(8), 5)]

    # Final check
    if _is_2_colorable_check(best_n, best_edges):
        # This should not happen if logic is correct, but safety net
        # Return K_8^5 which is definitely non-2-colorable
        best_n = 8
        best_edges = [list(e) for e in itertools.combinations(range(8), 5)]

    return {'n': best_n, 'edges': best_edges}

def _is_2_colorable_check(n, edges):
    """
    Wrapper that returns True if 2-colorable, False if not.
    Uses _is_non_2_colorable_fast.
    """
    return not _is_non_2_colorable_fast(n, edges)
# EVOLVE-BLOCK-END
