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
    
    # Sort edges by some heuristic? 
    # Random shuffle is standard, but maybe we can prioritize "central" edges?
    # Let's stick to random shuffle for robustness, but do multiple passes.
    
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
    deadline = start_time + 110  # Safety margin

    best_edges = None
    best_len = float('inf')
    best_n = 9

    # Try different n values. Smaller n might be denser, larger n might allow sparser non-2-colorable graphs.
    # The minimum m(5) is likely achieved for some n > 9.
    # Let's try n from 9 to 14.
    
    for n in range(9, 15):
        if time.time() > deadline - 10: # Reserve 10s for final checks
            break
            
        # Generate all possible edges
        all_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        
        # If the total number of edges is already larger than our best, we can still try to reduce it
        # But if n is large, K_n^(5) is huge. 
        # For n=12, C(12,5) = 792. 
        # Greedy removal from 792 edges might be slow.
        
        # Strategy: 
        # 1. If n is small (9, 10), start from K_n^(5).
        # 2. If n is larger, maybe start from a random subset? 
        #    Actually, greedy removal from K_n is the most reliable way to find a minimal subgraph.
        #    But it's slow.
        
        # Let's limit the number of restarts for larger n.
        num_restarts = 5 if n <= 10 else 3
        
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            rng = random.Random(seed * 100 + n)
            
            # For larger n, K_n is too big to process quickly in greedy removal?
            # Each check takes O(E * 2^N) roughly? No, backtracking is exponential in N.
            # N=12 is 4096 leaves max. Very fast.
            # So the bottleneck is the number of checks.
            # 792 edges * 792 checks = ~600k checks. 
            # If each check is 0.1ms, that's 60s. Too slow.
            
            # Optimization: 
            # We can use a "random subset" start for larger n?
            # If we start with a random subset of size K, and it's non-2-colorable, we reduce it.
            # If it's 2-colorable, we add edges?
            
            # Let's stick to n=9 and n=10 for the main search, as they are fast.
            # For n=11, 12, we might timeout.
            
            if n > 10 and seed > 0:
                continue # Only try first restart for larger n to save time?
                # Actually, let's just try n=9 and n=10 thoroughly.
                # The best known 51-edge graph likely has n around 9-11.
                
            reduced = _greedy_remove(n, all_edges, deadline, rng)

            if len(reduced) < best_len:
                best_len = len(reduced)
                best_edges = reduced[:]
                best_n = n

            # If we found a very good solution, we can stop early
            if best_len <= 35:
                break

    # Final verification
    if best_edges is None:
        # Fallback to K9
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_n = n

    if not _is_non_2_colorable_fast(best_n, best_edges):
        # Fallback to full K9 if something went wrong
        n = 9
        best_edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
