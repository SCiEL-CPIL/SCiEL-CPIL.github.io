import time
import random
import sys
from itertools import combinations

sys.setrecursionlimit(100000)


def _is_2_colorable(n, edges):
    """Exact check (backtracking + unit propagation) for a proper 2-colouring.
    Returns True iff the hypergraph IS 2-colorable."""
    if not edges:
        return True
    
    m = len(edges)
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    all_vars = (1 << n) - 1
    
    # Precompute for each edge: list of vertices
    edge_verts = [list(e) for e in edges]
    
    def backtrack(assign, unassigned):
        # Unit propagation loop
        changed = True
        while changed:
            changed = False
            for mask in edge_masks:
                assigned_in_edge = mask & ~unassigned
                unassigned_in_edge = mask & unassigned
                
                if unassigned_in_edge == 0:
                    if (assign & mask) == mask or (assign & mask) == 0:
                        return None
                else:
                    assigned_zero = assigned_in_edge & ~assign
                    assigned_one = assigned_in_edge & assign
                    unassigned_count = bin(unassigned_in_edge).count('1')
                    
                    if assigned_one == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign |= (1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
                    
                    if assigned_zero == 0 and unassigned_count == 1:
                        v = unassigned_in_edge.bit_length() - 1
                        assign &= ~(1 << v)
                        unassigned &= ~(1 << v)
                        changed = True
                        continue
        
        if unassigned == 0:
            for mask in edge_masks:
                if (assign & mask) == mask or (assign & mask) == 0:
                    return None
            return assign
        
        # Branch: pick unassigned variable with most constraints (heuristic)
        # Simple: pick highest-numbered unassigned
        v = unassigned.bit_length() - 1
        bit = 1 << v
        
        res = backtrack(assign | bit, unassigned & ~bit)
        if res is not None:
            return res
        return backtrack(assign & ~bit, unassigned & ~bit)
    
    return backtrack(0, all_vars) is not None


def _compute_edge_coloring_sets_fast(n, all_edges):
    """Compute for each 5-edge the bitset of colorings where it's monochromatic.
    
    For n=11, colorings are 0..2047. For edge {v0,v1,v2,v3,v4}:
    - All-0 colorings: vertices v0..v4 are 0, others free -> 2^(n-5) colorings
    - All-1 colorings: vertices v0..v4 are 1, others free -> 2^(n-5) colorings
    
    We compute these as Python ints (bitsets over 2^n bits).
    """
    num_colorings = 1 << n
    free_bits = n - 5
    # Better approach: for each edge, the all-0 colorings are those where
    # bits at positions v0..v4 are 0. The free bits are the other n-5 positions.
    # We can enumerate: for each assignment of free bits, construct the coloring.
    
    edge_sets = []
    for edge in all_edges:
        edge_set = 0
        # Free vertices
        free_verts = [v for v in range(n) if v not in edge]
        
        # All-0: edge vertices are 0, free vertices vary
        for free_mask_val in range(1 << len(free_verts)):
            coloring = 0
            for i, v in enumerate(free_verts):
                if free_mask_val & (1 << i):
                    coloring |= (1 << v)
            edge_set |= (1 << coloring)
        
        # All-1: edge vertices are 1, free vertices vary
        edge_val = 0
        for v in edge:
            edge_val |= (1 << v)
        for free_mask_val in range(1 << len(free_verts)):
            coloring = edge_val
            for i, v in enumerate(free_verts):
                if free_mask_val & (1 << i):
                    coloring |= (1 << v)
            edge_set |= (1 << coloring)
        
        edge_sets.append(edge_set)
    
    return edge_sets


def _greedy_set_cover(n, edge_sets, all_edges, deadline, rng=None):
    """Greedy set cover to find minimum edges covering all colorings."""
    universe = (1 << (1 << n)) - 1
    covered = 0
    uncovered = universe
    selected = []
    used = [False] * len(all_edges)
    
    # Precompute coverage sizes for initial ordering
    if rng is None:
        rng = random.Random(42)
    
    # Shuffle for variety
    indices = list(range(len(all_edges)))
    rng.shuffle(indices)
    
    while uncovered != 0:
        if time.time() > deadline:
            return None, covered
        
        best_idx = -1
        best_count = 0
        for idx in indices:
            if used[idx]:
                continue
            coverage = edge_sets[idx] & uncovered
            count = coverage.bit_count() if hasattr(coverage, 'bit_count') else bin(coverage).count('1')
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
        
        used[best_idx] = True
        selected.append(best_idx)
        covered |= edge_sets[best_idx]
        uncovered = universe & ~covered
    
    if uncovered == 0:
        return selected, covered
    return None, covered


def _local_search_reduce(n, edges, edge_sets, all_edges, deadline, max_iter=500):
    """Try to reduce edge count by removing edges that are still covered."""
    universe = (1 << (1 << n)) - 1
    current_edges = list(edges)
    current_set = set(range(len(current_edges)))
    
    # Compute coverage of current set
    covered = 0
    for e in current_edges:
        # Find index of e in all_edges
        # We need a mapping
        pass
    
    # This approach is complex. Instead, try: for each edge in current set,
    # check if removing it still covers all colorings.
    
    # Precompute: for each edge in current set, which colorings does it uniquely cover?
    # Actually, simpler: just try removing each edge and checking coverage.
    
    all_edges_set = set(tuple(e) for e in all_edges)
    edge_to_idx = {tuple(e): i for i, e in enumerate(all_edges)}
    
    current_tuples = [tuple(e) for e in current_edges]
    
    for attempt in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(len(current_tuples)):
            if time.time() > deadline:
                break
            
            # Try removing edge i
            test_set = current_tuples[:i] + current_tuples[i+1:]
            
            # Check if test_set covers all colorings
            test_covered = 0
            for t in test_set:
                idx = edge_to_idx[t]
                test_covered |= edge_sets[idx]
            
            if test_covered == universe:
                # Can remove edge i
                current_tuples = test_set
                improved = True
                break
        
        if not improved:
            break
    
    return [list(t) for t in current_tuples]


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Fallback: complete 5-graph on 9 vertices (126 edges)
    fallback_n = 9
    fallback_edges = [list(e) for e in combinations(range(fallback_n), 5)]
    
    # Primary approach: set cover on n=11
    n = 11
    all_edges = list(combinations(range(n), 5))
    
    print(f"Computing edge coloring sets for n={n}, {len(all_edges)} edges...", file=sys.stderr)
    t0 = time.time()
    edge_sets = _compute_edge_coloring_sets_fast(n, all_edges)
    print(f"  Done in {time.time()-t0:.2f}s", file=sys.stderr)
    
    if time.time() > deadline:
        return {'n': fallback_n, 'edges': fallback_edges}
    
    # Try greedy set cover with different seeds
    best_result = None
    best_count = float('inf')
    
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        if time.time() > deadline - 30:
            break
        
        rng = random.Random(seed)
        selected, covered = _greedy_set_cover(n, edge_sets, all_edges, deadline, rng)
        
        if selected is not None:
            edges = [list(all_edges[i]) for i in selected]
            count = len(edges)
            print(f"  Seed {seed}: {count} edges", file=sys.stderr)
            
            if count < best_count:
                # Verify non-2-colorability
                if not _is_2_colorable(n, edges):
                    best_result = edges
                    best_count = count
    
    if best_result is not None:
        # Try local search to reduce further
        if time.time() < deadline - 15:
            print(f"  Local search on {best_count} edges...", file=sys.stderr)
            reduced = _local_search_reduce(n, best_result, edge_sets, all_edges, deadline - 15)
            if len(reduced) < best_count:
                if not _is_2_colorable(n, reduced):
                    best_result = reduced
                    best_count = len(reduced)
                    print(f"  Reduced to {best_count} edges", file=sys.stderr)
        
        return {'n': n, 'edges': best_result}
    
    # Try n=10 as well (fewer colorings to cover: 1024)
    n10 = 10
    all_edges10 = list(combinations(range(n10), 5))
    
    if time.time() < deadline - 30:
        print(f"Trying n={n10}, {len(all_edges10)} edges...", file=sys.stderr)
        t0 = time.time()
        edge_sets10 = _compute_edge_coloring_sets_fast(n10, all_edges10)
        print(f"  Done in {time.time()-t0:.2f}s", file=sys.stderr)
        
        for seed in [42, 123, 456]:
            if time.time() > deadline - 10:
                break
            
            rng = random.Random(seed)
            selected, covered = _greedy_set_cover(n10, edge_sets10, all_edges10, deadline, rng)
            
            if selected is not None:
                edges = [list(all_edges10[i]) for i in selected]
                count = len(edges)
                print(f"  n=10 seed {seed}: {count} edges", file=sys.stderr)
                
                if count < best_count:
                    if not _is_2_colorable(n10, edges):
                        best_result = edges
                        best_count = count
    
    if best_result is not None:
        return {'n': n if best_count <= 51 else n10, 'edges': best_result}
    
    # Return fallback
    return {'n': fallback_n, 'edges': fallback_edges}
