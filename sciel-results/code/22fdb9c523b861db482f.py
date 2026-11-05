"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Generate all edges of Q7
    edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
    
    m = len(edges)  # 448
    
    # Build conflict graph: two edges conflict if they are opposite edges of a 2-face
    # Two parallel edges (same direction) conflict if their endpoints differ in exactly one other bit
    conflict = [set() for _ in range(m)]
    
    # Group edges by direction
    by_dir = [[] for _ in range(d)]
    for idx, (x, y) in enumerate(edges):
        diff = x ^ y
        dir_bit = diff.bit_length() - 1
        by_dir[dir_bit].append(idx)
    
    # For each direction, check pairs of parallel edges
    for dir_bit in range(d):
        dir_edges = by_dir[dir_bit]
        for i in range(len(dir_edges)):
            for j in range(i + 1, len(dir_edges)):
                idx1 = dir_edges[i]
                idx2 = dir_edges[j]
                x1, y1 = edges[idx1]
                x2, y2 = edges[idx2]
                # Check if they form a 2-face: endpoints differ in exactly one bit (other than dir_bit)
                # x1 and x2 should differ in exactly one bit, and that bit is not dir_bit
                diff12 = x1 ^ x2
                # The bit that differs must not be dir_bit
                if diff12 & (1 << dir_bit) == 0 and diff12.bit_count() == 1:
                    # They conflict
                    conflict[idx1].add(idx2)
                    conflict[idx2].add(idx1)
    
    # Pre-compute neighbors list for faster access
    neighbors = [list(c) for c in conflict]
    
    def greedy_mis(randomize=False):
        remaining = set(range(m))
        selected = set()
        
        while remaining:
            # Find edge with minimum degree in remaining
            min_deg = float('inf')
            candidates = []
            for idx in remaining:
                deg = len(conflict[idx] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [idx]
                elif deg == min_deg:
                    candidates.append(idx)
            
            if randomize and candidates:
                best = random.choice(candidates)
            else:
                best = candidates[0]
            
            selected.add(best)
            remaining.remove(best)
            # Remove all conflicts of best from remaining
            for c in conflict[best]:
                if c in remaining:
                    remaining.remove(c)
        
        return selected

    def local_improve(selected_set):
        selected_list = list(selected_set)
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_swap = None
            
            for f in selected_list:
                # If we remove f, which edges can we add?
                # An edge e can be added if e is not in selected_set and 
                # all its neighbors in the conflict graph are either f or not in selected_set.
                # Actually, e must not conflict with any vertex in (selected_set - {f}).
                # So e is a candidate if N(e) \ {f} is disjoint from selected_set.
                # And e must not be in selected_set.
                
                # Optimization: Only check edges that conflict with f?
                # No, removing f might allow adding edges that didn't conflict with f?
                # No, if e didn't conflict with f, it was already available if it didn't conflict with others.
                # So only edges that *did* conflict with f were blocked by f.
                # Thus, we only need to check neighbors of f.
                
                potential_adds = []
                for e in neighbors[f]:
                    if e in selected_set:
                        continue
                    # Check if e conflicts with any other vertex in selected_set
                    # e is not in selected_set, so we just check if N(e) intersects selected_set - {f}
                    # Since e is a neighbor of f, f is in N(e).
                    # We need N(e) \ {f} to be disjoint from selected_set.
                    # This is equivalent to: for all v in N(e), if v != f, v not in selected_set.
                    
                    # Quick check: if e has any neighbor in selected_set other than f, it's blocked.
                    blocked = False
                    for v in neighbors[e]:
                        if v != f and v in selected_set:
                            blocked = True
                            break
                    if not blocked:
                        potential_adds.append(e)
                
                gain = len(potential_adds) - 1
                if gain > best_gain:
                    best_gain = gain
                    best_swap = (f, potential_adds)
            
            if best_swap is not None:
                f, adds = best_swap
                selected_set.remove(f)
                for e in adds:
                    selected_set.add(e)
                improved = True
                selected_list = list(selected_set)
        
        return selected_set

    # Run multiple randomized greedy passes and local improvement
    best_set = set()
    
    # Run deterministic first
    s = greedy_mis(randomize=False)
    s = local_improve(s)
    if len(s) > len(best_set):
        best_set = s
        
    # Run randomized
    for _ in range(50):
        s = greedy_mis(randomize=True)
        s = local_improve(s)
        if len(s) > len(best_set):
            best_set = s

    # Convert back to edge list
    result = []
    for idx in best_set:
        x, y = edges[idx]
        result.append([x, y])
    
    return result
# EVOLVE-BLOCK-END
