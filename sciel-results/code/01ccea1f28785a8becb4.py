import time
import random
from itertools import combinations

def _edge_kills_mask(n, edge):
    """
    Return a bitmask of length 2^n where bit i is set if coloring i
    (represented as an n-bit integer) makes the given edge monochromatic.
    Coloring i: vertex v is color (i >> v) & 1.
    Edge is monochromatic if all 5 vertices have the same color.
    """
    mask = 0
    total = 1 << n
    # For each coloring, check if edge is monochromatic
    # Optimization: iterate only over colorings where the edge vertices agree
    # The 5 vertices in the edge must all be 0 or all be 1.
    # Fix the 5 edge vertices, iterate over the remaining n-5 vertices.
    edge_set = set(edge)
    other_verts = [v for v in range(n) if v not in edge_set]
    
    for color_val in [0, 1]:
        # Fix edge vertices to color_val
        # Iterate over all assignments of other vertices
        for other_mask in range(1 << len(other_verts)):
            # Build the full coloring
            c = 0
            for v in edge:
                c |= (color_val << v)
            for j, v in enumerate(other_verts):
                if (other_mask >> j) & 1:
                    c |= (1 << v)
            mask |= (1 << c)
    
    return mask

def _is_non_2colorable_backtrack(n, edges, deadline):
    """Verify non-2-colorability using backtracking with unit propagation."""
    edge_list = [tuple(sorted(e)) for e in edges]
    color = [-1] * n
    
    vert_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_list):
        for v in e:
            vert_edges[v].append(i)
    
    # Edge state: [count0, count1, unassigned]
    edge_state = []
    for e in edge_list:
        edge_state.append([0, 0, 5])
    
    def propagate():
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_list):
                c0, c1, un = edge_state[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and un == 1:
                    for v in e:
                        if color[v] == -1:
                            color[v] = 1
                            edge_state[i][0] -= 1
                            edge_state[i][1] += 1
                            edge_state[i][2] -= 1
                            # Update other edges containing v
                            for ei in vert_edges[v]:
                                if ei != i:
                                    edge_state[ei][0] -= 1
                                    edge_state[ei][1] += 1
                                    edge_state[ei][2] -= 1
                            changed = True
                elif c1 == 4 and un == 1:
                    for v in e:
                        if color[v] == -1:
                            color[v] = 0
                            edge_state[i][0] += 1
                            edge_state[i][1] -= 1
                            edge_state[i][2] -= 1
                            for ei in vert_edges[v]:
                                if ei != i:
                                    edge_state[ei][0] += 1
                                    edge_state[ei][1] -= 1
                                    edge_state[ei][2] -= 1
                            changed = True
        return True
    
    def get_next_vertex(idx):
        while idx < n and color[idx] != -1:
            idx += 1
        return idx
    
    def backtrack(idx):
        if time.time() > deadline:
            return 'timeout'
        idx = get_next_vertex(idx)
        if idx >= n:
            return True  # Found valid 2-coloring
        
        saved_color = color[:]
        saved_state = [s[:] for s in edge_state]
        
        for c in [0, 1]:
            color[idx] = c
            # Update edge states
            for ei in vert_edges[idx]:
                if c == 0:
                    edge_state[ei][0] += 1
                else:
                    edge_state[ei][1] += 1
                edge_state[ei][2] -= 1
            
            if propagate():
                result = backtrack(idx + 1)
                if result == 'timeout':
                    color[:] = saved_color
                    edge_state[:] = [s[:] for s in saved_state]
                    return 'timeout'
                if result == True:
                    color[:] = saved_color
                    edge_state[:] = [s[:] for s in saved_state]
                    return True
            
            color[:] = saved_color
            edge_state[:] = [s[:] for s in saved_state]
        
        return False
    
    result = backtrack(0)
    if result == 'timeout':
        return None
    return not result  # True if NOT 2-colorable

def construct():
    deadline = time.time() + 145
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Try different n values
    for n in range(9, 13):
        if time.time() > deadline - 15:
            break
        
        total_colorings = 1 << n
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        
        # Precompute kill masks for all edges
        kill_masks = []
        for e in all_edges:
            kill_masks.append(_edge_kills_mask(n, e))
        
        # Greedy set cover
        uncovered = (1 << total_colorings) - 1
        chosen = []
        used = [False] * num_edges
        
        while uncovered:
            if time.time() > deadline - 10:
                break
            best_idx = -1
            best_count_new = -1
            for i in range(num_edges):
                if used[i]:
                    continue
                new_cover = kill_masks[i] & uncovered
                cnt = new_cover.bit_count()
                if cnt > best_count_new:
                    best_count_new = cnt
                    best_idx = i
            
            if best_idx == -1:
                break
            
            chosen.append(best_idx)
            used[best_idx] = True
            uncovered &= ~kill_masks[best_idx]
        
        if not uncovered:
            # We have a valid covering
            current_edges = [list(all_edges[i]) for i in chosen]
            
            # Try to reduce further with local search
            # Simulated annealing on edge removal
            current_n = n
            current_edge_set = set(tuple(e) for e in current_edges)
            current_count = len(current_edges)
            
            # Verify it's actually non-2-colorable
            if time.time() > deadline - 5:
                if current_count < best_count:
                    best_edges = current_edges[:]
                    best_n = n
                    best_count = current_count
                continue
            
            is_valid = _is_non_2colorable_backtrack(current_n, current_edges, deadline - 5)
            if is_valid is not True:
                continue
            
            if current_count < best_count:
                best_edges = current_edges[:]
                best_n = n
                best_count = current_count
            
            # Try to remove edges one by one
            edge_list = list(current_edge_set)
            random.seed(42 + n)
            random.shuffle(edge_list)
            
            for edge_to_try in edge_list:
                if time.time() > deadline - 10:
                    break
                if len(current_edge_set) <= best_count:
                    break
                
                temp_set = current_edge_set - {edge_to_try}
                temp_edges = list(temp_set)
                
                # Quick check: does the union of kill masks still cover all?
                # This is a necessary but not sufficient condition
                # Actually for 2-colorability, the set cover condition is exactly what we need
                # But let's verify with backtracking for safety
                if time.time() > deadline - 8:
                    break
                
                is_valid = _is_non_2colorable_backtrack(current_n, temp_edges, deadline - 5)
                if is_valid is True:
                    current_edge_set = temp_set
                    current_count = len(current_edge_set)
                    if current_count < best_count:
                        best_edges = list(current_edge_set)
                        best_n = n
                        best_count = current_count
    
    # Fallback
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
    
    return {'n': best_n, 'edges': best_edges}
