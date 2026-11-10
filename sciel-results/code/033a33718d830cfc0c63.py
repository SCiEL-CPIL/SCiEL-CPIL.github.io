import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140  # Leave 10s margin for 150s limit
    
    n = 9
    all_edges = [tuple(e) for e in combinations(range(n), 5)]
    
    # Precompute edge index sets for fast lookup
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Fast 2-colorability checker using bitmask representation
    def is_non_2_colorable(n, edge_indices):
        """Returns True if the hypergraph with given edge indices is NOT 2-colorable."""
        m = len(edge_indices)
        if m == 0:
            return False
        
        # Build adjacency: for each vertex, which edges (by position in edge_indices) contain it
        v_to_eidx = [[] for _ in range(n)]
        for pos, eidx in enumerate(edge_indices):
            for v in all_edges[eidx]:
                v_to_eidx[v].append(pos)
        
        # colors: -1 unassigned, 0, 1
        colors = [-1] * n
        # Symmetry breaking
        colors[0] = 0
        
        # For each edge position, track count of 0s and 1s
        cnt0 = [0] * m
        cnt1 = [0] * m
        
        # Initialize counts
        for pos, eidx in enumerate(edge_indices):
            for v in all_edges[eidx]:
                if colors[v] == 0:
                    cnt0[pos] += 1
                elif colors[v] == 1:
                    cnt1[pos] += 1
        
        # Check initial state for conflicts
        for pos in range(m):
            if cnt0[pos] == 5 or cnt1[pos] == 5:
                return True  # Already monochromatic, so non-2-colorable
            # Unit propagation: if 4 of one color, force the 5th
            if cnt0[pos] == 4:
                # Find unassigned vertex in this edge
                for v in all_edges[edge_indices[pos]]:
                    if colors[v] == -1:
                        if not _propagate(v, 1, colors, cnt0, cnt1, v_to_eidx, edge_indices, all_edges):
                            return True
                        break
            elif cnt1[pos] == 4:
                for v in all_edges[edge_indices[pos]]:
                    if colors[v] == -1:
                        if not _propagate(v, 0, colors, cnt0, cnt1, v_to_eidx, edge_indices, all_edges):
                            return True
                        break
        
        # DFS with backtracking
        def dfs():
            # Find unassigned vertex - heuristic: pick one in most constrained edge
            best_v = -1
            best_score = -1
            for v in range(n):
                if colors[v] == -1:
                    # Score: number of edges where this vertex is the only unassigned one
                    score = 0
                    for pos in v_to_eidx[v]:
                        unassigned_count = 0
                        for u in all_edges[edge_indices[pos]]:
                            if colors[u] == -1:
                                unassigned_count += 1
                        if unassigned_count == 1:
                            score += 10  # This vertex is critical for this edge
                        elif unassigned_count == 2:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_v = v
            
            if best_v == -1:
                return True  # All assigned, found a valid coloring
            
            # Try color 0
            saved_colors = colors[:]
            saved_cnt0 = cnt0[:]
            saved_cnt1 = cnt1[:]
            
            if _propagate(best_v, 0, colors, cnt0, cnt1, v_to_eidx, edge_indices, all_edges):
                if dfs():
                    return True
            
            colors[:] = saved_colors
            cnt0[:] = saved_cnt0
            cnt1[:] = saved_cnt1
            
            # Try color 1
            saved_colors = colors[:]
            saved_cnt0 = cnt0[:]
            saved_cnt1 = cnt1[:]
            
            if _propagate(best_v, 1, colors, cnt0, cnt1, v_to_eidx, edge_indices, all_edges):
                if dfs():
                    return True
            
            colors[:] = saved_colors
            cnt0[:] = saved_cnt0
            cnt1[:] = saved_cnt1
            
            return False
        
        if dfs():
            return False  # Colorable
        else:
            return True  # Not colorable
    
    def _propagate(v, color, colors, cnt0, cnt1, v_to_eidx, edge_indices, all_edges):
        """Assign v to color with propagation. Returns False on conflict."""
        if colors[v] != -1:
            return colors[v] == color
        
        colors[v] = color
        
        # Process all edges containing v
        for pos in v_to_eidx[v]:
            if color == 0:
                cnt0[pos] += 1
                if cnt0[pos] == 5:
                    return False
                if cnt0[pos] == 4:
                    # Find unassigned vertex in this edge
                    for u in all_edges[edge_indices[pos]]:
                        if colors[u] == -1:
                            if not _propagate(u, 1, colors, cnt0, cnt1, v_to_eidx, edge_indices, all_edges):
                                return False
                            break
            else:
                cnt1[pos] += 1
                if cnt1[pos] == 5:
                    return False
                if cnt1[pos] == 4:
                    for u in all_edges[edge_indices[pos]]:
                        if colors[u] == -1:
                            if not _propagate(u, 0, colors, cnt0, cnt1, v_to_eidx, edge_indices, all_edges):
                                return False
                            break
        
        return True
    
    # Start with complete hypergraph
    best_edge_indices = set(range(len(all_edges)))
    best_count = len(best_edge_indices)
    
    # Multiple randomized greedy deletion passes
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        
        current = set(best_edge_indices)
        
        # Randomize edge order
        edges_list = list(current)
        random.shuffle(edges_list)
        
        improved = False
        for eidx in edges_list:
            if time.time() > deadline:
                break
            
            if eidx not in current:
                continue
            
            # Try removing this edge
            test_set = current - {eidx}
            test_list = list(test_set)
            
            if is_non_2_colorable(n, test_list):
                current = test_set
                improved = True
        
        if len(current) < best_count:
            best_edge_indices = current
            best_count = len(current)
            # If we reach 51, that's the known best
            if best_count <= 51:
                break
        
        # If no improvement, try a different strategy: start from current best with different randomization
        if not improved and pass_num > 10:
            # Try a more aggressive approach: remove multiple edges at once if possible
            # But for now, just continue with new randomization
            pass
    
    # Convert back to list of edges
    result_edges = [list(all_edges[i]) for i in sorted(best_edge_indices)]
    
    return {'n': n, 'edges': result_edges}
