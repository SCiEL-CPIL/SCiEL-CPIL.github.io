import time
import random
from itertools import combinations
import numpy as np

def is_2_colorable(n, edges):
    """
    Fast backtracking with unit propagation for 2-colorability of 5-uniform hypergraph.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
        
    m = len(edges)
    vertex_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_to_edges[v].append(i)
            
    colors = [-1] * n
    # edge_counts[i] = [count_0, count_1, count_unknown]
    edge_counts = [[0, 0, 5] for _ in range(m)]
    
    def propagate(queue):
        while queue:
            v, c = queue.pop()
            if colors[v] != -1:
                if colors[v] != c:
                    return False
                else:
                    continue
            
            colors[v] = c
            
            for e_idx in vertex_to_edges[v]:
                edge_counts[e_idx][2] -= 1
                edge_counts[e_idx][c] += 1
                
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if cU == 1:
                    e = edges[e_idx]
                    unknown_v = -1
                    for u in e:
                        if colors[u] == -1:
                            unknown_v = u
                            break
                    if unknown_v == -1:
                        continue
                    
                    if c0 == 4:
                        if colors[unknown_v] == 0:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 1))
                    elif c1 == 4:
                        if colors[unknown_v] == 1:
                            return False
                        if colors[unknown_v] == -1:
                            queue.append((unknown_v, 0))
        return True

    def backtrack():
        best_v = -1
        best_score = -1
        
        for v in range(n):
            if colors[v] != -1:
                continue
            score = 0
            for e_idx in vertex_to_edges[v]:
                c0 = edge_counts[e_idx][0]
                c1 = edge_counts[e_idx][1]
                cU = edge_counts[e_idx][2]
                if cU == 1:
                    if c0 == 4 or c1 == 4:
                        score += 10
                    else:
                        score += 1
                elif cU == 2:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_v = v
                
        if best_v == -1:
            return True
            
        v = best_v
        colors_choice = [0, 1]
        random.shuffle(colors_choice)
        
        for c in colors_choice:
            old_colors = colors.copy()
            old_edge_counts = [list(ec) for ec in edge_counts]
            
            if propagate([(v, c)]):
                if backtrack():
                    return True
            
            colors[:] = old_colors
            edge_counts[:] = old_edge_counts
            
        return False

    if propagate([]):
        return backtrack()
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_edge_count = 200
    
    # Try n=12 and n=11
    for n in [12, 11]:
        if time.time() > deadline - 20.0:
            break
            
        all_edges = list(combinations(range(n), 5))
        
        # Strategy: Start with all edges, delete as many as possible
        # This is a "Maximal 2-Colorable Subgraph" problem on the dual?
        # No, we want a MINIMAL set of edges that is NON-2-colorable.
        # This is the "Minimum Non-2-Colorable Hypergraph" problem.
        
        # Approach:
        # 1. Generate a random set of edges that is likely non-2-colorable.
        # 2. Use local search to minimize.
        
        # Better Approach:
        # Use the Set Cover idea but optimized.
        
        # Let's try a different heuristic:
        # Randomly select edges until it becomes non-2-colorable, then delete.
        
        # Or:
        # Start with a known structure.
        
        # Let's stick to the Set Cover but make it faster.
        # The previous implementation was slow due to Python overhead in the inner loop.
        
        # For n=12, subsets of size 5..7.
        # Sizes: 5, 6, 7.
        # Symmetry: S and V\S.
        # We only need to consider |S| <= n/2.
        # For n=12, |S| in {5, 6}.
        
        all_subsets = []
        for size in range(5, n//2 + 1):
            for s in combinations(range(n), size):
                all_subsets.append(s)
                
        # Note: For n=12, size 6 is self-complementary.
        # For n=11, sizes 5, 6. 5 and 6 are complementary.
        # We can just use sizes 5 to n//2.
        
        if not all_subsets:
            continue
            
        num_subsets = len(all_subsets)
        
        subsets_as_masks = []
        for s in all_subsets:
            mask = 0
            for v in s:
                mask |= (1 << v)
            subsets_as_masks.append(mask)
            
        # Precompute edge masks
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
            
        # Build edge_covers: for each edge, which subsets it kills.
        # An edge e kills S if e subset S OR e subset V\S.
        # Since we only have S with |S| <= n/2, e subset V\S implies |V\S| >= 5 => |S| <= n-5.
        # And e subset S implies |S| >= 5.
        
        edge_covers = [[] for _ in range(len(all_edges))]
        
        for e_idx, e_mask in enumerate(edge_masks):
            for s_idx, s_mask in enumerate(subsets_as_masks):
                # Check if e is monochromatic in S
                # e subset S: (e_mask & ~s_mask) == 0
                # e subset V\S: (e_mask & s_mask) == 0
                if (e_mask & ~s_mask) == 0 or (e_mask & s_mask) == 0:
                    edge_covers[e_idx].append(s_idx)
                    
        # Greedy Set Cover
        uncovered = set(range(num_subsets))
        chosen_edges = []
        
        # Precompute coverage counts
        # To speed up, we can maintain a count of uncovered subsets per edge.
        # But updating is expensive.
        # Let's use a simple greedy with periodic re-evaluation.
        
        while uncovered and time.time() < deadline - 20.0:
            best_e = -1
            best_count = -1
            
            # Optimization: Only check edges that cover at least 1 uncovered subset?
            # We can maintain a list of active edges.
            
            for e_idx in range(len(all_edges)):
                if not edge_covers[e_idx]:
                    continue
                count = 0
                for s_idx in edge_covers[e_idx]:
                    if s_idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_e = e_idx
                    
            if best_e == -1 or best_count == 0:
                break
                
            chosen_edges.append(best_e)
            for s_idx in edge_covers[best_e]:
                uncovered.discard(s_idx)
                
        if not uncovered:
            current_edges = [list(all_edges[e]) for e in chosen_edges]
            
            # Verify
            if not is_2_colorable(n, current_edges):
                if len(current_edges) < best_edge_count:
                    best_edge_count = len(current_edges)
                    best_result = {'n': n, 'edges': current_edges}
                    
                # Local Search: Delete edges
                improved = True
                while improved and time.time() < deadline - 10.0:
                    improved = False
                    # Shuffle order to avoid bias
                    indices = list(range(len(current_edges)))
                    random.shuffle(indices)
                    for i in indices:
                        test_edges = current_edges[:i] + current_edges[i+1:]
                        if not is_2_colorable(n, test_edges):
                            current_edges = test_edges
                            if len(current_edges) < best_edge_count:
                                best_edge_count = len(current_edges)
                                best_result = {'n': n, 'edges': current_edges}
                            improved = True
                            break
                            
    if best_result is not None:
        return best_result
        
    # Fallback: n=9, all edges (126)
    n_fb = 9
    all_edges_fb = list(combinations(range(n_fb), 5))
    return {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
