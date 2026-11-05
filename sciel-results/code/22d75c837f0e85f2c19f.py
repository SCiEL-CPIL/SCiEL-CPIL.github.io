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
    
    # Try different n values
    # n=11 is known to be good. n=10 might be too small? 
    # Let's try n=11 first.
    
    for n in [11, 12, 10]:
        if time.time() > deadline - 10.0:
            break
            
        # 1. Generate all valid subsets S (color 0 sets)
        # S must have size >= 5 and n-|S| >= 5
        # Actually, if |S| < 5, it can't contain an edge. If |V\S| < 5, it can't contain an edge.
        # So we only need to consider S where 5 <= |S| <= n-5.
        # If n < 10, there are no such S? 
        # If n=9, |S| can be 5,6,7. V\S has size 4,3,2. 
        # Wait, if |V\S| < 5, it cannot contain an edge. 
        # So if n=9, and S has size 5, V\S has size 4. 
        # Then V\S is automatically edge-free. 
        # So the only constraint is that S must be edge-free.
        # This simplifies things.
        
        all_subsets = []
        for size in range(5, n - 4): # size from 5 to n-5
            for s in combinations(range(n), size):
                all_subsets.append(s)
                
        # Also, due to symmetry, we can restrict to |S| <= n/2 to reduce universe size?
        # No, because edges in S and edges in V\S are distinct constraints.
        # But the condition is: (S contains edge) OR (V\S contains edge).
        # If we pick S, we cover all colorings where S is color 0.
        # Let's stick to the full set of S for correctness, or optimize by symmetry.
        # For n=11, sizes are 5, 6.
        # If S has size 5, V\S has size 6.
        # If S has size 6, V\S has size 5.
        # These are symmetric. We can just iterate all.
        
        if not all_subsets:
            continue
            
        num_subsets = len(all_subsets)
        
        # 2. For each edge, determine which subsets it "kills"
        # An edge e kills S if e is subset of S OR e is subset of V\S.
        # We represent subsets as frozensets or integers for speed.
        # Let's use integers (bitmasks).
        
        subsets_as_masks = []
        for s in all_subsets:
            mask = 0
            for v in s:
                mask |= (1 << v)
            subsets_as_masks.append(mask)
            
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        
        # Precompute edge masks
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
            
        # For each edge, calculate the list of subset indices it kills
        # killed[edge_idx] = list of subset indices
        # Or better: for each subset, list of edges that kill it.
        # For Set Cover, we want: for each element (subset), which sets (edges) cover it.
        
        subset_to_edges = [[] for _ in range(num_subsets)]
        
        for e_idx, e_mask in enumerate(edge_masks):
            # An edge e kills S if (e_mask & ~S_mask) == 0 (e subset of S) 
            # OR (e_mask & S_mask) == 0 (e subset of V\S)
            # Note: e is 5-uniform.
            for s_idx, s_mask in enumerate(subsets_as_masks):
                # e subset of S?
                if (e_mask & ~s_mask) == 0:
                    subset_to_edges[s_idx].append(e_idx)
                # e subset of V\S?
                elif (e_mask & s_mask) == 0:
                    subset_to_edges[s_idx].append(e_idx)
                    
        # 3. Greedy Set Cover
        # uncovered = set of all subset indices
        uncovered = set(range(num_subsets))
        chosen_edges = []
        
        # Precompute coverage size for each edge? 
        # Or just iterate.
        # To speed up, we can maintain count of how many uncovered subsets each edge covers.
        
        # Let's use a simple greedy: 
        # While uncovered is not empty:
        #   Pick edge e that covers max uncovered subsets
        #   Add e to chosen
        #   Remove covered subsets from uncovered
        
        # Optimization: Precompute for each edge the list of subsets it covers.
        edge_covers = [[] for _ in range(num_edges)]
        for s_idx, e_list in enumerate(subset_to_edges):
            for e_idx in e_list:
                edge_covers[e_idx].append(s_idx)
                
        # Sort edges by initial coverage? 
        # Greedy usually picks best at each step.
        
        while uncovered and time.time() < deadline - 10.0:
            best_e = -1
            best_count = -1
            
            # Iterate all edges, check how many of their covered subsets are in uncovered
            for e_idx in range(num_edges):
                count = 0
                for s_idx in edge_covers[e_idx]:
                    if s_idx in uncovered:
                        count += 1
                        if count > best_count:
                            # Early break if we already know we can't beat best? 
                            # No, we need max.
                            pass
                            
                if count > best_count:
                    best_count = count
                    best_e = e_idx
                    
            if best_e == -1 or best_count == 0:
                break # Cannot cover all subsets with available edges?
                
            chosen_edges.append(best_e)
            
            # Remove covered subsets
            for s_idx in edge_covers[best_e]:
                uncovered.discard(s_idx)
                
        if not uncovered:
            # We have a set of edges that kills all 2-colorings
            # Verify with backtracking to be sure
            current_edges = [list(all_edges[e]) for e in chosen_edges]
            
            # Check for duplicates? Set cover picks distinct edges.
            
            if not is_2_colorable(n, current_edges):
                if len(current_edges) < best_edge_count:
                    best_edge_count = len(current_edges)
                    best_result = {'n': n, 'edges': current_edges}
                    
                # Local search: try to remove edges
                # We can try to remove edges one by one and check if still non-2-colorable
                improved = True
                while improved and time.time() < deadline - 10.0:
                    improved = False
                    for i in range(len(current_edges)):
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
        
    # Fallback
    n_fb = 9
    all_edges_fb = list(combinations(range(n_fb), 5))
    return {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
