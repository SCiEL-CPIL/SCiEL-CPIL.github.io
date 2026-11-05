import time
import random
from itertools import combinations

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
    
    # Try n=11 first as it's a good balance
    for n in [11, 12, 10]:
        if time.time() > deadline - 15.0:
            break
            
        # Generate all subsets S with |S| <= n/2
        # Due to symmetry, covering S covers V\S
        # Condition for S: Edge in S OR Edge in V\S
        all_subsets = []
        max_size = n // 2
        for size in range(0, max_size + 1):
            for s in combinations(range(n), size):
                all_subsets.append(s)
                
        if not all_subsets:
            continue
            
        num_subsets = len(all_subsets)
        
        # Convert to masks
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
            
        # For each subset, determine which edges cover it
        # Edge e covers S if (e_mask & ~s_mask) == 0 (e subset of S) 
        # OR (e_mask & s_mask) == 0 (e subset of V\S)
        
        # We want to find a set of edges that covers all subsets.
        # This is Set Cover.
        # Elements: subsets
        # Sets: edges
        
        # To speed up greedy, precompute for each edge the list of subsets it covers.
        edge_covers = [[] for _ in range(num_edges)]
        
        for s_idx, s_mask in enumerate(subsets_as_masks):
            for e_idx, e_mask in enumerate(edge_masks):
                if (e_mask & ~s_mask) == 0 or (e_mask & s_mask) == 0:
                    edge_covers[e_idx].append(s_idx)
                    
        # Greedy Set Cover
        uncovered = set(range(num_subsets))
        chosen_edges = []
        
        # Sort edges by initial coverage to speed up first pick? 
        # Or just standard greedy.
        
        while uncovered and time.time() < deadline - 15.0:
            best_e = -1
            best_count = -1
            
            for e_idx in range(num_edges):
                if e_idx in chosen_edges:
                    continue
                count = 0
                for s_idx in edge_covers[e_idx]:
                    if s_idx in uncovered:
                        count += 1
                        if count > best_count:
                            # Early break if we can't beat best? No.
                            pass
                            
                if count > best_count:
                    best_count = count
                    best_e = e_idx
                    
            if best_e == -1 or best_count == 0:
                break # Cannot cover all?
                
            chosen_edges.append(best_e)
            
            # Remove covered subsets
            for s_idx in edge_covers[best_e]:
                uncovered.discard(s_idx)
                
        if not uncovered:
            current_edges = [list(all_edges[e]) for e in chosen_edges]
            
            # Verify
            if not is_2_colorable(n, current_edges):
                if len(current_edges) < best_edge_count:
                    best_edge_count = len(current_edges)
                    best_result = {'n': n, 'edges': current_edges}
                    
                # Local search: remove edges
                improved = True
                while improved and time.time() < deadline - 15.0:
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
