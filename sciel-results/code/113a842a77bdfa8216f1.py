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
    deadline = start_time + 135.0  # Safety margin: evaluator kills at 150s
    
    best_result = None
    best_edge_count = 200
    
    n = 11
    all_subsets = []
    max_size = n // 2
    for size in range(0, max_size + 1):
        for s in combinations(range(n), size):
            all_subsets.append(s)
            
    num_subsets = len(all_subsets)
    full_mask = (1 << num_subsets) - 1
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute cover masks for each edge
    # cover_mask[e_idx] has bit i set if edge e_idx kills subset i
    # Edge e kills subset S if e is subset of S OR e is subset of V\S
    cover_masks = [0] * num_edges
    
    # Optimization: Precompute masks for subsets
    subset_masks = [0] * num_subsets
    for i, s in enumerate(all_subsets):
        mask = 0
        for v in s:
            mask |= (1 << v)
        subset_masks[i] = mask
            
    for e_idx, e in enumerate(all_edges):
        e_mask = 0
        for v in e:
            e_mask |= (1 << v)
            
        cm = 0
        for s_idx, s_mask in enumerate(subset_masks):
            # Check if e is subset of S
            if (e_mask & ~s_mask) == 0:
                cm |= (1 << s_idx)
            # Check if e is subset of V\S
            elif (e_mask & s_mask) == 0:
                cm |= (1 << s_idx)
        cover_masks[e_idx] = cm
    
    def popcount(x):
        return bin(x).count('1')
        
    # Greedy Set Cover to get initial solution
    def greedy_cover():
        uncovered = full_mask
        chosen = []
        remaining_edges = list(range(num_edges))
        
        while uncovered and remaining_edges:
            best_e = -1
            best_gain = 0
            for e_idx in remaining_edges:
                gain = popcount(cover_masks[e_idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_e = e_idx
            if best_e == -1:
                break
            chosen.append(best_e)
            uncovered &= ~cover_masks[best_e]
            remaining_edges.remove(best_e)
            
        if uncovered == 0:
            return chosen
        return None

    # Local Search to minimize
    def local_search(initial_edges):
        if initial_edges is None:
            return None
            
        current_edges = list(initial_edges)
        current_mask = 0
        for e in current_edges:
            current_mask |= cover_masks[e]
            
        if current_mask != full_mask:
            return None
            
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # 1. Try to remove redundant edges
            to_remove = []
            for e in current_edges:
                if (current_mask & ~cover_masks[e]) == current_mask:
                    to_remove.append(e)
                    
            if to_remove:
                for e in to_remove:
                    current_edges.remove(e)
                    current_mask &= ~cover_masks[e]
                improved = True
                continue
                
            # 2. Try Swaps to enable future removals
            if not current_edges:
                break
                
            # Try multiple swap attempts before giving up on this round
            max_swap_attempts = min(len(current_edges), 20)
            for _ in range(max_swap_attempts):
                if time.time() > deadline - 5.0:
                    break
                    
                e_out = random.choice(current_edges)
                
                # Compute M_other = union of cover_masks of all edges except e_out
                M_other = 0
                for e in current_edges:
                    if e != e_out:
                        M_other |= cover_masks[e]
                        
                critical = cover_masks[e_out] & ~M_other
                
                if critical == 0:
                    continue
                
                # Find e_in not in current_edges such that cover_masks[e_in] covers critical
                current_set = set(current_edges)
                candidates = [e for e in range(num_edges) if e not in current_set]
                random.shuffle(candidates)
                
                swapped = False
                for e_in in candidates:
                    if (cover_masks[e_in] & critical) == critical:
                        # Verify the swap maintains full coverage
                        new_mask = 0
                        for e in current_edges:
                            if e != e_out:
                                new_mask |= cover_masks[e]
                        new_mask |= cover_masks[e_in]
                        
                        if new_mask == full_mask:
                            current_edges.remove(e_out)
                            current_edges.append(e_in)
                            current_mask = new_mask
                            swapped = True
                            break
                        
            if swapped:
                # After a swap, check if we can now remove edges
                to_remove = []
                for e in current_edges:
                    if (current_mask & ~cover_masks[e]) == current_mask:
                        to_remove.append(e)
                if to_remove:
                    for e in to_remove:
                        current_edges.remove(e)
                        current_mask &= ~cover_masks[e]
                    improved = True
                    break
        
        return current_edges

    # Main Search Loop
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
            
        init_edges = greedy_cover()
        if init_edges is None:
            continue
            
        result_edges = local_search(init_edges)
        if result_edges is not None:
            if len(result_edges) < best_edge_count:
                best_edge_count = len(result_edges)
                # Verify non-2-colorability
                edge_lists = [list(all_edges[e]) for e in result_edges]
                if not is_2_colorable(n, edge_lists):
                    best_result = {'n': n, 'edges': edge_lists}
    
    if best_result is not None:
        return best_result
        
    # Fallback: return a known non-2-colorable hypergraph
    n_fb = 9
    all_edges_fb = list(combinations(range(n_fb), 5))
    return {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
