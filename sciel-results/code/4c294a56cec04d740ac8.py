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
    
    # We focus on n=11 as it's the sweet spot for known constructions (e.g., 51 edges).
    # n=10 is too small (max 2-colorable with fewer edges? No, n=10 allows 2-colorable with many edges, 
    # but we want MIN edges for NON-2-colorable. 
    # Actually, smaller n usually requires FEWER edges to be non-2-colorable? 
    # Let's check: For n=9, total edges 126. 
    # For n=11, total edges 462.
    # The minimum number of edges for non-2-colorable 5-graphs is known to be around 50-60 for n=11.
    # For n=10, it might be lower or higher? 
    # Let's stick to n=11 which is a standard test case.
    
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
        
    # Greedy Set Cover to get initial solution
    def greedy_cover():
        uncovered = full_mask
        chosen = []
        remaining_edges = list(range(num_edges))
        
        while uncovered and remaining_edges:
            best_e = -1
            best_gain = 0
            # Randomize order to break ties and allow restarts
            random.shuffle(remaining_edges)
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

    def popcount(x):
        return bin(x).count('1')
        
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
            # An edge is redundant if removing it doesn't change the union mask
            to_remove = []
            for e in current_edges:
                # If (current_mask \ cover_masks[e]) == current_mask, then e is redundant
                # This is equivalent to: (current_mask & ~cover_masks[e]) == current_mask
                # Or: current_mask == (current_mask | cover_masks[e]) which is always true if e is in set? No.
                # We need to check if any bit is ONLY covered by e.
                # Let M = current_mask. Let M_e = M \ cover_masks[e].
                # If M_e == M, then e is redundant.
                # M_e = M & ~cover_masks[e]
                if (current_mask & ~cover_masks[e]) == current_mask:
                    to_remove.append(e)
                    
            if to_remove:
                for e in to_remove:
                    current_edges.remove(e)
                    current_mask &= ~cover_masks[e]
                improved = True
                continue
                
            # 2. Try Swaps to enable future removals
            # Pick a random edge to remove, find an edge to add that covers the "critical" bits
            if not current_edges:
                break
                
            e_out = random.choice(current_edges)
            # Critical bits are those covered by e_out but NOT by any other edge in current_edges
            # M_other = union of cover_masks of all edges except e_out
            # critical = cover_masks[e_out] & ~M_other
            # Instead of calculating M_other from scratch, note:
            # critical = cover_masks[e_out] & ~(current_mask \ cover_masks[e_out])? No.
            # Let's calculate M_other efficiently? 
            # For small sets, just loop.
            M_other = 0
            for e in current_edges:
                if e != e_out:
                    M_other |= cover_masks[e]
                    
            critical = cover_masks[e_out] & ~M_other
            
            if critical == 0:
                # e_out is redundant, should have been caught above
                continue
                
            # Find e_in not in current_edges such that cover_masks[e_in] covers critical
            # i.e., (cover_masks[e_in] & critical) == critical
            candidates = [e for e in range(num_edges) if e not in current_edges]
            random.shuffle(candidates)
            
            swapped = False
            for e_in in candidates:
                if (cover_masks[e_in] & critical) == critical:
                    # Perform swap
                    current_edges.remove(e_out)
                    current_edges.append(e_in)
                    # Update mask
                    # current_mask = (current_mask & ~cover_masks[e_out]) | cover_masks[e_in]
                    # Since e_out's critical bits are covered by e_in, and e_out's non-critical bits 
                    # are covered by others, the union should remain full.
                    # Let's verify:
                    new_mask = 0
                    for e in current_edges:
                        new_mask |= cover_masks[e]
                    if new_mask != full_mask:
                        # Rollback (shouldn't happen if logic is correct)
                        current_edges.remove(e_in)
                        current_edges.append(e_out)
                    else:
                        current_mask = new_mask
                        swapped = True
                        # Don't mark improved=True immediately, as size didn't change.
                        # But this allows further removals.
                        break
                        
            if not swapped:
                # If we can't swap, try a different e_out? 
                # To avoid infinite loops, we can limit swaps or use a counter.
                # For simplicity, we just break out of this inner loop and let the outer loop 
                # try again (which will pick a different e_out randomly next time if we don't exit).
                # But we need to ensure progress. 
                # Let's just continue the while loop. If no removals and no swaps lead to removals, 
                # we might be stuck. 
                # We can add a "stall" counter.
                pass
                
        return current_edges

    # Main Search Loop
    for n_try in [11, 10, 12]:
        if time.time() > deadline - 10.0:
            break
            
        if n_try != 11:
             # Re-generate for other n if needed, but 11 is priority
             # For now, hardcode 11 optimization. If 11 fails, try others.
             continue
             
        # Multiple restarts of Greedy + Local Search
        for restart in range(50):
            if time.time() > deadline - 5.0:
                break
                
            # Randomize the greedy choice slightly by shuffling edges before greedy
            # The greedy function already shuffles remaining_edges
            init_edges = greedy_cover()
            if init_edges is None:
                continue
                
            result_edges = local_search(init_edges)
            if result_edges is not None:
                if len(result_edges) < best_edge_count:
                    best_edge_count = len(result_edges)
                    # Convert indices to edge lists
                    best_result = {'n': n_try, 'edges': [list(all_edges[e]) for e in result_edges]}
                    
    if best_result is not None:
        return best_result
        
    # Fallback
    n_fb = 9
    all_edges_fb = list(combinations(range(n_fb), 5))
    return {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
