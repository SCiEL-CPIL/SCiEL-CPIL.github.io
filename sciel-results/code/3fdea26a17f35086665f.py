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
    
    def try_set_cover(n):
        """Use set-cover formulation: find min set of 5-edges hitting all mono subsets."""
        nonlocal best_result, best_edge_count, deadline
        
        if time.time() > deadline - 20.0:
            return
        
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        
        # Precompute edge bitmasks
        edge_masks = []
        for e in all_edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
        
        full_mask = (1 << n) - 1
        
        # Elements to cover: all subsets S with |S| <= n//2
        # A 5-edge e covers S if e ⊆ S (e mono in S) or e ⊆ V\S (e mono in complement)
        # i.e., (e_mask & ~s_mask) == 0 or (e_mask & s_mask) == 0
        
        max_size = n // 2
        all_subsets = []
        for size in range(0, max_size + 1):
            for s in combinations(range(n), size):
                mask = 0
                for v in s:
                    mask |= (1 << v)
                all_subsets.append(mask)
        
        num_subsets = len(all_subsets)
        
        if num_subsets == 0:
            return
        
        # For each edge, compute which subsets it covers
        # Use bitsets for efficiency when possible
        edge_covers = [set() for _ in range(num_edges)]
        
        for s_idx, s_mask in enumerate(all_subsets):
            comp_mask = full_mask ^ s_mask
            for e_idx, e_mask in enumerate(edge_masks):
                if (e_mask & ~s_mask) == 0 or (e_mask & comp_mask) == 0:
                    edge_covers[e_idx].add(s_idx)
        
        # Greedy set cover
        uncovered = set(range(num_subsets))
        chosen = []
        
        while uncovered and time.time() < deadline - 15.0:
            best_e = -1
            best_count = -1
            
            for e_idx in range(num_edges):
                if e_idx in chosen:
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
                
            chosen.append(best_e)
            for s_idx in edge_covers[best_e]:
                uncovered.discard(s_idx)
        
        if uncovered:
            return
        
        current_edges = [list(all_edges[e]) for e in chosen]
        
        # Verify
        if is_2_colorable(n, current_edges):
            return
        
        if len(current_edges) < best_edge_count:
            best_edge_count = len(current_edges)
            best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
        
        # Local search: delete edges
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            for i in range(len(current_edges)):
                if time.time() > deadline - 10.0:
                    break
                test_edges = current_edges[:i] + current_edges[i+1:]
                if not is_2_colorable(n, test_edges):
                    current_edges = test_edges
                    if len(current_edges) < best_edge_count:
                        best_edge_count = len(current_edges)
                        best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
                    improved = True
                    break
        
        # Swap moves
        for _ in range(30):
            if time.time() > deadline - 10.0 or len(current_edges) <= 30:
                break
            idx = random.randint(0, len(current_edges) - 1)
            edge_to_remove = current_edges[idx]
            edge_to_add = random.choice(all_edges)
            if edge_to_add == tuple(edge_to_remove):
                continue
            new_edges = [e for e in current_edges if e != edge_to_remove]
            new_edges.append(list(edge_to_add))
            if not is_2_colorable(n, new_edges):
                current_edges = new_edges
                # Try to remove again
                improved2 = True
                while improved2 and time.time() < deadline - 10.0:
                    improved2 = False
                    for i in range(len(current_edges)):
                        if time.time() > deadline - 10.0:
                            break
                        test_edges = current_edges[:i] + current_edges[i+1:]
                        if not is_2_colorable(n, test_edges):
                            current_edges = test_edges
                            if len(current_edges) < best_edge_count:
                                best_edge_count = len(current_edges)
                                best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
                            improved2 = True
                            break
    
    # Try different n values
    for n in [11, 12, 10, 13, 14]:
        if time.time() > deadline - 25.0:
            break
        try_set_cover(n)
        if best_edge_count <= 51:
            break
    
    # Fallback
    if best_result is None:
        n_fb = 9
        all_edges_fb = list(combinations(range(n_fb), 5))
        best_result = {'n': n_fb, 'edges': [list(e) for e in all_edges_fb]}
    
    return best_result
