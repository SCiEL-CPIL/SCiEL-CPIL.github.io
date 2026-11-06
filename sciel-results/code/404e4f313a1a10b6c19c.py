import time
import random
from itertools import combinations
import sys

def is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    # Brute force for n <= 14
    if n <= 14:
        edge_masks = []
        for e in edges:
            mask = 0
            for v in e:
                mask |= (1 << v)
            edge_masks.append(mask)
            
        for cm in range(1 << n):
            valid = True
            for em in edge_masks:
                # Monochromatic if intersection is 0 or full
                if (em & cm) == 0 or (em & cm) == em:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        # DPLL for n > 14
        m = len(edges)
        color = [-1] * n
        c0 = [0] * m
        c1 = [0] * m
        
        vertex_edges = [[] for _ in range(n)]
        edge_vertices = []
        for i, e in enumerate(edges):
            edge_vertices.append(e)
            for v in e:
                vertex_edges[v].append(i)
                
        def propagate():
            changed = True
            while changed:
                changed = False
                for i in range(m):
                    if c0[i] == 5 or c1[i] == 5:
                        return False
                    if c0[i] == 4 and c1[i] == 0:
                        for v in edge_vertices[i]:
                            if color[v] == -1:
                                color[v] = 1
                                c1[i] += 1
                                for other_e in vertex_edges[v]:
                                    c1[other_e] += 1
                                changed = True
                                if c1[i] == 5:
                                    return False
                                break
                    elif c1[i] == 4 and c0[i] == 0:
                        for v in edge_vertices[i]:
                            if color[v] == -1:
                                color[v] = 0
                                c0[i] += 1
                                for other_e in vertex_edges[v]:
                                    c0[other_e] += 1
                                changed = True
                                if c0[i] == 5:
                                    return False
                                break
            return True
        
        def backtrack():
            best_edge = -1
            min_unassigned = 6
            max_assigned = 0
            
            for i in range(m):
                unassigned = 5 - c0[i] - c1[i]
                assigned = c0[i] + c1[i]
                if unassigned < min_unassigned or (unassigned == min_unassigned and assigned > max_assigned):
                    min_unassigned = unassigned
                    max_assigned = assigned
                    best_edge = i
                    if unassigned == 0:
                        if c0[i] == 5 or c1[i] == 5:
                            return False
            
            if best_edge == -1:
                return True
                
            candidates = []
            for v in edge_vertices[best_edge]:
                if color[v] == -1:
                    candidates.append(v)
            
            if not candidates:
                return True
                
            for v in candidates:
                saved_c0 = c0[:]
                saved_c1 = c1[:]
                color[v] = 0
                ok = True
                for e_idx in vertex_edges[v]:
                    c0[e_idx] += 1
                    if c0[e_idx] == 5:
                        ok = False
                        break
                if ok and propagate():
                    if backtrack():
                        return True
                color[v] = -1
                c0[:] = saved_c0
                c1[:] = saved_c1
                
                saved_c0 = c0[:]
                saved_c1 = c1[:]
                color[v] = 1
                ok = True
                for e_idx in vertex_edges[v]:
                    c1[e_idx] += 1
                    if c1[e_idx] == 5:
                        ok = False
                        break
                if ok and propagate():
                    if backtrack():
                        return True
                color[v] = -1
                c0[:] = saved_c0
                c1[:] = saved_c1
            
            return False
        
        if not propagate():
            return False
            
        return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    for n in [11, 12, 13, 10, 9, 14]:
        if time.time() > deadline - 10:
            break
            
        all_possible = list(combinations(range(n), 5))
        
        if is_2_colorable(n, all_possible):
            continue
            
        max_attempts = 20
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                break
                
            curr = [list(e) for e in all_possible]
            edge_indices = list(range(len(curr)))
            random.shuffle(edge_indices)
            
            removed_any = True
            while removed_any and time.time() < deadline - 20:
                removed_any = False
                for i in edge_indices:
                    if time.time() > deadline - 20:
                        break
                    if i >= len(curr):
                        continue
                    
                    new_edges = curr[:i] + curr[i+1:]
                    if not is_2_colorable(n, new_edges):
                        curr = new_edges
                        edge_indices = list(range(len(curr)))
                        random.shuffle(edge_indices)
                        removed_any = True
                        break
                        
            if len(curr) < best_m:
                best_m = len(curr)
                best_edges = curr[:]
                best_n = n
                
        if best_edges and time.time() < deadline - 5:
            curr_edges = best_edges[:]
            curr_n = best_n
            all_edges_set = set(tuple(e) for e in all_possible)
            curr_set = set(tuple(e) for e in curr_edges)
            
            T = 5.0
            cooling = 0.95
            steps = 0
            max_steps = 3000
            
            while steps < max_steps and time.time() < deadline - 2:
                steps += 1
                move_type = random.random()
                new_set = set(curr_set)
                
                if move_type < 0.5:
                    if not curr_set: continue
                    e_to_remove = random.choice(list(curr_set))
                    new_set.remove(e_to_remove)
                elif move_type < 0.8:
                    available = list(all_edges_set - curr_set)
                    if not available: continue
                    e_to_add = random.choice(available)
                    new_set.add(e_to_add)
                else:
                    if not curr_set: continue
                    e_to_remove = random.choice(list(curr_set))
                    available = list(all_edges_set - curr_set)
                    if not available: continue
                    e_to_add = random.choice(available)
                    new_set.remove(e_to_remove)
                    new_set.add(e_to_add)
                    
                new_edges_list = [list(e) for e in new_set]
                is_valid_new = not is_2_colorable(curr_n, new_edges_list)
                
                if is_valid_new:
                    delta = len(new_set) - len(curr_set)
                    if delta < 0:
                        curr_set = new_set
                        if len(curr_set) < best_m:
                            best_m = len(curr_set)
                            best_edges = [list(e) for e in curr_set]
                            best_n = curr_n
                    else:
                        if random.random() < (1.0 / (1.0 + delta * T)):
                            curr_set = new_set
                else:
                    pass
                    
                T *= cooling
                
        if best_m <= 51:
            break

    if best_edges is None:
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
