import random
import time
from itertools import combinations

def is_2_colorable(n, edges, deadline=None):
    """
    Checks if a 5-uniform hypergraph is 2-colorable using backtracking
    with unit propagation. Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    edge_set = [frozenset(e) for e in edges]
    num_edges = len(edge_set)
    
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_set):
        for v in e:
            vertex_edges[v].append(i)
    
    color = [-1] * n
    edge_zeros = [0] * num_edges
    edge_ones = [0] * num_edges
    
    def propagate_and_check():
        """Check for conflicts and apply simple unit propagation."""
        for e_idx in range(num_edges):
            zeros = edge_zeros[e_idx]
            ones = edge_ones[e_idx]
            if zeros == 5 or ones == 5:
                return False
            unassigned = []
            for v in edge_set[e_idx]:
                if color[v] == -1:
                    unassigned.append(v)
            if len(unassigned) == 1:
                v = unassigned[0]
                if zeros == 4 and ones == 0:
                    if color[v] == 0:
                        return False
                    if color[v] == -1:
                        color[v] = 1
                        edge_ones[e_idx] += 1
                        for other_e in vertex_edges[v]:
                            edge_ones[other_e] += 1
                elif zeros == 0 and ones == 4:
                    if color[v] == 1:
                        return False
                    if color[v] == -1:
                        color[v] = 0
                        edge_zeros[e_idx] += 1
                        for other_e in vertex_edges[v]:
                            edge_zeros[other_e] += 1
        return True
    
    def backtrack(v_idx):
        if deadline and time.time() > deadline:
            return True
        
        v = -1
        for i in range(v_idx, n):
            if color[i] == -1:
                v = i
                break
        if v == -1:
            for e_idx in range(num_edges):
                if edge_zeros[e_idx] == 5 or edge_ones[e_idx] == 5:
                    return False
            return True
        
        # Try 0
        color[v] = 0
        saved = [(edge_zeros[e], edge_ones[e]) for e in range(num_edges)]
        valid = True
        for e_idx in vertex_edges[v]:
            edge_zeros[e_idx] += 1
            if edge_zeros[e_idx] == 5:
                valid = False
                break
        if valid and propagate_and_check():
            if backtrack(v + 1):
                return True
        # Restore
        for e_idx in range(num_edges):
            edge_zeros[e_idx], edge_ones[e_idx] = saved[e_idx]
        
        # Try 1
        color[v] = 1
        saved = [(edge_zeros[e], edge_ones[e]) for e in range(num_edges)]
        valid = True
        for e_idx in vertex_edges[v]:
            edge_ones[e_idx] += 1
            if edge_ones[e_idx] == 5:
                valid = False
                break
        if valid and propagate_and_check():
            if backtrack(v + 1):
                return True
        # Restore
        for e_idx in range(num_edges):
            edge_zeros[e_idx], edge_ones[e_idx] = saved[e_idx]
        
        color[v] = -1
        return False
    
    return backtrack(0)

def is_2_colorable_fast(n, edges):
    """
    Checks if a 5-uniform hypergraph is 2-colorable.
    Uses brute force for n <= 12, backtracking for larger n.
    """
    if not edges:
        return True
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    if n <= 12:
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                if (mask & em) == 0 or (mask & em) == em:
                    valid = False
                    break
            if valid:
                return True
        return False
    else:
        return is_2_colorable(n, edges)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_edges = None
    best_n = 9
    best_m = 10**9
    
    n_start = 9
    edges_start = [list(e) for e in combinations(range(n_start), 5)]
    if not is_2_colorable_fast(n_start, edges_start):
        best_edges = edges_start[:]
        best_m = len(edges_start)
        
    for n in range(9, 16):
        if time.time() > deadline - 5:
            break
            
        all_possible = list(combinations(range(n), 5))
        num_edges_total = len(all_possible)
        
        for density in [0.7, 0.5, 0.3, 0.2, 0.1]:
            if time.time() > deadline - 5:
                break
            m_target = int(num_edges_total * density)
            if m_target >= best_m:
                continue
            
            for attempt in range(30):
                if time.time() > deadline - 5:
                    break
                
                random.shuffle(all_possible)
                candidate_edges = [list(e) for e in all_possible[:m_target]]
                
                if not is_2_colorable_fast(n, candidate_edges):
                    current_edges = candidate_edges[:]
                    improved = True
                    while improved and time.time() < deadline - 15:
                        improved = False
                        indices = list(range(len(current_edges)))
                        random.shuffle(indices)
                        for i in indices:
                            if time.time() > deadline - 15:
                                break
                            new_edges = current_edges[:i] + current_edges[i+1:]
                            if not is_2_colorable_fast(n, new_edges):
                                current_edges = new_edges
                                improved = True
                                break
                                
                    if len(current_edges) < best_m:
                        best_m = len(current_edges)
                        best_edges = current_edges[:]
                        best_n = n
                        if best_m <= 30:
                            break
            if best_m <= 30:
                break
    
    if best_edges is None:
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_n = 9
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
