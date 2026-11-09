import time
from random import Random

def _random_graph(n, p, seed):
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _maximal_cliques_bit(n, adj_masks, deadline):
    """
    Finds all maximal cliques in a graph represented by adjacency bitmasks.
    Uses Bron-Kerbosch algorithm with pivoting.
    Returns a list of lists of vertices.
    """
    cliques = []
    
    def bron_kerbosch(C, P, X):
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(C) >= 2:
                cliques.append(C[:])
            return
        
        U = P | X
        if U == 0:
            return
            
        # Find pivot with max common neighbors in P
        pivot = None
        max_common = -1
        u_temp = U
        while u_temp:
            bit = u_temp & (-u_temp)
            idx = bit.bit_length() - 1
            common = P & adj_masks[idx]
            count = common.bit_count()
            if count > max_common:
                max_common = count
                pivot = idx
            u_temp -= bit
            
        if pivot is None:
            return
            
        # Candidates are vertices in P not adjacent to pivot
        candidates = P & ~adj_masks[pivot]
        
        c_temp = candidates
        while c_temp:
            bit = c_temp & (-c_temp)
            v = bit.bit_length() - 1
            c_temp -= bit
            
            C.append(v)
            new_P = P & adj_masks[v]
            new_X = X & adj_masks[v]
            
            bron_kerbosch(C, new_P, new_X)
            
            C.pop()
            P &= ~bit
            X |= bit

    initial_P = (1 << n) - 1
    bron_kerbosch([], initial_P, 0)
    
    return cliques

def _solve_instance(n, p, seed, deadline):
    edges = _random_graph(n, p, seed)
    if not edges:
        return []
    
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    max_cliques = _maximal_cliques_bit(n, adj_masks, deadline)
    
    if not max_cliques:
        return [list(e) for e in edges]
        
    clique_edges = []
    for C in max_cliques:
        e_set = set()
        for i in range(len(C)):
            for j in range(i+1, len(C)):
                u, v = C[i], C[j]
                if u > v: u, v = v, u
                e_set.add((u, v))
        clique_edges.append(e_set)
        
    remaining_edges = set(edges)
    result_cliques = []
    
    while remaining_edges:
        if time.time() > deadline:
            for e in remaining_edges:
                result_cliques.append(list(e))
            break
            
        best_idx = -1
        best_count = 0
        
        for i, e_set in enumerate(clique_edges):
            count = len(remaining_edges & e_set)
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_idx == -1:
            for e in remaining_edges:
                result_cliques.append(list(e))
            break
            
        result_cliques.append(max_cliques[best_idx])
        remaining_edges.difference_update(clique_edges[best_idx])
        
    return result_cliques

def construct():
    deadline = time.time() + 110
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = _solve_instance(n, p, seed, deadline)
        result.append(cliques)
    return result
