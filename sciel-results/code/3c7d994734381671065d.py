"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, permutations

def _precompute_cycles():
    cycles = []
    seen = set()
    for perm in permutations(range(1, 6)):
        cycle = (0,) + perm
        edges = frozenset(frozenset((cycle[i], cycle[(i+1)%6])) for i in range(6))
        if edges not in seen:
            seen.add(edges)
            cycles.append(cycle)
    return cycles

_CYCLES = _precompute_cycles()

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def _check_no_c6_with_new_vertex(v, i, c, adj, adj_list):
        """
        Check if setting adj[i][v]=c creates a C6 involving v.
        """
        for d in range(i):
            if adj[d][v] != c:
                continue
            
            N_i = adj_list[c][i]
            N_d = adj_list[c][d]
            
            if not N_i or not N_d:
                continue
                
            for a in N_i:
                if a == d:
                    continue
                for c_node in N_d:
                    if c_node == a or c_node == i or c_node == d:
                        continue
                        
                    N_a = adj_list[c][a]
                    N_c_node = adj_list[c][c_node]
                    
                    if N_a & N_c_node:
                        for b in (N_a & N_c_node):
                            if b != i and b != d and b != a and b != c_node:
                                return False
        return True

    def _constrained_backtrack(n_target, deadline):
        adj = [[-1]*n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        
        def _search(v):
            if time.time() > deadline:
                return False
            
            if v == n_target:
                return True
            
            if v == 0:
                return _search(v + 1)
            
            def _assign(idx):
                if time.time() > deadline:
                    return False
                
                if idx == v:
                    return _search(v + 1)
                
                i = idx
                for c in range(4):
                    if not _check_no_c6_with_new_vertex(v, i, c, adj, adj_list):
                        continue
                    
                    adj[i][v] = c
                    adj[v][i] = c
                    adj_list[c][i].add(v)
                    adj_list[c][v].add(i)
                    
                    if _assign(idx + 1):
                        return True
                    
                    adj[i][v] = -1
                    adj[v][i] = -1
                    adj_list[c][i].remove(v)
                    adj_list[c][v].remove(i)
                
                return False
            
            return _assign(0)
        
        if _search(0):
            colors = []
            for i in range(n_target):
                for j in range(i+1, n_target):
                    colors.append(adj[i][j])
            return colors
        return None

    def _verify(colors, n):
        if n < 6:
            return True
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        for c in range(4):
            for subset in combinations(range(n), 6):
                for cycle in _CYCLES:
                    all_edges = True
                    for k in range(6):
                        a = subset[cycle[k]]
                        b = subset[cycle[(k+1)%6]]
                        if adj[a][b] != c:
                            all_edges = False
                            break
                    if all_edges:
                        return False
        return True

    best_n = 1
    best_colors = []
    
    for n_target in range(2, 18):
        if time.time() > deadline - 10:
            break
        
        colors = _constrained_backtrack(n_target, deadline - 5)
        if colors is not None:
            if _verify(colors, n_target):
                best_n = n_target
                best_colors = colors
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
