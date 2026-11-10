"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

_ADJ = None
_EDGE_SET = None
_EDGE_EXISTS = None

def _init():
    global _ADJ, _EDGE_SET, _EDGE_EXISTS
    if _ADJ is not None:
        return
    n = P['n']
    adj = defaultdict(list)
    eset = set()
    edge_exists = [[False]*n for _ in range(n)]
    for a, b in P['edges']:
        adj[a].append(b)
        adj[b].append(a)
        eset.add((min(a,b), max(a,b)))
        edge_exists[a][b] = True
        edge_exists[b][a] = True
    _ADJ = adj
    _EDGE_SET = eset
    _EDGE_EXISTS = edge_exists

def _find_hamiltonian_cycles_in_subset(vertices, max_cycles=10, deadline=None):
    if deadline is None:
        deadline = time.time() + 5
    k = len(vertices)
    if k < 3:
        return []
    
    v_set = set(vertices)
    v_map = {v: i for i, v in enumerate(vertices)}
    inv_map = {i: v for i, v in enumerate(vertices)}
    
    adj_sub = [[False]*k for _ in range(k)]
    for v in vertices:
        i = v_map[v]
        for u in _ADJ[v]:
            if u in v_set:
                j = v_map[u]
                adj_sub[i][j] = True
                adj_sub[j][i] = True
                
    edge_count = sum(sum(row) for row in adj_sub) // 2
    if edge_count < k:
        return []
        
    global_used = set()
    cycles = []
    
    def dfs_global(path, local_used):
        if time.time() > deadline:
            return None
        if len(path) == k:
            last = path[-1]
            first = path[0]
            if adj_sub[last][first]:
                e = (min(last, first), max(last, first))
                if e not in local_used and e not in global_used:
                    return list(path)
            return None
        cur = path[-1]
        neighbors = [j for j in range(k) if adj_sub[cur][j] and j not in path]
        def score(j):
            cnt = 0
            for nn in range(k):
                if nn not in path and adj_sub[j][nn]:
                    e = (min(j, nn), max(j, nn))
                    if e not in local_used and e not in global_used:
                        cnt += 1
            return cnt
        neighbors.sort(key=score)
        
        for nb in neighbors:
            e = (min(cur, nb), max(cur, nb))
            if e in local_used or e in global_used:
                continue
            new_path = path + [nb]
            new_local_used = local_used | {e}
            res = dfs_global(new_path, new_local_used)
            if res is not None:
                return res
        return None

    while len(cycles) < max_cycles:
        if time.time() > deadline:
            break
        res = dfs_global([0], set())
        if res is None:
            break
        cycle_vertices = [inv_map[i] for i in res]
        for i in range(k):
            a = res[i]
            b = res[(i+1) % k]
            e = (min(a, b), max(a, b))
            global_used.add(e)
        cycles.append(cycle_vertices)
        
    return cycles

def construct():
    _init()
    deadline = time.time() + 85
    best_cycles = []
    best_count = 0
    n = P['n']
    
    for k in range(5, 9):
        if time.time() > deadline - 10:
            break
        
        if k <= 6:
            subs = list(itertools.combinations(range(n), k))
        else:
            random.seed(42)
            subs = []
            seen = set()
            while len(subs) < 5000 and time.time() < deadline - 20:
                s = tuple(sorted(random.sample(range(n), k)))
                if s not in seen:
                    seen.add(s)
                    subs.append(s)
                    
        for subset in subs:
            if time.time() > deadline - 5:
                break
            e_count = 0
            s_set = set(subset)
            for v in subset:
                for u in subset:
                    if v < u and _EDGE_EXISTS[v][u]:
                        e_count += 1
            if e_count < 2 * k:
                continue
                
            cycles = _find_hamiltonian_cycles_in_subset(list(subset), max_cycles=10, deadline=deadline-5)
            if len(cycles) > best_count:
                best_count = len(cycles)
                best_cycles = cycles
                if best_count >= 10:
                    return [[int(v) for v in c] for c in best_cycles]
                    
    if not best_cycles:
        for i in range(n):
            for j in range(i+1, n):
                if not _EDGE_EXISTS[i][j]: continue
                for l in range(j+1, n):
                    if _EDGE_EXISTS[i][l] and _EDGE_EXISTS[j][l]:
                        best_cycles = [[i, j, l]]
                        best_count = 1
                        break
                if best_cycles: break
            if best_cycles: break
            
    return [[int(v) for v in c] for c in best_cycles]

# EVOLVE-BLOCK-END
