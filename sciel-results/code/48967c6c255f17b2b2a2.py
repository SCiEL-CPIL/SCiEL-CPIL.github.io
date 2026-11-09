import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 110

def _is_2_colorable(n, edges):
    edge_sets = [frozenset(e) for e in edges]
    colors = [-1] * n
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_sets):
        for v in e:
            vertex_edges[v].append(i)
            
    def propagate():
        changed = True
        while changed:
            changed = False
            for i, e in enumerate(edge_sets):
                c0 = 0
                c1 = 0
                uncolored = []
                for v in e:
                    if colors[v] == 0:
                        c0 += 1
                    elif colors[v] == 1:
                        c1 += 1
                    else:
                        uncolored.append(v)
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and len(uncolored) == 1:
                    colors[uncolored[0]] = 1
                    changed = True
                elif c1 == 4 and len(uncolored) == 1:
                    colors[uncolored[0]] = 0
                    changed = True
        return True

    def backtrack():
        uncolored_vertices = [v for v in range(n) if colors[v] == -1]
        if not uncolored_vertices:
            for e in edge_sets:
                c = colors[e[0]]
                if all(colors[v] == c for v in e):
                    return False
            return True
        def score(v):
            s = 0
            for ei in vertex_edges[v]:
                e = edge_sets[ei]
                colored_in_e = sum(1 for u in e if colors[u] != -1)
                s += colored_in_e
            return s
        v = max(uncolored_vertices, key=score)
        colors[v] = 0
        if propagate() and backtrack():
            return True
        colors[v] = 1
        if propagate() and backtrack():
            return True
        colors[v] = -1
        return False

    return backtrack()

def construct():
    n = 9
    base_edges = [list(e) for e in combinations(range(n), 5)]
    
    best_edges = base_edges
    best_len = len(base_edges)
    
    # Try multiple random shuffles for pruning
    for seed in range(10):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        random.seed(seed)
        edges = base_edges.copy()
        random.shuffle(edges)
        
        i = 0
        while i < len(edges):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            test_edges = edges[:i] + edges[i+1:]
            if not _is_2_colorable(n, test_edges):
                edges = test_edges
                # do not increment i
            else:
                i += 1
        
        if len(edges) < best_len:
            best_len = len(edges)
            best_edges = edges
    
    return {'n': n, 'edges': best_edges}
