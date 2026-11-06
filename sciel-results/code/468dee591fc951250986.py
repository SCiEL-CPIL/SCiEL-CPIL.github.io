import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n = 16
    
    all_4sets = list(combinations(range(n), 4))
    n4 = len(all_4sets)
    
    vertex_in_4sets = {v: [] for v in range(n)}
    for idx, s in enumerate(all_4sets):
        for v in s:
            vertex_in_4sets[v].append(idx)
    
    # Precompute disjointness: for each pair of 4-sets, are they disjoint?
    disjoint = [[False] * n4 for _ in range(n4)]
    for i in range(n4):
        si = set(all_4sets[i])
        for j in range(i+1, n4):
            if not si & set(all_4sets[j]):
                disjoint[i][j] = disjoint[j][i] = True
    
    edge_set = set()
    for a in range(3):
        for b in range(3, 16):
            edge_set.add((a, b))
    
    degree = [[0] * n4 for _ in range(n)]
    for u, v in edge_set:
        for A_idx in vertex_in_4sets[u]:
            if v not in all_4sets[A_idx]:
                degree[v][A_idx] += 1
        for A_idx in vertex_in_4sets[v]:
            if u not in all_4sets[A_idx]:
                degree[u][A_idx] += 1
    
    def check_q3(u, v):
        for A_idx in vertex_in_4sets[u]:
            A = all_4sets[A_idx]
            for B_idx in vertex_in_4sets[v]:
                if not disjoint[A_idx][B_idx]:
                    continue
                ok = True
                for a in A:
                    d = degree[a][B_idx] + (1 if a == u else 0)
                    if d != 3:
                        ok = False
                        break
                if not ok:
                    continue
                for b in all_4sets[B_idx]:
                    d = degree[b][A_idx] + (1 if b == v else 0)
                    if d != 3:
                        ok = False
                        break
                if ok:
                    return True
        for A_idx in vertex_in_4sets[v]:
            A = all_4sets[A_idx]
            for B_idx in vertex_in_4sets[u]:
                if not disjoint[A_idx][B_idx]:
                    continue
                ok = True
                for a in A:
                    d = degree[a][B_idx] + (1 if a == v else 0)
                    if d != 3:
                        ok = False
                        break
                if not ok:
                    continue
                for b in all_4sets[B_idx]:
                    d = degree[b][A_idx] + (1 if b == u else 0)
                    if d != 3:
                        ok = False
                        break
                if ok:
                    return True
        return False
    
    def add_edge(u, v):
        edge_set.add((u, v))
        for A_idx in vertex_in_4sets[u]:
            if v not in all_4sets[A_idx]:
                degree[v][A_idx] += 1
        for A_idx in vertex_in_4sets[v]:
            if u not in all_4sets[A_idx]:
                degree[u][A_idx] += 1
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if (u, v) not in edge_set:
            if not check_q3(u, v):
                add_edge(u, v)
    
    if time.time() < deadline - 10:
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edge_set]
            if not missing:
                break
            u, v = random.choice(missing)
            if not check_q3(u, v):
                add_edge(u, v)
    
    return [[u, v] for u, v in edge_set]
