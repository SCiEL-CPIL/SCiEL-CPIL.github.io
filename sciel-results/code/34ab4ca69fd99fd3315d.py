import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 18
    
    # Initial graph: K6,6,6
    edges = set()
    for i in range(6):
        for j in range(6, 12):
            edges.add((i, j))
            edges.add((j, i))
        for k in range(12, 18):
            edges.add((i, k))
            edges.add((k, i))
    for i in range(6, 12):
        for k in range(12, 18):
            edges.add((i, k))
            edges.add((k, i))
            
    def to_list(edges_set):
        return [[u, v] for u, v in edges_set if u < v]

    def is_k4_free(edge_set):
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    for w in range(v+1, n):
                        if adj[u][w] and adj[v][w]:
                            for x in range(w+1, n):
                                if adj[u][x] and adj[v][x] and adj[w][x]:
                                    return False
        return True

    def has_tf_set_of_size(edge_set, size):
        adj = [[False]*n for _ in range(n)]
        for u, v in edge_set:
            adj[u][v] = True
            adj[v][u] = True
        for comb in combinations(range(n), size):
            is_tf = True
            for i in range(len(comb)):
                u = comb[i]
                for j in range(i+1, len(comb)):
                    v = comb[j]
                    if adj[u][v]:
                        for k in range(j+1, len(comb)):
                            w = comb[k]
                            if adj[u][w] and adj[v][w]:
                                is_tf = False
                                break
                if not is_tf:
                    break
            if is_tf:
                return True
        return False

    def calc_score(edge_set):
        for s in range(18, 0, -1):
            if has_tf_set_of_size(edge_set, s):
                return s
        return 0

    best_edges = set(edges)
    best_score = calc_score(best_edges)
    
    current_edges = set(edges)
    current_score = best_score
    
    while time.time() < deadline:
        # Swap
        if len(current_edges) > 0:
            edge_to_remove = random.choice(list(current_edges))
            u, v = edge_to_remove
            while True:
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and (x, y) not in current_edges:
                    break
            
            new_edges = set(current_edges)
            new_edges.remove((u, v))
            new_edges.remove((v, u))
            new_edges.add((x, y))
            new_edges.add((y, x))
            
            if is_k4_free(new_edges):
                # Check if it improves
                # Heuristic: check if it has TF set of size 11
                if current_score > 10:
                    if not has_tf_set_of_size(new_edges, 11):
                        new_score = calc_score(new_edges)
                        if new_score < current_score:
                            current_edges = new_edges
                            current_score = new_score
                            if current_score < best_score:
                                best_edges = set(current_edges)
                                best_score = current_score
                else:
                    new_score = calc_score(new_edges)
                    if new_score < current_score:
                        current_edges = new_edges
                        current_score = new_score
                        if current_score < best_score:
                            best_edges = set(current_edges)
                            best_score = current_score

    return to_list(best_edges)
