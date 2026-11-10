import random
import itertools
import time
import math

def construct():
    n = 20
    
    triples = list(itertools.combinations(range(n), 3))
    
    def is_valid(adj):
        for A in triples:
            common = adj[A[0]] & adj[A[1]] & adj[A[2]]
            if len(common) >= 3:
                return False
        return True

    def count_edges(adj):
        return sum(len(adj[i]) for i in range(n)) // 2

    def greedy_construct():
        adj = [set() for _ in range(n)]
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            adj[u].add(v)
            adj[v].add(u)
            if not is_valid(adj):
                adj[u].remove(v)
                adj[v].remove(u)
        return adj

    def hill_climb(adj, max_iters=2000):
        current_adj = [set(s) for s in adj]
        current_edges = count_edges(current_adj)
        best_adj = [set(s) for s in current_adj]
        best_edges = current_edges
        
        for _ in range(max_iters):
            improved = False
            # Try adding an edge
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in current_adj[i]]
            random.shuffle(missing)
            
            for u, v in missing:
                current_adj[u].add(v)
                current_adj[v].add(u)
                if is_valid(current_adj):
                    current_edges += 1
                    improved = True
                    break
                else:
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
            
            if improved:
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [set(s) for s in current_adj]
            else:
                # Try a swap to escape local optimum
                existing = [(i, j) for i in range(n) for j in current_adj[i] if i < j]
                if not existing: break
                random.shuffle(existing)
                for u, v in existing[:5]:
                    current_adj[u].remove(v)
                    current_adj[v].remove(u)
                    
                    # Try to add a different edge
                    for x, y in missing:
                        if (x==u and y==v) or (x==v and y==u): continue
                        current_adj[x].add(y)
                        current_adj[y].add(x)
                        if is_valid(current_adj):
                            # Accepted swap, edge count same, but structure changed
                            break
                        else:
                            current_adj[x].remove(y)
                            current_adj[y].remove(x)
                    else:
                        # No edge could be added, revert swap
                        current_adj[u].add(v)
                        current_adj[v].add(u)

        return best_adj, best_edges

    best_adj = None
    best_edges = 0
    
    for trial in range(100):
        random.seed(trial)
        adj = greedy_construct()
        e = count_edges(adj)
        if e > best_edges:
            best_edges = e
            best_adj = [set(s) for s in adj]
        
        # Refine
        r_adj, r_e = hill_climb(adj, 500)
        if r_e > best_edges:
            best_edges = r_e
            best_adj = r_adj

    if best_adj is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]

    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
    return edges
