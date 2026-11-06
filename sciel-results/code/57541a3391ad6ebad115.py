import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def find_maximal_cliques(n, edges):
        # Build adjacency list and neighbor sets
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        cliques = []
        
        # Bron-Kerbosch with pivot
        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2: # Only keep cliques of size >= 2
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose u from P union X with max |P intersect adj[u]|
            # For simplicity, just pick any u from P union X if P is empty, else from P
            # Standard heuristic: max degree in P
            U = P.union(X)
            if not U:
                return
            
            # Find pivot u
            max_count = -1
            pivot = None
            for u in U:
                count = len(P.intersection(adj[u]))
                if count > max_count:
                    max_count = count
                    pivot = u
            
            # Recurse on P \ adj[pivot]
            # If pivot is in P, we iterate over P \ adj[pivot]
            # If pivot is not in P (i.e., in X), we iterate over all P
            # Wait, the standard BK says:
            # If P is empty, return.
            # Choose pivot u from P union X.
            # For each v in P \ adj[u]:
            #    bronk(R U {v}, P intersect adj[v], X intersect adj[v])
            #    P = P \ {v}
            #    X = X U {v}
            
            if pivot is not None and pivot in P:
                candidates = P.difference(adj[pivot])
            else:
                candidates = P.copy()
                
            # We need to iterate and modify P and X
            # To avoid modification issues during iteration, we can use a list
            # But P and X are sets.
            # We will iterate over a snapshot of candidates
            
            for v in list(candidates):
                if v not in P: continue # In case it was removed? No, candidates is subset of P.
                bronk(R.union({v}), P.intersection(adj[v]), X.intersection(adj[v]))
                P.discard(v)
                X.add(v)
                
        bronk(frozenset(), set(range(n)), set())
        return cliques

    def greedy_partition(n, edges, cliques):
        # Map edge to index for fast lookup? Or just a set of tuples.
        uncovered = set(edges)
        partition = []
        
        # Sort cliques by size descending
        # Ties broken by sum of vertex degrees? Or just size.
        cliques_sorted = sorted(cliques, key=lambda c: len(c), reverse=True)
        
        for c in cliques_sorted:
            if time.time() > deadline:
                break
            # Check if all edges in clique are uncovered
            valid = True
            for u, v in combinations(sorted(c), 2):
                if (u, v) not in uncovered and (v, u) not in uncovered:
                    valid = False
                    break
            if valid:
                partition.append(list(c))
                for u, v in combinations(sorted(c), 2):
                    uncovered.discard((u, v))
                    uncovered.discard((v, u))
                    
        # Cover remaining edges with 2-cliques
        for u, v in uncovered:
            partition.append([u, v])
            
        return partition

    instances = [
        (32, 0.6, 601),
        (64, 0.55, 602)
    ]
    
    results = []
    for n, p, seed in instances:
        edges = random_graph(n, p, seed)
        # Find maximal cliques
        cliques = find_maximal_cliques(n, edges)
        # Greedy partition
        part = greedy_partition(n, edges, cliques)
        results.append(part)
        
    return results
