"""Clique partition solver for dense random graphs."""
import time
import sys
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 110.0  # safety margin before 120s kill

    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_maximal_cliques(n, adj):
        """Bron-Kerbosch with pivot to find all maximal cliques."""
        cliques = []
        
        def bron_kerbosch(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            # Choose pivot
            if P:
                pivot = max(P, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                candidates = P
            for v in list(candidates):
                bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return cliques

    def greedy_clique_partition(n, edges, adj, cliques, time_limit):
        """Greedy set cover: pick clique covering most uncovered edges."""
        edge_set = set(edges)
        uncovered = set(edge_set)
        partition = []
        
        # Precompute edges in each clique
        clique_edges = {}
        for c in cliques:
            if len(c) < 2:
                continue
            ce = set()
            cl = sorted(c)
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    e = (cl[i], cl[j])
                    if e in edge_set:
                        ce.add(e)
            if ce:
                clique_edges[frozenset(c)] = ce
        
        # Sort cliques by size descending as initial heuristic
        sorted_cliques = sorted(clique_edges.keys(), key=lambda c: -len(c))
        
        while uncovered:
            if time.time() > time_limit:
                break
            best_clique = None
            best_count = 0
            for c in sorted_cliques:
                ce = clique_edges[c]
                count = len(ce & uncovered)
                if count > best_count:
                    best_count = count
                    best_clique = c
                    if best_count == len(ce):
                        break
            if best_clique is None or best_count == 0:
                # Fallback: single edges
                for e in list(uncovered):
                    partition.append(list(e))
                    uncovered.discard(e)
                break
            partition.append(sorted(best_clique))
            uncovered -= clique_edges[best_clique]
        
        # Handle remaining uncovered edges as singletons
        for e in uncovered:
            partition.append(list(e))
        
        return partition

    def local_search_improve(n, edges, adj, partition, time_limit):
        """Try to merge cliques or swap edges to reduce count."""
        # This is a simple improvement: try to find if any two cliques can be merged
        # (i.e., their union is also a clique)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            new_partition = []
            i = 0
            while i < len(partition):
                c1 = set(partition[i])
                j = i + 1
                merged = False
                while j < len(partition):
                    c2 = set(partition[j])
                    union = c1 | c2
                    # Check if union is a clique
                    is_clique = True
                    ul = sorted(union)
                    for a in range(len(ul)):
                        for b in range(a + 1, len(ul)):
                            if (ul[a], ul[b]) not in set(edges):
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        new_partition.append(ul)
                        merged = True
                        # skip j
                        j += 1
                        c1 = union
                    else:
                        j += 1
                if not merged:
                    new_partition.append(sorted(c1))
                i += 1
            if len(new_partition) < len(partition):
                partition = new_partition
                improved = True
        return partition

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = random_graph(n, p, seed)
        adj = build_adjacency(n, edges)
        
        # Find maximal cliques
        cliques = find_maximal_cliques(n, adj)
        
        # Greedy partition
        partition = greedy_clique_partition(n, edges, adj, cliques, DEADLINE - 5)
        
        # Try local search improvement
        partition = local_search_improve(n, edges, adj, partition, DEADLINE - 2)
        
        results.append(partition)
    
    return results
