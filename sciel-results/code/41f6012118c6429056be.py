"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    time_start = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def greedy_clique_partition(edges, adj, n, deadline):
        """
        Greedy approach: repeatedly find a maximal clique covering the most uncovered edges.
        """
        uncovered = set(edges)
        cliques = []

        # Precompute some structure
        # For speed, we'll use a greedy heuristic:
        # Pick the edge with the highest "clique potential" (i.e., both endpoints have high degree in uncovered graph)
        
        while uncovered and time.time() < deadline:
            # Find the best edge to start a clique from
            # Heuristic: pick edge (u,v) that maximizes the size of the common neighborhood in uncovered graph
            best_edge = None
            best_score = -1
            
            # Sample a subset of edges if too many
            if len(uncovered) > 500:
                sample = random.sample(list(uncovered), 500)
            else:
                sample = list(uncovered)
            
            for (u, v) in sample:
                # Common neighbors in uncovered graph
                # A vertex w can be added if (u,w) and (v,w) are both in uncovered
                common = adj[u] & adj[v]
                # Filter to only vertices where both edges are still uncovered
                candidates = set()
                for w in common:
                    if (min(u,w), max(u,w)) in uncovered and (min(v,w), max(v,w)) in uncovered:
                        candidates.add(w)
                
                # Score: size of candidates + 2 (for u and v themselves)
                score = len(candidates) + 2
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
            
            if best_edge is None:
                # Fallback: just pick any edge
                best_edge = next(iter(uncovered))
                best_score = 2
            
            u, v = best_edge
            # Build the clique greedily
            clique = {u, v}
            
            # Get current common neighbors in uncovered graph
            common = adj[u] & adj[v]
            candidates = list(common)
            # Sort by degree in uncovered subgraph for better greedy choice
            def deg_in_uncovered(w):
                count = 0
                for x in clique:
                    if (min(x,w), max(x,w)) in uncovered:
                        count += 1
                return count
            
            candidates.sort(key=lambda w: deg_in_uncovered(w), reverse=True)
            
            for w in candidates:
                if time.time() > deadline:
                    break
                # Check if w can be added to clique
                can_add = True
                for x in clique:
                    if (min(x,w), max(x,w)) not in uncovered:
                        can_add = False
                        break
                if can_add:
                    clique.add(w)
                    # Remove edges covered by adding w
                    for x in list(clique - {w}):
                        e = (min(x,w), max(x,w))
                        if e in uncovered:
                            uncovered.discard(e)
            
            # Remove all edges within the clique
            for x, y in combinations(sorted(clique), 2):
                uncovered.discard((x, y))
            
            cliques.append(sorted(clique))
        
        return cliques

    def try_improve(cliques, edges, adj, n, deadline, max_rounds=50):
        """
        Try to merge cliques: if C1 ∪ C2 is a clique, merge them.
        """
        improved = True
        round_num = 0
        while improved and time.time() < deadline and round_num < max_rounds:
            improved = False
            round_num += 1
            for i in range(len(cliques)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(cliques)):
                    if time.time() > deadline:
                        break
                    C = set(cliques[i]) | set(cliques[j])
                    # Check if C is a clique
                    is_clique = True
                    for a, b in combinations(sorted(C), 2):
                        if (a, b) not in edges:
                            is_clique = False
                            break
                    if is_clique:
                        cliques[i] = sorted(C)
                        cliques.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return cliques

    def solve_instance(n, p, seed, deadline):
        edges, adj = random_graph(n, p, seed)
        cliques = greedy_clique_partition(edges, adj, n, deadline)
        cliques = try_improve(cliques, edges, adj, n, deadline)
        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() >= deadline:
            # Fallback: just return all edges as 2-cliques
            edges, _ = random_graph(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed, deadline))

    return result
# EVOLVE-BLOCK-END
