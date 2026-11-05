import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    def generate_edges(n, p, seed):
        """Generate edges of a random graph with given parameters."""
        edges = set()
        rng = random.Random(seed)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def build_adj(n, edges):
        """Build adjacency list from edge set."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_maximal_clique(start_vertex, adj, remaining_edges):
        """
        Find a maximal clique starting from start_vertex using greedy expansion.
        Returns the clique as a set of vertices.
        """
        if start_vertex not in remaining_edges and not any((start_vertex, v) in remaining_edges or (v, start_vertex) in remaining_edges for v in range(len(adj))):
            # No edges incident to start_vertex in remaining graph
            return set()
        
        # Start with the vertex itself
        clique = {start_vertex}
        
        # Candidates are neighbors of start_vertex that have edges in remaining graph
        candidates = set()
        for v in adj[start_vertex]:
            if (min(start_vertex, v), max(start_vertex, v)) in remaining_edges:
                candidates.add(v)
        
        # Greedy expansion
        while candidates:
            # Pick the candidate with the most connections to other candidates
            best_v = None
            best_score = -1
            for v in candidates:
                # Count how many other candidates are adjacent to v in the remaining graph
                score = 0
                for u in candidates:
                    if u == v:
                        continue
                    if (min(u, v), max(u, v)) in remaining_edges:
                        score += 1
                # Also consider degree in remaining graph as tiebreaker
                if score > best_score or (score == best_score and best_v is not None and len(adj[v]) > len(adj[best_v])):
                    best_score = score
                    best_v = v
            
            if best_v is None:
                break
            
            clique.add(best_v)
            
            # Remove best_v and its non-neighbors from candidates
            non_neighbors = set()
            for u in candidates:
                if u != best_v:
                    if (min(u, best_v), max(u, best_v)) not in remaining_edges:
                        non_neighbors.add(u)
            
            candidates = candidates - {best_v} - non_neighbors
        
        return clique

    def greedy_clique_partition(n, edges_set):
        """
        Find a clique partition using a weighted set cover approach.
        1. Generate candidate cliques using greedy expansion from high-degree vertices.
        2. Select cliques greedily to maximize uncovered edge coverage.
        """
        if not edges_set:
            return []
        
        remaining_edges = set(edges_set)
        adj = build_adj(n, edges_set)
        partition = []
        
        # Track which edges are covered
        covered_edges = set()
        
        # Generate candidate cliques
        candidates = []
        
        # Compute degrees
        degrees = [0] * n
        for u, v in edges_set:
            degrees[u] += 1
            degrees[v] += 1
        
        # Sort vertices by degree descending
        vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Generate candidate cliques from top vertices
        max_candidates = min(100, len(vertices_by_degree))
        for start_v in vertices_by_degree[:max_candidates]:
            if time.time() - start_time > DEADLINE:
                break
            clique = find_maximal_clique(start_v, adj, edges_set)
            if len(clique) >= 2:
                # Check if this clique has at least one edge in the original graph
                clique_edges = set()
                clique_list = list(clique)
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        u, v = clique_list[i], clique_list[j]
                        edge = (min(u, v), max(u, v))
                        if edge in edges_set:
                            clique_edges.add(edge)
                if clique_edges:
                    candidates.append((clique, clique_edges))
        
        # Also try to generate cliques from pairs of high-degree vertices
        for i in range(min(20, len(vertices_by_degree))):
            for j in range(i + 1, min(20, len(vertices_by_degree))):
                if time.time() - start_time > DEADLINE:
                    break
                v1, v2 = vertices_by_degree[i], vertices_by_degree[j]
                if (min(v1, v2), max(v1, v2)) in edges_set:
                    # Try to expand from this pair
                    clique = {v1, v2}
                    candidates_set = set()
                    for v in adj[v1]:
                        if v != v2 and (min(v, v2), max(v, v2)) in edges_set:
                            candidates_set.add(v)
                    
                    while candidates_set:
                        best_v = None
                        best_score = -1
                        for v in candidates_set:
                            score = 0
                            for u in candidates_set:
                                if u == v:
                                    continue
                                if (min(u, v), max(u, v)) in edges_set:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_v = v
                        if best_v is None:
                            break
                        clique.add(best_v)
                        non_neighbors = set()
                        for u in candidates_set:
                            if u != best_v:
                                if (min(u, best_v), max(u, best_v)) not in edges_set:
                                    non_neighbors.add(u)
                        candidates_set = candidates_set - {best_v} - non_neighbors
                    
                    if len(clique) >= 2:
                        clique_edges = set()
                        clique_list = list(clique)
                        for i2 in range(len(clique_list)):
                            for j2 in range(i2 + 1, len(clique_list)):
                                u, v = clique_list[i2], clique_list[j2]
                                edge = (min(u, v), max(u, v))
                                if edge in edges_set:
                                    clique_edges.add(edge)
                        if clique_edges:
                            candidates.append((clique, clique_edges))
        
        # Greedy set cover: select cliques that cover the most uncovered edges
        uncovered_edges = set(edges_set)
        
        while uncovered_edges:
            if time.time() - start_time > DEADLINE:
                # Fallback: remaining edges as 2-cliques
                for e in uncovered_edges:
                    partition.append(list(e))
                break
            
            best_clique = None
            best_coverage = 0
            best_clique_edges = None
            
            for clique, clique_edges in candidates:
                # Count how many edges in this clique are still uncovered
                coverage = len(clique_edges & uncovered_edges)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_clique = clique
                    best_clique_edges = clique_edges
            
            if best_clique is None or best_coverage == 0:
                # No candidate clique covers any uncovered edge
                # Fallback: use a 2-clique for one uncovered edge
                e = next(iter(uncovered_edges))
                partition.append(list(e))
                uncovered_edges.discard(e)
                continue
            
            # Add the selected clique to the partition
            partition.append(sorted(best_clique))
            
            # Remove covered edges from uncovered set
            uncovered_edges -= best_clique_edges
        
        return partition

    def solve_instance(n, p, seed):
        edges_set = generate_edges(n, p, seed)
        return greedy_clique_partition(n, edges_set)

    result1 = solve_instance(32, 0.6, 601)
    if time.time() - start_time > DEADLINE:
        # Fallback for second instance
        edges2 = generate_edges(64, 0.55, 602)
        result2 = [[u, v] for (u, v) in edges2]
        return [result1, result2]
    
    result2 = solve_instance(64, 0.55, 602)
    return [result1, result2]
