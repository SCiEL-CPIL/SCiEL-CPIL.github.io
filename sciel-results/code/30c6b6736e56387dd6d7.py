import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def is_clique(vertices, adj):
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if vertices[j] not in adj[vertices[i]]:
                    return False
        return True

    def initial_partition(edges, adj, n):
        """Create initial partition by grouping edges into maximal cliques greedily."""
        remaining = set(edges)
        cliques = []
        
        # Sort edges by degree sum for better initial grouping
        def edge_degree_sum(e):
            u, v = e
            return len(adj[u]) + len(adj[v])
        
        remaining_list = sorted(remaining, key=edge_degree_sum, reverse=True)
        
        for edge in remaining_list:
            if edge not in remaining:
                continue
            u, v = edge
            # Try to extend this edge into a clique
            candidates = sorted(adj[u] & adj[v], key=lambda x: len(adj[x]), reverse=True)
            clique = [u, v]
            for w in candidates:
                if (u, w) in remaining or (w, u) in remaining or (v, w) in remaining or (w, v) in remaining:
                    # Check if w can be added to maintain clique property with all in clique
                    can_add = True
                    for x in clique:
                        if w not in adj[x]:
                            can_add = False
                            break
                    if can_add:
                        clique.append(w)
            
            if len(clique) >= 2:
                cliques.append(clique)
                # Remove all edges within this clique
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        a, b = clique[i], clique[j]
                        if a > b:
                            a, b = b, a
                        remaining.discard((a, b))
            else:
                cliques.append([u, v])
                remaining.discard(edge)
        
        # Add any remaining edges as individual cliques
        for edge in remaining:
            cliques.append(list(edge))
        
        return cliques

    def validate_partition(cliques, edges, adj):
        """Check if partition is valid."""
        edge_count = {}
        for clique in cliques:
            if len(clique) < 2:
                return False
            if not is_clique(clique, adj):
                return False
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    a, b = clique[i], clique[j]
                    if a > b:
                        a, b = b, a
                    edge = (a, b)
                    edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # Every edge must appear exactly once
        if set(edge_count.keys()) != set(edges):
            return False
        for count in edge_count.values():
            if count != 1:
                return False
        return True

    def score_partition(cliques):
        return len(cliques)

    def simulated_annealing(cliques, edges, adj, n, time_budget):
        """Improve partition using simulated annealing."""
        current = cliques[:]
        current_score = score_partition(current)
        best = current[:]
        best_score = current_score
        
        T = 100.0
        T_min = 0.01
        cooling = 0.995
        
        rng = random.Random(42)
        
        def get_clique_index_by_edge(edge):
            for i, clique in enumerate(current):
                for j in range(len(clique)):
                    for k in range(j + 1, len(clique)):
                        a, b = clique[j], clique[k]
                        if (a, b) == edge or (b, a) == edge:
                            return i
            return -1
        
        while time.time() < time_budget and T > T_min:
            move_type = rng.choice(['split', 'merge', 'transfer'])
            
            if move_type == 'split':
                # Split a clique into two
                if len(current) < 2:
                    T *= cooling
                    continue
                idx = rng.randrange(len(current))
                clique = current[idx]
                if len(clique) < 3:
                    T *= cooling
                    continue
                split_point = rng.randrange(1, len(clique))
                part1 = clique[:split_point]
                part2 = clique[split_point:]
                
                if len(part1) < 2 or len(part2) < 2:
                    T *= cooling
                    continue
                if not is_clique(part1, adj) or not is_clique(part2, adj):
                    T *= cooling
                    continue
                
                new_current = current[:]
                new_current[idx] = part1
                new_current.append(part2)
                new_score = score_partition(new_current)
                
                delta = new_score - current_score
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current = new_current
                    current_score = new_score
                    if current_score < best_score:
                        best = current[:]
                        best_score = current_score
            
            elif move_type == 'merge':
                # Merge two cliques if they form a clique
                if len(current) < 2:
                    T *= cooling
                    continue
                i, j = rng.sample(range(len(current)), 2)
                c1, c2 = current[i], current[j]
                merged = list(set(c1) | set(c2))
                if len(merged) < 2:
                    T *= cooling
                    continue
                if not is_clique(merged, adj):
                    T *= cooling
                    continue
                
                # Check that all edges in merged are covered
                merged_edges = set()
                for a in merged:
                    for b in merged:
                        if a < b:
                            merged_edges.add((a, b))
                
                # Check no edge is duplicated
                other_edges = set()
                for k, clique in enumerate(current):
                    if k == i or k == j:
                        continue
                    for a in clique:
                        for b in clique:
                            if a < b:
                                other_edges.add((a, b))
                
                if merged_edges & other_edges:
                    T *= cooling
                    continue
                
                new_current = current[:]
                # Remove i and j, add merged
                del new_current[max(i, j)]
                del new_current[min(i, j)]
                new_current.append(merged)
                new_score = score_partition(new_current)
                
                delta = new_score - current_score
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current = new_current
                    current_score = new_score
                    if current_score < best_score:
                        best = current[:]
                        best_score = current_score
            
            elif move_type == 'transfer':
                # Move a vertex from one clique to another
                if len(current) < 2:
                    T *= cooling
                    continue
                src_idx = rng.randrange(len(current))
                src_clique = current[src_idx]
                if len(src_clique) < 3:
                    T *= cooling
                    continue
                
                vertex = rng.choice(src_clique)
                new_src = [v for v in src_clique if v != vertex]
                if len(new_src) < 2:
                    T *= cooling
                    continue
                if not is_clique(new_src, adj):
                    T *= cooling
                    continue
                
                # Find a target clique to add vertex to
                dst_idx = -1
                for k in range(len(current)):
                    if k == src_idx:
                        continue
                    dst_clique = current[k]
                    new_dst = list(set(dst_clique) | {vertex})
                    if is_clique(new_dst, adj):
                        # Check no edge duplication
                        dst_edges = set()
                        for a in new_dst:
                            for b in new_dst:
                                if a < b:
                                    dst_edges.add((a, b))
                        
                        src_edges = set()
                        for a in new_src:
                            for b in new_src:
                                if a < b:
                                    src_edges.add((a, b))
                        
                        other_edges = set()
                        for m in range(len(current)):
                            if m == src_idx or m == k:
                                continue
                            for a in current[m]:
                                for b in current[m]:
                                    if a < b:
                                        other_edges.add((a, b))
                        
                        if not (dst_edges & src_edges) and not (dst_edges & other_edges):
                            dst_idx = k
                            break
                
                if dst_idx == -1:
                    T *= cooling
                    continue
                
                new_current = current[:]
                new_current[src_idx] = new_src
                dst_clique = new_current[dst_idx]
                new_dst = list(set(dst_clique) | {vertex})
                new_current[dst_idx] = new_dst
                new_score = score_partition(new_current)
                
                delta = new_score - current_score
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current = new_current
                    current_score = new_score
                    if current_score < best_score:
                        best = current[:]
                        best_score = current_score
            
            T *= cooling
        
        return best

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = generate_graph(n, p, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        initial = initial_partition(edges, adj, n)
        
        # Validate initial
        if not validate_partition(initial, edges, adj):
            # Fallback: each edge as its own clique
            initial = [list(e) for e in edges]
        
        time_per_instance = (deadline - time.time()) / 2.0
        improved = simulated_annealing(initial, edges, adj, n, time.time() + time_per_instance)
        
        # Final validation
        if not validate_partition(improved, edges, adj):
            improved = [list(e) for e in edges]
        
        results.append(improved)
    
    return results
