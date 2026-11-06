import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 138  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_max_clique(vertices, adj, time_limit):
        """Bron-Kerbosch with pivot to find maximum clique."""
        best_clique = []
        
        def bk(R, P, X):
            nonlocal best_clique
            if time.time() > time_limit:
                return
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            if not P:
                return
            
            # Choose pivot u from P ∪ X to maximize |N(u) ∩ P|
            pivot = None
            max_count = -1
            for u in P:
                count = len(adj.get(u, set()) & P)
                if count > max_count:
                    max_count = count
                    pivot = u
            for u in X:
                count = len(adj.get(u, set()) & P)
                if count > max_count:
                    max_count = count
                    pivot = u
            
            if pivot is None:
                return
            
            # Recurse on P \ N(pivot)
            candidates = P - adj.get(pivot, set())
            for v in list(candidates):
                if time.time() > time_limit:
                    return
                bk(R | {v}, P & adj.get(v, set()), X & adj.get(v, set()))
                P.discard(v)
                X.add(v)
        
        bk(set(), set(vertices), set())
        return best_clique

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        # Build full adjacency
        full_adj = {u: set() for u in range(n)}
        for u, v in edges:
            full_adj[u].add(v)
            full_adj[v].add(u)
        
        remaining_edges = set(edges)
        partition = []
        instance_deadline = start_time + time_budget
        
        while remaining_edges and time.time() < instance_deadline:
            # Build subgraph adjacency from remaining edges
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
            
            if not active_vertices:
                break
            
            sub_adj = {u: set() for u in active_vertices}
            for u, v in remaining_edges:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            # Find maximum clique in the subgraph
            clique = find_max_clique(active_vertices, sub_adj, instance_deadline)
            
            if not clique or len(clique) < 2:
                # Fallback: take any edge
                u, v = next(iter(remaining_edges))
                clique = [u, v]
            
            clique_set = set(clique)
            partition.append(sorted(clique_set))
            
            # Remove all edges within this clique
            to_remove = set()
            clique_list = sorted(clique_set)
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    if (a, b) in remaining_edges:
                        to_remove.add((a, b))
            remaining_edges -= to_remove
        
        # Fallback for any remaining edges
        if remaining_edges:
            for u, v in remaining_edges:
                partition.append([u, v])
        
        # Local refinement: try to merge pairs of cliques if their union is a clique
        merged = True
        while merged and time.time() < instance_deadline:
            merged = False
            new_partition = []
            i = 0
            while i < len(partition):
                if i + 1 < len(partition):
                    c1 = set(partition[i])
                    c2 = set(partition[i + 1])
                    union = c1 | c2
                    # Check if union is a clique in the original graph
                    is_clique = True
                    ulist = sorted(union)
                    for a in ulist:
                        for b in ulist:
                            if a < b and (a, b) not in edges:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique and len(union) >= 2:
                        new_partition.append(ulist)
                        i += 2
                        merged = True
                        continue
                new_partition.append(partition[i])
                i += 1
            partition = new_partition
        
        return partition

    result1 = solve_instance(32, 0.6, 601, 50)
    result2 = solve_instance(64, 0.55, 602, 85)
    
    return [result1, result2]
