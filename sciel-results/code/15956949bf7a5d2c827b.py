"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def edge_set_to_dict(edges):
        d = {}
        for u, v in edges:
            d[(u, v)] = True
            d[(v, u)] = True
        return d

    def is_clique(vertices, adj):
        if len(vertices) < 2:
            return False
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if (vertices[i], vertices[j]) not in adj:
                    return False
        return True

    def greedy_clique_partition(edges, adj, n):
        """Greedy: repeatedly find largest clique among remaining edges."""
        remaining = set()
        for e in edges:
            remaining.add(tuple(sorted(e)))
        
        cliques = []
        
        while remaining:
            # Find the largest clique
            best_clique = None
            best_size = 1
            
            # Try to build cliques from a random starting edge
            edge_list = list(remaining)
            random.shuffle(edge_list)
            
            for edge in edge_list:
                u, v = edge
                # Try to extend {u, v} to a larger clique
                # Get common neighbors of u and v
                common = set()
                for w in range(n):
                    if w != u and w != v:
                        if (u, w) in adj and (v, w) in adj:
                            common.add(w)
                
                # Greedily add vertices to form a clique
                clique = [u, v]
                candidates = list(common)
                random.shuffle(candidates)
                
                for w in candidates:
                    can_add = True
                    for x in clique:
                        if (w, x) not in adj:
                            can_add = False
                            break
                    if can_add:
                        clique.append(w)
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
                    if best_size >= 10:  # good enough
                        break
            
            if best_clique is None or len(best_clique) < 2:
                # Fallback: take a single edge
                edge = edge_list[0]
                best_clique = list(edge)
            
            # Remove all edges in this clique from remaining
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    e = tuple(sorted((best_clique[i], best_clique[j])))
                    remaining.discard(e)
            
            cliques.append(best_clique)
        
        return cliques

    def try_merge_cliques(cliques, adj, n):
        """Try to merge pairs of cliques if their union is a clique."""
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(cliques):
                j = i + 1
                while j < len(cliques):
                    # Check if union of cliques[i] and cliques[j] is a clique
                    union = list(set(cliques[i]) | set(cliques[j]))
                    if is_clique(union, adj):
                        # Merge them
                        cliques[i] = union
                        cliques.pop(j)
                        changed = True
                        break
                    j += 1
                if changed:
                    break
                i += 1
        return cliques

    def solve_instance(n, p, seed):
        edges = generate_graph(n, p, seed)
        adj = edge_set_to_dict(edges)
        
        # Greedy partition
        cliques = greedy_clique_partition(edges, adj, n)
        
        # Try to merge
        cliques = try_merge_cliques(cliques, adj, n)
        
        # Multiple restarts to find better solution
        best_cliques = cliques
        best_count = len(cliques)
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE:
            return best_cliques
        
        # Try different random seeds for shuffling
        for attempt in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            cliques = greedy_clique_partition(edges, adj, n)
            cliques = try_merge_cliques(cliques, adj, n)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
        
        return best_cliques

    result1 = solve_instance(32, 0.6, 601)
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
# EVOLVE-BLOCK-END
