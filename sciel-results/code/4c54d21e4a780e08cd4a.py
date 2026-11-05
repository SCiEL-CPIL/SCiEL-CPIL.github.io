"""Clique partition of dense random graphs - minimize number of cliques."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    _start = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill
    
    def generate_edges(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges
    
    def greedy_clique_partition(edges, n):
        """Find a clique partition greedily by finding large cliques."""
        if not edges:
            return []
        
        remaining = set(edges)
        cliques = []
        
        # Build adjacency for quick clique checking
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        while remaining:
            # Find a large clique greedily
            # Pick a vertex with high degree in remaining graph
            rem_adj = [set() for _ in range(n)]
            rem_vertices = set()
            for u, v in remaining:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
                rem_vertices.add(u)
                rem_vertices.add(v)
            
            # Greedy: start from vertex with highest degree
            if not rem_vertices:
                break
            
            # Try to build a large clique starting from each high-degree vertex
            best_clique = None
            best_size = 0
            
            # Sort vertices by degree in remaining graph
            sorted_verts = sorted(rem_vertices, key=lambda x: len(rem_adj[x]), reverse=True)
            
            # Try building clique from top candidates
            for start in sorted_verts[:min(20, len(sorted_verts))]:
                clique = [start]
                candidates = set(rem_adj[start])
                
                while candidates:
                    # Pick candidate with highest degree within candidates
                    best_cand = max(candidates, key=lambda x: len(rem_adj[x] & candidates))
                    clique.append(best_cand)
                    candidates = candidates & rem_adj[best_cand]
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if best_clique is None or len(best_clique) < 2:
                # Take any edge as a 2-clique
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                continue
            
            # Add this clique
            cliques.append(best_clique)
            
            # Remove all edges within this clique
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    if u > v:
                        u, v = v, u
                    remaining.discard((u, v))
            
            # Check deadline
            if time.time() - _start > _DEADLINE - 5.0:
                break
        
        # Handle any remaining edges as 2-cliques
        for e in remaining:
            cliques.append(list(e))
        
        return cliques
    
    def refine_partition(cliques, edges, n, max_time=30.0):
        """Try to merge adjacent cliques to reduce count."""
        _refine_start = time.time()
        edge_set = set(edges)
        
        # Check if two cliques can be merged
        def can_merge(c1, c2, adj):
            """Check if union of c1 and c2 forms a clique."""
            union = set(c1) | set(c2)
            for u in union:
                for v in union:
                    if u < v and (u, v) not in adj:
                        return False
            return True
        
        # Build adjacency from edges
        adj = set()
        for u, v in edges:
            if u < v:
                adj.add((u, v))
            else:
                adj.add((v, u))
        
        changed = True
        while changed and (time.time() - _refine_start) < max_time:
            changed = False
            new_cliques = []
            i = 0
            while i < len(cliques):
                merged = False
                for j in range(i + 1, len(cliques)):
                    if can_merge(cliques[i], cliques[j], adj):
                        new_cliques.append(sorted(set(cliques[i]) | set(cliques[j])))
                        cliques.pop(j)
                        merged = True
                        changed = True
                        break
                if not merged:
                    new_cliques.append(cliques[i])
                i += 1
            cliques = new_cliques
        
        return cliques
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = generate_edges(n, p, seed)
        cliques = greedy_clique_partition(edges, n)
        
        # Try to refine if time permits
        if time.time() - _start < _DEADLINE - 10.0:
            cliques = refine_partition(cliques, edges, n, max_time=15.0)
        
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
