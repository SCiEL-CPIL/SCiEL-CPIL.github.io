"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # 140s to leave safety margin before 150s kill
    
    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges
    
    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency as sets for fast lookup
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Track which edges are still uncovered
        edge_set = set(edges)
        remaining = set(edges)
        
        cliques = []
        
        # Greedy: repeatedly find a large clique in the remaining edge subgraph
        while remaining and time.time() < deadline - 5:
            # Find a good clique by trying each vertex as a seed
            best_clique = None
            best_size = 0
            
            # Try each vertex as potential seed
            for start in range(n):
                if not any((start, v) in remaining or (v, start) in remaining for v in range(n) if v != start):
                    continue
                
                # Greedily build a clique starting from 'start'
                # Only include edges that are in 'remaining'
                clique = [start]
                candidates = set(adj[start]) & set(v for u, v in remaining if u == start or v == start)
                # Actually, let's be more careful: candidates must be connected to all current clique members via remaining edges
                
                # Better approach: build clique where every pair has an edge in remaining
                clique_set = {start}
                # Get neighbors of start that have edges to start in remaining
                possible = set()
                for u, v in remaining:
                    if u == start:
                        possible.add(v)
                    elif v == start:
                        possible.add(u)
                
                # Greedily add vertices
                changed = True
                while changed:
                    changed = False
                    new_possible = set()
                    for cand in possible:
                        if cand in clique_set:
                            continue
                        # Check if cand is connected to all in clique_set via remaining edges
                        valid = True
                        for member in clique_set:
                            if member == cand:
                                continue
                            if (min(member, cand), max(member, cand)) not in remaining:
                                valid = False
                                break
                        if valid:
                            new_possible.add(cand)
                    if new_possible:
                        # Add the one with highest degree in the subgraph for better clique
                        # Just add all that are valid? No, we need a clique, so we need to be selective
                        # Let's just pick one greedily - the one with most connections to other possible
                        best_cand = None
                        best_score = -1
                        for cand in new_possible:
                            score = 0
                            for other in new_possible:
                                if other != cand and (min(cand, other), max(cand, other)) in remaining:
                                    score += 1
                            if score > best_score:
                                best_score = score
                                best_cand = cand
                        if best_cand is not None:
                            clique_set.add(best_cand)
                            possible.discard(best_cand)
                            changed = True
                
                if len(clique_set) > best_size:
                    best_size = len(clique_set)
                    best_clique = list(clique_set)
            
            if best_clique is None or len(best_clique) < 2:
                # Fallback: take a single edge
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                continue
            
            # Remove all edges within best_clique from remaining
            clique_set = set(best_clique)
            to_remove = []
            for u, v in remaining:
                if u in clique_set and v in clique_set:
                    to_remove.append((u, v))
            
            if not to_remove:
                # Shouldn't happen, but fallback
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                continue
            
            cliques.append(best_clique)
            remaining.difference_update(to_remove)
        
        # Handle any remaining edges as individual edge-cliques
        for e in remaining:
            cliques.append(list(e))
        
        return cliques
    
    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Emergency fallback
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))
    
    return result
# EVOLVE-BLOCK-END
