"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    def instance(n, k, p, seed):
        rng = Random(seed)
        # Generate internal edges of the clique C
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        # Generate cross edges between C and I
        cross_edges = [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return edges, cross_edges

    def find_largest_clique(vertices, edge_set):
        """
        Finds a large clique in the subgraph induced by 'vertices' 
        using edges present in 'edge_set'.
        Uses a simple greedy heuristic with local improvement.
        """
        if not vertices:
            return []
        
        # Build adjacency within vertices based on edge_set
        # For speed, precompute neighbors in the remaining graph
        # But edge_set changes, so we check membership.
        
        # Greedy: Start with all vertices, remove those that break clique property?
        # Or build up.
        # Let's try: Pick vertex with max degree in H_rem, add it, restrict to its neighbors, repeat.
        
        # Sort by degree in H_rem
        degs = {}
        for u in vertices:
            deg = 0
            for v in vertices:
                if u != v and (min(u, v), max(u, v)) in edge_set:
                    deg += 1
            degs[u] = deg
        
        # Greedy construction
        clique = []
        candidates = list(vertices)
        
        while candidates:
            # Pick vertex with max degree among candidates
            u = max(candidates, key=lambda x: degs[x])
            clique.append(u)
            
            # Next candidates must be neighbors of u in H_rem
            # And must be in candidates
            new_candidates = []
            for v in candidates:
                if v != u and (min(u, v), max(u, v)) in edge_set:
                    new_candidates.append(v)
            
            # Update degrees for new candidates? 
            # Strictly, degrees should be relative to new_candidates.
            # For simplicity, we just filter.
            candidates = new_candidates
            
        return clique

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    for n, k, p, seed in specs:
        if time.time() > deadline - 5:
            # Fallback to simple solution if running out of time
            # Just return the simple solution for remaining instances
            # But we need to return valid results for all 3.
            # If we are here, we might have incomplete results.
            # Better to just compute the simple solution for all if we start late?
            # No, we process sequentially.
            # If we hit deadline, we must return something valid.
            # The simple solution is always valid.
            # Let's ensure we have a fallback.
            pass

        internal_edges, cross_edges = instance(n, k, p, seed)
        
        # Convert to sets for fast lookup
        edge_set = set(internal_edges)
        
        # Map cross edges by I-vertex
        # I-vertices are k to n-1
        neighbors = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            neighbors[v].add(u)
            
        cliques = []
        
        # Process each I-vertex
        for v in range(k, n):
            if time.time() > deadline - 2:
                # Fallback: add singletons for remaining
                for u in neighbors[v]:
                    cliques.append([u, v])
                # And handle remaining internal edges at the end
                continue
                
            N_v = neighbors[v]
            remaining_N_v = set(N_v)
            
            while remaining_N_v:
                # Find largest clique in H_rem induced by remaining_N_v
                # H_rem edges are in edge_set
                S = find_largest_clique(remaining_N_v, edge_set)
                
                if len(S) >= 2:
                    # Add clique {v} U S
                    cliques.append([v] + list(S))
                    
                    # Remove edges of S from edge_set
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            u, w = S[i], S[j]
                            e = (min(u, w), max(u, w))
                            edge_set.discard(e)
                            
                    # Remove S from remaining_N_v
                    remaining_N_v -= set(S)
                else:
                    # No clique of size >= 2 found (or S is size 1)
                    # This means no edges in edge_set connect any pair in remaining_N_v
                    # So we must use singletons
                    for u in remaining_N_v:
                        cliques.append([u, v])
                    remaining_N_v = set()
        
        # Cover remaining internal edges
        if edge_set:
            # Find all vertices incident to remaining edges
            V_rem = set()
            for u, w in edge_set:
                V_rem.add(u)
                V_rem.add(w)
            if V_rem:
                cliques.append(list(V_rem))
        
        results.append(cliques)

    return results
# EVOLVE-BLOCK-END
