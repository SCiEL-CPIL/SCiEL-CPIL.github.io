"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        """Find a clique partition for one split graph instance."""
        # Build adjacency
        rng = Random(seed)
        # Clique part: 0..k-1, all connected
        # Independent set: k..n-1, no edges among them
        # Cross edges: for each v in [k, n), for each u in [0, k), edge with prob p
        
        # Build neighbor sets for independent set vertices
        cross_neighbors = {}  # v -> set of neighbors in clique part
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if rng.random() < p:
                    nbrs.add(u)
            cross_neighbors[v] = nbrs
        
        # Track which edges are covered
        # Edge types: (u,v) with u<v
        # Clique-part edges: both in [0,k)
        # Cross edges: one in [0,k), one in [k,n)
        
        # We'll use a greedy approach:
        # 1. For each independent-set vertex, form cliques covering its cross edges
        # 2. Then partition remaining clique-part edges
        
        # Track which clique-part edges are still uncovered
        # Use a set of uncovered clique-part edges
        uncovered_clique_edges = set()
        for u in range(k):
            for v in range(u+1, k):
                uncovered_clique_edges.add((u, v))
        
        cliques = []
        
        # For each independent-set vertex, greedily form cliques
        # Sort by degree (number of cross neighbors) descending for better grouping
        indep_vertices = sorted(range(k, n), key=lambda v: len(cross_neighbors[v]), reverse=True)
        
        for v in indep_vertices:
            nbrs = set(cross_neighbors[v])
            # We need to cover edges (v, u) for u in nbrs
            # Each clique containing v can include a subset S of nbrs
            # The clique {v} ∪ S covers cross edges (v,u) for u in S, and clique-part edges within S
            # We want to maximize S size, but only if all clique-part edges within S are still uncovered
            
            # Greedy: repeatedly pick the largest subset S of remaining nbrs such that
            # all edges within S are in uncovered_clique_edges
            remaining = set(nbrs)
            
            while remaining:
                # Try to find a large clique in the remaining set where all internal edges are uncovered
                # Since the clique part is complete, any subset is a clique in the graph sense
                # But we need all internal edges to be uncovered
                # Start with the full remaining set, try to build a maximal valid subset
                
                # Greedy: sort remaining by some heuristic, add vertices one by one
                # A vertex u can be added if all edges from u to already-chosen vertices are uncovered
                chosen = []
                for u in sorted(remaining):
                    # Check if all edges from u to chosen vertices are uncovered
                    can_add = True
                    for w in chosen:
                        edge = (min(u,w), max(u,w))
                        if edge not in uncovered_clique_edges:
                            can_add = False
                            break
                    if can_add:
                        chosen.append(u)
                
                if not chosen:
                    # Shouldn't happen since single vertex is always valid
                    chosen = [sorted(remaining)[0]]
                
                # Form clique {v} ∪ chosen
                clique = sorted([v] + chosen)
                cliques.append(clique)
                
                # Remove covered edges from uncovered_clique_edges
                for i in range(len(chosen)):
                    for j in range(i+1, len(chosen)):
                        edge = (min(chosen[i], chosen[j]), max(chosen[i], chosen[j]))
                        uncovered_clique_edges.discard(edge)
                
                # Remove chosen from remaining
                for u in chosen:
                    remaining.discard(u)
            
            # Check deadline
            if time.time() > DEADLINE:
                break
        
        # Now partition remaining uncovered clique-part edges into cliques
        # Each clique is a subset of [0, k) where all pairwise edges are in uncovered_clique_edges
        # Greedy: repeatedly find a maximal clique in the remaining edge graph
        
        while uncovered_clique_edges:
            # Build adjacency of the remaining edge graph on vertices [0, k)
            # Two vertices are adjacent if their edge is in uncovered_clique_edges
            # Find a maximal clique greedily
            
            # Get all vertices that appear in uncovered edges
            vertices_in_edges = set()
            for (u, v) in uncovered_clique_edges:
                vertices_in_edges.add(u)
                vertices_in_edges.add(v)
            
            if not vertices_in_edges:
                break
            
            # Greedy maximal clique
            # Sort vertices by degree in remaining graph
            deg = {v: 0 for v in vertices_in_edges}
            adj = {v: set() for v in vertices_in_edges}
            for (u, v) in uncovered_clique_edges:
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
            
            # Greedy: pick highest degree vertex, then add neighbors that form clique
            sorted_verts = sorted(vertices_in_edges, key=lambda x: deg[x], reverse=True)
            
            clique = []
            for v in sorted_verts:
                if v not in vertices_in_edges:
                    continue
                # Check if v can be added: all edges from v to clique members must be in uncovered_clique_edges
                can_add = True
                for w in clique:
                    edge = (min(v,w), max(v,w))
                    if edge not in uncovered_clique_edges:
                        can_add = False
                        break
                if can_add:
                    clique.append(v)
            
            if len(clique) < 2:
                # Take any edge
                edge = next(iter(uncovered_clique_edges))
                clique = [edge[0], edge[1]]
            
            cliques.append(sorted(clique))
            
            # Remove covered edges
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    edge = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                    uncovered_clique_edges.discard(edge)
            
            # Check deadline
            if time.time() > DEADLINE:
                break
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE - 5:
            # Not enough time, use fallback (individual edges)
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            cliques = solve_one(*spec)
            results.append(cliques)
    
    # Convert to proper types
    final_result = []
    for cliques in results:
        converted = []
        for clique in cliques:
            converted.append([int(v) for v in clique])
        final_result.append(converted)
    
    return final_result
# EVOLVE-BLOCK-END
