# EVOLVE-BLOCK-START
import time
from random import Random

def _build_graph(n, p, seed):
    rng = Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _greedy_clique_partition_v3(adj, deadline):
    """
    Structural change: Global Candidate Selection.
    Instead of picking the single best vertex to extend, we generate multiple
    candidate cliques (one for each of the top-degree vertices in the remaining
    subgraph) and select the one that covers the most remaining edges.
    """
    n = len(adj)
    remaining = set()
    for u in range(n):
        for v in adj[u]:
            if v > u:
                remaining.add((u, v))
    
    if not remaining:
        return []
    
    cliques = []
    active_vertices = set()
    for u, v in remaining:
        active_vertices.add(u)
        active_vertices.add(v)
        
    # To speed up, we maintain a list of active vertices.
    # We will repeatedly:
    # 1. Identify top K active vertices by degree in the remaining subgraph.
    # 2. For each, generate a maximal clique containing it.
    # 3. Pick the clique that covers the most remaining edges.
    # 4. Remove those edges.
    
    K = 10 # Number of candidates to evaluate
    
    while remaining and time.time() < deadline:
        if not active_vertices:
            break
            
        # Calculate degrees in the remaining subgraph for active vertices
        # We only need degrees for active vertices
        degrees = {v: 0 for v in active_vertices}
        for u, v in remaining:
            if u in active_vertices and v in active_vertices:
                degrees[u] += 1
                degrees[v] += 1
        
        # Get top K vertices
        top_vertices = sorted(active_vertices, key=lambda v: degrees[v], reverse=True)
        if not top_vertices:
            break
        candidates = top_vertices[:K]
        
        best_clique = None
        best_coverage = 0
        best_clique_edges = set()
        
        for start_v in candidates:
            # Generate a maximal clique starting from start_v
            # Greedy strategy: sort neighbors of start_v by their degree in remaining subgraph
            # and add if connected to all current clique members
            
            # Neighbors of start_v in remaining subgraph
            neighbors = [v for v in adj[start_v] if (min(start_v, v), max(start_v, v)) in remaining]
            
            # Sort neighbors by degree (descending)
            neighbors.sort(key=lambda v: degrees.get(v, 0), reverse=True)
            
            current_clique = [start_v]
            # For efficiency, maintain a set of valid candidates
            valid_cands = set(neighbors)
            
            for v in neighbors:
                if v not in valid_cands:
                    continue
                # Check if v is connected to all in current_clique
                is_connected = True
                for u in current_clique:
                    if (min(u, v), max(u, v)) not in remaining:
                        is_connected = False
                        break
                if is_connected:
                    current_clique.append(v)
                    # Update valid_cands: remove v, and remove any neighbor of v that is not in remaining
                    valid_cands.discard(v)
                    for w in valid_cands:
                        if (min(v, w), max(v, w)) not in remaining:
                            valid_cands.discard(w)
            
            # Calculate coverage: number of edges in current_clique that are in remaining
            clique_edges = set()
            for i in range(len(current_clique)):
                for j in range(i + 1, len(current_clique)):
                    u, v = current_clique[i], current_clique[j]
                    e = (min(u, v), max(u, v))
                    if e in remaining:
                        clique_edges.add(e)
            
            coverage = len(clique_edges)
            if coverage > best_coverage:
                best_coverage = coverage
                best_clique = current_clique
                best_clique_edges = clique_edges
                
        if best_clique and best_coverage > 0:
            # Remove edges
            for e in best_clique_edges:
                remaining.discard(e)
            cliques.append(best_clique)
            
            # Update active vertices
            # A vertex is active if it has at least one remaining edge
            # We can just recalculate active_vertices from remaining
            active_vertices = set()
            for u, v in remaining:
                active_vertices.add(u)
                active_vertices.add(v)
        else:
            # Fallback: take a single edge
            if not remaining:
                break
            edge = next(iter(remaining))
            u, v = edge
            cliques.append([u, v])
            remaining.discard(edge)
            active_vertices = set()
            for a, b in remaining:
                active_vertices.add(a)
                active_vertices.add(b)
                
    # Cover any remaining edges as pairs
    for edge in list(remaining):
        if time.time() > deadline:
            break
        u, v = edge
        if (u, v) in remaining:
            cliques.append([u, v])
            remaining.discard((u, v))
            
    return cliques

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin before 150s limit
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback: trivial partition
            adj = _build_graph(n, p, seed)
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        cliques.append([u, v])
            results.append(cliques)
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _greedy_clique_partition_v3(adj, deadline)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
