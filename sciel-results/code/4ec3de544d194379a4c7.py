"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict, deque

def construct():
    """
    Decompose each instance graph into edge-disjoint simple paths covering all edges.
    Uses Random Restart Greedy with tie-breaking to minimize path count.
    """
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_into_paths(n, edges, rng_seed=0):
        """
        Decompose a graph into edge-disjoint simple paths.
        Strategy: 
        1. Build adjacency structure with edge IDs.
        2. Greedily extend paths by finding available edges at endpoints.
        3. Start new paths for remaining edges.
        4. Use randomization in tie-breaking to explore different solutions.
        """
        rng = random.Random(rng_seed)
        
        # Map edge to unique ID and build adjacency
        edge_to_id = {}
        adj = defaultdict(list)  # vertex -> list of (neighbor, edge_id)
        for idx, (u, v) in enumerate(edges):
            edge_to_id[(u, v)] = idx
            adj[u].append((v, idx))
            adj[v].append((u, idx))
        
        used_edges = [False] * len(edges)
        paths = []
        
        # Build a set of unused edges for quick lookup
        unused_edges = set(range(len(edges)))
        
        # For each vertex, maintain the count of unused incident edges
        unused_count = [0] * n
        for idx in range(len(edges)):
            u, v = edges[idx]
            unused_count[u] += 1
            unused_count[v] += 1
        
        # For each vertex, maintain a set of available (neighbor, edge_id)
        avail = [set() for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            avail[u].add((v, idx))
            avail[v].add((u, idx))
        
        while unused_edges:
            # Pick an unused edge to start a new path
            # Heuristic: pick an edge where both endpoints have few unused edges
            # (i.e., "bottleneck" edges first) to avoid stranding them
            # We sample to keep it fast, but with randomization in tie-breaking
            
            # To ensure we don't miss critical edges, we check all if small, else sample
            candidates_edges = list(unused_edges)
            if len(candidates_edges) > 50:
                candidates_edges = rng.sample(candidates_edges, 50)
            
            best_edge = None
            best_score = float('inf')
            for eidx in candidates_edges:
                u, v = edges[eidx]
                score = unused_count[u] + unused_count[v]
                if score < best_score:
                    best_score = score
                    best_edge = eidx
                elif score == best_score and rng.random() < 0.5:
                    best_edge = eidx
            
            if best_edge is None:
                best_edge = next(iter(unused_edges))
            
            u, v = edges[best_edge]
            path = [u, v]
            used_edges[best_edge] = True
            unused_edges.discard(best_edge)
            
            # Update avail and unused_count
            avail[u].discard((v, best_edge))
            avail[v].discard((u, best_edge))
            unused_count[u] -= 1
            unused_count[v] -= 1
            
            # Try to extend from both ends
            # Extend from start (u side)
            while avail[u]:
                candidates = list(avail[u])
                # Sort by unused_count of neighbor, descending
                # Add small random noise to break ties and explore
                candidates.sort(key=lambda x: (unused_count[x[0]], rng.random()), reverse=True)
                
                next_v, eidx = candidates[0]
                # Check if next_v is already in path (to ensure simplicity)
                if next_v in path:
                    # Try another candidate
                    found = False
                    for cand in candidates[1:]:
                        nv, ei = cand
                        if nv not in path:
                            next_v, eidx = nv, ei
                            found = True
                            break
                    if not found:
                        break
                else:
                    found = True
                
                if not found:
                    break
                
                path.insert(0, next_v)
                used_edges[eidx] = True
                unused_edges.discard(eidx)
                avail[u].discard((next_v, eidx))
                avail[next_v].discard((u, eidx))
                unused_count[u] -= 1
                unused_count[next_v] -= 1
                u = next_v
            
            # Extend from end (v side)
            while avail[v]:
                candidates = list(avail[v])
                candidates.sort(key=lambda x: (unused_count[x[0]], rng.random()), reverse=True)
                next_u, eidx = candidates[0]
                if next_u in path:
                    found = False
                    for cand in candidates[1:]:
                        nu, ei = cand
                        if nu not in path:
                            next_u, eidx = nu, ei
                            found = True
                            break
                    if not found:
                        break
                else:
                    found = True
                
                if not found:
                    break
                
                path.append(next_u)
                used_edges[eidx] = True
                unused_edges.discard(eidx)
                avail[v].discard((next_u, eidx))
                avail[next_u].discard((v, eidx))
                unused_count[v] -= 1
                unused_count[next_u] -= 1
                v = next_u
            
            paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        best_paths = None
        best_count = float('inf')
        
        # Run multiple random restarts
        for r in range(20):
            paths = decompose_into_paths(n, edges, rng_seed=r)
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
        
        results.append(best_paths)
    
    return results
# EVOLVE-BLOCK-END
