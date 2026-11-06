import time
import sys
from collections import defaultdict, deque
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Get all vertices
        vertices = set(adj.keys())
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
            
        # 1. Greedy Trail Cover
        # We want to cover all edges with minimum number of simple paths.
        # A key insight: In a graph, the minimum number of edge-disjoint paths covering all edges
        # is related to the number of odd-degree vertices. 
        # Specifically, if we have k odd-degree vertices, we can pair them up and find paths between pairs.
        # But we need to cover ALL edges.
        # A simpler greedy approach: 
        # While edges remain:
        #   Pick a vertex with max degree (or random).
        #   Perform a greedy walk, always moving to a neighbor with unused edges, 
        #   preferring to avoid dead ends if possible (Warnsdorff-like heuristic).
        #   Record the path.
        #   Remove used edges.
        
        remaining_edges = set(edges)
        paths = []
        
        # To make it deterministic and effective, we'll prioritize starting from odd-degree vertices
        # or high-degree vertices.
        
        def get_degree(v):
            return sum(1 for nb in adj[v] if tuple(sorted((v, nb))) in remaining_edges)
        
        while remaining_edges:
            if time.time() > deadline - 5:
                # Fallback: trivial paths
                for e in remaining_edges:
                    paths.append([e[0], e[1]])
                break
            
            # Find best start vertex
            # Heuristic: Start from a vertex with high degree, or if possible, an odd-degree one
            # that hasn't been "used up" recently? 
            # Actually, just picking a vertex with max remaining degree is a good start.
            start_candidates = [v for v in vertices if get_degree(v) > 0]
            if not start_candidates:
                break
            
            # Sort by degree descending, tie-break by vertex id for determinism
            start_candidates.sort(key=lambda v: (-get_degree(v), v))
            start = start_candidates[0]
            
            # Greedy walk
            path = [start]
            current = start
            used_in_path = {start}
            
            while True:
                # Find next neighbor
                # Heuristic: Choose neighbor that leads to the longest possible continuation?
                # That's expensive. 
                # Simpler: Choose neighbor with max remaining degree (greedy).
                # Or: Choose neighbor that is not a dead end if possible.
                
                next_v = None
                best_score = -1
                for nb in adj[current]:
                    e = tuple(sorted((current, nb)))
                    if e in remaining_edges and nb not in used_in_path:
                        # Score: prefer neighbors with higher degree (more options later)
                        # Also, avoid moving to a leaf if we have other options?
                        deg_nb = get_degree(nb)
                        # If nb is a leaf (deg 1), moving there ends the path.
                        # If we have other options, maybe avoid it?
                        # But if we don't move there now, we might have to start a new path there.
                        # Actually, moving to a leaf is fine if it's the only option or if it closes a nice segment.
                        
                        # Let's use a simple heuristic:
                        # 1. If nb has degree 1, score it low (unless it's the only option).
                        # 2. Otherwise, score by degree.
                        
                        # Check if nb has other unused edges
                        other_edges = 0
                        for nb2 in adj[nb]:
                            e2 = tuple(sorted((nb, nb2)))
                            if e2 in remaining_edges and nb2 != current:
                                other_edges += 1
                                
                        if other_edges == 0:
                            score = 0 # Dead end
                        else:
                            score = other_edges + 10 # Prefer non-dead ends
                        
                        if score > best_score:
                            best_score = score
                            next_v = nb
                
                if next_v is None:
                    break
                
                # Move
                e = tuple(sorted((current, next_v)))
                remaining_edges.discard(e)
                
                path.append(next_v)
                used_in_path.add(next_v)
                current = next_v
            
            if len(path) > 1:
                paths.append(path)
            else:
                # Should not happen if remaining_edges is not empty and start has degree > 0
                # But just in case, handle isolated edge?
                pass

        # 2. Local Search / Optimization
        # The greedy approach might produce suboptimal paths.
        # We can try to merge paths if possible.
        # Two paths can be merged if the end of one is adjacent to the start of the other,
        # and the edge connecting them is not used? No, the edge must be in the graph and not in any path?
        # Wait, the paths are edge-disjoint and cover all edges.
        # So we cannot use an edge that is already in a path.
        # However, we can *re-route* parts of paths to merge them.
        
        # A simpler improvement: 
        # If we have two paths P1=[u1..uk] and P2=[v1..vm], and there is an edge (uk, v1) in the graph,
        # but that edge is NOT in P1 or P2... wait, all edges are covered.
        # So (uk, v1) must be in some path. If it's in P1 or P2, it's already used.
        # So we can't just merge them using an existing edge unless we break the existing paths.
        
        # Alternative: 2-Swap
        # This is complex. Given time constraints, let's stick to a robust greedy
        # with a better heuristic for the greedy walk.
        
        # Let's refine the greedy: 
        # Instead of just max degree, let's try to extend from BOTH ends of the current path?
        # No, the path is built sequentially.
        
        # Let's try a different greedy: 
        # 1. Find all odd degree vertices.
        # 2. Pair them up arbitrarily.
        # 3. For each pair, find a path between them using only unused edges?
        # This is the "Chinese Postman" like idea but for paths.
        # Actually, a known result: The minimum number of paths to cover all edges is 
        # max(1, ceil(odd/2)) if the graph is connected? No, that's for trails.
        # For simple paths, it's harder.
        
        # Given the small size (n=30), we can afford a slightly smarter search.
        # But the current greedy is O(E * (N+D)). 
        # Let's add a simple "path shortening" or "merging" check.
        
        # Actually, the most impactful change is often in the *initial* decomposition.
        # The current greedy picks a start, walks greedily.
        # Let's try to pick the start more wisely: 
        # Start from a leaf? No, leaves are dead ends.
        # Start from a high-degree node is good.
        
        # Let's implement a simple local improvement:
        # Try to swap the order of edges in the path? No.
        
        # Let's just return the greedy result. It's often close to optimal for sparse graphs.
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 5:
            break
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
    
    # If we ran out of time, fill remaining with trivial edge paths
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results
