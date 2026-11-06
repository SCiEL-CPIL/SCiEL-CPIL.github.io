import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        """
        Decompose graph into edge-disjoint paths covering all edges.
        Strategy:
        1. Build adjacency with edge IDs.
        2. Find all odd-degree vertices.
        3. If no odd-degree vertices, the graph is Eulerian in each component.
           We can form cycles (trails that start and end at same vertex), 
           but since we need simple paths (vertex lists with no repeated vertices),
           we cannot use cycles directly as paths unless we break them.
           
           However, the problem says "simple" paths. In graph theory, a simple path
           has no repeated vertices. An Eulerian cycle is not a simple path unless
           the graph is a single cycle.
           
           So we need a different approach:
           - Use a greedy algorithm to find long trails (walks with no repeated edges)
           - But the requirement is "simple" paths (no repeated vertices).
           
           Wait, let me re-read: "Paths must be simple, use graph edges, be edge-disjoint and cover all edges."
           Simple path = no repeated vertices.
           
           This is the path decomposition problem where each path is a simple path.
           Gallai's conjecture is about decomposing connected graphs into paths.
           The standard interpretation allows the paths to be simple (no repeated vertices).
           
           For sparse ER graphs, most vertices have degree 2 or 3.
           Degree 2 vertices can be "smoothed" out.
           
           Better approach:
           1. Identify all edges.
           2. Use a greedy algorithm that repeatedly picks a vertex of degree >= 2,
              walks along unused edges as far as possible (always choosing an unused
              edge that leads to a vertex not yet in the current path to maintain simplicity),
              and records the path.
           3. Repeat until all edges are covered.
           
           But ensuring simplicity (no repeated vertices) while maximizing path length is tricky.
           
           Alternative: Since the graphs are sparse and small (n=20,25,30), we can use
           a more sophisticated approach:
           
           1. Build the graph.
           2. Use a heuristic: repeatedly find a longest simple path in the remaining graph
              (using DFS with pruning), remove its edges, repeat.
           3. This is expensive but n is small.
           
           Actually, for n=30, the longest simple path search could be expensive.
           Let's use a simpler greedy:
           - Start from a vertex with highest degree in remaining graph.
           - Do a DFS to find a long simple path (not necessarily longest, but greedy).
           - Remove those edges.
           
           Even simpler and effective for sparse graphs:
           - While edges remain:
             - Pick an edge (u,v) that is in the remaining graph.
             - Try to extend from u in one direction and v in the other direction
               as far as possible using unused edges, maintaining simplicity.
             - This forms a path.
             - Remove the path's edges.
           
           Let's implement a greedy trail extension that ensures simplicity.
        """
        # Build adjacency list with edge tracking
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        while remaining_edges:
            # Pick a vertex with the highest degree in the remaining graph
            deg = {v: len(adj[v] & {u for u in adj[v] if tuple(sorted((v, u))) in remaining_edges or tuple(sorted((u, v))) in remaining_edges}) for v in adj}
            # Simpler: just count remaining edges per vertex
            rem_deg = defaultdict(int)
            for u, v in remaining_edges:
                rem_deg[u] += 1
                rem_deg[v] += 1
            
            if not rem_deg:
                break
                
            # Start from vertex with max remaining degree
            start = max(rem_deg.keys(), key=lambda x: rem_deg[x])
            
            # Greedy walk: always move to a neighbor with unused edges, prefer higher degree
            path = [start]
            used_in_path = {start}
            current = start
            path_edges = []
            
            while True:
                # Find neighbors of current that have an unused edge to current
                candidates = []
                for nb in adj[current]:
                    edge = tuple(sorted((current, nb)))
                    if edge in remaining_edges and nb not in used_in_path:
                        candidates.append((rem_deg[nb], nb))
                
                if not candidates:
                    break
                
                # Pick neighbor with highest remaining degree
                candidates.sort(reverse=True)
                _, next_v = candidates[0]
                path.append(next_v)
                used_in_path.add(next_v)
                edge = tuple(sorted((current, next_v)))
                path_edges.append(edge)
                current = next_v
            
            # The path is complete. Remove its edges from remaining.
            for e in path_edges:
                remaining_edges.discard(e)
            
            # Update rem_deg
            for u, v in path_edges:
                rem_deg[u] -= 1
                rem_deg[v] -= 1
            
            paths.append(path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback: return what we have, or use trivial decomposition
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
