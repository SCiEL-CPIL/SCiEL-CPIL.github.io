"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque

def construct():
    """
    Solve the minimum edge-disjoint path cover problem for three fixed graphs.
    Uses a flow-based approach: model as min-cost flow on a directed graph.
    """
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def min_path_cover(n, edges):
        """
        Find minimum number of edge-disjoint paths covering all edges.
        
        Approach: 
        - The minimum number of paths needed to cover all edges in a connected graph 
          with 2k odd-degree vertices is at least k (each path has 2 endpoints, which 
          must be odd-degree vertices, unless the path is a cycle).
        - Actually, by Gallai's conjecture, for a connected graph with maximum degree 
          <= 3, the minimum number of paths is ceil(d/2) where d is the number of 
          odd-degree vertices. But we can do better with a flow formulation.
        
        Better approach using flow:
        - Create a flow network where each undirected edge becomes two directed arcs.
        - We want to decompose the edge set into paths. This is equivalent to finding
          a set of paths that cover all edges exactly once.
        
        Actually, let's use a simpler combinatorial approach:
        - In any graph, the minimum number of edge-disjoint paths covering all edges
          equals max(1, number_of_odd_vertices / 2) if the graph is connected? No, that's
          not quite right either.
        
        Let me think more carefully. The key insight is:
        - A path decomposition of a graph G into k paths means we can pair up some of
          the odd-degree vertices as endpoints of paths, and the rest are internal.
        - The minimum number of paths is at least ceil(odd_count / 2).
        - For many graphs, this lower bound is achievable.
        
        Strategy:
        1. Count odd-degree vertices. Let m = number of odd-degree vertices.
        2. Lower bound: max(1, m // 2)
        3. Try to achieve this lower bound by finding a set of m//2 disjoint paths
           connecting pairs of odd-degree vertices, then the remaining edges form cycles
           or can be absorbed into these paths.
        
        Actually, a cleaner approach:
        - Add a super source S and super sink T.
        - For each vertex v, if deg(v) is odd, we need v to be an endpoint of exactly 
          one path (or we can think of it as having imbalance 1).
        - Model as a flow problem where we want to route flow such that each edge is 
          used exactly once.
        
        Let's use the following formulation:
        - Direct each undirected edge in both directions.
        - For each vertex v:
          - If deg(v) is even: net flow = 0
          - If deg(v) is odd: net flow = ±1 (we need to decide which)
        - Each directed arc has capacity 1 and cost 1.
        - We want to minimize the number of paths, which equals (number of odd vertices)/2
          if we can find a valid flow.
        
        The minimum number of paths is exactly m/2 where m is the number of odd-degree
          vertices, provided the graph is connected (which it is). This is a known result:
          every connected graph with 2k odd-degree vertices can be decomposed into k 
          edge-disjoint paths.
        
        So the answer is simply m // 2 (or 1 if m == 0, i.e., all degrees even, Eulerian).
        
        But we need to actually construct the paths!
        
        Algorithm to construct k = m//2 paths:
        1. Find all odd-degree vertices.
        2. Pair them up arbitrarily (or optimally).
        3. Use a modified Euler tour approach:
           - Add k "dummy" edges pairing up the odd vertices.
           - Now all vertices have even degree, so the graph has an Eulerian circuit.
           - Find the Eulerian circuit.
           - Remove the dummy edges; the circuit breaks into k paths.
        
        This is a classic technique!
        """
        # Build adjacency
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Count degrees
        deg = {v: len(adj[v]) for v in range(n)}
        
        # Find odd-degree vertices
        odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
        m = len(odd_vertices)
        
        if m == 0:
            # Eulerian circuit
            # Find Eulerian circuit using Hierholzer's algorithm
            edge_set = set()
            for u, v in edges:
                edge_set.add((u, v))
                edge_set.add((v, u))
            
            circuit = []
            stack = [0]  # Start from any vertex (graph is connected)
            adj_dict = defaultdict(list)
            for u, v in edges:
                adj_dict[u].append(v)
                adj_dict[v].append(u)
            
            # Use Hierholzer's
            used = set()
            def euler_tour(start):
                stack = [start]
                path = []
                while stack:
                    v = stack[-1]
                    if adj_dict[v]:
                        # Find an unused edge
                        next_v = None
                        for i, w in enumerate(adj_dict[v]):
                            e = tuple(sorted((v, w)))
                            if e not in used:
                                next_v = w
                                break
                        if next_v is not None:
                            e = tuple(sorted((v, next_v)))
                            used.add(e)
                            # Remove from adjacency
                            adj_dict[v].remove(next_v)
                            adj_dict[next_v].remove(v)
                            stack.append(next_v)
                        else:
                            path.append(stack.pop())
                    else:
                        path.append(stack.pop())
                return path
            
            circuit = euler_tour(0)
            # Convert circuit to edge list
            paths = []
            for i in range(len(circuit) - 1):
                paths.append([circuit[i], circuit[i+1]])
            # But this gives individual edges, not the full path
            # Actually, the Eulerian circuit is a single cycle, so it's one path (cycle)
            # Return it as a single path (cycle)
            return [circuit]
        
        else:
            # Pair up odd vertices
            # Simple pairing: just pair consecutive ones
            pairs = []
            for i in range(0, m, 2):
                pairs.append((odd_vertices[i], odd_vertices[i+1]))
            
            # Add dummy edges
            dummy_edges = set(pairs)
            
            # Build augmented graph
            adj_aug = defaultdict(list)
            for u, v in edges:
                adj_aug[u].append(v)
                adj_aug[v].append(u)
            for u, v in pairs:
                adj_aug[u].append(v)
                adj_aug[v].append(u)
            
            # Find Eulerian circuit in augmented graph
            used = set()
            for u, v in edges:
                used.add(tuple(sorted((u, v))))
            for u, v in pairs:
                used.add(tuple(sorted((u, v))))
            
            def euler_tour_aug(start):
                stack = [start]
                path = []
                while stack:
                    v = stack[-1]
                    if adj_aug[v]:
                        next_v = None
                        for w in adj_aug[v]:
                            e = tuple(sorted((v, w)))
                            if e not in used:
                                next_v = w
                                break
                        if next_v is not None:
                            e = tuple(sorted((v, next_v)))
                            used.add(e)
                            adj_aug[v].remove(next_v)
                            adj_aug[next_v].remove(v)
                            stack.append(next_v)
                        else:
                            path.append(stack.pop())
                    else:
                        path.append(stack.pop())
                return path
            
            circuit = euler_tour_aug(0)
            
            # Remove dummy edges to get paths
            # The circuit is a sequence of vertices. When we encounter a dummy edge,
            # we split the circuit at that point.
            paths = []
            current_path = []
            for i in range(len(circuit) - 1):
                u, v = circuit[i], circuit[i+1]
                e = tuple(sorted((u, v)))
                current_path.append(u)
                if e in dummy_edges:
                    current_path.append(v)
                    paths.append(current_path)
                    current_path = []
            if current_path:
                current_path.append(circuit[-1])
                paths.append(current_path)
            
            # Filter out paths that are just dummy edges (shouldn't happen but be safe)
            paths = [p for p in paths if len(p) >= 2]
            
            return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = min_path_cover(n, edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
