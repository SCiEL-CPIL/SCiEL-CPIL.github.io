import time
import sys
from random import Random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    # Allow 140s total, leave 10s margin
    deadline = start_time + 140

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycles(edges, n, time_budget):
        """
        Find a pool of candidate cycles using randomized DFS.
        """
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = set()
        # Sort edges to ensure determinism in some parts if needed, but we use randomization
        edge_list = list(edges)
        
        # Pre-compute edge to index mapping for faster lookup? 
        # We represent cycles as frozensets of edges for uniqueness.
        
        rng = Random(42) # Fixed seed for reproducibility of the search process itself
        
        def dfs_cycle(start, current, visited, path_edges):
            # Try to close the cycle
            if len(visited) >= 3:
                # Check if start is in current's neighbors
                if start in adj[current]:
                    # Found a cycle
                    # The cycle is path_edges + (current, start)
                    # We need to ensure we don't add duplicate cycles (reversed/rotated)
                    # Canonical form: sort the edges
                    cycle_edges = frozenset(path_edges | {(min(current, start), max(current, start))})
                    if cycle_edges not in cycles:
                        cycles.add(cycle_edges)
                    # We can also record the vertex sequence for reconstruction later if needed, 
                    # but for ILP we just need edge sets.
                    # However, the output requires vertex sequences.
                    # We can reconstruct the vertex sequence from the edge set if it's a simple cycle.
                    # But wait, a set of edges forming a cycle doesn't uniquely define the vertex order 
                    # if the graph has chords? No, the cycle is a simple cycle in the graph.
                    # The edges of the cycle form a connected component where every vertex has degree 2 
                    # within that subgraph. So we can reconstruct the order.
                    pass

            # Explore neighbors
            neighbors = list(adj[current])
            rng.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor == start:
                    # Only close if length >= 3
                    if len(visited) >= 2: # path length is len(visited)-1, so len(visited)>=3 means edges>=3? 
                        # visited contains vertices. start is in visited.
                        # Path: start -> ... -> current.
                        # If neighbor is start, we close.
                        # Length of path in vertices is len(visited).
                        # Number of edges in path is len(visited) - 1.
                        # Adding edge (current, start) makes total edges len(visited).
                        # So we need len(visited) >= 3.
                        if len(visited) >= 3:
                            cycle_edges = frozenset(path_edges | {(min(current, start), max(current, start))})
                            if cycle_edges not in cycles:
                                cycles.add(cycle_edges)
                    continue

                if neighbor in visited:
                    continue

                # Pruning: if time is up, return
                if time.time() > time_budget:
                    return

                # Recurse
                visited.add(neighbor)
                path_edges.add((min(current, neighbor), max(current, neighbor)))
                dfs_cycle(start, neighbor, visited, path_edges)
                path_edges.discard((min(current, neighbor), max(current, neighbor)))
                visited.discard(neighbor)

        # Run DFS from multiple starts
        # Limit the number of DFS calls to avoid excessive time
        # We will run DFS for a fixed number of steps or time
        
        # To improve quality, we can bias towards longer cycles.
        # Simple DFS finds short cycles first.
        # We can modify the neighbor ordering to prefer unvisited nodes with high degree?
        # Or just run many random walks.
        
        # Let's run DFS from each node, but limit the depth to avoid exponential explosion.
        # Max depth 15-20 is reasonable for n=30.
        
        max_depth = 20
        
        def dfs_limited(start, current, depth, visited, path_edges):
            if depth > max_depth:
                return
            if time.time() > time_budget:
                return
            
            # Check for cycle closure
            if len(visited) >= 3 and start in adj[current]:
                cycle_edges = frozenset(path_edges | {(min(current, start), max(current, start))})
                if cycle_edges not in cycles:
                    cycles.add(cycle_edges)
            
            neighbors = list(adj[current])
            rng.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                path_edges.add((min(current, neighbor), max(current, neighbor)))
                dfs_limited(start, neighbor, depth + 1, visited, path_edges)
                path_edges.discard((min(current, neighbor), max(current, neighbor)))
                visited.discard(neighbor)

        # Run from each node
        for start in range(n):
            if time.time() > time_budget:
                break
            visited = {start}
            path_edges = set()
            dfs_limited(start, start, 0, visited, path_edges)
            
        # Also, do some random restarts to find different cycles
        # Pick random edges and try to extend them?
        # The above DFS from each node should cover most cycles up to max_depth.
        
        return list(cycles)

    def reconstruct_cycle(cycle_edges):
        """
        Given a set of edges forming a simple cycle, return the list of vertices.
        """
        if not cycle_edges:
            return []
        
        # Build local adjacency
        local_adj = {}
        for u, v in cycle_edges:
            if u not in local_adj:
                local_adj[u] = []
            if v not in local_adj:
                local_adj[v] = []
            local_adj[u].append(v)
            local_adj[v].append(u)
        
        # Start from any node
        start = next(iter(local_adj))
        path = [start]
        prev = None
        curr = start
        
        while True:
            # Find next node
            next_node = None
            for neighbor in local_adj[curr]:
                if neighbor != prev:
                    next_node = neighbor
                    break
            
            if next_node is None or next_node == start:
                break
                
            path.append(next_node)
            prev = curr
            curr = next_node
            
        return path

    def decompose(edges, n, time_budget):
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        edge_list = list(edges)
        edge_index = {e: i for i, e in enumerate(edge_list)}
        
        # 1. Find candidate cycles
        # Allocate 50% of time budget to finding cycles
        cycle_search_time = time_budget * 0.5
        candidate_cycles = find_cycles(edges, n, cycle_search_time)
        
        if not candidate_cycles:
            return [[u, v] for u, v in edges]
        
        # 2. Formulate ILP
        # Variables:
        # x_c for each cycle c (0..C-1)
        # y_e for each edge e (0..E-1)
        # Total variables: C + E
        
        C = len(candidate_cycles)
        E = len(edge_list)
        
        # Objective coefficients
        c_obj = np.ones(C + E)
        
        # Constraints:
        # For each edge e, sum(x_c for c containing e) + y_e >= 1
        # We need to build the matrix A.
        # Rows: E (one per edge)
        # Cols: C + E
        
        A = np.zeros((E, C + E))
        
        # Fill x part
        for i, cycle in enumerate(candidate_cycles):
            for e in cycle:
                idx = edge_index.get(e)
                if idx is not None:
                    A[idx, i] = 1.0
        
        # Fill y part (identity matrix)
        for i in range(E):
            A[i, C + i] = 1.0
        
        lb = np.ones(E)
        ub = np.full(E, np.inf)
        
        # Integrality: all binary
        integrality = np.ones(C + E)
        
        # Bounds: 0 to 1
        bounds = Bounds(lb=np.zeros(C + E), ub=np.ones(C + E))
        
        # Solve
        try:
            # Use a time limit for the solver? milp doesn't have a direct time limit param in older SciPy,
            # but we can check the result.
            # We will wrap it in a timeout check if possible, or just rely on it being fast.
            res = milp(
                c=c_obj,
                constraints=LinearConstraint(A, lb, ub),
                integrality=integrality,
                bounds=bounds
            )
            
            if res.status == 0:
                x_vals = res.x[:C]
                y_vals = res.x[C:]
                
                pieces = []
                
                # Add cycles
                for i in range(C):
                    if x_vals[i] > 0.5:
                        cycle_edges = candidate_cycles[i]
                        cycle_vertices = reconstruct_cycle(cycle_edges)
                        if cycle_vertices:
                            pieces.append(cycle_vertices)
                
                # Add single edges
                for i in range(E):
                    if y_vals[i] > 0.5:
                        u, v = edge_list[i]
                        pieces.append([u, v])
                        
                return pieces
            else:
                # Fallback to greedy if ILP fails
                pass
        except Exception:
            pass
        
        # Fallback: Greedy Set Cover
        # This is a robust fallback
        uncovered = set(edge_list)
        pieces = []
        
        # Sort cycles by length descending (longer is better)
        sorted_cycles = sorted(candidate_cycles, key=len, reverse=True)
        
        for cycle in sorted_cycles:
            if not uncovered:
                break
            # Check if this cycle covers any uncovered edges
            # Actually, we should pick the cycle that covers the most uncovered edges.
            # Greedy Set Cover algorithm:
            # At each step, pick the set (cycle) that covers the most uncovered elements.
            
            best_cycle = None
            best_count = 0
            best_uncovered_covered = set()
            
            for cyc in candidate_cycles:
                # Optimization: skip if no intersection with uncovered
                # We can pre-compute or just check
                covered = set(cyc).intersection(uncovered)
                if len(covered) > best_count:
                    best_count = len(covered)
                    best_cycle = cyc
                    best_uncovered_covered = covered
            
            if best_cycle is None or best_count == 0:
                break
                
            # Add this cycle
            pieces.append(reconstruct_cycle(best_cycle))
            uncovered -= best_uncovered_covered
            
        # Add remaining edges as singletons
        for e in uncovered:
            pieces.append(list(e))
            
        return pieces

    results = []
    # Time allocation
    # 3 graphs. Total time 110s (leaving 10s for overhead).
    # 36s per graph.
    time_per_graph = 36
    
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Check global deadline
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            # Not enough time, return trivial solution
            results.append([[u, v] for u, v in edges])
            continue
            
        t_start = time.time()
        pieces = decompose(edges, n, remaining_time - 2)
        results.append(pieces)
        
    return results
