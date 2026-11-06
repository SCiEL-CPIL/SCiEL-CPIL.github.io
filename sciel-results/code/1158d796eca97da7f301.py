"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_lp(edges, time_limit=30.0):
        if not edges:
            return []
        
        m = len(edges)
        # Map vertices to 0..V-1
        verts = sorted(set([v for e in edges for v in e]))
        v_map = {v: i for i, v in enumerate(verts)}
        V = len(verts)
        
        # Build adjacency matrix
        adj = np.zeros((V, V), dtype=int)
        for u, v in edges:
            adj[v_map[u]][v_map[v]] = 1
            adj[v_map[v]][v_map[u]] = 1
            
        # Degrees
        deg = adj.sum(axis=1)
        
        # Initial solution: each edge is a path.
        # Path count = m.
        # We can improve by merging paths.
        # Merging two paths is possible if they share no vertices and there is an edge between their endpoints.
        # Actually, let's use a different LP formulation.
        
        # Let x_{uv} be 1 if edge (u,v) is used as a "link" in the path decomposition.
        # This doesn't directly work.
        
        # Let's use the flow formulation for path cover.
        # Minimize number of paths.
        # This is equivalent to maximizing the number of "links" between paths.
        # Or minimizing the number of start vertices.
        
        # A path cover is a set of vertex-disjoint paths that cover all vertices?
        # NO. The task is EDGE decomposition.
        # "Paths must be simple, use graph edges, be edge-disjoint and cover all edges."
        # Vertices CAN be shared between paths.
        
        # So it's an edge decomposition into paths.
        # This is the "Path Decomposition" problem.
        # For a graph G, the minimum number of paths in an edge decomposition is related to the maximum number of odd-degree vertices?
        # No, that's for Eulerian trails.
        
        # Gallai's Conjecture: Any connected graph on n vertices can be decomposed into at most floor(n/2) paths.
        # This is about VERTEX disjoint paths?
        # "Gallai's conjecture: Every connected graph with n vertices has a path decomposition into at most floor(n/2) paths."
        # Usually, path decomposition implies vertex-disjoint paths covering all vertices?
        # Or edge-disjoint paths covering all edges?
        # The task says: "edge-disjoint and cover all edges".
        # And "Paths must be simple".
        
        # If vertices can be shared, the problem is easier.
        # If vertices must be disjoint, it's harder.
        # The prompt says: "Paths must be simple, use graph edges, be edge-disjoint and cover all edges."
        # It does NOT say vertex-disjoint.
        # However, Gallai's conjecture is typically about vertex-disjoint paths covering all vertices (Path Cover Number).
        # Wait, let's re-read carefully.
        # "Path decompositions of three fixed connected graphs (Gallai's conjecture)"
        # "Paths must be simple, use graph edges, be edge-disjoint and cover all edges."
        
        # If it's edge-disjoint, and vertices can repeat, then a single Eulerian trail (if exists) is 1 path.
        # If not Eulerian, we can add edges to make it Eulerian, then remove them.
        # The number of trails needed is max(1, odd_degree_vertices / 2).
        # But these are TRAILS, not necessarily PATHS (simple).
        # A trail can repeat vertices. A path cannot.
        
        # If the requirement is for PATHS (simple), and they are edge-disjoint covering all edges:
        # This is the "Minimum Path Decomposition" of edges.
        # Since vertices can be shared, we can always decompose into paths?
        # Yes. Any trail can be broken into paths if it has repeated vertices.
        # But breaking a trail into simple paths might increase the count.
        
        # Actually, if vertices can be shared, the minimum number of paths is at most the number of odd-degree vertices / 2.
        # Because we can pair up odd-degree vertices and find trails between them.
        # Each trail can be decomposed into simple paths.
        # But the trails themselves might not be simple.
        
        # Let's assume the standard interpretation for such puzzles:
        # Vertex-disjoint paths covering all vertices?
        # "cover all edges" -> No, it explicitly says cover all edges.
        
        # Let's look at the seed program.
        # It extracts paths from edges.
        # It does not enforce vertex-disjointness.
        # So vertices CAN be shared.
        
        # If vertices can be shared, the problem is:
        # Decompose edges into minimum number of simple paths.
        
        # Strategy:
        # 1. Find an Eulerian decomposition (trails).
        # 2. Break trails into simple paths.
        # 3. Optimize the breaking.
        
        # Or, use a greedy approach that tries to make paths as long as possible.
        # The current program does this with Warnsdorff.
        
        # Let's try a different strategy:
        # Use a matching-based approach?
        # No, let's stick to a robust heuristic that is different from Warnsdorff.
        # How about "Longest Path First" with dynamic edge removal?
        
        # Or, let's try to improve the current Warnsdorff by using a better tie-breaking or start vertex selection.
        # The directive asks for a MATERIALLY DIFFERENT strategy.
        
        # Idea: Use a local search on the set of paths.
        # Start with a greedy solution.
        # Try to merge two paths if possible?
        # Merging two paths P1 and P2 is possible if there is an edge (u,v) such that u is in P1 and v is in P2?
        # No, that would create a cycle or a longer path, but we must use the edge (u,v) which is NOT in the paths yet?
        # No, all edges are covered.
        # So we can only MERGE paths by removing a constraint? No.
        
        # We can REARRANGE edges.
        
        # Let's try a Simulated Annealing approach on the edge assignment to paths.
        # State: assignment of each edge to a path ID.
        # Constraint: Each path must be simple and valid.
        # Objective: Minimize number of path IDs.
        
        # This is hard to implement efficiently.
        
        # Alternative:
        # Use the fact that the graphs are small (n=20, 25, 30).
        # Use a Branch and Bound?
        # Might be too slow.
        
        # Let's go with a different heuristic:
        # "Heaviest Edge First" or "Bottleneck Edge First".
        # Edges that are "bottlenecks" (bridge-like) should be in short paths?
        # Or long paths?
        
        # Let's try a strategy based on **Eulerian Trail Decomposition** followed by **Path Splitting**.
        # 1. Compute degrees.
        # 2. Pair up odd-degree vertices arbitrarily.
        # 3. Find trails between pairs (using Hierholzer's or DFS).
        # 4. Each trail is a sequence of edges.
        # 5. Split each trail into simple paths.
        # 6. Total paths = sum of splits.
        
        # This guarantees a valid decomposition.
        # The number of paths is at most the number of trails + sum of (length of trail - 1) in worst case?
        # No, splitting a trail of length L into simple paths takes at most L paths (each edge a path).
        # But usually much fewer.
        
        # Let's implement this.
        
        # Step 1: Identify odd-degree vertices.
        odd_verts = [v for v in range(V) if deg[v] % 2 == 1]
        
        # Step 2: Pair them up.
        # To minimize paths, we want long trails.
        # So pair vertices that are far apart?
        # Or just random pairing?
        
        # Let's use random pairing with restarts.
        
        def get_trails(pairing):
            # Copy adjacency
            adj_copy = [set() for _ in range(V)]
            for u in range(V):
                for v in range(V):
                    if adj[u][v]:
                        adj_copy[u].add(v)
                        
            trails = []
            
            # Find Eulerian trails for each pair
            for u, v in pairing:
                # Add virtual edge (u,v)
                adj_copy[u].add(v)
                adj_copy[v].add(u)
                
                # Find Eulerian circuit in the modified graph?
                # No, we just want a trail from u to v in the original graph.
                # This is equivalent to an Eulerian trail in the graph where we added edge (u,v).
                # But we only care about the edges in the original graph.
                
                # Actually, Hierholzer's algorithm finds an Eulerian circuit.
                # If we add edge (u,v), the graph is Eulerian (all degrees even).
                # Find circuit, remove edge (u,v), get trail from u to v.
                
                # But we need to remove edges as we go.
                
                # Let's just do a DFS from u to v, removing edges.
                # This is not guaranteed to find the longest trail.
                
                pass

        # This is getting complicated.
        
        # Let's go back to the heuristic.
        # The current program uses Warnsdorff.
        # Let's try **Randomized Greedy with Edge Weighting**.
        # Assign weights to edges.
        # Always pick the edge with the highest weight.
        # Start a path from one end.
        # Extend as far as possible.
        # Remove used edges.
        # Update weights?
        
        # Or, simply: **Multiple Restarts of Random Greedy**.
        # Pick a random edge.
        # Extend in both directions randomly.
        
        # Let's implement a cleaner version of the current approach but with a different heuristic:
        # **Degree-based Greedy**: Always extend to the neighbor with the HIGHEST remaining degree.
        # This is the opposite of Warnsdorff (which picks lowest onward moves).
        # High degree means more options, so maybe it leads to longer paths?
        
        def solve_greedy_high_deg(edges, max_restarts=100, time_limit=10.0):
            if not edges:
                return []
            
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            best_paths = None
            best_count = float('inf')
            start_time = time.time()
            
            for restart in range(max_restarts):
                if time.time() > start_time + time_limit:
                    break
                    
                rng = random.Random(restart * 98765 + 12345)
                current_edges = set(edges)
                paths = []
                
                while current_edges:
                    # Pick a random edge to start
                    e = rng.choice(list(current_edges))
                    u, v = e
                    
                    # Decide direction
                    if rng.random() < 0.5:
                        start_v, end_v = u, v
                    else:
                        start_v, end_v = v, u
                        
                    path = [start_v]
                    visited = {start_v}
                    last_v = start_v
                    current_edges.discard(e)
                    
                    while True:
                        # Find neighbors in current_edges
                        candidates = []
                        for nb in adj[last_v]:
                            e2 = tuple(sorted((last_v, nb)))
                            if e2 in current_edges:
                                # Heuristic: Pick neighbor with HIGHEST remaining degree
                                deg_nb = len([e3 for e3 in current_edges if nb in e3])
                                candidates.append((deg_nb, nb))
                                
                        if not candidates:
                            break
                            
                        # Pick the one with highest degree
                        candidates.sort(key=lambda x: x[0], reverse=True)
                        next_v = candidates[0][1]
                        
                        path.append(next_v)
                        visited.add(next_v)
                        e2 = tuple(sorted((last_v, next_v)))
                        current_edges.discard(e2)
                        last_v = next_v
                        
                    if len(path) > 1:
                        paths.append(path)
                        
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = [p[:] for p in paths]
                    
            return best_paths if best_paths else []

        # Let's also try a **Backtracking** approach for small residual graphs?
        # No, too slow.
        
        # Let's combine Warnsdorff and High-Degree.
        
        return solve_greedy_high_deg(edges, max_restarts=200, time_limit=time_limit)

    def solve_mixed(edges, time_limit=30.0):
        # Run both Warnsdorff and High-Degree and pick the best.
        # But we don't have Warnsdorff here. Let's implement a simple one.
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        def greedy_warnsdorff(max_restarts=50, tl=10.0):
            best_paths = None
            best_count = float('inf')
            start_time = time.time()
            
            for restart in range(max_restarts):
                if time.time() > start_time + tl:
                    break
                rng = random.Random(restart * 111 + 222)
                current_edges = set(edges)
                paths = []
                
                while current_edges:
                    degs = defaultdict(int)
                    for u, v in current_edges:
                        degs[u] += 1
                        degs[v] += 1
                    if not degs: break
                    max_deg = max(degs.values())
                    candidates = [v for v, d in degs.items() if d == max_deg]
                    start_v = rng.choice(candidates)
                    
                    path = [start_v]
                    visited = {start_v}
                    last_v = start_v
                    
                    while True:
                        neighbors = []
                        for nb in adj[last_v]:
                            e = tuple(sorted((last_v, nb)))
                            if e in current_edges:
                                onward = 0
                                for nn in adj[nb]:
                                    if nn not in visited and nn != last_v:
                                        e2 = tuple(sorted((nb, nn)))
                                        if e2 in current_edges:
                                            onward += 1
                                neighbors.append((onward, nb))
                        if not neighbors: break
                        neighbors.sort(key=lambda x: x[0])
                        min_onward = neighbors[0][0]
                        best_nbs = [nb for on, nb in neighbors if on == min_onward]
                        next_v = rng.choice(best_nbs)
                        
                        path.append(next_v)
                        visited.add(next_v)
                        e = tuple(sorted((last_v, next_v)))
                        current_edges.discard(e)
                        last_v = next_v
                        
                    if len(path) > 1:
                        paths.append(path)
                        
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = [p[:] for p in paths]
                    
            return best_paths if best_paths else []

        def greedy_high_deg(max_restarts=50, tl=10.0):
            best_paths = None
            best_count = float('inf')
            start_time = time.time()
            
            for restart in range(max_restarts):
                if time.time() > start_time + tl:
                    break
                rng = random.Random(restart * 333 + 444)
                current_edges = set(edges)
                paths = []
                
                while current_edges:
                    e = rng.choice(list(current_edges))
                    u, v = e
                    if rng.random() < 0.5:
                        start_v, end_v = u, v
                    else:
                        start_v, end_v = v, u
                        
                    path = [start_v]
                    visited = {start_v}
                    last_v = start_v
                    current_edges.discard(e)
                    
                    while True:
                        candidates = []
                        for nb in adj[last_v]:
                            e2 = tuple(sorted((last_v, nb)))
                            if e2 in current_edges:
                                deg_nb = len([e3 for e3 in current_edges if nb in e3])
                                candidates.append((deg_nb, nb))
                        if not candidates: break
                        candidates.sort(key=lambda x: x[0], reverse=True)
                        next_v = candidates[0][1]
                        
                        path.append(next_v)
                        visited.add(next_v)
                        e2 = tuple(sorted((last_v, next_v)))
                        current_edges.discard(e2)
                        last_v = next_v
                        
                    if len(path) > 1:
                        paths.append(path)
                        
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = [p[:] for p in paths]
                    
            return best_paths if best_paths else []

        # Run both
        p1 = greedy_warnsdorff(max_restarts=100, tl=time_limit/2)
        p2 = greedy_high_deg(max_restarts=100, tl=time_limit/2)
        
        if p1 is None: return p2
        if p2 is None: return p1
        
        return p1 if len(p1) <= len(p2) else p2

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for i, spec in enumerate(specs):
        g_edges = instance(*spec)
        time_left = deadline - time.time()
        if time_left < 5:
            # Fallback
            paths = []
            current_edges = set(g_edges)
            while current_edges:
                e = next(iter(current_edges))
                paths.append(list(e))
                current_edges.discard(e)
        else:
            tl = min(30.0, time_left / 3.0)
            paths = solve_mixed(g_edges, time_limit=tl)
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-END
