import time
import random
from collections import defaultdict, deque
from random import Random

_DEADLINE = 148.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def construct():
    def instance_graph(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def eulerian_circuit(adj, start):
        """Find Eulerian circuit using Hierholzer's algorithm.
        adj: dict of vertex -> list of neighbors (mutable, edges consumed)
        Returns list of vertices in the circuit.
        """
        # Use stack-based Hierholzer
        circuit = []
        stack = [start]
        visited_edges = set()
        
        while stack:
            v = stack[-1]
            # Find an unused edge from v
            found = False
            while adj[v]:
                u = adj[v].pop()
                if (min(v,u), max(v,u)) not in visited_edges:
                    visited_edges.add((min(v,u), max(v,u)))
                    stack.append(u)
                    found = True
                    break
            if not found:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def decompose_graph(edges, n, max_time=10.0):
        if not edges:
            return []
        
        start_time = time.time()
        
        # Build adjacency with edge tracking
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Compute degrees
        deg = {i: len(adj[i]) for i in range(n)}
        odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
        
        # If no odd vertices, we can have a single Eulerian circuit (1 path if it's a path, or 1 circuit)
        # Actually, if all even, we need 1 path (or 0 if no edges)
        # If odd vertices exist, we need at least len(odd_vertices)/2 paths
        
        # Strategy: 
        # 1. Pair odd vertices
        # 2. Add virtual edges between pairs
        # 3. Find Eulerian circuit
        # 4. Split at virtual edges to get paths
        # 5. Fix non-simple paths
        
        best_paths = None
        best_count = float('inf')
        
        # Try multiple pairings if there are odd vertices
        if odd_vertices:
            num_pairs = len(odd_vertices) // 2
            
            # Generate several random pairings
            num_trials = 50
            trial = 0
            while trial < num_trials and time.time() - start_time < max_time and _remaining() > 2.0:
                # Randomly shuffle odd vertices and pair them
                shuffled = odd_vertices[:]
                random.shuffle(shuffled)
                pairs = [(shuffled[2*i], shuffled[2*i+1]) for i in range(num_pairs)]
                
                # Build augmented adjacency
                aug_adj = {i: list(adj[i]) for i in range(n)}
                virtual_edges = set()
                for u, v in pairs:
                    aug_adj[u].append(v)
                    aug_adj[v].append(u)
                    virtual_edges.add((min(u,v), max(u,v)))
                
                # Find Eulerian circuit
                # Start from a vertex with degree > 0
                start_v = None
                for v in range(n):
                    if aug_adj[v]:
                        start_v = v
                        break
                if start_v is None:
                    continue
                
                circuit = eulerian_circuit(aug_adj, start_v)
                
                if not circuit or len(circuit) < 2:
                    continue
                
                # Split circuit at virtual edges
                # A virtual edge is between consecutive vertices in the circuit
                # We want to split such that each segment between virtual edges is a path
                paths = []
                current_path = [circuit[0]]
                
                for i in range(len(circuit) - 1):
                    u, v = circuit[i], circuit[i+1]
                    current_path.append(v)
                    if (min(u,v), max(u,v)) in virtual_edges:
                        if len(current_path) >= 2:
                            paths.append(current_path[:])
                        current_path = [v]
                
                # The last segment
                if len(current_path) >= 2:
                    paths.append(current_path[:])
                
                # Now fix non-simple paths
                fixed_paths = []
                for path in paths:
                    # Check if path is simple (no repeated vertices)
                    if len(set(path)) == len(path):
                        fixed_paths.append(path)
                    else:
                        # Split at first repeated vertex
                        seen = set()
                        sub_path = []
                        for v in path:
                            if v in seen:
                                if len(sub_path) >= 2:
                                    fixed_paths.append(sub_path[:])
                                sub_path = [v]
                            else:
                                sub_path.append(v)
                                seen.add(v)
                        if len(sub_path) >= 2:
                            fixed_paths.append(sub_path[:])
                
                # Verify all edges are covered
                covered = set()
                for path in fixed_paths:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        covered.add((min(u,v), max(u,v)))
                
                # Check if we covered all original edges
                all_edges = set((min(u,v), max(u,v)) for u, v in edges)
                if covered == all_edges:
                    # Try to merge adjacent paths
                    merged = fixed_paths
                    # Check if any two paths can be merged (one ends where another begins, and the connecting edge exists)
                    changed = True
                    while changed:
                        changed = False
                        new_merged = []
                        i = 0
                        while i < len(merged):
                            if i + 1 < len(merged):
                                p1 = merged[i]
                                p2 = merged[i+1]
                                # Check if p1 ends at p2's start or p2 ends at p1's start
                                # and the edge between them exists in original graph
                                if p1[-1] == p2[0]:
                                    # Check if edge exists
                                    if (min(p1[-1], p2[0]), max(p1[-1], p2[0])) in all_edges:
                                        # But wait, this edge is already used in p1 or p2?
                                        # Actually if p1 ends at v and p2 starts at v, the edge (p1[-2], p1[-1]) is in p1, and (p2[0], p2[1]) is in p2.
                                        # We can't merge unless there's an unused edge.
                                        # Actually, merging means concatenating: p1 + p2[1:]
                                        # This is valid if p1[-1] == p2[0] and the resulting path is simple and uses only existing edges.
                                        # But the edge (p1[-1], p2[1]) must exist and not be already used.
                                        # This is complex. Let's skip merging for now.
                                        pass
                                # Similarly for other direction
                            new_merged.append(merged[i])
                            i += 1
                        merged = new_merged
                    
                    if len(merged) < best_count:
                        best_count = len(merged)
                        best_paths = [p[:] for p in merged]
                
                trial += 1
        
        # If no odd vertices or the above failed, try greedy
        if best_paths is None:
            # Greedy fallback
            remaining = set((min(u,v), max(u,v)) for u, v in edges)
            paths = []
            
            while remaining and time.time() - start_time < max_time and _remaining() > 1.0:
                # Pick a random edge to start
                edge = random.choice(list(remaining))
                u, v = edge
                remaining.discard(edge)
                
                path = [u, v]
                current = v
                
                while True:
                    # Find available neighbors
                    available = []
                    for nb in adj[current]:
                        e = (min(current, nb), max(current, nb))
                        if e in remaining:
                            # Check if nb is not already in path
                            if nb not in path:
                                available.append(nb)
                    
                    if not available:
                        break
                    
                    # Choose neighbor with minimum remaining degree
                    min_deg = float('inf')
                    best_nb = None
                    for nb in available:
                        rem_deg = 0
                        for x in adj[nb]:
                            e = (min(nb, x), max(nb, x))
                            if e in remaining:
                                rem_deg += 1
                        if rem_deg < min_deg:
                            min_deg = rem_deg
                            best_nb = nb
                    
                    u, v = current, best_nb
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                    path.append(v)
                    current = v
                
                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
        
        if best_paths is None:
            # Last resort: each edge as a path
            return [[u, v] for u, v in edges]
        
        return best_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = instance_graph(n, p, seed)
        max_t = 12.0
        if _remaining() < max_t:
            max_t = max(1.0, _remaining() - 1.0)
            
        paths = decompose_graph(edges, n, max_time=max_t)
        results.append(paths)
        
    return results
