import time
import random
from collections import defaultdict, deque
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def instance_graph(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_greedy(edges, n, max_restarts=20):
        """
        Greedy vertex-start extension with min-degree heuristic.
        """
        if not edges:
            return []
        
        best_paths = None
        best_count = float('inf')
        
        # Pre-compute initial degrees for start selection bias
        init_degree = defaultdict(int)
        for u, v in edges:
            init_degree[u] += 1
            init_degree[v] += 1
            
        odd_vertices = [v for v in range(n) if init_degree[v] % 2 == 1]
        
        # We want to try different start strategies
        # 1. Start from odd vertices
        # 2. Start from max degree vertices
        # 3. Random starts
        
        start_candidates = []
        if odd_vertices:
            start_candidates.extend(odd_vertices)
        # Add a few high degree vertices
        sorted_by_deg = sorted(range(n), key=lambda x: init_degree[x], reverse=True)
        start_candidates.extend(sorted_by_deg[:5])
        # Add some random vertices
        rng = Random(42)
        for _ in range(5):
            start_candidates.append(rng.randint(0, n-1))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_starts = []
        for v in start_candidates:
            if v not in seen:
                seen.add(v)
                unique_starts.append(v)
                
        for restart in range(max_restarts):
            # Shuffle the start candidates for variety in restarts
            rng.shuffle(unique_starts)
            
            # For each restart, we will greedily build paths
            # We track used edges
            used_edges = set()
            for u, v in edges:
                used_edges.add((u, v))
                used_edges.add((v, u))
                
            remaining_edges = list(used_edges)
            # We need a mutable structure for remaining edges
            # Let's use a set of tuples (u, v) with u < v for storage, 
            # but for lookup we need both directions.
            # Actually, let's just maintain a list of (u, v) where u<v is the canonical form.
            
            canon_edges = set()
            for u, v in edges:
                canon_edges.add(tuple(sorted((u, v))))
                
            # We will remove edges from canon_edges as we use them.
            # To make removal fast, we work with the set.
            
            paths = []
            
            # We will pick start vertices from the list, but we need to ensure we cover all edges.
            # A better loop: while canon_edges is not empty:
            #   Pick a start vertex.
            #   Extend path.
            
            # To bias towards odd vertices, we can prioritize them if they still have unused edges.
            
            current_edges = set(canon_edges)
            
            while current_edges:
                # Find a start vertex
                # Prefer vertices with odd degree in the *remaining* graph?
                # Or just pick from unique_starts that still have unused edges?
                
                # Let's calculate remaining degrees to pick a good start
                rem_degree = defaultdict(int)
                for u, v in current_edges:
                    rem_degree[u] += 1
                    rem_degree[v] += 1
                    
                # Find a vertex with max remaining degree, tie-break by odd degree
                start_v = None
                max_deg = -1
                for v in range(n):
                    if rem_degree[v] > 0:
                        # Heuristic: prefer odd degree in remaining graph?
                        # Actually, Gallai suggests endpoints should be odd in the *original* graph.
                        # But as we remove edges, degrees change.
                        # Let's just pick max degree to start a long path.
                        if rem_degree[v] > max_deg:
                            max_deg = rem_degree[v]
                            start_v = v
                            
                if start_v is None:
                    break # Should not happen if current_edges is not empty
                    
                # Build path starting from start_v
                path = [start_v]
                visited_in_path = {start_v}
                current_v = start_v
                
                while True:
                    # Find neighbors of current_v that have unused edges
                    neighbors = []
                    for u, v in current_edges:
                        if u == current_v:
                            neighbors.append(v)
                        elif v == current_v:
                            neighbors.append(u)
                            
                    if not neighbors:
                        break
                    
                    # Choose next vertex
                    # Heuristic: Min remaining degree (to keep graph open)
                    # But we must avoid visited_in_path to ensure simplicity
                    valid_neighbors = [nb for nb in neighbors if nb not in visited_in_path]
                    
                    if not valid_neighbors:
                        break
                    
                    # Select neighbor with min remaining degree
                    # If tie, pick random or first
                    min_deg = float('inf')
                    next_v = None
                    for nb in valid_neighbors:
                        # Calculate degree of nb in current_edges
                        deg = 0
                        for u, v in current_edges:
                            if u == nb or v == nb:
                                deg += 1
                        if deg < min_deg:
                            min_deg = deg
                            next_v = nb
                            
                    if next_v is None:
                        break
                        
                    # Move to next_v
                    path.append(next_v)
                    visited_in_path.add(next_v)
                    
                    # Remove edge (current_v, next_v) from current_edges
                    edge_to_remove = tuple(sorted((current_v, next_v)))
                    current_edges.remove(edge_to_remove)
                    
                    current_v = next_v
                    
            paths.append(path)
            
            # Check if all edges are covered
            # The loop `while current_edges` ensures we stop when no edges are left.
            # So paths should cover all edges.
            
            # Verify simplicity and edge disjointness (should be true by construction)
            # But let's double check edge coverage
            covered = set()
            for p in paths:
                for i in range(len(p) - 1):
                    covered.add(tuple(sorted((p[i], p[i+1]))))
            
            if covered == canon_edges:
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = [p[:] for p in paths]
                    
        if best_paths:
            return best_paths
        else:
            # Fallback: each edge as a path
            return [[u, v] for u, v in edges]

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            edges = instance_graph(*spec)
            n = spec[0]
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance_graph(*spec)
        n = spec[0]
        
        # Use the greedy solver
        paths = solve_greedy(edges, n, max_restarts=50)
        results.append(paths)
        
    return results
