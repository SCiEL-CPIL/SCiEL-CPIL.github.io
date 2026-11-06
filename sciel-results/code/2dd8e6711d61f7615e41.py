"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges, beam_width=50, max_restarts=10):
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        best_paths = None
        best_count = float('inf')
        
        # Restart loop to find a better decomposition
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Use a different random seed for tie-breaking in this restart
            rng = random.Random(restart * 1000 + 42)
            
            current_edges = set(edges)
            paths = []
            
            while current_edges:
                # Beam Search for the next path
                # State: (path_list, used_edges_set)
                # We want to find a long simple path in the remaining graph.
                
                # Initial states: start from every vertex?
                # Too many. Start from vertices with high degree or random.
                # To be efficient, we can just pick a random start vertex.
                
                # Actually, for beam search of a path:
                # We expand a path.
                # Beam width K.
                
                # To speed up, we only start from a few candidates.
                # Or, we run beam search from the vertex with the highest degree in remaining graph.
                
                # Let's try: Start from all vertices? No, n=30.
                # We can run beam search from a specific start vertex.
                # Which start vertex? The one that leads to the longest path.
                # We can run a quick greedy to estimate, or just try all?
                # Trying all starts with beam search might be slow.
                
                # Strategy: 
                # 1. Pick a random edge to start the path?
                # 2. Or pick a vertex.
                
                # Let's pick the vertex with the highest degree in the remaining graph.
                # If tie, random.
                
                degrees = {v: len([e for e in current_edges if v in e]) for v in set().union(*current_edges) if current_edges}
                if not degrees:
                    break
                start_v = max(degrees, key=lambda x: (degrees[x], rng.random()))
                
                # Beam Search
                # State: (path, used_edges)
                # We track used edges to ensure simplicity (no vertex repeat implies no edge repeat, but we need to check vertex repeat).
                
                beam = [([start_v], frozenset())]
                
                for _ in range(50): # Max path length
                    new_beam = []
                    for path, used_e in beam:
                        last_v = path[-1]
                        # Find neighbors
                        neighbors = []
                        for u, v in current_edges:
                            if u == last_v and v not in path:
                                neighbors.append(v)
                            elif v == last_v and u not in path:
                                neighbors.append(u)
                        
                        if not neighbors:
                            # Dead end, keep this path as a candidate
                            new_beam.append((path, used_e))
                        else:
                            for nb in neighbors:
                                new_path = path + [nb]
                                new_used_e = used_e | frozenset([tuple(sorted((last_v, nb)))])
                                new_beam.append((new_path, new_used_e))
                    
                    if not new_beam:
                        break
                        
                    # Sort by length (longer is better)
                    new_beam.sort(key=lambda x: len(x[0]), reverse=True)
                    
                    # Keep top K
                    beam = new_beam[:beam_width]
                    
                    # If all paths in beam are dead ends (same length as previous?), we can stop?
                    # No, we stop when we can't extend any.
                    # Check if any path extended
                    prev_lengths = set(len(p[0]) for p in beam)
                    curr_lengths = set(len(p[0]) for p in new_beam)
                    if not (curr_lengths - prev_lengths):
                        # No extension happened
                        break

                # Pick the longest path from the beam
                # There might be duplicates.
                best_path = max(beam, key=lambda x: len(x[0]))[0]
                
                if len(best_path) < 2:
                    # Isolated vertex? Should not happen if edges exist.
                    break
                
                paths.append(best_path)
                
                # Remove edges from current_edges
                path_edges = set()
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    path_edges.add(tuple(sorted((u, v))))
                
                for e in path_edges:
                    current_edges.discard(e)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                
            # Early stop if we hit the theoretical lower bound?
            # Lower bound: max(1, sum(max(0, deg-2)) / 2 + odd/2) ... 
            # Just keep searching for a bit.
            
        if best_paths is None:
            # Fallback: greedy
            return greedy_solve(edges)
            
        return best_paths

    def greedy_solve(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        current_edges = set(edges)
        paths = []
        
        while current_edges:
            # Pick vertex with highest degree
            degrees = {v: len(adj[v] & set().union(*current_edges)) for v in set().union(*current_edges) if current_edges}
            if not degrees:
                break
            start_v = max(degrees, key=lambda x: degrees[x])
            
            path = [start_v]
            visited = {start_v}
            last_v = start_v
            
            while True:
                # Find unvisited neighbor
                next_v = None
                for nb in adj[last_v]:
                    if nb not in visited:
                        # Check if edge exists in current_edges
                        e = tuple(sorted((last_v, nb)))
                        if e in current_edges:
                            next_v = nb
                            break
                if next_v is None:
                    break
                path.append(next_v)
                visited.add(next_v)
                current_edges.discard(tuple(sorted((last_v, next_v))))
                last_v = next_v
            
            if len(path) > 1:
                paths.append(path)
            else:
                # Stuck, pick another start
                # This simple greedy might fail to cover all edges if it gets stuck early
                # Fallback to edge-by-edge for remaining
                for e in list(current_edges):
                    paths.append(list(e))
                current_edges = set()
                break
        
        return paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            # Return whatever we have? Or fail?
            # Must return valid result.
            # If we run out of time, we should have saved the best so far.
            # But we process sequentially.
            # If we don't finish, we can't return valid paths for the last graph.
            # So we must ensure we finish.
            # The beam search is fast for n=30.
            pass
        
        g_edges = instance(*spec)
        paths = solve_graph(g_edges)
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-END
