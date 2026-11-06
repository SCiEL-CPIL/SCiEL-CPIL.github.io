import random
import time
import sys

def construct():
    # Instance generator from verifier
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(edges, n, max_restarts=500, time_limit=45):
        start_time = time.time()
        best_paths = None
        best_count = float('inf')
        
        # Pre-compute initial adjacency
        # We will work with a set of edges for fast removal
        
        for r in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
                
            current_edges = set(edges)
            if not current_edges:
                return []
                
            # Build adjacency list
            adj = {i: set() for i in range(n)}
            for u, v in current_edges:
                adj[u].add(v)
                adj[v].add(u)
                
            paths = []
            
            # Greedy path finding
            while current_edges:
                # Find start vertex: max degree
                start = max(adj, key=lambda x: len(adj[x]))
                
                path = [start]
                visited_in_path = {start}
                
                while True:
                    last = path[-1]
                    # Get neighbors in current_edges not in path
                    candidates = []
                    for v in adj[last]:
                        if v not in visited_in_path:
                            # Check if edge (last, v) is in current_edges
                            e1 = (last, v) if last < v else (v, last)
                            if e1 in current_edges:
                                candidates.append(v)
                    
                    if not candidates:
                        break
                    
                    # Heuristic: pick candidate with max degree in current graph
                    # To avoid getting stuck, maybe prefer higher degree
                    best_v = max(candidates, key=lambda x: len(adj[x]))
                    
                    # Add to path
                    path.append(best_v)
                    visited_in_path.add(best_v)
                    
                    # Remove edge
                    e = (last, best_v) if last < best_v else (best_v, last)
                    current_edges.remove(e)
                    adj[last].remove(best_v)
                    adj[best_v].remove(last)
                    
                paths.append(path)
            
            # Verify solution
            # 1. Simple paths (guaranteed by construction)
            # 2. Edge disjoint and cover all edges
            # Check coverage
            covered_edges = set()
            valid = True
            for p in paths:
                for i in range(len(p) - 1):
                    u, v = p[i], p[i+1]
                    e = (u, v) if u < v else (v, u)
                    if e in covered_edges:
                        valid = False
                        break
                    covered_edges.add(e)
                if not valid:
                    break
            
            if valid and covered_edges == set(edges):
                count = len(paths)
                if count < best_count:
                    best_count = count
                    best_paths = [list(p) for p in paths]
                    
            # Optional: Try to improve with local search? 
            # For now, greedy with restarts should be decent.
            
        if best_paths is None:
            # Fallback: each edge as a path
            return [[u, v] for u, v in edges]
            
        return best_paths

    # Main logic
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    # Allocate time budget
    total_time_limit = 110 # 10s margin
    per_graph_limit = total_time_limit / 3
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        paths = solve_graph(edges, n, max_restarts=1000, time_limit=per_graph_limit)
        results.append(paths)
        
    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(p) for p in res)
    print(total)
