"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import heapq
from collections import deque

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = deque([i])
            visited[i] = True
            while queue:
                node = queue.popleft()
                comp.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(comp)
    
    # Find the largest component
    largest_comp = max(components, key=len)
    comp_set = set(largest_comp)
    comp_size = len(largest_comp)
    
    # If the largest component is small, just return it as a path
    if comp_size <= 2:
        return largest_comp
    
    # Build bipartite graph for matching relaxation
    # We want to find a long path. A path in a graph with m vertices has m-1 edges.
    # The internal vertices of a path have degree 2 in the path, endpoints have degree 1.
    # We can model this as: each vertex can be an endpoint (degree 1) or internal (degree 2).
    # This is related to finding a maximum path cover.
    
    # Let's use a different approach: iterative deepening DFS with heuristic ordering
    # But first, let's try to use the matching relaxation idea.
    
    # For a path, we need to select a subset of vertices and arrange them in a sequence.
    # The key insight: if we can find a matching that covers all but 2 vertices in a certain way...
    
    # Actually, let's use a more direct approach: 
    # 1. Identify vertices with low degree (potential bottlenecks)
    # 2. Use DFS with backtracking, prioritizing vertices that lead to dead ends
    
    # Let's compute degrees within the component
    deg = {v: len(adj[v] & comp_set) for v in comp_set}
    
    # Sort vertices by degree (low degree first - these are more constrained)
    sorted_vertices = sorted(comp_set, key=lambda v: (deg[v], v))
    
    # Try to find a Hamiltonian path using backtracking with heuristics
    # But for 100 vertices, pure backtracking might be too slow.
    # Let's use a greedy approach with local search.
    
    # Strategy: Start from a low-degree vertex, greedily extend the path
    # choosing the neighbor with the lowest "slack" (fewest unused neighbors)
    
    best_path = []
    
    # Try starting from each vertex, but prioritize low-degree vertices
    for start in sorted_vertices:
        # Greedy DFS with backtracking (limited depth for speed)
        path = [start]
        used = {start}
        
        # Use a stack-based DFS with heuristic ordering
        # At each step, choose the neighbor with the fewest unused neighbors
        # (to avoid getting stuck)
        
        while True:
            current = path[-1]
            candidates = [v for v in adj[current] if v in comp_set and v not in used]
            
            if not candidates:
                break
            
            # Sort candidates by their "slack" - fewer unused neighbors first
            # This is the "matching slack" idea: vertices with low slack are more constrained
            candidates.sort(key=lambda v: (len(adj[v] & comp_set - used), v))
            
            # Try the most constrained candidate first
            next_v = candidates[0]
            path.append(next_v)
            used.add(next_v)
        
        if len(path) > len(best_path):
            best_path = path
        
        # Early termination if we found a Hamiltonian path
        if len(best_path) == comp_size:
            break
    
    # If we didn't find a Hamiltonian path, try to improve with local search
    # Try different starting points and different tie-breaking strategies
    
    if len(best_path) < comp_size:
        # Try multiple randomized attempts
        rng = random.Random(42)
        for attempt in range(50):
            # Random start
            start = rng.choice(sorted_vertices)
            path = [start]
            used = {start}
            
            while True:
                current = path[-1]
                candidates = [v for v in adj[current] if v in comp_set and v not in used]
                
                if not candidates:
                    break
                
                # Randomized choice with bias toward low-slack vertices
                # Weight by inverse slack
                slacks = [len(adj[v] & comp_set - used) + 1 for v in candidates]
                # Lower slack = higher priority, so use 1/(slack+1) as weight
                weights = [1.0 / s for s in slacks]
                next_v = rng.choices(candidates, weights=weights, k=1)[0]
                path.append(next_v)
                used.add(next_v)
            
            if len(path) > len(best_path):
                best_path = path
                if len(best_path) == comp_size:
                    break
    
    # Final attempt: try to extend the best path found so far
    if len(best_path) < comp_size:
        # Try to insert missing vertices into the path
        missing = comp_set - set(best_path)
        
        for _ in range(100):
            improved = False
            # Try inserting each missing vertex at each position
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                for m in missing:
                    if m in adj[u] and m in adj[v]:
                        new_path = best_path[:i + 1] + [m] + best_path[i + 1:]
                        if len(new_path) > len(best_path):
                            best_path = new_path
                            missing = comp_set - set(best_path)
                            improved = True
                            break
                if improved:
                    break
            if not improved or len(best_path) == comp_size:
                break
    
    # Ensure we return a valid path
    # Verify the path is valid
    for i in range(len(best_path) - 1):
        assert best_path[i + 1] in adj[best_path[i]], f"Invalid edge: {best_path[i]}-{best_path[i + 1]}"
    
    return best_path
# EVOLVE-BLOCK-END
