"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

# Pre-compute adjacency lists and host edge set
N = P['n']
adj = [[] for _ in range(N)]
host_edges_set = set()
for u, v in P['edges']:
    adj[u].append(v)
    adj[v].append(u)
    host_edges_set.add((min(u, v), max(u, v)))

def find_hamiltonian_cycle(adj, n, start_vertex=None, deadline=None, rng=None):
    """
    Finds a Hamiltonian cycle in the graph defined by adj.
    Uses backtracking with Warnsdorff's rule and optional randomization.
    Returns a list of vertices representing the cycle, or None if not found.
    """
    if n < 3:
        return None
    
    if start_vertex is None:
        start_vertex = 0
    
    path = [start_vertex]
    visited = {start_vertex}
    
    # Precompute neighbor lists for faster access
    adj_set = [set(lst) for lst in adj]
    
    def dfs(current, depth):
        if deadline is not None and time.time() > deadline:
            return False
        
        if len(path) == n:
            # Check if we can close the cycle
            if start_vertex in adj_set[current]:
                return True
            return False
        
        # Get unvisited neighbors
        candidates = []
        for neighbor in adj[current]:
            if neighbor not in visited:
                # Warnsdorff: count unvisited neighbors of the neighbor
                unvisited_count = sum(1 for nb in adj[neighbor] if nb not in visited)
                if rng is not None:
                    # Add small random perturbation to break ties
                    candidates.append((unvisited_count + rng.random() * 0.1, neighbor))
                else:
                    candidates.append((unvisited_count, neighbor))
        
        # Sort by fewest unvisited neighbors (Warnsdorff's rule)
        candidates.sort()
        
        for _, next_vertex in candidates:
            path.append(next_vertex)
            visited.add(next_vertex)
            
            if dfs(next_vertex, depth + 1):
                return True
            
            path.pop()
            visited.remove(next_vertex)
            
        return False

    if dfs(start_vertex, 0):
        return path
    return None

def find_forced_cycle(adj, n, start, second, deadline=None, rng=None):
    """
    Finds a Hamiltonian cycle that must start with start -> second.
    """
    if n < 3:
        return None
    
    path = [start, second]
    visited = {start, second}
    
    adj_set = [set(lst) for lst in adj]
    
    def dfs(current, depth):
        if deadline is not None and time.time() > deadline:
            return False
        
        if len(path) == n:
            if start in adj_set[current]:
                return True
            return False
        
        candidates = []
        for neighbor in adj[current]:
            if neighbor not in visited:
                unvisited_count = sum(1 for nb in adj[neighbor] if nb not in visited)
                if rng is not None:
                    candidates.append((unvisited_count + rng.random() * 0.1, neighbor))
                else:
                    candidates.append((unvisited_count, neighbor))
        
        candidates.sort()
        
        for _, next_vertex in candidates:
            path.append(next_vertex)
            visited.add(next_vertex)
            
            if dfs(next_vertex, depth + 1):
                return True
            
            path.pop()
            visited.remove(next_vertex)
            
        return False

    if dfs(second, 1):
        return path
    return None

def construct():
    # Deadline guard - keep 10s safety margin
    start_time = time.time()
    deadline = start_time + 55
    
    best_result = []
    
    # Strategy: Try to find 3 edge-disjoint Hamiltonian cycles
    # The graph is 6-regular with 21 vertices and 63 edges
    # Each Hamiltonian cycle uses 21 edges, so 3 cycles would use all 63 edges
    
    def try_decomposition(rng_seed=None, max_restarts=5):
        """Try to find 3 edge-disjoint Hamiltonian cycles using randomized restarts."""
        if rng_seed is not None:
            rng = random.Random(rng_seed)
        else:
            rng = random.Random()
        
        # Try multiple restarts with different randomizations
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Build initial adjacency
            curr_adj = [list(lst) for lst in adj]
            removed_edges = set()
            
            cycles = []
            
            # Find Cycle 1
            found_cycle1 = False
            # Try different second vertices to diversify
            neighbors_0 = list(curr_adj[0])
            rng.shuffle(neighbors_0)
            
            for second_v in neighbors_0:
                if time.time() > deadline:
                    break
                
                cycle1 = find_forced_cycle(curr_adj, N, 0, second_v, deadline, rng)
                if cycle1:
                    found_cycle1 = True
                    cycles.append(cycle1)
                    for i in range(len(cycle1)):
                        u = cycle1[i]
                        v = cycle1[(i + 1) % len(cycle1)]
                        removed_edges.add((u, v))
                        removed_edges.add((v, u))
                    break
            
            if not found_cycle1:
                continue
            
            # Find Cycle 2
            curr_adj_1 = [[] for _ in range(N)]
            for u in range(N):
                for v in adj[u]:
                    if (u, v) not in removed_edges:
                        curr_adj_1[u].append(v)
            
            found_cycle2 = False
            neighbors_0_1 = list(curr_adj_1[0])
            rng.shuffle(neighbors_0_1)
            
            for second_v in neighbors_0_1:
                if time.time() > deadline:
                    break
                
                cycle2 = find_forced_cycle(curr_adj_1, N, 0, second_v, deadline, rng)
                if cycle2:
                    found_cycle2 = True
                    cycles.append(cycle2)
                    for i in range(len(cycle2)):
                        u = cycle2[i]
                        v = cycle2[(i + 1) % len(cycle2)]
                        removed_edges.add((u, v))
                        removed_edges.add((v, u))
                    break
            
            if not found_cycle2:
                continue
            
            # Find Cycle 3
            curr_adj_2 = [[] for _ in range(N)]
            for u in range(N):
                for v in adj[u]:
                    if (u, v) not in removed_edges:
                        curr_adj_2[u].append(v)
            
            # After removing 2 Hamiltonian cycles from a 6-regular graph,
            # we have a 2-regular graph. If connected, it's a single cycle.
            # Check if it's a single cycle covering all vertices
            if all(len(lst) == 2 for lst in curr_adj_2):
                # It's 2-regular. Check if it's connected (single cycle)
                visited = set()
                if curr_adj_2[0]:
                    stack = [0]
                    visited.add(0)
                    while stack:
                        node = stack.pop()
                        for nb in curr_adj_2[node]:
                            if nb not in visited:
                                visited.add(nb)
                                stack.append(nb)
                
                if len(visited) == N:
                    # It's a single cycle - extract it
                    cycle3 = [0]
                    prev = -1
                    curr = 0
                    while len(cycle3) < N:
                        next_v = None
                        for nb in curr_adj_2[curr]:
                            if nb != prev:
                                next_v = nb
                                break
                        if next_v is None or next_v in cycle3:
                            break
                        cycle3.append(next_v)
                        prev = curr
                        curr = next_v
                    
                    if len(cycle3) == N and cycle3[0] in curr_adj_2[cycle3[-1]]:
                        cycles.append(cycle3)
            
            if len(cycles) == 3:
                return cycles
        
        return cycles
    
    # First, try deterministic approach
    result = try_decomposition(rng_seed=42, max_restarts=3)
    
    if len(result) < 3:
        # Try with different seeds
        for seed in [0, 1, 7, 13, 100, 123, 456, 789, 1000, 2024]:
            if time.time() > deadline:
                break
            result = try_decomposition(rng_seed=seed, max_restarts=2)
            if len(result) == 3:
                break
    
    # Verify and format result
    if result:
        # Convert to list of lists of Python ints
        formatted = []
        for c in result:
            formatted.append([int(v) for v in c])
        
        # Verify all cycles use same vertex set of size N
        if formatted:
            vset = set(formatted[0])
            if len(vset) != N:
                return []
            for c in formatted:
                if set(c) != vset:
                    return []
        
        # Verify edge-disjointness and host edges
        used_edges = set()
        for c in formatted:
            for i in range(len(c)):
                u = c[i]
                v = c[(i + 1) % len(c)]
                edge = (min(u, v), max(u, v))
                if edge in used_edges:
                    return []
                used_edges.add(edge)
                if edge not in host_edges_set:
                    return []
        
        best_result = formatted
    
    return best_result

# EVOLVE-BLOCK-END
