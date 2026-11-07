"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_base_adj():
    n = P['n']
    adj = [[False] * n for _ in range(n)]
    for u, v in P['edges']:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _get_candidate_edges(base_adj):
    """Get all edges that can be added without creating triangles."""
    n = P['n']
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if base_adj[u][v]:
                continue
            # Check if adding (u,v) creates a triangle
            creates_triangle = False
            for w in range(n):
                if base_adj[u][w] and base_adj[v][w]:
                    creates_triangle = True
                    break
            if not creates_triangle:
                candidates.append((u, v))
    return candidates

def _check_triangle_free(adj, added_edges):
    """Verify no triangles exist in the augmented graph."""
    n = P['n']
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                for k in range(j + 1, n):
                    if adj[i][k] and adj[j][k]:
                        return False
    return True

def _compute_diameter(adj):
    """Compute graph diameter via BFS from each node."""
    n = P['n']
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in range(n):
                if adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d > max_dist:
                max_dist = d
    return max_dist

def _is_valid(adj, added_edges):
    """Check if solution is valid: triangle-free, connected, diameter <= 4."""
    if not _check_triangle_free(adj, added_edges):
        return False
    diam = _compute_diameter(adj)
    return diam <= 4

def _assign_bipartition(n, base_adj):
    """Assign vertices to two sets A and B such that base path edges cross the partition.
    For a path, we can alternate: even indices in A, odd in B."""
    partition = [0] * n  # 0 = A, 1 = B
    for i in range(n):
        partition[i] = i % 2
    return partition

def _simulated_annealing(base_adj, candidates, deadline):
    """Use simulated annealing to find minimum edge set satisfying constraints."""
    n = P['n']
    start_time = time.time()
    
    # Initial solution: try to find a feasible solution greedily first
    # Start with a random subset of candidates
    random.seed(42)
    
    best_solution = None
    best_edges = []
    
    # Try multiple random initializations
    for init_attempt in range(5):
        if time.time() > deadline:
            break
        
        # Random initial set of edges
        current_edges = set()
        # Add some random candidates
        num_initial = random.randint(3, 8)
        for _ in range(num_initial):
            if candidates:
                e = random.choice(candidates)
                current_edges.add(e)
        
        # Build adjacency
        adj = [row[:] for row in base_adj]
        for u, v in current_edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check if valid
        if _check_triangle_free(adj, list(current_edges)):
            diam = _compute_diameter(adj)
            if diam <= 4:
                best_edges = list(current_edges)
                best_solution = best_edges[:]
        
        # Simulated annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        max_iterations = 5000
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Choose a move: add, remove, or swap an edge
            move_type = random.random()
            
            if move_type < 0.3 and len(current_edges) < 15:
                # Add an edge
                if candidates:
                    e = random.choice(candidates)
                    if e not in current_edges:
                        old_size = len(current_edges)
                        current_edges.add(e)
                        
                        # Check triangle-free
                        u, v = e
                        creates_tri = False
                        for w in range(n):
                            if adj[u][w] and adj[v][w]:
                                creates_tri = True
                                break
                        
                        if creates_tri:
                            current_edges.discard(e)
                            continue
                        
                        adj[u][v] = True
                        adj[v][u] = True
                        
                        diam = _compute_diameter(adj)
                        new_cost = len(current_edges) + max(0, diam - 4) * 10
                        old_cost = old_size + max(0, _compute_diameter([row[:] for row in base_adj]) - 4) * 10
                        
                        # Simplified: just check if valid and prefer fewer edges
                        if diam <= 4:
                            if not best_solution or len(current_edges) < len(best_solution):
                                best_solution = list(current_edges)
                                best_edges = best_solution[:]
                        elif random.random() < min(1.0, T / 10.0):
                            # Accept worse solution with probability
                            pass
                        else:
                            # Revert
                            current_edges.discard(e)
                            adj[u][v] = False
                            adj[v][u] = False
                            
            elif move_type < 0.6 and len(current_edges) > 0:
                # Remove an edge
                e = random.choice(list(current_edges))
                u, v = e
                current_edges.discard(e)
                adj[u][v] = False
                adj[v][u] = False
                
                diam = _compute_diameter(adj)
                if diam <= 4:
                    if not best_solution or len(current_edges) < len(best_solution):
                        best_solution = list(current_edges)
                        best_edges = best_solution[:]
                else:
                    # Revert if too many bad
                    if random.random() > min(1.0, T / 10.0):
                        current_edges.add(e)
                        adj[u][v] = True
                        adj[v][u] = True
                        
            else:
                # Swap: remove one, add another
                if candidates and len(current_edges) > 0:
                    e_remove = random.choice(list(current_edges))
                    e_add = random.choice(candidates)
                    if e_add not in current_edges:
                        u1, v1 = e_remove
                        u2, v2 = e_add
                        
                        # Check triangle for new edge
                        creates_tri = False
                        for w in range(n):
                            if adj[u2][w] and adj[v2][w]:
                                creates_tri = True
                                break
                        
                        if not creates_tri:
                            current_edges.discard(e_remove)
                            adj[u1][v1] = False
                            adj[v1][u1] = False
                            
                            current_edges.add(e_add)
                            adj[u2][v2] = True
                            adj[v2][u2] = True
                            
                            diam = _compute_diameter(adj)
                            if diam <= 4:
                                if not best_solution or len(current_edges) < len(best_solution):
                                    best_solution = list(current_edges)
                                    best_edges = best_solution[:]
                            else:
                                if random.random() > min(1.0, T / 10.0):
                                    # Revert
                                    current_edges.discard(e_add)
                                    adj[u2][v2] = False
                                    adj[v2][u2] = False
                                    current_edges.add(e_remove)
                                    adj[u1][v1] = True
                                    adj[v1][u1] = True
                            
            T *= cooling
            if T < T_min:
                T = T_min
    
    return best_edges

def construct():
    deadline = time.time() + 80  # 80s deadline with 10s safety margin
    
    base_adj = _build_base_adj()
    candidates = _get_candidate_edges(base_adj)
    
    # First, try the known good solution
    known_solution = [
        [12, 1], [12, 3], [12, 6], [12, 9],
        [12, 15], [12, 18], [12, 21], [12, 23]
    ]
    
    # Verify known solution
    adj_test = [row[:] for row in base_adj]
    valid = True
    for u, v in known_solution:
        if adj_test[u][v]:
            valid = False
            break
        for w in range(P['n']):
            if adj_test[u][w] and adj_test[v][w]:
                valid = False
                break
        if not valid:
            break
        adj_test[u][v] = True
        adj_test[v][u] = True
    
    if valid:
        diam = _compute_diameter(adj_test)
        if diam <= 4:
            return known_solution
    
    # Use simulated annealing to find better solution
    best_edges = _simulated_annealing(base_adj, candidates, deadline)
    
    if best_edges is not None:
        # Convert to list of lists
        result = [[u, v] for u, v in best_edges]
        return result
    else:
        # Fallback: return empty or known solution
        return known_solution

# EVOLVE-BLOCK-END
