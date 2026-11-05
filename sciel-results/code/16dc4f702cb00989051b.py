import time
import sys
import numpy as np
import random
from scipy.optimize import milp, LinearConstraint, Bounds

# EVOLVE-BLOCK-START
def construct():
    n = 16
    target = 65
    deadline = time.time() + 118.0  # safety margin

    # Helper to compute score
    def compute_score(edges):
        if not edges:
            return float('inf')
        deg = [0] * n
        adj = [[False] * n for _ in range(n)]
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            adj[u][v] = True
            adj[v][u] = True
        
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    for w in range(v + 1, n):
                        if adj[u][w] and adj[v][w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    # 1. Precompute all possible edges and triangles
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    m = len(all_edges)
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # Incidence matrix: A[i, e] = 1 if edge e is incident to vertex i
    A = np.zeros((n, m), dtype=int)
    for i, (u, v) in enumerate(all_edges):
        A[u, i] = 1
        A[v, i] = 1

    # Find all triangles (static set of all possible triangles in K16)
    triangles = []
    for u in range(n):
        for v in range(u + 1, n):
            for w in range(v + 1, n):
                triangles.append((u, v, w))
    
    t_count = len(triangles)
    
    # 2. Define ILP Variables
    # x[e] for e in range(m): 1 if edge is present
    # y[u] for u in range(n): degree of vertex u
    # T: max triangle degree sum
    # Total variables: m + n + 1
    
    # Objective: minimize T
    c = np.zeros(m + n + 1)
    c[m + n] = 1.0  # Minimize T
    
    # Constraints:
    # 1. Exactly 65 edges: sum(x) = 65
    # 2. Degree consistency: y[u] = sum(A[u, e] * x[e]) for all u
    # 3. Triangle constraint: For each triangle k with vertices (u, v, w):
    #    y[u] + y[v] + y[w] <= T
    
    num_rows = 1 + n + t_count
    num_vars = m + n + 1
    
    lb = np.zeros(num_rows)
    ub = np.zeros(num_rows)
    
    # Constraint 0: Sum x = 65
    lb[0] = 65
    ub[0] = 65
    
    constraint_matrix = np.zeros((num_rows, num_vars))
    constraint_matrix[0, :m] = 1.0
    
    for u in range(n):
        row_idx = 1 + u
        constraint_matrix[row_idx, :m] = -A[u, :]
        constraint_matrix[row_idx, m + u] = 1.0
        
    for k in range(t_count):
        row_idx = 1 + n + k
        u, v, w = triangles[k]
        constraint_matrix[row_idx, m + u] = 1.0
        constraint_matrix[row_idx, m + v] = 1.0
        constraint_matrix[row_idx, m + w] = 1.0
        constraint_matrix[row_idx, m + n] = -1.0
        ub[row_idx] = 0.0
        lb[row_idx] = -np.inf
        
    lb_vars = np.zeros(num_vars)
    ub_vars = np.zeros(num_vars)
    ub_vars[:m] = 1.0
    ub_vars[m:m+n] = n - 1
    ub_vars[m+n] = 3 * (n - 1)
    
    integrality = np.zeros(num_vars)
    integrality[:m] = 1.0
    
    constraints = LinearConstraint(constraint_matrix, lb, ub)
    bounds = Bounds(lb_vars, ub_vars)
    
    # Fallback construction
    def get_seed_edges():
        # Complete bipartite K8,8 plus one edge in one part
        edges = [[u, v] for u in range(8) for v in range(8, 16)]
        edges.append([0, 1])
        return edges

    best_edges = get_seed_edges()
    best_score = compute_score(best_edges)
    
    # Try ILP with tighter bounds to guide the solver
    # We know the seed score is 24. We want to find something < 24.
    # Let's try forcing T <= 23 first.
    
    # We will run ILP with an upper bound on T to see if a solution exists.
    # If it does, great. If not, we try local search.
    
    # Strategy:
    # 1. Try ILP with T <= 23. If success, return.
    # 2. If fail, run local search starting from seed.
    # 3. If local search finds better, return.
    # 4. Return best found.
    
    # ILP Attempt 1: T <= 23
    # Note: milp doesn't allow changing constraints easily without rebuilding, 
    # but we can just set the bound on variable T.
    # However, the constraint matrix is fixed. The variable bound on T is in `bounds`.
    # So we can just call milp with modified bounds? No, bounds is passed in.
    # We can create a new Bounds object.
    
    # But wait, the constraints `y[u]+y[v]+y[w] <= T` are already there.
    # If we set ub_vars[m+n] = 23, the solver will try to find a solution with T <= 23.
    # If no solution exists, it will return infeasible.
    
    try:
        bounds_t23 = Bounds(lb_vars, ub_vars.copy())
        bounds_t23.upper[m+n] = 23.0
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds_t23, time_limit=30)
        if res.status == 0:
            x_sol = res.x[:m]
            edges = []
            for i in range(m):
                if x_sol[i] > 0.5:
                    edges.append(list(all_edges[i]))
            if len(edges) == target:
                score = compute_score(edges)
                if score < best_score:
                    best_edges = edges
                    best_score = score
                return {'n': n, 'edges': best_edges}
        # If infeasible or other status, proceed to local search
    except Exception as e:
        pass

    # Local Search
    # Start from best_edges (seed)
    current_edges = best_edges.copy()
    current_score = best_score
    adj = [[False] * n for _ in range(n)]
    deg = [0] * n
    for u, v in current_edges:
        adj[u][v] = True
        adj[v][u] = True
        deg[u] += 1
        deg[v] += 1
        
    def update_score_fast(adj, deg):
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    for w in range(v + 1, n):
                        if adj[u][w] and adj[v][w]:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    # Precompute edge indices for fast access
    edge_set = set(tuple(e) for e in current_edges)
    
    # Local search loop
    # We want to minimize max triangle sum.
    # Move: remove edge (u,v), add edge (x,y).
    # Check if score improves.
    
    # To be efficient, we can track the current max triangle sum.
    # But recomputing is O(n^3) = 4096 ops, which is fast enough for many iterations.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Generate random moves
        # Sample a few random moves per iteration to avoid getting stuck in local minima?
        # Or do full scan? Full scan is 120*120/2 ~ 7000 moves. 4096*7000 ~ 28M ops. Might be slow in Python.
        # Let's do random sampling of moves.
        
        # Actually, let's try a few random restarts or just random moves.
        # Random moves: pick random edge to remove, random non-edge to add.
        
        for _ in range(1000): # 1000 random moves per pass
            if time.time() > deadline - 5:
                break
            
            # Pick random edge to remove
            if not current_edges:
                break
            idx_rm = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx_rm]
            
            # Pick random non-edge to add
            # Try a few random candidates
            added = False
            for _ in range(10): # try 10 random candidates
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if x > y:
                    x, y = y, x
                if (x, y) in edge_set:
                    continue
                
                # Apply move
                # Remove (u,v)
                adj[u][v] = False
                adj[v][u] = False
                deg[u] -= 1
                deg[v] -= 1
                edge_set.discard((u, v))
                edge_set.discard((v, u)) # just in case, though we store canonical
                
                # Add (x,y)
                adj[x][y] = True
                adj[y][x] = True
                deg[x] += 1
                deg[y] += 1
                edge_set.add((x, y))
                
                # Compute new score
                new_score = update_score_fast(adj, deg)
                
                if new_score < current_score:
                    # Accept
                    current_score = new_score
                    # Update current_edges list
                    current_edges.pop(idx_rm)
                    current_edges.append([x, y])
                    improved = True
                    break
                else:
                    # Revert
                    adj[x][y] = False
                    adj[y][x] = False
                    deg[x] -= 1
                    deg[y] -= 1
                    edge_set.discard((x, y))
                    
                    adj[u][v] = True
                    adj[v][u] = True
                    deg[u] += 1
                    deg[v] += 1
                    edge_set.add((u, v))
            # If we found an improvement, we broke the inner loop, so we continue to next random move attempt
            # But we should probably restart the pass or just continue.
            
    # Final verification
    final_score = compute_score(current_edges)
    if final_score < best_score:
        best_edges = current_edges
        best_score = final_score
        
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
