import numpy as np
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 20
    deadline = time.time() + 140  # Safety margin
    
    # 1. Try MILP
    best_edges_list = None
    best_edges_count = 0
    
    try:
        # Build MILP
        # Variables: x_{uv} (190) + y_{uvw} (3420)
        num_pairs = n * (n - 1) // 2
        num_y = num_pairs * (n - 2)
        total_vars = num_pairs + num_y
        
        # Objective: Maximize sum x_{uv}
        c = np.zeros(total_vars)
        c[:num_pairs] = -1  # Minimize -sum x
        
        # Constraints
        A_ub = []
        b_ub = []
        
        # Mapping for x
        x_idx = {}
        idx = 0
        for u in range(n):
            for v in range(u + 1, n):
                x_idx[(u, v)] = idx
                idx += 1
        
        # Mapping for y
        y_idx = {}
        idx = num_pairs
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(n):
                    if w != u and w != v:
                        # Ensure consistent ordering for y? y_{uvw} corresponds to pair (u,v) and vertex w.
                        # Since x is symmetric, x_{uw} is x_{min(u,w), max(u,w)}.
                        # We just need a unique index for (u,v,w) where {u,v} is the pair.
                        y_idx[(u, v, w)] = idx
                        idx += 1
                        
        # Linearization constraints
        # y_{uvw} <= x_{uw}  =>  y_{uvw} - x_{uw} <= 0
        # y_{uvw} <= x_{vw}  =>  y_{uvw} - x_{vw} <= 0
        # y_{uvw} >= x_{uw} + x_{vw} - 1 => -y_{uvw} + x_{uw} + x_{vw} <= 1
        
        rows = []
        rhs = []
        
        for (u, v) in x_idx:
            for w in range(n):
                if w == u or w == v:
                    continue
                y_i = y_idx[(u, v, w)]
                x_uw_i = x_idx[(min(u, w), max(u, w))]
                x_vw_i = x_idx[(min(v, w), max(v, w))]
                
                # Row 1: y - x_uw <= 0
                row1 = np.zeros(total_vars)
                row1[y_i] = 1
                row1[x_uw_i] = -1
                rows.append(row1)
                rhs.append(0)
                
                # Row 2: y - x_vw <= 0
                row2 = np.zeros(total_vars)
                row2[y_i] = 1
                row2[x_vw_i] = -1
                rows.append(row2)
                rhs.append(0)
                
                # Row 3: -y + x_uw + x_vw <= 1
                row3 = np.zeros(total_vars)
                row3[y_i] = -1
                row3[x_uw_i] = 1
                row3[x_vw_i] = 1
                rows.append(row3)
                rhs.append(1)
                
        # Sum constraint: sum_w y_{uvw} <= 2
        for (u, v) in x_idx:
            row_sum = np.zeros(total_vars)
            for w in range(n):
                if w == u or w == v:
                    continue
                y_i = y_idx[(u, v, w)]
                row_sum[y_i] = 1
            rows.append(row_sum)
            rhs.append(2)
            
        A_ub = np.array(rows)
        b_ub = np.array(rhs)
        
        # Bounds
        integrality = np.ones(total_vars) # All binary
        bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
        
        # Solve
        res = milp(c=c, constraints=LinearConstraint(A_ub, ub=b_ub), integrality=integrality, bounds=bounds, options={"time_limit": 100})
        
        if res.status == 0 or res.status == 3: # Optimal or Time Limit with incumbent
            if res.x is not None:
                x_sol = res.x[:num_pairs].astype(int)
                # Round to nearest integer
                x_sol = np.round(x_sol).astype(int)
                
                # Reconstruct adjacency matrix
                A = np.zeros((n, n), dtype=int)
                idx = 0
                for u in range(n):
                    for v in range(u + 1, n):
                        if x_sol[idx] == 1:
                            A[u, v] = 1
                            A[v, u] = 1
                        idx += 1
                            
                # Validate
                if is_valid(A):
                    best_edges_list = edges_from_matrix(A)
                    best_edges_count = len(best_edges_list)
                    
    except Exception as e:
        print(f"MILP failed: {e}")
        
    # 2. Fallback / Polish with Heuristic
    if best_edges_count == 0 or time.time() < deadline - 10:
        # Run heuristic to get a baseline or improve
        A_heur = solve_heuristic(n, time_limit=30)
        edges_heur = edges_from_matrix(A_heur)
        if len(edges_heur) > best_edges_count:
            best_edges_list = edges_heur
            best_edges_count = len(edges_heur)
            
        # Local search on the best found
        if best_edges_list:
            A_best = matrix_from_edges(best_edges_list)
            A_best, _ = local_search(A_best, deadline - 5)
            final_edges = edges_from_matrix(A_best)
            if len(final_edges) > best_edges_count:
                best_edges_list = final_edges
                best_edges_count = len(final_edges)
                
    if best_edges_list is None:
        # Absolute fallback: 5 disjoint K4s
        best_edges_list = []
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    best_edges_list.append([u, v])
                    
    return best_edges_list

def is_valid(A):
    K = A @ A
    np.fill_diagonal(K, 0)
    return np.max(K) <= 2

def edges_from_matrix(A):
    edges = []
    n = A.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if A[u, v] == 1:
                edges.append([u, v])
    return edges

def matrix_from_edges(edges):
    n = 20
    A = np.zeros((n, n), dtype=int)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    return A

def solve_heuristic(n, time_limit):
    # Greedy initialization
    A = np.zeros((n, n), dtype=int)
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                u = 4 * k + a
                v = 4 * k + b
                A[u, v] = 1
                A[v, u] = 1
                
    # Greedy fill
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 0:
                    if try_add_edge(A, u, v):
                        changed = True
    return A

def try_add_edge(A, u, v):
    n = A.shape[0]
    if A[u, v] == 1:
        return False
    for w in range(n):
        if w == u or w == v:
            continue
        # Check pair (u, w)
        if A[v, w] == 1:
            cn_uw = int(np.dot(A[u, :], A[:, w]))
            if cn_uw >= 2:
                return False
        # Check pair (v, w)
        if A[u, w] == 1:
            cn_vw = int(np.dot(A[v, :], A[:, w]))
            if cn_vw >= 2:
                return False
    A[u, v] = 1
    A[v, u] = 1
    return True

def local_search(A, deadline):
    best_A = A.copy()
    best_edges = int(np.sum(A)) // 2
    n = A.shape[0]
    
    while time.time() < deadline:
        # Random restarts or moves
        # Try adding a random edge
        non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
        if not non_edges:
            break
        u, v = random.choice(non_edges)
        if try_add_edge(A, u, v):
            current_edges = int(np.sum(A)) // 2
            if current_edges > best_edges:
                best_edges = current_edges
                best_A = A.copy()
        else:
            # Swap: remove random edge, try to add u,v
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 1]
            if not edges: break
            ru, rv = random.choice(edges)
            A_backup = A.copy()
            A[ru, rv] = 0
            A[rv, ru] = 0
            try_add_edge(A, u, v)
            # Try to add another
            non_edges_now = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
            for _ in range(3):
                if not non_edges_now: break
                x, y = random.choice(non_edges_now)
                if try_add_edge(A, x, y):
                    non_edges_now.remove((x, y))
            
            current_edges = int(np.sum(A)) // 2
            if current_edges > best_edges:
                best_edges = current_edges
                best_A = A.copy()
            elif current_edges < best_edges:
                A[:] = A_backup
    return best_A, best_edges
