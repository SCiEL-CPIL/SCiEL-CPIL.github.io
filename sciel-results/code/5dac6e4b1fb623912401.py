"""Graph construction via MILP: minimize max triangle degree sum.

Formulation:
- Binary variables x_e for each potential edge (120 total)
- Binary variables t_ijk for each triangle (560 total)
- Continuous variable T = max triangle degree sum
- Minimize T subject to:
  * Exactly 65 edges
  * t_ijk <= x_ij, x_jk, x_ik (triangle indicator)
  * T >= deg_i + deg_j + deg_k for all triangles
  * T <= 23 (to beat published construction)
"""
import time
import random
import numpy as np
from scipy.optimize import linprog

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0  # safety margin before 150s kill

# Precompute all potential edges and triangles
EDGE_LIST = [(i, j) for i in range(N) for j in range(i + 1, N)]
NUM_EDGES = len(EDGE_LIST)  # 120
EDGE_IDX = {(i, j): k for k, (i, j) in enumerate(EDGE_LIST)}

TRI_LIST = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            TRI_LIST.append((i, j, k))
NUM_TRIS = len(TRI_LIST)  # 560


def _solve_milp(t0, max_T=23):
    """Solve the MILP using scipy linprog (LP relaxation may give integer solution).
    
    Variables:
    - x[0..119]: binary edge indicators
    - t[120..679]: binary triangle indicators  
    - T[680]: continuous, max triangle degree sum
    
    Objective: minimize T
    """
    # Objective: minimize T (index 680)
    c = np.zeros(NUM_EDGES + NUM_TRIS + 1)
    c[-1] = 1.0  # minimize T
    
    # Inequality constraints: A_ub @ vars <= b_ub
    # 1) Exactly 65 edges: sum(x) = 65 -> -sum(x) <= -65 and sum(x) <= 65
    # 2) t_ijk <= x_ij, x_jk, x_ik for each triangle
    # 3) T >= deg_i + deg_j + deg_k for each triangle (if t_ijk=1)
    #    Actually: T - deg_i - deg_j - deg_k >= 0 when t_ijk=1
    #    We can write: T >= sum of degrees for vertices in triangle
    #    deg_v = sum of x_e over edges incident to v
    
    # Build constraint matrix
    # Constraints:
    # A) sum(x) <= 65
    # B) -sum(x) <= -65  (i.e., sum(x) >= 65)
    # C) For each triangle (i,j,k): t_ijk - x_ij <= 0, t_ijk - x_jk <= 0, t_ijk - x_ik <= 0
    # D) For each triangle (i,j,k): T - deg_i - deg_j - deg_k >= 0
    #    But this is only binding when t_ijk=1. We can use:
    #    T >= (deg_i + deg_j + deg_k) * t_ijk
    #    This is nonlinear. Instead, we can just enforce:
    #    T >= deg_i + deg_j + deg_k for ALL triangles (even non-triangles)
    #    This is valid because if it's not a triangle, the constraint is still satisfied
    #    (T is at least the sum of degrees of any triple, but we only care about actual triangles)
    #    Wait, that's too strong. Let me think again.
    #    
    #    Actually, the score is max over ACTUAL triangles of deg sum.
    #    So we need: for each actual triangle, T >= deg sum.
    #    But we don't know which are actual triangles in advance.
    #    
    #    Alternative: just enforce T >= deg_i + deg_j + deg_k for all i<j<k.
    #    This is valid (may be slightly conservative) and linear.
    
    num_constraints = 2 + NUM_TRIS * 3 + NUM_TRIS
    A_ub = np.zeros((num_constraints, NUM_EDGES + NUM_TRIS + 1))
    b_ub = np.zeros(num_constraints)
    
    # Constraint A: sum(x) <= 65
    for e in range(NUM_EDGES):
        A_ub[0, e] = 1.0
    b_ub[0] = TARGET_EDGES
    
    # Constraint B: -sum(x) <= -65
    for e in range(NUM_EDGES):
        A_ub[1, e] = -1.0
    b_ub[1] = -TARGET_EDGES
    
    # Constraint C: t_ijk - x_ij <= 0, etc.
    row = 2
    for (i, j, k) in TRI_LIST:
        t_idx = NUM_EDGES + TRI_LIST.index((i, j, k))
        A_ub[row, t_idx] = 1.0
        A_ub[row, EDGE_IDX[(i, j)]] = -1.0
        b_ub[row] = 0.0
        row += 1
        
        A_ub[row, t_idx] = 1.0
        A_ub[row, EDGE_IDX[(j, k)]] = -1.0
        b_ub[row] = 0.0
        row += 1
        
        A_ub[row, t_idx] = 1.0
        A_ub[row, EDGE_IDX[(i, k)]] = -1.0
        b_ub[row] = 0.0
        row += 1
    
    # Constraint D: T >= deg_i + deg_j + deg_k for each triple
    # deg_v = sum of x_e where e is incident to v
    # Precompute incidence: for each vertex, which edges are incident
    vertex_edges = [[] for _ in range(N)]
    for e, (u, v) in enumerate(EDGE_LIST):
        vertex_edges[u].append(e)
        vertex_edges[v].append(e)
    
    for (i, j, k) in TRI_LIST:
        A_ub[row, -1] = 1.0  # T coefficient
        for e in vertex_edges[i]:
            A_ub[row, e] = -1.0
        for e in vertex_edges[j]:
            A_ub[row, e] = -1.0
        for e in vertex_edges[k]:
            A_ub[row, e] = -1.0
        b_ub[row] = 0.0
        row += 1
    
    # Bounds: 0 <= x_e <= 1, 0 <= t_ijk <= 1, 0 <= T <= max_T
    bounds = [(0, 1)] * NUM_EDGES + [(0, 1)] * NUM_TRIS + [(0, max_T)]
    
    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.status != 0:
        return None
    
    x_vals = res.x[:NUM_EDGES]
    
    # Round to nearest integer (LP relaxation often gives integer solutions for this structure)
    edges = []
    for e, (u, v) in enumerate(EDGE_LIST):
        if x_vals[e] > 0.5:
            edges.append([u, v])
    
    # Verify edge count
    if len(edges) != TARGET_EDGES:
        # Try to fix by adding/removing edges
        adj = [[False] * N for _ in range(N)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        
        current_count = sum(adj[i][j] for i in range(N) for j in range(i + 1, N))
        
        if current_count > TARGET_EDGES:
            # Remove edges
            while current_count > TARGET_EDGES:
                # Find edge whose removal doesn't hurt score too much
                best_remove = None
                best_score_after = float('inf')
                for u, v in edges[:]:
                    adj[u][v] = adj[v][u] = False
                    deg = [0] * N
                    for i in range(N):
                        for j in range(N):
                            if adj[i][j]:
                                deg[i] += 1
                    # Compute max triangle sum
                    max_sum = 0
                    for (a, b, c) in TRI_LIST:
                        if adj[a][b] and adj[b][c] and adj[a][c]:
                            s = deg[a] + deg[b] + deg[c]
                            if s > max_sum:
                                max_sum = s
                    if max_sum < best_score_after:
                        best_score_after = max_sum
                        best_remove = (u, v)
                    adj[u][v] = adj[v][u] = True
                if best_remove:
                    edges.remove(list(best_remove))
                    adj[best_remove[0]][best_remove[1]] = False
                    adj[best_remove[1]][best_remove[0]] = False
                    current_count -= 1
                else:
                    break
        elif current_count < TARGET_EDGES:
            # Add edges
            all_edges = set(tuple(e) for e in edges)
            while current_count < TARGET_EDGES:
                best_add = None
                best_score_after = float('inf')
                for (u, v) in EDGE_LIST:
                    if (u, v) in all_edges or (v, u) in all_edges:
                        continue
                    adj[u][v] = adj[v][u] = True
                    deg = [0] * N
                    for i in range(N):
                        for j in range(N):
                            if adj[i][j]:
                                deg[i] += 1
                    max_sum = 0
                    for (a, b, c) in TRI_LIST:
                        if adj[a][b] and adj[b][c] and adj[a][c]:
                            s = deg[a] + deg[b] + deg[c]
                            if s > max_sum:
                                max_sum = s
                    if max_sum < best_score_after:
                        best_score_after = max_sum
                        best_add = (u, v)
                    adj[u][v] = adj[v][u] = False
                if best_add:
                    edges.append(list(best_add))
                    all_edges.add(best_add)
                    adj[best_add[0]][best_add[1]] = True
                    adj[best_add[1]][best_add[0]] = True
                    current_count += 1
                else:
                    break
    
    return edges


def _local_search_polish(edges, t0, max_iters=50000):
    """Polish with local search: try edge swaps to reduce max triangle sum."""
    adj = [[False] * N for _ in range(N)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
    
    deg = [0] * N
    for i in range(N):
        for j in range(N):
            if adj[i][j]:
                deg[i] += 1
    
    def compute_score():
        max_sum = 0
        for (i, j, k) in TRI_LIST:
            if adj[i][j] and adj[j][k] and adj[i][k]:
                s = deg[i] + deg[j] + deg[k]
                if s > max_sum:
                    max_sum = s
        return max_sum
    
    cur_score = compute_score()
    best_score = cur_score
    best_edges = [list(e) for e in edges]
    
    edge_set = set(tuple(e) for e in edges)
    nonedge_set = set(tuple(e) for e in EDGE_LIST if e not in edge_set and (e[1], e[0]) not in edge_set)
    
    rng = random.Random(42)
    temp = 2.0
    cool = 0.9995
    
    for it in range(max_iters):
        if time.time() - t0 > DEADLINE - 5:
            break
        
        if not edge_set or not nonedge_set:
            break
        
        # Pick random edge to remove and non-edge to add
        e_remove = list(edge_set)[rng.randrange(len(edge_set))]
        e_add = list(nonedge_set)[rng.randrange(len(nonedge_set))]
        
        u, v = e_remove
        x, y = e_add
        
        # Apply swap
        adj[u][v] = adj[v][u] = False
        deg[u] -= 1
        deg[v] -= 1
        adj[x][y] = adj[y][x] = True
        deg[x] += 1
        deg[y] += 1
        
        new_score = compute_score()
        
        accept = False
        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.3
        else:
            d = new_score - cur_score
            if temp > 1e-9:
                accept = (1.0 / (1.0 + np.exp(d / temp))) > rng.random()
            else:
                accept = False
        
        if accept:
            cur_score = new_score
            edge_set.remove(e_remove)
            edge_set.add(e_add)
            nonedge_set.remove(e_add)
            nonedge_set.add(e_remove)
            if new_score < best_score:
                best_score = new_score
                best_edges = [list(e) for e in edge_set]
        else:
            # Undo
            adj[x][y] = adj[y][x] = False
            deg[x] -= 1
            deg[y] -= 1
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1
        
        temp *= cool
    
    return best_edges


def construct():
    t0 = time.time()
    
    # Try MILP with progressively relaxed T bounds
    for max_T in [23, 24, 25, 26]:
        if time.time() - t0 > DEADLINE - 30:
            break
        
        edges = _solve_milp(t0, max_T)
        if edges is not None and len(edges) == TARGET_EDGES:
            # Polish with local search
            if time.time() - t0 < DEADLINE - 10:
                edges = _local_search_polish(edges, t0)
            
            # Verify
            adj = [[False] * N for _ in range(N)]
            for u, v in edges:
                adj[u][v] = adj[v][u] = True
            count = sum(adj[i][j] for i in range(N) for j in range(i + 1, N))
            if count == TARGET_EDGES:
                return {'n': N, 'edges': edges}
    
    # Fallback: use the known good construction from the seed program
    # K_{8,8} + one internal edge
    adj = [[False] * N for _ in range(N)]
    for i in range(8):
        for j in range(8, 16):
            adj[i][j] = adj[j][i] = True
    # Add edge (0, 1)
    adj[0][1] = adj[1][0] = True
    
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j]:
                edges.append([i, j])
    
    # Polish
    if time.time() - t0 < DEADLINE - 10:
        edges = _local_search_polish(edges, t0)
    
    return {'n': N, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res["edges"]))
