"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Directed edges for ILP
    # Map directed edge (u, v) to an index
    dir_edges = []
    dir_edge_index = {}
    for u, v in edges:
        dir_edges.append((u, v))
        dir_edge_index[(u, v)] = len(dir_edges) - 1
        dir_edges.append((v, u))
        dir_edge_index[(v, u)] = len(dir_edges) - 1
        
    m_edges = len(dir_edges)
    
    def check_feasibility(branches):
        """
        Check if there exist vertex-disjoint paths for all pairs in branches.
        Returns (feasible, paths_dict) where paths_dict maps (u,v) to list of vertices.
        """
        if len(branches) < 2:
            return False, {}
            
        k = len(branches)
        pairs = list(combinations(branches, 2))
        num_pairs = len(pairs)
        
        # Variables: x[e, p] for each directed edge e and pair p
        # Total variables: m_edges * num_pairs
        num_vars = m_edges * num_pairs
        
        # Objective: minimize 0
        c = np.zeros(num_vars)
        
        # Constraints
        # 1. Flow conservation
        # 2. Vertex capacity
        # 3. Branch exclusion
        
        A_rows = []
        lb = []
        ub = []
        
        branch_set = set(branches)
        
        # Helper to get variable index
        def var_idx(e_idx, p_idx):
            return p_idx * m_edges + e_idx
            
        # Flow conservation constraints
        # For each pair p, and each vertex v:
        # sum(out) - sum(in) = 1 if v is source, -1 if v is sink, 0 otherwise
        for p_idx, (u, v) in enumerate(pairs):
            for vert in range(n):
                if vert == u:
                    # out - in = 1
                    row = np.zeros(num_vars)
                    for e_idx, (eu, ev) in enumerate(dir_edges):
                        if eu == vert:
                            row[var_idx(e_idx, p_idx)] = 1
                        elif ev == vert:
                            row[var_idx(e_idx, p_idx)] = -1
                    A_rows.append(row)
                    lb.append(1)
                    ub.append(1)
                elif vert == v:
                    # out - in = -1
                    row = np.zeros(num_vars)
                    for e_idx, (eu, ev) in enumerate(dir_edges):
                        if eu == vert:
                            row[var_idx(e_idx, p_idx)] = 1
                        elif ev == vert:
                            row[var_idx(e_idx, p_idx)] = -1
                    A_rows.append(row)
                    lb.append(-1)
                    ub.append(-1)
                else:
                    # out - in = 0
                    row = np.zeros(num_vars)
                    for e_idx, (eu, ev) in enumerate(dir_edges):
                        if eu == vert:
                            row[var_idx(e_idx, p_idx)] = 1
                        elif ev == vert:
                            row[var_idx(e_idx, p_idx)] = -1
                    A_rows.append(row)
                    lb.append(0)
                    ub.append(0)
        
        # Vertex capacity constraints
        # For each non-branch vertex v:
        # sum of flows entering v <= 1 (for all paths)
        # Actually, sum of x[e, p] for e entering v, for all p <= 1
        for vert in range(n):
            if vert in branch_set:
                continue
            row = np.zeros(num_vars)
            for e_idx, (eu, ev) in enumerate(dir_edges):
                if ev == vert:
                    for p_idx in range(num_pairs):
                        row[var_idx(e_idx, p_idx)] = 1
            A_rows.append(row)
            lb.append(0)
            ub.append(1)
            
        # Branch exclusion constraints
        # For each pair p (u, v) and each branch b not in {u, v}:
        # x[e, p] = 0 for any e entering b
        for p_idx, (u, v) in enumerate(pairs):
            for b in branch_set:
                if b == u or b == v:
                    continue
                for e_idx, (eu, ev) in enumerate(dir_edges):
                    if ev == b:
                        row = np.zeros(num_vars)
                        row[var_idx(e_idx, p_idx)] = 1
                        A_rows.append(row)
                        lb.append(0)
                        ub.append(0)
                        
        if not A_rows:
            return True, {}
            
        A = np.array(A_rows)
        lb_arr = np.array(lb)
        ub_arr = np.array(ub)
        
        constraints = LinearConstraint(A, lb_arr, ub_arr)
        bounds = Bounds(np.zeros(num_vars), np.ones(num_vars))
        
        try:
            res = milp(c=c, constraints=constraints, integrality=np.ones(num_vars), bounds=bounds)
        except Exception:
            return False, {}
            
        if res.status != 0:
            return False, {}
            
        # Extract paths
        x = res.x
        paths_dict = {}
        for p_idx, (u, v) in enumerate(pairs):
            path = [u]
            curr = u
            while curr != v:
                found_next = False
                for e_idx, (eu, ev) in enumerate(dir_edges):
                    if eu == curr and x[var_idx(e_idx, p_idx)] > 0.5:
                        path.append(ev)
                        curr = ev
                        found_next = True
                        break
                if not found_next:
                    return False, {} # Should not happen if feasible
            paths_dict[(u, v)] = path
            
        return True, paths_dict

    # Search for maximum branches
    # Use greedy with backtracking
    # Sort vertices by degree
    vertices = list(range(n))
    vertices.sort(key=lambda v: len(adj[v]), reverse=True)
    
    best_branches = []
    best_paths = {}
    
    def search(current_branches, candidates):
        nonlocal best_branches, best_paths
        
        if time.time() > deadline:
            return
            
        # Check if current is better
        if len(current_branches) > len(best_branches):
            # Verify feasibility one more time to be safe, or trust previous check
            # We only add if feasible, so current is feasible
            best_branches = list(current_branches)
            # We need to recover paths for best_branches
            # We can store them during search or recompute
            # Recomputing is safer
            feasible, paths = check_feasibility(best_branches)
            if feasible:
                best_paths = paths
                
        if not candidates:
            return
            
        # Try to add each candidate
        for i, v in enumerate(candidates):
            new_branches = current_branches + [v]
            # Quick check: if new_branches is feasible
            feasible, _ = check_feasibility(new_branches)
            if feasible:
                search(new_branches, candidates[i+1:])
                
    # Initial guess: try max clique or just start with empty and add
    # To speed up, we can start with a small feasible set
    # Let's just start with empty and let search find it
    # But search space is 2^24 which is large.
    # We need a better heuristic.
    
    # Heuristic: Start with a random vertex, then add greedily
    # Or start with the max clique
    # Let's find a max clique using Bron-Kerbosch (fast for n=24)
    def max_clique():
        max_c = []
        def bronk(R, P_set, X_set):
            nonlocal max_c
            if time.time() > deadline:
                return
            if not P_set and not X_set:
                if len(R) > len(max_c):
                    max_c = list(R)
                return
            U = P_set.union(X_set)
            if not U: return
            pivot = max(U, key=lambda u: len(P_set.intersection(adj[u])))
            candidates = list(P_set.difference(adj[pivot]))
            for v in candidates:
                bronk(R.union({v}), P_set.intersection(adj[v]), X_set.intersection(adj[v]))
                P_set.remove(v)
                X_set.add(v)
        bronk(set(), set(range(n)), set())
        return max_c
        
    init_clique = max_clique()
    # Check if init_clique is feasible
    if init_clique:
        feasible, paths = check_feasibility(init_clique)
        if feasible:
            best_branches = init_clique
            best_paths = paths
        else:
            # Try to fix it by removing vertices
            # Just start with empty for simplicity in search, but use clique as seed for ordering
            pass
            
    # Run search starting from empty, but prioritize high degree vertices
    # The search function above is O(2^n) worst case.
    # For n=24, this is too slow if not pruned well.
    # The feasibility check is the bottleneck.
    
    # Alternative: Iterate k from n down to 1.
    # For each k, check if any subset of size k is feasible.
    # This is also hard.
    
    # Let's stick to the greedy search but with a time limit.
    # The search above adds vertices one by one.
    # To make it faster, we can limit the depth or width.
    
    # Let's try a different search: Local Search
    # 1. Start with a feasible set (e.g., clique or empty)
    # 2. Try to add a random vertex. If feasible, keep.
    # 3. Try to swap a vertex.
    
    # Given the small n, the backtracking search with ILP pruning might work if the ILP is fast.
    # ILP for k=5 has ~10 pairs, ~1000 vars. Fast.
    # ILP for k=10 has ~45 pairs, ~4500 vars. Still fast.
    
    # Let's run the backtracking search.
    search([], vertices)
    
    if not best_branches:
        # Fallback
        if edges:
            best_branches = [edges[0][0], edges[0][1]]
            best_paths = {(best_branches[0], best_branches[1]): best_branches}
        else:
            best_branches = [0, 1]
            best_paths = {(0, 1): [0, 1]}
            
    # Ensure paths are recovered for best_branches
    if not best_paths:
        feasible, paths = check_feasibility(best_branches)
        if feasible:
            best_paths = paths
        else:
            # Last resort
            best_branches = [0, 1] if n > 1 else [0]
            best_paths = {(0, 1): [0, 1]} if n > 1 else {}

    paths_list = []
    for u, v in combinations(best_branches, 2):
        if (u, v) in best_paths:
            paths_list.append([int(x) for x in best_paths[(u, v)]])
        elif (v, u) in best_paths:
            paths_list.append([int(x) for x in best_paths[(v, u)]])
        else:
            paths_list.append([int(u), int(v)])
            
    return {'branches': [int(b) for b in best_branches], 'paths': paths_list}

# EVOLVE-BLOCK-END
