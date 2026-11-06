import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin

    def is_valid(adj):
        n = adj.shape[0]
        # Check for K5
        for combo in itertools.combinations(range(n), 5):
            if all(adj[i][j] == 1 for i, j in itertools.combinations(combo, 2)):
                return False
        # Check for Independent Set 5
        for combo in itertools.combinations(range(n), 5):
            if all(adj[i][j] == 0 for i, j in itertools.combinations(combo, 2)):
                return False
        return True

    def to_edges(adj):
        n = adj.shape[0]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == 1:
                    edges.append((i, j))
        return edges

    def paley_37():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

    def get_4_sets(adj):
        n = adj.shape[0]
        four_cliques = []
        four_indeps = []
        for combo in itertools.combinations(range(n), 4):
            is_clique = True
            is_indep = True
            for i, j in itertools.combinations(combo, 2):
                if adj[i][j] == 1:
                    is_indep = False
                else:
                    is_clique = False
                if not is_clique and not is_indep:
                    break
            if is_clique:
                four_cliques.append(combo)
            elif is_indep:
                four_indeps.append(combo)
        return four_cliques, four_indeps

    def try_add_vertex_milp(adj, four_cliques, four_indeps, time_limit):
        n = adj.shape[0]
        
        if not four_cliques and not four_indeps:
            return np.ones(n, dtype=int) # Any assignment works? No, need to check K5/Ind5 with new vertex.
            # If no 4-sets, then no K5/Ind5 possible with 1 new vertex?
            # K5 needs 4 existing + 1 new. If no 4-clique, no K5.
            # Ind5 needs 4 existing + 1 new. If no 4-indep, no Ind5.
            # So any assignment is valid.

        # Formulate MILP
        # Variables x_i for i in 0..n-1
        # c = 0 (feasibility)
        c = np.zeros(n)
        
        # Constraints
        # 1. For each 4-clique C: sum(x_i for i in C) <= 3
        # 2. For each 4-indep I: sum(x_i for i in I) >= 1
        
        A_ub_list = []
        b_ub_list = []
        A_eq_list = []
        b_eq_list = []
        
        # Inequalities: A_ub @ x <= b_ub
        # For cliques: sum <= 3
        for clique in four_cliques:
            row = np.zeros(n)
            for v in clique:
                row[v] = 1
            A_ub_list.append(row)
            b_ub_list.append(3)
            
        # For indeps: sum >= 1  =>  -sum <= -1
        for indep in four_indeps:
            row = np.zeros(n)
            for v in indep:
                row[v] = -1
            A_ub_list.append(row)
            b_ub_list.append(-1)
            
        if A_ub_list:
            A_ub = np.array(A_ub_list)
            b_ub = np.array(b_ub_list)
        else:
            A_ub = None
            b_ub = None
            
        constraints = LinearConstraint(A_ub, -np.inf, b_ub) if A_ub is not None else []
        
        # Bounds: 0 <= x_i <= 1
        integrality = np.ones(n) # All integer
        
        try:
            res = milp(c, constraints=constraints, integrality=integrality, 
                       bounds=Bounds(0, 1), 
                       options={"time_limit": time_limit - time.time()})
            
            if res.success and res.x is not None:
                # res.x is float, round to int
                x = np.round(res.x).astype(int)
                return x
            else:
                return None
        except Exception:
            return None

    # Main logic
    seed_n, seed_edges = paley_37()
    best_n = seed_n
    best_edges = seed_edges
    
    adj = np.zeros((seed_n, seed_n), dtype=int)
    for i, j in seed_edges:
        adj[i][j] = 1
        adj[j][i] = 1
        
    four_cliques, four_indeps = get_4_sets(adj)
    
    target_n = 42
    for current_n in range(seed_n, target_n):
        if time.time() > deadline - 30:
            break
            
        added = False
        # Try multiple times (MILP is deterministic, so retries might not help unless we randomize or change formulation)
        # But we can try different initial points? MILP doesn't take initial points in scipy.
        # So just one attempt per vertex?
        # Or maybe randomize the order of constraints?
        
        attempt_deadline = time.time() + 20.0 # 20s per attempt
        
        c = try_add_vertex_milp(adj, four_cliques, four_indeps, attempt_deadline)
        
        if c is not None:
            new_n = current_n + 1
            new_adj = np.zeros((new_n, new_n), dtype=int)
            new_adj[:current_n, :current_n] = adj
            new_adj[:current_n, current_n] = c
            new_adj[current_n, :current_n] = c
            
            best_n = new_n
            best_edges = to_edges(new_adj)
            
            four_cliques, four_indeps = get_4_sets(new_adj)
            adj = new_adj
            added = True
            
        if not added:
            break
            
    if best_n > seed_n:
        check_adj = np.zeros((best_n, best_n), dtype=int)
        for i, j in best_edges:
            check_adj[i][j] = 1
            check_adj[j][i] = 1
        if is_valid(check_adj):
            return best_n, best_edges
        else:
            return seed_n, seed_edges
            
    return seed_n, seed_edges
