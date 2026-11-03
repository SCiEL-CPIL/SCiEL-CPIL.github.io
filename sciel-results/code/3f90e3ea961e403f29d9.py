"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def _build_supports():
    """Generate all 4-element subsets of {0,...,10} as bitmasks."""
    supports = []
    indices = range(D)
    for comb in combinations(indices, 4):
        mask = 0
        for idx in comb:
            mask |= (1 << idx)
        supports.append(mask)
    return supports

def _walksat_maxsat(n_vars, clauses, deadline):
    """
    Solve Max-SAT using WalkSAT heuristic.
    clauses: list of lists of (var, polarity). polarity +1 means var must be True, -1 means var must be False.
    Here, we want to maximize the number of selected supports such that no pair conflicts.
    Actually, standard Max-SAT formulation:
    Variables x_i = 1 if support i is selected.
    Constraint: For every conflicting pair (i, j), NOT (x_i AND x_j).
    Clause: (NOT x_i OR NOT x_j).
    We want to satisfy as many such clauses as possible while maximizing sum(x_i)?
    No, Max-SAT usually maximizes satisfied clauses.
    If we just maximize satisfied clauses of (NOT x_i OR NOT x_j), the trivial solution x_i=0 for all i satisfies all clauses.
    We need to maximize the number of True variables subject to constraints.
    This is equivalent to finding a Maximum Independent Set.
    
    Alternative WalkSAT formulation for MIS:
    We can use a penalty-based approach or just treat it as a constraint satisfaction where we want to add nodes.
    However, the prompt specifically asks for a Max-SAT formulation where clauses enforce no two selected supports share 3 indices.
    
    Let's refine the WalkSAT strategy for MIS:
    State: Binary vector x of length n.
    Objective: Maximize sum(x_i) - lambda * sum(conflicts(x_i, x_j))
    Or, more simply, use WalkSAT to satisfy the hard constraints of a clique/independent set while trying to keep the set large.
    
    Actually, a common trick is to use WalkSAT on the "conflict graph" to find a large independent set by treating the problem as:
    Minimize the number of "violations" (edges in the induced subgraph) while keeping the number of nodes high.
    
    Let's implement a specific WalkSAT variant for Maximum Independent Set:
    1. Start with a random assignment or a greedy solution.
    2. Define cost(x) = -sum(x_i) + penalty * (number of edges in induced subgraph of x).
    3. Flip a variable x_i.
    4. If cost decreases (or with some probability if it increases), accept.
    
    To align with "Max-SAT ... clauses enforce that no two selected supports share three indices":
    The clauses are C_ij = (not x_i or not x_j) for all conflicting pairs (i,j).
    We want to satisfy ALL these clauses (which defines an Independent Set).
    Among all satisfying assignments, we want the one with the most True variables.
    
    WalkSAT typically operates on a fixed set of clauses to find a satisfying assignment.
    If we use the standard WalkSAT on the conflict clauses, it will find *some* independent set.
    To make it *maximum*, we can add soft clauses or use a weighted approach, or simply restart with different random seeds and keep the best (largest) independent set found.
    Given the time limit, we can run WalkSAT multiple times, each time trying to find a satisfying assignment, and track the maximum size found.
    To bias towards larger sets, we can initialize with a dense random assignment and let WalkSAT repair violations.
    """
    start_time = time.time()
    best_size = 0
    best_solution = None
    
    # Pre-compute conflict pairs (edges in conflict graph)
    # clauses: list of (i, j) where i, j conflict
    # In Max-SAT terms, clause is (¬xi ∨ ¬xj)
    
    # We don't have the explicit list of all pairs passed in, but we can compute it or pass it.
    # For efficiency, let's assume we have a function to get neighbors or we compute cost on the fly.
    # Since n=330, n^2/2 ~ 54k pairs. We can precompute the conflict matrix or list of edges.
    
    # Let's assume we pass the adjacency list or conflict list to this function.
    # I will modify the signature to accept conflict_pairs.
    
    pass # Placeholder, will implement in main logic

def construct():
    deadline = time.time() + 110
    
    # 1. Generate supports
    supports = _build_supports()
    n_sup = len(supports)
    
    # 2. Build conflict graph
    # Conflict if intersection size is 3
    conflict_pairs = []
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        for j in range(i + 1, n_sup):
            inter = supports[i] & supports[j]
            if inter.bit_count() == 3:
                conflict_pairs.append((i, j))
                adj[i].add(j)
                adj[j].add(i)
    
    # 3. WalkSAT-based Max-SAT / MIS Solver
    # Strategy:
    # We treat the problem as finding an assignment x that satisfies all clauses (¬xi ∨ ¬xj) for (i,j) in conflict_pairs.
    # This ensures x corresponds to an Independent Set.
    # We want to maximize the number of True variables.
    # 
    # WalkSAT heuristic:
    # - Start with a random assignment biased towards True (to encourage larger sets).
    # - While unsatisfied clauses exist:
    #   - Pick a random unsatisfied clause.
    #   - Flip a variable in that clause with probability p (WalkSAT parameter).
    #   - Otherwise, flip the variable that minimizes the number of newly violated clauses (MinConflicts).
    # - When a satisfying assignment is found, record its size.
    # - Repeat with different random seeds/initializations to find larger sets.
    
    best_mis = set()
    
    def get_violations(x):
        """Return list of violated clauses (i, j) where x[i]=1 and x[j]=1."""
        violations = []
        for i, j in conflict_pairs:
            if x[i] and x[j]:
                violations.append((i, j))
        return violations
    
    def count_violations_for_var(var, x, neighbors):
        """Count how many clauses involving 'var' would be violated if var is flipped."""
        # If var is currently 0, flipping to 1 might violate edges to neighbors that are 1.
        # If var is currently 1, flipping to 0 will satisfy all edges involving var.
        # We want to minimize total violations.
        # Current violations involving var:
        # If x[var] == 1: violations are edges (var, u) where x[u]==1.
        # If x[var] == 0: violations are 0 (since clause ¬xi ∨ ¬xj is satisfied if xi=0).
        
        # Let's calculate delta in violations if we flip var.
        # Current state:
        if x[var]:
            # Currently 1. Flipping to 0 removes all violations involving var.
            # Number of violations removed = number of neighbors u with x[u]==1.
            current_violations_involving_var = sum(1 for u in neighbors[var] if x[u])
            new_violations_involving_var = 0
            delta = new_violations_involving_var - current_violations_involving_var
        else:
            # Currently 0. Flipping to 1 adds violations for neighbors u with x[u]==1.
            current_violations_involving_var = 0
            new_violations_involving_var = sum(1 for u in neighbors[var] if x[u])
            delta = new_violations_involving_var - current_violations_involving_var
            
        return delta

    p_walk = 0.5 # Probability of random flip in WalkSAT
    
    for restart in range(1000): # Limit restarts to ensure we finish
        if time.time() > deadline - 5:
            break
            
        # Initialize x
        # Bias towards True to try to find large sets
        # Probability of True: start high, e.g., 0.9
        p_true = 0.8
        x = np.random.rand(n_sup) < p_true
        
        # Local search loop
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            violations = get_violations(x)
            if not violations:
                # Found a valid Independent Set
                current_size = np.sum(x)
                if current_size > len(best_mis):
                    best_mis = set(np.where(x)[0])
                break # Restart with new random init to find potentially larger sets
                
            # Pick a random violated clause
            vi, vj = violations[np.random.randint(len(violations))]
            
            # Decide which variable to flip
            if np.random.random() < p_walk:
                # Random flip
                var = vi if np.random.random() < 0.5 else vj
            else:
                # MinConflicts: Flip the variable that results in the smallest increase in total violations
                # We only consider vi and vj for simplicity (standard WalkSAT)
                delta_i = count_violations_for_var(vi, x, adj)
                delta_j = count_violations_for_var(vj, x, adj)
                
                # Note: delta is change in violations. We want to minimize total violations.
                # So we pick the one with smaller delta (could be negative, meaning it fixes other things).
                if delta_i < delta_j:
                    var = vi
                else:
                    var = vj
                    
            x[var] = not x[var]
            
        # If we didn't find a solution in max_steps, we can still check the current state
        # But usually WalkSAT finds a solution if one exists.
        # If no solution found, the current x is invalid. We ignore it for best_mis.
        # However, we might want to keep the "least violating" state? No, we need valid MIS.
        # So we only update best_mis if we found a valid one.

    # Fallback: If WalkSAT didn't find anything (unlikely), use a simple greedy
    if not best_mis:
        # Greedy MIS
        alive = set(range(n_sup))
        while alive:
            v = min(alive, key=lambda u: len(adj[u].intersection(alive)))
            best_mis.add(v)
            alive.difference_update(adj[v])
            alive.remove(v)

    # 4. Generate vectors for the MIS supports
    vectors = []
    for idx in best_mis:
        mask = supports[idx]
        inds = [i for i in range(D) if (mask >> i) & 1]
        # To maximize the set, we should include ALL sign variations?
        # The prompt says: "Construct nonzero integer vectors... whose minimum pairwise distance is at least the maximum vector norm."
        # If we pick a support S, we can pick any vector with support S and entries +/-1.
        # Norm^2 = 4.
        # If we pick two vectors u, v with the SAME support S:
        # u.v can be 4 (if same), 2 (if 2 signs differ), 0 (if 4 signs differ? No, 4 entries, if 2 same 2 diff, dot=0. If 1 same 3 diff, dot=-2. If 0 same 4 diff, dot=-4).
        # Wait, if u and v have the same support, they are distinct vectors.
        # Max dot product for distinct vectors with same support:
        # u = (1,1,1,1), v = (1,1,1,-1) -> dot = 1+1+1-1 = 2.
        # Distance^2 = 4 + 4 - 2(2) = 4. >= 4. Valid.
        # u = (1,1,1,1), v = (1,1,-1,-1) -> dot = 1+1-1-1 = 0.
        # Distance^2 = 8. Valid.
        # So ALL sign variations for a given support are mutually compatible?
        # Let's check u=(1,1,1,1) and v=(-1,-1,-1,-1). dot = -4. Dist^2 = 8 - 2(-4) = 16. Valid.
        # So yes, for a fixed support, all 16 sign vectors form a valid clique (mutually compatible set).
        
        # Therefore, for each support in the MIS, we can add all 16 vectors.
        for sign_mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for j, idx_val in enumerate(inds):
                if sign_mask & (1 << j):
                    v[idx_val] = 1
                else:
                    v[idx_val] = -1
            vectors.append(v)
            
    S4 = np.array(vectors, dtype=np.int64)
    
    # 5. Try to add Shell 2 vectors (weight 2)
    shell2_vecs = []
    for i in range(D):
        for j in range(i+1, D):
            v1 = np.zeros(D, dtype=np.int64)
            v1[i] = 1; v1[j] = -1
            shell2_vecs.append(v1)
            v2 = np.zeros(D, dtype=np.int64)
            v2[i] = -1; v2[j] = 1
            shell2_vecs.append(v2)
    shell2 = np.array(shell2_vecs, dtype=np.int64)
    
    if len(S4) > 0:
        # Check compatibility of shell2 with S4
        # Norm shell2 = sqrt(2). Norm S4 = 2.
        # Dist^2 >= max_norm^2 = 4.
        # ||u||^2 + ||v||^2 - 2u.v >= 4
        # 2 + 4 - 2u.v >= 4 => 6 - 2u.v >= 4 => 2u.v <= 2 => u.v <= 1.
        # So we need shell2 vectors with dot product <= 1 with all S4 vectors.
        
        dots_24 = shell2 @ S4.T
        max_ip_24 = dots_24.max(axis=1)
        valid_2_mask = max_ip_24 <= 1
        S2_candidates = shell2[valid_2_mask]
        
        n2 = len(S2_candidates)
        if n2 > 0:
            # Now select a subset of S2_candidates that are mutually compatible.
            # Norm S2 = sqrt(2).
            # Dist^2 >= 4.
            # 2 + 2 - 2u.v >= 4 => 4 - 2u.v >= 4 => u.v <= 0.
            # So we need S2 vectors with dot product <= 0 with each other.
            
            dots_22 = S2_candidates @ S2_candidates.T
            # We want a subset where off-diagonal entries are <= 0.
            # This is a Maximum Independent Set on the graph where edge exists if dot > 0.
            
            # Build conflict graph for S2
            n_cand = len(S2_candidates)
            adj2 = [set() for _ in range(n_cand)]
            for i in range(n_cand):
                for j in range(i+1, n_cand):
                    if dots_22[i, j] > 0:
                        adj2[i].add(j)
                        adj2[j].add(i)
                        
            # Greedy MIS for S2 (small enough)
            alive2 = set(range(n_cand))
            selected2 = []
            while alive2:
                # Pick node with min degree
                v_idx = min(alive2, key=lambda u: len(adj2[u].intersection(alive2)))
                selected2.append(v_idx)
                alive2.difference_update(adj2[v_idx])
                alive2.remove(v_idx)
                
            S2 = S2_candidates[selected2]
            result = np.vstack([S4, S2])
        else:
            result = S4
    else:
        result = S4
        
    return result

# EVOLVE-BLOCK-END
