# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import milp, Bounds, LinearConstraint

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a large square-difference-free set modulo 205.
    
    Strategy:
    1. Compute the conflict graph (edges connect pairs with forbidden difference).
    2. Use a hybrid approach:
       a. Run MILP to get an exact or high-quality initial solution.
       b. Use a specialized local search (1-add, 1-swap, 2-for-1, 1-for-2) to try to improve 
          upon the MILP solution, in case the solver hit a time limit or returned a 
          sub-optimal solution due to numerical issues or solver settings.
       c. If MILP fails or times out, use randomized greedy + local search.
    3. The local search is optimized for speed using bitsets (integers) for adjacency.
    """
    m = 205
    k = 2
    
    # Step 1: Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # Build adjacency as bitmasks for fast local search
    # adj_mask[i] is an integer where bit j is set if i and j are adjacent
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i - d) % m
            if i != j:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # Precompute the list of forbidden pairs for MILP constraints
    edges = []
    for i in range(m):
        for j in range(i + 1, m):
            if (adj_mask[i] >> j) & 1:
                edges.append((i, j))
    
    def is_valid(subset):
        if len(subset) == 0:
            return False
        if len(set(subset)) != len(subset):
            return False
        if any(x < 0 or x >= m for x in subset):
            return False
        s = set(subset)
        for a in subset:
            for b in subset:
                if a == b:
                    continue
                d = (a - b) % m
                if d in forbidden:
                    return False
        return True

    best_set = []
    best_size = 0
    
    def update_best(subset):
        nonlocal best_set, best_size
        if is_valid(subset):
            if len(subset) > best_size:
                best_set = list(subset)
                best_size = len(subset)

    # --- Local Search Helper ---
    # Uses bitmasks for speed.
    # state: set of nodes in current solution
    # We try to add a node, or swap one node for another, etc.
    
    def local_search(initial_set, max_iter=5000):
        current = set(initial_set)
        current_mask = 0
        for v in current:
            current_mask |= (1 << v)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Try 1-move: Add a node
            # A node v can be added if it has no neighbors in current set.
            # i.e., adj_mask[v] & current_mask == 0
            for v in range(m):
                if v not in current:
                    if (adj_mask[v] & current_mask) == 0:
                        current.add(v)
                        current_mask |= (1 << v)
                        improved = True
                        break
            
            # Try 2-move: Swap one node out, one in
            # Remove u, add v.
            # Valid if:
            # 1. u in current, v not in current
            # 2. v is not adjacent to any node in (current - {u})
            #    i.e., adj_mask[v] & (current_mask ^ (1 << u)) == 0
            
            if not improved:
                # Try to find a swap
                found_swap = False
                for u in list(current):
                    # Mask of current set without u
                    mask_without_u = current_mask ^ (1 << u)
                    # Find v not in current such that adj_mask[v] & mask_without_u == 0
                    for v in range(m):
                        if v in current:
                            continue
                        if (adj_mask[v] & mask_without_u) == 0:
                            # Swap u with v
                            current.remove(u)
                            current.add(v)
                            current_mask ^= (1 << u)
                            current_mask |= (1 << v)
                            improved = True
                            found_swap = True
                            break
                    if found_swap:
                        break
            
            # Try 3-move: Remove 2 nodes, add 1 node (2-for-1 swap)
            # This can help escape local optima where no single swap works.
            if not improved and len(current) >= 2:
                found_2for1 = False
                u_list = list(current)
                for i_idx in range(len(u_list)):
                    u1 = u_list[i_idx]
                    for j_idx in range(i_idx + 1, len(u_list)):
                        u2 = u_list[j_idx]
                        # Mask of current set without u1 and u2
                        mask_without_u1u2 = current_mask ^ (1 << u1) ^ (1 << u2)
                        # Find v not in current such that adj_mask[v] & mask_without_u1u2 == 0
                        for v in range(m):
                            if v in current:
                                continue
                            if (adj_mask[v] & mask_without_u1u2) == 0:
                                # Also ensure v is not adjacent to u1 or u2 (already checked by mask)
                                # Remove u1, u2; add v
                                current.remove(u1)
                                current.remove(u2)
                                current.add(v)
                                current_mask ^= (1 << u1)
                                current_mask ^= (1 << u2)
                                current_mask |= (1 << v)
                                improved = True
                                found_2for1 = True
                                break
                        if found_2for1:
                            break
                    if found_2for1:
                        break
            
            # Try 4-move: Remove 1 node, add 2 nodes (1-for-2 swap)
            # This can increase the set size if the removed node is "blocking" two potential additions.
            if not improved and len(current) >= 1:
                found_1for2 = False
                for u in list(current):
                    # Mask of current set without u
                    mask_without_u = current_mask ^ (1 << u)
                    # Find two nodes v1, v2 not in current such that:
                    # 1. v1 is compatible with (current - {u})
                    # 2. v2 is compatible with (current - {u})
                    # 3. v1 is compatible with v2
                    candidates = []
                    for v in range(m):
                        if v in current:
                            continue
                        if (adj_mask[v] & mask_without_u) == 0:
                            candidates.append(v)
                    
                    # Try all pairs from candidates
                    for idx1 in range(len(candidates)):
                        v1 = candidates[idx1]
                        for idx2 in range(idx1 + 1, len(candidates)):
                            v2 = candidates[idx2]
                            # Check if v1 and v2 are compatible
                            if (adj_mask[v1] >> v2) & 1:
                                continue
                            # Also check v2 and v1 (should be same if symmetric, but let's be safe)
                            # Actually adj_mask is symmetric for undirected graph, so one check is enough
                            
                            # Remove u; add v1, v2
                            current.remove(u)
                            current.add(v1)
                            current.add(v2)
                            current_mask ^= (1 << u)
                            current_mask |= (1 << v1)
                            current_mask |= (1 << v2)
                            improved = True
                            found_1for2 = True
                            break
                        if found_1for2:
                            break
                    if found_1for2:
                        break

        return list(current)

    deadline = time.time() + 580.0
    
    # --- MILP Approach ---
    try:
        n_vars = m
        c = -np.ones(n_vars)
        
        n_constraints = len(edges)
        if n_constraints > 0:
            A = np.zeros((n_constraints, n_vars))
            for idx, (i, j) in enumerate(edges):
                A[idx, i] = 1
                A[idx, j] = 1
            
            bounds = Bounds(lb=0, ub=1)
            integrality = np.ones(n_vars)
            cons = LinearConstraint(A, ub=np.ones(n_constraints))
            
            start_time = time.time()
            # Give MILP 60 seconds
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": 60})
            
            if res.status == 0 or res.status == 1: # Optimal or Feasible
                solution = res.x.astype(int)
                subset = [i for i in range(n_vars) if solution[i] == 1]
                update_best(subset)
            else:
                # Try to use the incumbent if available?
                # milp doesn't easily expose incumbent in all versions/statuses simply.
                # We'll rely on fallback.
                pass
    except Exception:
        pass

    # --- Local Search Refinement ---
    # If we have a solution from MILP, refine it.
    if best_size > 0:
        refined = local_search(best_set, max_iter=1000)
        update_best(refined)
    
    # --- Randomized Greedy + Local Search Fallback ---
    # If MILP failed or we want to explore more, run random greedy starts.
    # This is a different design family: Stochastic Local Search (Simulated Annealing-like or simple Hill Climbing)
    # from multiple random starts.
    
    if time.time() < deadline:
        for _ in range(20):
            if time.time() > deadline:
                break
            # Random greedy construction
            order = list(range(m))
            random.shuffle(order)
            current = []
            current_mask = 0
            for candidate in order:
                if (adj_mask[candidate] & current_mask) == 0:
                    current.append(candidate)
                    current_mask |= (1 << candidate)
            
            # Local search from this random start
            refined = local_search(current, max_iter=2000)
            update_best(refined)

    if best_set:
        return np.array(best_set, dtype=np.int64)
    
    return np.array([0], dtype=np.int64)

# EVOLVE-BLOCK-END
