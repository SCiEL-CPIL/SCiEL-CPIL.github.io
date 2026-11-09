import time
import math
import sys
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2v not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
    
    n = len(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    
    # Build conflict edges
    # conflicts[i] is a list of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    conflicts = [[] for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if valid_nodes[i] + valid_nodes[j] in squares:
                conflicts[i].append(j)
                conflicts[j].append(i)
                edges.append((i, j))
    
    # If no edges, all nodes are independent
    if not edges:
        return valid_nodes
    
    # Try to solve using MILP for Minimum Vertex Cover
    # Minimize sum(x_i) subject to x_i + x_j >= 1 for all (i,j) in E
    # x_i in {0, 1}
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Number of variables
        m = n
        
        # Objective: minimize sum(x_i)
        c = np.ones(m)
        
        # Constraints: x_i + x_j >= 1 for each edge
        # We need to build the A matrix for A_ub @ x <= b_ub or use LinearConstraint
        # LinearConstraint(A, lb, ub) imposes lb <= A @ x <= ub
        # We have x_i + x_j >= 1  =>  -x_i - x_j <= -1
        # So A_ub rows are -1 at i, -1 at j, 0 elsewhere. b_ub is -1.
        
        num_edges = len(edges)
        # For efficiency, we can use sparse matrices if available, but dense is okay for n~250
        # However, num_edges can be large. Let's use a sparse representation if possible.
        # scipy.sparse is available.
        from scipy.sparse import csr_matrix
        
        # Build A matrix for constraints
        # A is (num_edges, m)
        row_indices = []
        col_indices = []
        data = []
        
        for idx, (u, v) in enumerate(edges):
            row_indices.extend([idx, idx])
            col_indices.extend([u, v])
            data.extend([-1, -1])
        
        A = csr_matrix((data, (row_indices, col_indices)), shape=(num_edges, m))
        
        # Constraint: -x_i - x_j <= -1
        # LinearConstraint(A, lb=-np.inf, ub=-np.ones(num_edges))
        # Wait, A @ x <= -1
        # So lb is -inf, ub is -1 for each row.
        
        constraints = LinearConstraint(A, lb=-np.inf, ub=-np.ones(num_edges))
        
        # Bounds: x_i in [0, 1]
        integrality = np.ones(m) # All variables are integer
        bounds = Bounds(lb=0, ub=1)
        
        # Solve
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 80, "presolve": True}
        )
        
        if res.status == 0: # Optimal
            # The solution is the Minimum Vertex Cover
            # x_i = 1 means node i is in the cover (excluded from Independent Set)
            # x_i = 0 means node i is NOT in the cover (included in Independent Set)
            # Due to numerical issues, we might need to round or check values.
            # milp returns integer solutions if integrality is set.
            
            cover_mask = res.x
            # The independent set is the complement
            independent_set_indices = []
            for i in range(m):
                if cover_mask[i] < 0.5: # Not in cover
                    independent_set_indices.append(valid_nodes[i])
            
            if independent_set_indices:
                # Verify the solution
                valid = True
                for i in range(len(independent_set_indices)):
                    for j in range(i, len(independent_set_indices)):
                        s = independent_set_indices[i] + independent_set_indices[j]
                        r = math.isqrt(s)
                        if r * r == s:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return independent_set_indices
                else:
                    # Fallback if verification fails (should not happen)
                    pass
            
    except Exception as e:
        pass
    
    # Fallback: Optimized Greedy + Local Search + B&B
    # This part is kept from the original but optimized
    
    # Build bitmask conflicts for fast B&B
    conflict_masks = [0] * n
    for i in range(n):
        for j in conflicts[i]:
            conflict_masks[i] |= (1 << j)
    
    def greedy_mis():
        available = (1 << n) - 1
        chosen = 0
        while available:
            # Pick vertex with lowest degree in remaining subgraph
            best_v = -1
            best_deg = float('inf')
            temp = available
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                deg = bin(conflict_masks[v] & available).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lsb
            chosen |= (1 << best_v)
            available &= ~(1 << best_v | conflict_masks[best_v])
        return chosen
    
    best_mask = greedy_mis()
    best_size = bin(best_mask).count('1')
    
    # Local Search
    def local_search(initial_mask):
        current = initial_mask
        current_size = bin(current).count('1')
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding
            for v in range(n):
                if not (current & (1 << v)):
                    if not (conflict_masks[v] & current):
                        current |= (1 << v)
                        current_size += 1
                        improved = True
            # Try swaps (1-for-1)
            # Optimization: Only try swaps if we are stuck
            if not improved:
                for v in range(n):
                    if current & (1 << v):
                        current_without_v = current & ~(1 << v)
                        # Find a candidate u not in current that doesn't conflict with current_without_v
                        # And is "better" (e.g., lower index or just any)
                        # To save time, we just check if we can add any u
                        # Actually, a swap increases size only if we remove 1 and add 1? No, size stays same.
                        # We want to increase size. So we need to remove 1 and add 2? Or remove 1 and add 1 is not an improvement.
                        # The previous code tried swaps but `new_size` was calculated as `bin(current_without_v | (1 << u)).count('1')`
                        # which is `current_size`. So it never improved size.
                        # To improve size, we need to find a set of removals and additions that increases count.
                        # Simplest: Try to add any vertex. If none, try to remove one vertex and add two?
                        # Or just rely on B&B.
                        pass
        return current, bin(current).count('1')
    
    best_mask, best_size = local_search(best_mask)
    
    # B&B
    def bnb(available_mask, current_size, chosen_mask, best):
        if time.time() > deadline:
            return
        if current_size + bin(available_mask).count('1') <= best[0]:
            return
        if available_mask == 0:
            if current_size > best[0]:
                best[0] = current_size
                best[1] = chosen_mask
            return
        
        # Pick vertex with highest degree in available
        best_v = -1
        best_deg = -1
        temp = available_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            deg = bin(conflict_masks[v] & available_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            return
        
        v_bit = 1 << best_v
        
        # Include
        new_avail = available_mask & ~(v_bit | conflict_masks[best_v])
        bnb(new_avail, current_size + 1, chosen_mask | v_bit, best)
        
        # Exclude
        new_avail2 = available_mask & ~v_bit
        bnb(new_avail2, current_size, chosen_mask, best)
        
    best = [best_size, best_mask]
    all_mask = (1 << n) - 1
    bnb(all_mask, 0, 0, best)
    
    best_mask = best[1]
    result_nodes = []
    for i in range(n):
        if best_mask & (1 << i):
            result_nodes.append(valid_nodes[i])
            
    if not result_nodes:
        result_nodes = [valid_nodes[0]]
        
    return result_nodes
