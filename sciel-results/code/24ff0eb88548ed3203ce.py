import time
import numpy as np
import random
from math import gcd

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy: Voltage lift of Petersen graph with Min-Conflicts solver.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    def remaining():
        return deadline - time.time()
    
    # ---------------- Girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        """Verify no cycle shorter than g using non-backtracking BFS from every vertex."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        
        for s in range(n):
            if time.time() > time_limit:
                return -1
            
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            
            if min_cycle < g:
                return min_cycle
        
        return min_cycle
    
    # ---------------- Voltage lift of Petersen graph ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None
        
        # Petersen graph edges (10 vertices, 15 edges)
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
        ]
        
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Generate all non-backtracking closed walks of length < g
        max_L = g - 1
        walk_vectors = set()
        
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0]*15)]
            while stack:
                current, prev, length, counts = stack.pop()
                
                if length >= max_L:
                    continue
                
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1
                    
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    
                    if next_v == start_v and new_l >= 3:
                        walk_vectors.add(tuple(new_counts))
                    
                    stack.append((next_v, current, new_l, new_counts))
        
        if not walk_vectors:
            return None
            
        # Filter out zero vectors and normalize
        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c):
                continue
            first_non_zero = -1
            for i in range(15):
                if c[i] != 0:
                    first_non_zero = i
                    break
            if first_non_zero != -1:
                if c[first_non_zero] < 0:
                    c = tuple(-x for x in c)
                unique_constraints.add(c)
                
        C = np.array(list(unique_constraints), dtype=np.int64)
        if C.size == 0:
            return None
        
        num_constraints, num_vars = C.shape
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
            return True
        
        def min_conflicts_solve(m, max_iters=100000):
            """
            Min-conflicts local search.
            """
            # Initialize
            v = np.random.randint(1, m, size=num_vars).astype(np.int64)
            
            # Pre-compute rows of C for faster access?
            # We need to calculate conflicts for each variable.
            # Conflict for variable j is the number of constraints i where sum(C[i, :] * v) % m == 0.
            # Changing v[j] affects all constraints.
            
            def get_conflicts(v):
                # Returns array of conflict counts for each variable
                # This is O(num_constraints * num_vars) which might be slow.
                # Better: calculate total sum for each constraint, then for each variable,
                # check which constraints are violated.
                
                # Actually, we can just calculate the number of violated constraints total,
                # and for a specific variable, the number of violated constraints involving it.
                
                # Let's use a simpler approach:
                # 1. Calculate residues for all constraints.
                # 2. A constraint is violated if residue == 0.
                # 3. For variable j, it is "conflicted" if it is part of any violated constraint.
                # 4. We want to minimize the number of violated constraints.
                
                residues = np.mod(C @ v, m)
                violated = (residues == 0)
                
                # Count conflicts per variable
                # A variable j is in conflict if there exists i such that violated[i] and C[i,j] != 0
                # Actually, standard Min-Conflicts counts the number of constraints violated *by the variable*.
                # Since each constraint is a sum, if the sum is 0, all variables in that sum are "at fault".
                
                conflict_counts = np.zeros(num_vars, dtype=np.int64)
                for i in range(num_constraints):
                    if violated[i]:
                        # Add 1 to all non-zero variables in this constraint
                        # This is slow in Python.
                        pass
                
                # Optimized:
                # We can just return the total number of violated constraints as the cost.
                # And for variable selection, pick the one that appears in the most violated constraints.
                
                # Let's stick to total cost for simplicity and speed, and pick random variable among those with high involvement.
                return np.sum(violated)

            # For Min-Conflicts, we need to evaluate the cost of changing a single variable.
            # If we change v[j] from old to new, the change in cost is:
            # For each constraint i where C[i,j] != 0:
            #   old_res = (sum - old*C[i,j] + new*C[i,j]) % m
            #   if old_res == 0 and new_res != 0: cost decreases by 1
            #   if old_res != 0 and new_res == 0: cost increases by 1
            
            def calc_cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)
            
            best_v = v.copy()
            best_cost = calc_cost(v)
            
            if best_cost == 0:
                return v, 0
            
            # Precompute non-zero indices for each column (variable)
            var_constraints = [[] for _ in range(num_vars)]
            for j in range(num_vars):
                for i in range(num_constraints):
                    if C[i, j] != 0:
                        var_constraints[j].append(i)
            
            # Precompute coefficients
            # C is (constraints, vars)
            
            for it in range(max_iters):
                if remaining() < 5:
                    break
                
                # Pick a variable with maximum conflicts
                # To do this efficiently, we can just pick a random variable, 
                # or we can approximate. 
                # Let's pick a random variable to keep it simple and fast.
                # Or, we can pick the variable that appears in the most currently violated constraints.
                
                # Calculate current violated constraints
                residues = np.mod(C @ v, m)
                violated_idx = np.where(residues == 0)[0]
                
                if len(violated_idx) == 0:
                    return v, 0
                
                # Count how many violated constraints each variable is in
                # This is O(num_violated * avg_constraint_width)
                var_violation_count = np.zeros(num_vars, dtype=np.int64)
                for i in violated_idx:
                    row = C[i]
                    non_zero_vars = np.nonzero(row)[0]
                    var_violation_count[non_zero_vars] += 1
                
                # Pick variable with max violation count
                max_count = np.max(var_violation_count)
                if max_count == 0:
                    # Should not happen if there are violations
                    j = random.randint(0, num_vars - 1)
                else:
                    candidates = np.where(var_violation_count == max_count)[0]
                    j = candidates[random.randint(0, len(candidates) - 1)]
                
                # Try all values for j
                best_new_val = v[j]
                best_new_cost = best_cost
                best_delta = 0
                
                # We only need to check values that change the residue of the constraints involving j
                # Current residues for constraints involving j:
                involved_constraints = var_constraints[j]
                
                # Calculate current contribution of j to these constraints
                # res_i = (Sum - v[j]*C[i,j]) % m
                # New res_i = (res_i + (new_val - v[j])*C[i,j]) % m
                
                # To avoid O(m * num_constraints) per step, we can sample values?
                # No, m is small (53-100). O(100 * 50) = 5000 ops per step.
                # 10000 steps * 5000 = 50M ops. Might be too slow in Python.
                
                # Optimization:
                # Instead of trying all m values, we can try a few random ones?
                # Min-Conflicts usually tries all values for the selected variable.
                # Let's try all values but optimize the inner loop.
                
                old_val = v[j]
                
                # Pre-compute base sum for involved constraints without v[j]
                # base_i = (C[i, :] @ v - old_val * C[i, j]) % m
                # This is expensive to recompute.
                
                # Let's just use the full cost function for the best few candidates?
                # Or just pick a random new value and accept if it improves? (Hill Climbing)
                # Min-Conflicts: try all values, pick best.
                
                # Let's try a subset of values if m is large? m is small.
                # Let's try to speed up the cost delta calculation.
                
                # Get current residues for involved constraints
                # We have `residues` from above.
                # For each involved constraint i:
                #   old_res = residues[i]
                #   coeff = C[i, j]
                #   new_res = (old_res + (new_val - old_val) * coeff) % m
                
                # We want to minimize sum(new_res == 0).
                
                # Let's iterate new_val from 1 to m-1
                # This is O(m * len(involved_constraints))
                
                min_cost = float('inf')
                best_val = old_val
                
                # To speed up, we can break ties randomly.
                
                for new_val in range(1, m):
                    if new_val == old_val:
                        continue
                    
                    # Calculate change in cost
                    delta = 0
                    for i in involved_constraints:
                        coeff = C[i, j]
                        old_res = residues[i]
                        new_res = (old_res + (new_val - old_val) * coeff) % m
                        
                        if old_res == 0 and new_res != 0:
                            delta -= 1
                        elif old_res != 0 and new_res == 0:
                            delta += 1
                    
                    current_cost = best_cost + delta
                    if current_cost < min_cost:
                        min_cost = current_cost
                        best_val = new_val
                        # Early termination if cost is 0?
                        if min_cost == 0:
                            break
                
                if min_cost < best_cost:
                    v[j] = best_val
                    best_cost = min_cost
                    best_v = v.copy()
                    
                    if best_cost == 0:
                        return v, 0
                else:
                    # No improvement found. Random restart or random move?
                    # Standard Min-Conflicts: if stuck, random restart.
                    # But we can also try a random move to escape local minima.
                    if random.random() < 0.1:
                        v[j] = random.randint(1, m - 1)
                        best_cost = calc_cost(v)
                        if best_cost < 0: # Should not happen
                            pass
                    else:
                        # Random restart
                        v = np.random.randint(1, m, size=num_vars).astype(np.int64)
                        best_cost = calc_cost(v)
                        if best_cost == 0:
                            return v, 0
                        best_v = v.copy()
            
            return best_v, best_cost

        # Try primes in increasing order
        m = 53
        while m < 200 and remaining() > 10:
            if is_prime(m):
                # Try multiple restarts
                for restart in range(5):
                    if remaining() < 5:
                        break
                    
                    v, cost = min_conflicts_solve(m, max_iters=5000)
                    
                    if cost == 0:
                        # Construct graph
                        n = 10 * m
                        edges = []
                        for u, v_base in pet_edges:
                            e_idx = edge_index[(u, v_base)]
                            vol = v[e_idx]
                            for layer in range(m):
                                u_lift = u * m + layer
                                v_lift = v_base * m + (layer + vol) % m
                                if u_lift < v_lift:
                                    edges.append((u_lift, v_lift))
                                else:
                                    edges.append((v_lift, u_lift))
                        
                        # Verify
                        gc = check_girth(n, edges, g, deadline - 2)
                        if gc >= g:
                            return (n, edges)
            
            m += 1
        
        return None
    
    # ---------------- LUW fallback ----------------
    def luw_fallback(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)
        
        start = (0, (None,) + (0,) * d)
        index, order, edges_luw = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_luw.append((index[vertex], index[other]))
        return len(order), edges_luw

    # ---------------- Main ----------------
    res = voltage_lift_petersen()
    if res is not None:
        return res
    
    return luw_fallback(k, g)

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
