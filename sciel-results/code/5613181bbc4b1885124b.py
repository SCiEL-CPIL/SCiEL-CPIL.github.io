"""Rewritten program for strong edge colourings of bounded-degree graphs."""
# EVOLVE-BLOCK-START
import random
import time
import sys

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_instance(edges, n, time_limit=35):
        E = len(edges)
        if E == 0:
            return []
        
        # Build adjacency sets for original graph
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute conflict graph
        # conflict[i] is a set of indices j such that edge i and edge j conflict
        conflict = [set() for _ in range(E)]
        
        for i in range(E):
            u1, v1 = edges[i]
            for j in range(i + 1, E):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    if u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]:
                        conflict[i].add(j)
                        conflict[j].add(i)

        # Greedy coloring to get an upper bound
        def greedy_coloring():
            # Order by conflict degree descending
            order = list(range(E))
            order.sort(key=lambda x: len(conflict[x]), reverse=True)
            colors = [-1] * E
            for idx in order:
                used_colors = set()
                for j in conflict[idx]:
                    if colors[j] != -1:
                        used_colors.add(colors[j])
                c = 0
                while c in used_colors:
                    c += 1
                colors[idx] = c
            return colors

        best_valid_colors = greedy_coloring()
        best_k = max(best_valid_colors) + 1 if E > 0 else 0
        
        start_time = time.time()
        
        # Try to improve using Backtracking CSP
        # We try k from a lower bound to best_k - 1
        # Lower bound: max degree in conflict graph + 1 is an upper bound on chromatic number? No.
        # Max degree + 1 is an upper bound.
        # Lower bound: 1. Or clique number.
        # Let's try k from max(1, best_k - 5) to best_k - 1?
        # Or just iterate from 1 to best_k - 1?
        # If best_k is 20, trying 1..19 is 19 runs.
        # Each run has a time limit.
        
        # To be efficient, we can use a binary search if we had a decision oracle, 
        # but backtracking is not monotonic in time (harder k is slower).
        # Usually, k close to chi is hardest. k >> chi is easy.
        # So we should try k close to the lower bound first?
        # If k is too small, backtracking will fail quickly (dead ends).
        # If k is feasible, it might take time to find the solution.
        
        # Strategy:
        # 1. Estimate a better LB. 
        #    The size of the maximum independent set in the conflict graph? No.
        #    The chromatic number of the conflict graph is the answer.
        #    Let's just try k from max(1, int(0.8 * best_k)) to best_k - 1?
        #    No, if the greedy is bad, best_k might be 30 while chi is 20.
        #    Let's try k from 1 to best_k - 1, but with increasing time limits?
        #    Or just a fixed small time limit for each k, prioritizing smaller k?
        
        # Let's allocate time_per_k = time_limit / best_k
        # And iterate k from 1 to best_k - 1.
        
        # Optimization: If we find a solution for k, we can stop?
        # No, we want the MINIMUM k.
        # So if we find a solution for k=10, we should check if k=9 works?
        # Yes. So we must iterate from 1 upwards.
        # If we find a solution for k, we update best_k and best_valid_colors.
        # We continue to k+1? No, if k works, we have a valid solution.
        # We want to know if k-1 works. But we already tried k-1 in the previous iteration.
        # If k-1 failed, it might be due to time limit.
        # So we should continue?
        # If we find a solution for k=10, we know chi <= 10.
        # We tried 1..9. If any of them succeeded, we would have found it earlier.
        # So if we reach k=10 and succeed, it means 1..9 failed (either impossible or timeout).
        # If they failed due to timeout, they might still be feasible.
        # But we don't have time to re-check them with more time.
        # So we accept k=10 as the best found.
        # We can break the loop because any k > 10 is worse.
        
        time_per_k = time_limit / max(1, best_k)
        
        for k in range(1, best_k):
            if time.time() - start_time > time_limit:
                break
                
            # Run Backtracking for k colors
            # Time limit for this specific k
            t_limit = time_per_k
            
            # CSP Variables: 0..E-1
            # Domains: set of colors
            # Constraints: if i and j conflict, colors[i] != colors[j]
            
            # Heuristic: Order variables by degree (most constrained first)
            var_order = sorted(range(E), key=lambda x: len(conflict[x]), reverse=True)
            
            # We will use a recursive backtracking with forward checking.
            # To avoid recursion depth issues and allow time checking, we use an explicit stack or just recursion with a counter.
            # Given E <= 200, recursion depth is manageable.
            
            assignment = [-1] * E
            domains = [set(range(k)) for _ in range(E)]
            
            # Pre-compute neighbors for each variable in the order of assignment?
            # No, just use the conflict sets.
            
            nodes_visited = 0
            max_nodes = 100000 # Limit nodes per k to prevent hang
            
            def backtrack():
                nonlocal nodes_visited
                nodes_visited += 1
                if nodes_visited > max_nodes:
                    return False
                if time.time() - start_time > start_time + t_limit:
                    return False
                
                # Find unassigned variable with smallest domain (MRV)
                min_idx = -1
                min_size = float('inf')
                for i in range(E):
                    if assignment[i] == -1:
                        if len(domains[i]) < min_size:
                            min_size = len(domains[i])
                            min_idx = i
                            if min_size == 0:
                                return False # Dead end
                            
                if min_idx == -1:
                    return True # All assigned
                
                # Try each color in domain
                # Heuristic: Try colors that appear most often in neighbors? (LCV)
                # For now, just iterate
                for c in sorted(domains[min_idx]):
                    # Check if c is consistent with assigned neighbors
                    consistent = True
                    for j in conflict[min_idx]:
                        if assignment[j] == c:
                            consistent = False
                            break
                    if not consistent:
                        continue
                        
                    # Assign
                    assignment[min_idx] = c
                    domains[min_idx].clear()
                    
                    # Propagate
                    backtrack_ok = True
                    # For each unassigned neighbor, remove c from domain
                    # If domain becomes empty, fail
                    # We need to restore domains on backtrack
                    # This is complex. Let's use a simpler check without full propagation first.
                    # Just check consistency with assigned neighbors.
                    # This is "Backtracking with Forward Checking" but without maintaining domains explicitly in a way that's easy to undo.
                    # Let's stick to simple backtracking: check against assigned neighbors.
                    
                    if backtrack():
                        return True
                    else:
                        # If we used propagation, we'd undo it here.
                        # Since we didn't propagate, we just unassign.
                        pass
                    assignment[min_idx] = -1
                    domains[min_idx].add(c) # Restore? No, we didn't modify neighbors.
                    # Wait, if we don't propagate, domains don't change.
                    # So we just need to restore assignment[min_idx].
                    
                return False

            # The simple backtracking above is slow.
            # Let's use a better heuristic: DSATUR.
            # Actually, let's just use the simple one but with a better variable ordering (static).
            # The MRV dynamic selection is O(E) per step, which is fine.
            
            # Let's optimize the inner loop.
            # Instead of checking all neighbors every time, we can maintain a count of used colors per neighbor?
            # Too complex.
            
            # Let's try running the backtracking.
            # Note: The `backtrack` function above has a bug in logic (it doesn't properly handle the domain restoration if we were to propagate, but we aren't).
            # It just checks `assignment[j] == c`. This is correct for simple backtracking.
            
            # However, `domains[min_idx]` is not updated to reflect that `c` is taken?
            # We don't need to update domains if we just check against `assignment`.
            # But MRV uses `len(domains[i])`. If we don't update domains, MRV is not "Minimum Remaining Values" in the CSP sense (which considers constraints).
            # It's just "Minimum Initial Domain".
            # This is weak.
            
            # Let's implement a slightly better backtracking:
            # 1. Order variables statically by degree.
            # 2. Assign in that order.
            # 3. Check consistency.
            
            var_order = sorted(range(E), key=lambda x: len(conflict[x]), reverse=True)
            
            def backtrack_static(idx):
                nonlocal nodes_visited
                nodes_visited += 1
                if nodes_visited > max_nodes:
                    return False
                if time.time() - start_time > start_time + t_limit:
                    return False
                
                if idx == E:
                    return True
                
                i = var_order[idx]
                
                for c in range(k):
                    # Check if c is used by any assigned neighbor
                    conflict_found = False
                    for j in conflict[i]:
                        if assignment[j] == c:
                            conflict_found = True
                            break
                    if not conflict_found:
                        assignment[i] = c
                        if backtrack_static(idx + 1):
                            return True
                        assignment[i] = -1
                        
                return False

            # Run it
            assignment = [-1] * E
            if backtrack_static(0):
                # Found a valid coloring with k colors (or fewer)
                # Compact colors
                used_colors = set(assignment)
                num_used = len(used_colors)
                if num_used < best_k:
                    best_k = num_used
                    mapping = {c: idx for idx, c in enumerate(sorted(used_colors))}
                    best_valid_colors = [mapping[c] for c in assignment]
                # Since we iterate k upwards, if we found a solution for k,
                # it means all k' < k failed (or timed out).
                # We can break because we want the minimum k.
                # If k' < k was feasible but timed out, we missed it.
                # But we can't help that.
                # And any k'' > k is worse.
                break

        return best_valid_colors

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colors = solve_instance(edges, n, time_limit=35)
        out.append(colors)
    return out
# EVOLVE-BLOCK-END
