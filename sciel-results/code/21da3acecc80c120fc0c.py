import time
import random
from collections import defaultdict
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Generate graph with max degree d
        # The prompt implies bounded_degree_graph(n, d, seed)
        # We replicate the generation logic from the seed program
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, adj):
        """
        Build conflict list for strong edge coloring.
        Two edges conflict if they share a vertex or are joined by an edge.
        """
        m = len(edges)
        # Map edge index to endpoints
        # adj: vertex -> set of neighbors
        # We need to check for each pair of edges if they conflict.
        # m is small (<= 240), so O(m^2) is fine.
        conflict = [set() for _ in range(m)]
        
        # Pre-compute adjacency sets for fast lookup
        # adj is already a dict of sets
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                # 1. Share vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # 2. Joined by an edge
                # Check if any endpoint of e1 is adjacent to any endpoint of e2
                # u1-u2, u1-v2, v1-u2, v1-v2
                if (u2 in adj[u1] or v2 in adj[u1] or
                    u2 in adj[v1] or v2 in adj[v1]):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict

    def greedy_coloring(conflict, m):
        """
        Greedy coloring using DSATUR heuristic.
        Returns color list and num colors used.
        """
        color = [-1] * m
        # degree in conflict graph
        deg = [len(conflict[i]) for i in range(m)]
        
        # Pre-compute neighbors for faster access?
        # conflict is list of sets.
        
        for _ in range(m):
            best_edge = -1
            best_sat = -1
            best_deg = -1
            
            for i in range(m):
                if color[i] != -1:
                    continue
                # Count distinct colors in neighbors
                sat_colors = set()
                for j in conflict[i]:
                    if color[j] != -1:
                        sat_colors.add(color[j])
                sat = len(sat_colors)
                
                if sat > best_sat or (sat == best_sat and deg[i] > best_deg):
                    best_sat = sat
                    best_deg = deg[i]
                    best_edge = i
            
            if best_edge == -1:
                break
            
            # Find smallest available color
            used = set()
            for j in conflict[best_edge]:
                if color[j] != -1:
                    used.add(color[j])
            
            c = 0
            while c in used:
                c += 1
            
            color[best_edge] = c
        
        num_colors = max(color) + 1 if m > 0 else 0
        return color, num_colors

    def local_search_reduce(color, conflict, m, max_iters=5000):
        """
        Try to reduce the number of colors by reassigning edges from the highest color class.
        """
        best = color[:]
        best_nc = max(best) + 1 if best else 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            cur_nc = max(best) + 1 if best else 0
            if cur_nc <= 1:
                break
            
            target = cur_nc - 1
            target_edges = [i for i in range(m) if best[i] == target]
            
            if not target_edges:
                continue
            
            # Try to reassign all target edges to other colors
            # Sort target edges by degree (easiest first)
            target_edges.sort(key=lambda i: len(conflict[i]))
            
            success = True
            trial = best[:]
            
            for idx in target_edges:
                # Find available colors for idx
                neighbor_colors = set()
                for j in conflict[idx]:
                    neighbor_colors.add(trial[j])
                
                found = False
                # Try colors 0 to target-1
                for c in range(target):
                    if c not in neighbor_colors:
                        trial[idx] = c
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if success:
                best = trial
                # Loop continues to try to remove the new highest color (which is now target-1)
            else:
                # If failed, try random swap?
                # For now, just break or try a different order?
                # To avoid getting stuck, we can shuffle target_edges order or try different color assignments
                # Simple improvement: try all permutations of small target sets? No, too slow.
                # Just break for this iteration and let the outer loop handle it?
                # Actually, if we can't remove the top color, we are stuck at this count.
                # We should try to modify the coloring to make it possible.
                # Pick a random edge from target, try to move it to a random valid color, then retry.
                if target_edges:
                    idx = random.choice(target_edges)
                    neighbor_colors = set()
                    for j in conflict[idx]:
                        neighbor_colors.add(best[j])
                    available = [c for c in range(target) if c not in neighbor_colors]
                    if available:
                        best[idx] = random.choice(available)
                        # Retry removal
                        trial = best[:]
                        # Re-sort?
                        target_edges.sort(key=lambda i: len(conflict[i]))
                        success = True
                        for idx2 in target_edges:
                            if trial[idx2] != target:
                                continue
                            neighbor_colors2 = set()
                            for j in conflict[idx2]:
                                neighbor_colors2.add(trial[j])
                            found2 = False
                            for c in range(target):
                                if c not in neighbor_colors2:
                                    trial[idx2] = c
                                    found2 = True
                                    break
                            if not found2:
                                success = False
                                break
                        if success:
                            best = trial
                        else:
                            best[idx] = target # revert
                            break
                    else:
                        break

        return best

    def backtrack_coloring(conflict, m, k, time_limit=5.0):
        """
        Try to color the graph with exactly k colors using backtracking with forward checking.
        Returns color list if successful, else None.
        """
        if m == 0:
            return []
        
        # Order variables by degree (most constrained first)
        # We will use a dynamic MRV heuristic.
        
        # Initial domains: all edges can take any color 0..k-1
        domains = [set(range(k)) for _ in range(m)]
        
        # Pre-compute conflict sets as lists for iteration
        conflict_list = [list(conflict[i]) for i in range(m)]
        
        # Heuristic ordering: static order by degree
        # Dynamic MRV is better but expensive to maintain.
        # Given small m, we can recalculate MRV at each step.
        
        color = [-1] * m
        assigned = [False] * m
        
        # To speed up, we can use a simple static ordering first:
        # Sort edges by degree in conflict graph
        order = sorted(range(m), key=lambda i: len(conflict[i]), reverse=True)
        
        # However, MRV is much stronger.
        # Let's implement a simple backtracking with MRV and Forward Checking.
        
        # To avoid deep recursion issues and overhead, we use an explicit stack or recursion with limit.
        # m is up to 240. Recursion depth 240 is fine in Python if optimized, but might be slow.
        
        # We'll use a recursive function with a node counter and time check.
        nodes_visited = 0
        start_bt_time = time.time()
        
        # Forward checking:
        # When assigning color c to edge i:
        # 1. Remove c from domain of all neighbors j of i.
        # 2. If any domain becomes empty, fail.
        
        # To make this efficient, we maintain domains as sets.
        
        def solve(idx, current_depth):
            nonlocal nodes_visited
            nodes_visited += 1
            
            if time.time() - start_bt_time > time_limit:
                return False
            
            if idx == m:
                return True
            
            # Find MRV variable among unassigned
            # Since we are using a static 'order' list, we just pick the next unassigned in that order?
            # No, MRV requires scanning all unassigned.
            # For m=240, scanning 240 items at each of 240 steps is 240^2 = 57k ops.
            # 57k * branching factor... might be okay.
            
            best_var = -1
            min_domain_size = k + 1
            
            for v in order:
                if assigned[v]:
                    continue
                if len(domains[v]) < min_domain_size:
                    min_domain_size = len(domains[v])
                    best_var = v
                    if min_domain_size == 1:
                        break
            
            if best_var == -1:
                return True # All assigned
            
            if len(domains[best_var]) == 0:
                return False
            
            # Try colors in domains[best_var]
            # Value Ordering: Least Constraining Value
            # We can just iterate sorted domains
            
            for c in sorted(domains[best_var]):
                # Assign c to best_var
                color[best_var] = c
                assigned[best_var] = True
                
                # Forward Checking
                saved_domains = {}
                valid = True
                for nb in conflict_list[best_var]:
                    if assigned[nb]:
                        continue
                    if c in domains[nb]:
                        if nb not in saved_domains:
                            saved_domains[nb] = domains[nb].copy()
                        domains[nb].discard(c)
                        if len(domains[nb]) == 0:
                            valid = False
                            break
                
                if valid:
                    if solve(idx + 1, current_depth + 1):
                        return True
                
                # Backtrack
                for nb, dom in saved_domains.items():
                    domains[nb] = dom
                
                assigned[best_var] = False
                color[best_var] = -1
        
        # Initial call
        # We need to handle the 'idx' correctly. Since we pick MRV dynamically, idx is just a counter or we can ignore it and check if all assigned.
        # Let's modify solve to not take idx, but check if all assigned.
        
        def solve_mrv():
            nonlocal nodes_visited
            nodes_visited += 1
            if time.time() - start_bt_time > time_limit:
                return False
            
            # Find MRV
            best_var = -1
            min_domain_size = k + 1
            all_assigned = True
            for v in range(m):
                if not assigned[v]:
                    all_assigned = False
                    if len(domains[v]) < min_domain_size:
                        min_domain_size = len(domains[v])
                        best_var = v
                        if min_domain_size == 1:
                            break
            
            if all_assigned:
                return True
            
            if best_var == -1 or len(domains[best_var]) == 0:
                return False
            
            for c in sorted(domains[best_var]):
                color[best_var] = c
                assigned[best_var] = True
                
                saved_domains = {}
                valid = True
                for nb in conflict_list[best_var]:
                    if assigned[nb]:
                        continue
                    if c in domains[nb]:
                        if nb not in saved_domains:
                            saved_domains[nb] = domains[nb].copy()
                        domains[nb].discard(c)
                        if len(domains[nb]) == 0:
                            valid = False
                            break
                
                if valid:
                    if solve_mrv():
                        return True
                
                for nb, dom in saved_domains.items():
                    domains[nb] = dom
                
                assigned[best_var] = False
                color[best_var] = -1
            
            return False

        if solve_mrv():
            return color
        return None

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict(edges, adj)
        
        # 1. Greedy Initial
        color, nc = greedy_coloring(conflict, m)
        
        # 2. Local Search to reduce
        # Allocate some time to local search
        local_time_limit = 5.0
        start_ls = time.time()
        while time.time() - start_ls < local_time_limit:
            new_color = local_search_reduce(color, conflict, m, max_iters=100)
            if max(new_color) < max(color):
                color = new_color
            else:
                # If no improvement in a while, break?
                # local_search_reduce already tries to reduce. If it fails, it returns same.
                # We can break if no improvement in last few iterations.
                # For simplicity, we just run it a few times.
                break # Or continue? Let's break if no improvement.
        
        current_nc = max(color) + 1
        
        # 3. Backtracking to try to reduce further
        # Try k from current_nc - 1 down to d
        # Allocate remaining time to backtracking
        bt_time_limit = max(0, deadline - time.time() - 5.0)
        
        # If we have time, try to prove optimality or find better
        for k in range(current_nc - 1, d - 1, -1):
            if time.time() > deadline - 5:
                break
            
            # Allocate a portion of time for this k
            # If k is close to current, it's more likely to succeed.
            # Give more time to larger k? No, we want to find the smallest k.
            # So we start from current_nc - 1 (easiest to find if possible) and go down.
            # Wait, if we find a solution for k, we can stop and try k-1.
            # So we iterate downwards.
            
            # Limit time per k?
            # Let's just run backtrack with a time limit.
            # If it finds a solution, we update color and nc, and continue to next k.
            # If it fails, we stop trying smaller k? No, failure for k doesn't mean k-1 fails.
            # But usually, if k is hard, k-1 is harder.
            # We can just try them in order.
            
            bt_limit = min(bt_time_limit, 10.0) # Cap per k
            bt_color = backtrack_coloring(conflict, m, k, time_limit=bt_limit)
            
            if bt_color is not None:
                color = bt_color
                current_nc = k
                # Update bt_time_limit? No, total time is managed by deadline.
                # Continue to try k-1
            else:
                # If backtracking failed for k, it's unlikely to succeed for k-1 quickly?
                # Actually, if k is the minimum, k-1 will fail.
                # If k is not the minimum, k-1 might succeed.
                # But if backtracking timed out, we don't know.
                # We should probably stop if we've spent too much time.
                # Let's break if we've spent > 50% of remaining time on this k and failed.
                # Or just break to be safe?
                # Let's break. The local search already did the heavy lifting.
                break
        
        return color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 30:
            break
        
        n, d, seed = spec
        out.append(solve_instance(n, d, seed))
    
    while len(out) < 3:
        out.append([])
    
    return out
