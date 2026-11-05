import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    def build_conflict_graph(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            for x in adj[u]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            for x in adj[v]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        conflicts[i].add(edge_to_idx[e2])
            conflicts[i].discard(i)
        return [list(c) for c in conflicts]

    def greedy_color_order(order, conflicts):
        m = len(order)
        color_of = [-1] * m
        for i in order:
            forbidden = set()
            for j in conflicts[i]:
                if color_of[j] != -1:
                    forbidden.add(color_of[j])
            c = 0
            while c in forbidden:
                c += 1
            color_of[i] = c
        return color_of, max(color_of) + 1

    def branch_and_bound(m, conflicts, initial_bound, deadline, rng):
        # conflicts is a list of lists of ints
        # We will maintain a list of sets of available colors for each uncolored edge?
        # No, we will use a simple backtracking with a global color array.
        # To make it fast, we use a specific branching heuristic.
        
        # Order edges by descending degree? Or dynamic?
        # Dynamic DSATUR: pick uncolored edge with fewest available colors.
        
        colors = [-1] * m
        best = initial_bound
        
        # Pre-compute adjacency for quick forbidden check?
        # conflicts[i] is list of neighbors.
        
        # We need a way to quickly find available colors.
        # Since max colors is small (<= 50), we can just check 0..best-1.
        
        # To speed up, we can maintain a count of how many edges have each color?
        # No, we need per-edge forbidden colors.
        
        # Let's use a recursive function.
        # To avoid recursion depth issues, m is small (~100-200).
        
        def dfs(colored_count):
            nonlocal best
            if time.time() > deadline:
                return False # Timeout signal
            
            if colored_count == m:
                # All colored
                # The number of colors used is max(colors) + 1
                # But we are coloring sequentially, so we can track max used.
                # Actually, if we just want to minimize max color, we can check at the end.
                # However, B&B usually prunes if current_max >= best.
                # We need to track current_max.
                return True # Found a solution (we'll update best outside or here)
            
            # Find uncolored edge with fewest available colors (DSATUR-like)
            min_avail_count = float('inf')
            best_edge = -1
            best_avail = None
            
            # Optimization: Only check edges that are "frontier"? 
            # No, checking all uncolored is O(m * K). m=200, K=50. 10k ops.
            # Recursion depth 200. 200 * 10k = 2M ops per level? No.
            # It's O(m*K) per node. Number of nodes is the bottleneck.
            
            for i in range(m):
                if colors[i] == -1:
                    # Calculate available colors
                    forbidden = set()
                    for j in conflicts[i]:
                        if colors[j] != -1:
                            forbidden.add(colors[j])
                    
                    # Count available colors < best
                    # Actually, we only care about colors < best.
                    # If no color < best is available, this branch is dead (prune).
                    avail = []
                    for c in range(best):
                        if c not in forbidden:
                            avail.append(c)
                    
                    if not avail:
                        return False # Prune: no valid color for this edge within current best
                    
                    if len(avail) < min_avail_count:
                        min_avail_count = len(avail)
                        best_edge = i
                        best_avail = avail
                        
                        if min_avail_count == 1:
                            break # Can't do better than 1
            
            if best_edge == -1:
                # Should not happen if colored_count < m
                return False
            
            # Branch on best_edge
            # Sort available colors? Maybe try smallest first to find good solutions fast?
            # Or random?
            # Smallest first usually finds a valid coloring quickly, reducing 'best'.
            rng.shuffle(best_avail)
            
            for c in best_avail:
                colors[best_edge] = c
                if dfs(colored_count + 1):
                    # Success
                    # Check if we improved best
                    # The current coloring uses colors up to max(colors).
                    # But we restricted to < best.
                    # So max color used is <= best - 1.
                    # We can update best to max(colors) + 1?
                    # Actually, we know a solution exists with < best colors.
                    # So best can be reduced to max(color used in this solution) + 1.
                    # But we don't track max efficiently.
                    # However, since we only tried colors < best, the max color used is < best.
                    # So we can set best = best - 1?
                    # Not necessarily. We might have used color 0, 1, 2... best-2.
                    # We should compute max.
                    # But computing max is O(m).
                    # Let's just return True and let the caller handle it?
                    # No, we need to update 'best' to prune future branches.
                    
                    # Let's compute max color used.
                    cur_max = -1
                    for col in colors:
                        if col > cur_max:
                            cur_max = col
                    new_best = cur_max + 1
                    if new_best < best:
                        best = new_best
                    
                    # If we found a solution with 1 color (impossible for non-empty), stop?
                    # If best == 1, we can't do better.
                    if best == 1:
                        return True
                    # Continue searching for even better?
                    # No, if we found a solution, we can backtrack and try other branches
                    # to see if we can find one with fewer colors.
                    # But wait, if we found a solution with K colors, we update best to K.
                    # Then we backtrack. The parent will try other colors.
                    # This is standard B&B.
                else:
                    # Timeout or failure
                    if time.time() > deadline:
                        return False
                colors[best_edge] = -1
            
            return True # Finished all branches for this edge (didn't timeout)

        # Initial call
        # We need to handle the case where we find a solution.
        # The dfs returns True if it finished or found a solution.
        # It returns False on timeout.
        
        # To make it work correctly:
        # If dfs returns True, it means we explored the space (partially or fully) without timeout.
        # 'best' holds the best found so far.
        
        # We need to be careful: if we find a solution, we update best.
        # Then we continue searching.
        
        dfs(0)
        
        return best, colors if colors.count(-1) == 0 else None

    def solve_instance(edges, adj, seed_offset):
        m = len(edges)
        if m == 0:
            return [], 0
        
        conflicts = build_conflict_graph(edges, adj)
        
        # 1. Get a good upper bound using Greedy
        # Try a few orderings
        best_ub = float('inf')
        best_init = None
        
        order1 = list(range(m))
        order1.sort(key=lambda i: len(conflicts[i]), reverse=True)
        c1, s1 = greedy_color_order(order1, conflicts)
        if s1 < best_ub:
            best_ub = s1
            best_init = c1[:]
            
        order2 = list(range(m))
        order2.sort(key=lambda i: len(conflicts[i]))
        c2, s2 = greedy_color_order(order2, conflicts)
        if s2 < best_ub:
            best_ub = s2
            best_init = c2[:]
            
        rng = random.Random(seed_offset)
        for _ in range(10):
            order3 = list(range(m))
            rng.shuffle(order3)
            c3, s3 = greedy_color_order(order3, conflicts)
            if s3 < best_ub:
                best_ub = s3
                best_init = c3[:]
        
        # 2. Branch and Bound
        # We have a deadline.
        # Allocate time based on m?
        # m=40, d=4 -> ~80 edges.
        # m=60, d=5 -> ~150 edges.
        # m=80, d=6 -> ~240 edges.
        
        # B&B is exponential. We need to be careful.
        # Let's give it a chunk of time.
        
        # We'll run B&B with the current best_ub as the initial bound.
        # If it finds a better one, great.
        
        # To avoid getting stuck on one instance, we'll time slice.
        # But the task gives 150s total.
        # Let's just run B&B until deadline or success.
        
        # We need a RNG for B&B shuffling.
        rng_bb = random.Random(seed_offset + 1000)
        
        # We pass the deadline to B&B.
        # But we have 3 instances.
        # We should allocate time fairly.
        
        return best_ub, best_init

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        graphs.append((edges, adj))
        
    results = []
    
    # We will process each instance.
    # We can't easily time-slice a single B&B run without modifying it.
    # So we will run a "quick" B&B for each, with a time limit.
    
    time_per_instance = (deadline - time.time()) / len(graphs)
    
    for idx, (edges, adj) in enumerate(graphs):
        m = len(edges)
        if m == 0:
            results.append([])
            continue
            
        conflicts = build_conflict_graph(edges, adj)
        
        # Get initial bound
        best_ub, best_init = solve_instance(edges, adj, idx)
        
        # Now run B&B with a time limit
        # We need to modify B&B to stop after time_per_instance
        # But B&B is recursive. We need a global stop flag or deadline check.
        
        # Let's implement a specific B&B for this instance with a deadline.
        
        deadline_bb = time.time() + time_per_instance * 0.9 # Safety
        
        # We'll use the B&B function defined above, but it needs to be adapted
        # to return the best coloring found.
        
        # The previous B&B sketch didn't return the coloring if it timed out mid-way.
        # It returns best and colors (if complete).
        # We need a version that keeps the best coloring found.
        
        def bb_search(m, conflicts, initial_bound, deadline, rng):
            colors = [-1] * m
            best = initial_bound
            best_colors = None
            
            # We need to track the best coloring found.
            # In the DFS, when we reach a leaf (all colored), we update best_colors.
            
            def dfs(colored_count):
                nonlocal best, best_colors
                if time.time() > deadline:
                    return False
                
                if colored_count == m:
                    cur_max = max(colors)
                    if cur_max + 1 < best:
                        best = cur_max + 1
                        best_colors = colors[:]
                        if best == 1:
                            return True
                    return True
                
                min_avail_count = float('inf')
                best_edge = -1
                best_avail = None
                
                for i in range(m):
                    if colors[i] == -1:
                        forbidden = set()
                        for j in conflicts[i]:
                            if colors[j] != -1:
                                forbidden.add(colors[j])
                        
                        avail = [c for c in range(best) if c not in forbidden]
                        
                        if not avail:
                            return False
                        
                        if len(avail) < min_avail_count:
                            min_avail_count = len(avail)
                            best_edge = i
                            best_avail = avail
                            if min_avail_count == 1:
                                break
                
                if best_edge == -1:
                    return False
                
                rng.shuffle(best_avail)
                
                for c in best_avail:
                    colors[best_edge] = c
                    if dfs(colored_count + 1):
                        if time.time() > deadline:
                            return False
                    else:
                        if time.time() > deadline:
                            return False
                    colors[best_edge] = -1
                
                return True
            
            dfs(0)
            
            # If best_colors is None, it means we didn't find a better solution than initial_bound
            # Or we timed out before finding any solution?
            # Wait, if we timed out, we might have found a solution.
            # If we didn't find any solution better than initial_bound, we should return the initial one.
            # But we don't have the initial coloring here.
            # We should pass it in or return None and handle outside.
            
            return best, best_colors

        # Run B&B
        rng_bb = random.Random(idx + 1000)
        bb_best, bb_colors = bb_search(m, conflicts, best_ub, deadline_bb, rng_bb)
        
        if bb_colors is not None:
            results.append(bb_colors)
        else:
            results.append(best_init)
            
        # Update time_per_instance for remaining instances?
        # No, we just use the calculated slice.
        
    return results

