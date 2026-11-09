import time
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    def solve_instance(n, d, seed, time_limit):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Precompute adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Precompute conflict sets as lists of indices for faster iteration
        # conflict_sets[i] contains all j != i such that edge i and edge j conflict
        conflict_sets = [set() for _ in range(m)]
        for i in range(m):
            ui, vi = edges[i]
            for j in range(i + 1, m):
                uj, vj = edges[j]
                # Check if they share a vertex
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict_sets[i].add(j)
                    conflict_sets[j].add(i)
                else:
                    # Check if they are joined by an edge
                    if ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)

        # Convert to lists for faster access
        conflict_lists = [list(s) for s in conflict_sets]

        def greedy_dsatur():
            colors = [-1] * m
            uncolored = set(range(m))
            while uncolored:
                best_i, best_score = None, (-1, -1)
                for i in uncolored:
                    neighbor_colors = set()
                    for j in conflict_lists[i]:
                        if colors[j] != -1:
                            neighbor_colors.add(colors[j])
                    sat = len(neighbor_colors)
                    deg = sum(1 for j in conflict_lists[i] if j in uncolored)
                    score = (sat, deg)
                    if score > best_score:
                        best_score = score
                        best_i = i
                i = best_i
                taken = set()
                for j in conflict_lists[i]:
                    if colors[j] != -1:
                        taken.add(colors[j])
                c = 0
                while c in taken:
                    c += 1
                colors[i] = c
                uncolored.remove(i)
            return colors

        init_colors = greedy_dsatur()
        best_colors = init_colors[:]
        best_num_colors = max(init_colors) + 1 if init_colors else 0

        # Simulated Annealing
        # Objective: Minimize number of colors used.
        # State: list of colors for each edge.
        # Move: Pick a random edge, change its color to a random valid color (or any color, penalize conflicts).
        # Actually, a better move for minimization is to try to recolor an edge to an existing color if possible,
        # or to a new color if it reduces the total palette?
        # Let's use a standard SA approach:
        # Energy = Total number of distinct colors used + penalty for conflicts (if we allow invalid states temporarily? No, let's keep valid).
        # If we keep valid, we can only move to valid states.
        # Move: Pick edge i. Try to change color to c' != color[i].
        # If c' is not in the palette, it increases palette size (bad).
        # If c' is in the palette, it might decrease it if no other edge uses color[i] anymore.
        # Let's define Cost = Number of distinct colors.
        # We want to minimize this.
        
        # To allow exploration, we can allow "invalid" moves? No, strong edge coloring requires validity.
        # So we restrict moves to valid recolorings.
        # A valid recoloring of edge i to color c is one where no neighbor j of i has color c.
        
        # We also want to try to reduce the number of colors.
        # Strategy:
        # 1. Identify "removable" colors: colors used by only one edge? No, that doesn't help directly unless we merge.
        # 2. Try to recolor edges to merge color classes.
        
        # Let's use a different heuristic:
        # Local Search:
        # 1. Try to recolor an edge to a color that is already used, if valid. This doesn't change count.
        # 2. Try to recolor an edge to a NEW color? No, that increases count.
        # 3. The key is to eliminate a color.
        #    If a color k is used by a set of edges S_k.
        #    If we can recolor all edges in S_k to other colors, we eliminate k.
        #    This is a "reassignment" problem.
        
        # Alternative: Simulated Annealing on the number of colors.
        # Start with K = best_num_colors.
        # Try to find a valid coloring with K-1 colors.
        # If successful, set K = K-1 and repeat.
        # This is a feasibility search.
        
        # Let's implement a randomized local search for feasibility with K colors.
        def try_feasibility(K, time_budget):
            if K <= 0:
                return None
            
            # Initialize with random assignment from 0..K-1
            # To improve start, we can use the current best_colors and map them to 0..K-1 if possible?
            # Or just random.
            # Random is better for diversity in SA.
            
            # Let's use a "repair" strategy or just random restarts.
            # Given the time limit, we do many short runs.
            
            best_local = None
            best_local_cost = float('inf')
            
            # SA Parameters
            T0 = 1.0
            Tend = 0.01
            alpha = 0.99 # Cooling rate
            # We don't know the number of steps in advance, so we run until time runs out or convergence.
            
            # We need a cost function that is 0 if valid, >0 if invalid.
            # But we want to minimize colors. So we fix K and search for validity.
            # Cost = Number of conflicts.
            
            def count_conflicts(colors):
                conflicts = 0
                for i in range(m):
                    ci = colors[i]
                    for j in conflict_lists[i]:
                        if j > i: # Count each pair once
                            if colors[j] == ci:
                                conflicts += 1
                return conflicts
            
            # We want to find a state with 0 conflicts.
            
            # Multiple restarts
            num_restarts = 50
            time_per_restart = time_budget / num_restarts
            
            for r in range(num_restarts):
                if time.time() > start_time + time_limit:
                    break
                
                # Initialize
                # Use greedy DSATUR but limited to K colors?
                # If DSATUR uses > K colors, it's a good sign it's hard, but random might work.
                # Let's try a "Randomized DSATUR" or just Random.
                # Random is fast.
                colors = [random.randint(0, K - 1) for _ in range(m)]
                
                # Quick check
                if count_conflicts(colors) == 0:
                    return colors
                
                # SA Loop
                T = T0
                step = 0
                max_steps = 2000 # Cap steps per restart to ensure progress
                
                while step < max_steps and T > Tend:
                    step += 1
                    if time.time() > start_time + time_limit:
                        return best_local if best_local is not None else None
                    
                    # Pick random edge i
                    i = random.randint(0, m - 1)
                    old_c = colors[i]
                    
                    # Pick new color c
                    # To be efficient, pick a color that is "promising".
                    # If we pick random, we might not improve.
                    # Let's pick a color that minimizes new conflicts for this edge.
                    # But that's expensive.
                    # Let's just pick a random color != old_c.
                    new_c = random.randint(0, K - 1)
                    if new_c == old_c:
                        new_c = (new_c + 1) % K
                    
                    # Calculate delta cost
                    # Delta = (Conflicts involving i with new_c) - (Conflicts involving i with old_c)
                    # Note: Conflicts are pairwise.
                    # old_conflicts_i = sum(1 for j in conflict_lists[i] if colors[j] == old_c)
                    # new_conflicts_i = sum(1 for j in conflict_lists[i] if colors[j] == new_c)
                    
                    old_conf = 0
                    new_conf = 0
                    for j in conflict_lists[i]:
                        if colors[j] == old_c:
                            old_conf += 1
                        if colors[j] == new_c:
                            new_conf += 1
                    
                    delta = new_conf - old_conf
                    
                    if delta <= 0 or random.random() < np.exp(-delta / T):
                        colors[i] = new_c
                        
                        # Check if solved
                        if delta == 0 and old_conf == 0:
                            # We might have solved it if total conflicts was 0?
                            # We don't track total conflicts efficiently here.
                            # Let's just periodically check.
                            pass
                        
                        # Periodic full check
                        if step % 100 == 0:
                            total_conf = count_conflicts(colors)
                            if total_conf == 0:
                                return colors
                            # Track best local
                            if total_conf < best_local_cost:
                                best_local_cost = total_conf
                                best_local = colors[:]
                    
                    T *= alpha
                
                # End of restart
                total_conf = count_conflicts(colors)
                if total_conf < best_local_cost:
                    best_local_cost = total_conf
                    best_local = colors[:]
                
                if best_local_cost == 0:
                    return best_local

            return best_local if best_local_cost == 0 else None

        # Strategy:
        # 1. Get initial K from DSATUR.
        # 2. Try to reduce K by 1 using SA feasibility search.
        # 3. If successful, repeat.
        # 4. Stop when time runs out or K reaches a lower bound (e.g., chromatic index of line graph? No, induced matching number).
        
        current_K = best_num_colors
        # Lower bound: At least 1.
        # We can also use a simple lower bound: max degree of conflict graph?
        # Conflict graph chromatic number is a lower bound.
        # But computing that is hard.
        
        # Allocate time to each K reduction attempt.
        # We have time_limit.
        # Let's give 5 seconds to each attempt.
        
        while current_K > 1:
            if time.time() > start_time + time_limit - 2.0:
                break
            
            time_for_attempt = min(5.0, start_time + time_limit - 2.0 - time.time())
            if time_for_attempt < 0.5:
                break
            
            result = try_feasibility(current_K - 1, time_for_attempt)
            
            if result is not None:
                # Valid coloring with current_K - 1
                current_K -= 1
                best_colors = result
                # Update best
            else:
                # Failed to find coloring with current_K - 1
                # It might be infeasible, or we just didn't find it.
                # If we have time, we could try again?
                # But if it's infeasible, we stop.
                # If it's just hard, we might stop and return current_K.
                # Given the small graphs, if SA fails quickly, it's likely infeasible.
                # Let's stop to be safe and return the valid coloring we have.
                break

        return best_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        remaining = deadline - time.time()
        if remaining <= 0:
            # Fallback: simple greedy
            edges = instance(n, d, seed)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colors = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colors.items() if set(f) & near}
                c = 0
                while c in taken:
                    c += 1
                colors[e] = c
            results.append([colors[e] for e in edges])
            continue
        
        # Allocate time
        # 3 instances.
        # Give each 40s?
        time_limit = min(40.0, remaining - 5.0)
        if len(results) == 2: # Last instance
            time_limit = remaining - 2.0
        
        result = solve_instance(n, d, seed, time_limit)
        results.append(result)
        
        if time.time() > deadline - 2.0:
            pass

    return results
