"""Strong edge colourings of three fixed bounded-degree graphs."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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

    def build_conflicts(edges, adj):
        """Build conflict neighbor lists and bitmasks."""
        m = len(edges)
        if m == 0:
            return [], []
        
        conflict_bits = [0] * m
        conflict_list = [[] for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflict_bits[i] |= (1 << j)
                    conflict_bits[j] |= (1 << i)
                    conflict_list[i].append(j)
                    conflict_list[j].append(i)
                    continue
                if (u2 in adj.get(u1, set()) or v2 in adj.get(u1, set()) or
                    u2 in adj.get(v1, set()) or v2 in adj.get(v1, set())):
                    conflict_bits[i] |= (1 << j)
                    conflict_bits[j] |= (1 << i)
                    conflict_list[i].append(j)
                    conflict_list[j].append(i)
        
        return conflict_bits, conflict_list

    def dsatur_greedy_fast(conflict_list, m, rng):
        """
        Fast DSATUR using incremental saturation tracking.
        Returns (color_array, num_colors).
        """
        color = [-1] * m
        # sat_mask[i] = bitmask of colors used by colored neighbors of i
        sat_mask = [0] * m
        colored_count = [0] * m  # how many neighbors of i are colored
        num_colors = 0
        
        for _ in range(m):
            if time.time() > deadline - 20:
                break
            best_edge = -1
            best_sat = -1
            best_deg = -1
            
            # Find uncolored edge with max saturation, tie-break by degree
            for i in range(m):
                if color[i] != -1:
                    continue
                sat = bin(sat_mask[i]).count('1')
                deg = len(conflict_list[i])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_edge = i
            
            if best_edge == -1:
                break
            
            # Find smallest available color
            used = sat_mask[best_edge]
            c = 0
            while (used >> c) & 1:
                c += 1
            
            color[best_edge] = c
            if c >= num_colors:
                num_colors = c + 1
            
            # Update saturation masks of uncolored neighbors
            for j in conflict_list[best_edge]:
                if color[j] == -1:
                    sat_mask[j] |= (1 << c)
                    colored_count[j] += 1
        
        return color, num_colors

    def try_remove_color_fast(color, conflict_list, target_color, m):
        """
        Try to eliminate target_color by reassigning its edges to other colors.
        Returns True if successful (mutates color in place).
        """
        target_edges = [i for i in range(m) if color[i] == target_color]
        if not target_edges:
            return True
        
        # Sort by fewest conflicts first
        target_edges.sort(key=lambda i: len(conflict_list[i]))
        
        for idx in target_edges:
            # Find available colors
            neighbor_colors = set()
            for j in conflict_list[idx]:
                if color[j] != -1:
                    neighbor_colors.add(color[j])
            
            found = False
            max_c = max(color) if any(c != -1 for c in color) else 0
            for c in range(max_c + 1):
                if c == target_color:
                    continue
                if c not in neighbor_colors:
                    color[idx] = c
                    found = True
                    break
            
            if not found:
                return False
        
        return True

    def kempe_swap(color, conflict_list, target_color, m, rng):
        """
        Try a Kempe-chain style swap: find a color c such that swapping
        target_color and c in a connected component allows removal of target_color.
        """
        target_edges = [i for i in range(m) if color[i] == target_color]
        if not target_edges:
            return False
        
        max_c = max(color) if any(c != -1 for c in color) else 0
        
        for c in range(max_c + 1):
            if c == target_color:
                continue
            
            # For each target edge, check if it can take color c
            for idx in target_edges:
                neighbor_colors = set()
                for j in conflict_list[idx]:
                    if color[j] != -1:
                        neighbor_colors.add(color[j])
                if c not in neighbor_colors:
                    # idx can take color c directly
                    # Try: change idx to c, then try to remove target_color
                    trial = color[:]
                    trial[idx] = c
                    if try_remove_color_fast(trial, conflict_list, target_color, m):
                        color[:] = trial
                        return True
        
        return False

    def local_search(color, conflict_list, m, max_iters=3000):
        """
        Local search: repeatedly try to remove the highest color.
        """
        best = color[:]
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            cur_nc = max(best) + 1 if best else 0
            if cur_nc <= 1:
                break
            
            target = cur_nc - 1
            target_edges = [i for i in range(m) if best[i] == target]
            
            if not target_edges:
                continue
            
            # Try direct reassignment
            trial = best[:]
            if try_remove_color_fast(trial, conflict_list, target, m):
                best = trial
                continue
            
            # Try Kempe-style swap
            if kempe_swap(best, conflict_list, target, m, random):
                continue
            
            # Random perturbation
            idx = random.choice(target_edges)
            neighbor_colors = set()
            for j in conflict_list[idx]:
                if best[j] != -1:
                    neighbor_colors.add(best[j])
            available = [c for c in range(cur_nc) if c != target and c not in neighbor_colors]
            if available:
                best[idx] = random.choice(available)
                trial = best[:]
                if try_remove_color_fast(trial, conflict_list, target, m):
                    best = trial
        
        return best

    def polish(color, conflict_list, m):
        """
        Final polish: multiple passes trying to remove colors from high to low.
        """
        best = color[:]
        
        for _ in range(8):
            if time.time() > deadline - 5:
                break
            
            cur_nc = max(best) + 1 if best else 0
            
            for target in range(cur_nc - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                target_edges = [i for i in range(m) if best[i] == target]
                if not target_edges:
                    continue
                
                trial = best[:]
                if try_remove_color_fast(trial, conflict_list, target, m):
                    best = trial
                    break
        
        return best

    def solve_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict_bits, conflict_list = build_conflicts(edges, adj)
        
        # Multiple restarts with different random seeds
        best_color = None
        best_nc = float('inf')
        
        num_restarts = max(3, min(10, int(time_budget / 15)))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                break
            
            rng = random.Random(seed + restart * 1000)
            color, nc = dsatur_greedy_fast(conflict_list, m, rng)
            
            if nc < best_nc:
                best_nc = nc
                best_color = color[:]
            
            # Local search
            color = local_search(color, conflict_list, m, max_iters=2000)
            nc = max(color) + 1 if color else 0
            
            if nc < best_nc:
                best_nc = nc
                best_color = color[:]
        
        # Polish the best
        if best_color is not None:
            best_color = polish(best_color, conflict_list, m)
        
        return best_color if best_color is not None else []

    # Allocate time budget per instance
    total_budget = deadline - start_time - 5  # 5s safety
    # Weight by expected difficulty (more edges = harder)
    weights = [40*4, 60*5, 80*6]  # proportional to n*d
    total_w = sum(weights)
    budgets = [total_budget * w / total_w for w in weights]
    
    out = []
    for i, spec in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        n, d, seed = spec
        remaining = deadline - time.time() - 5
        budget = min(budgets[i], remaining)
        if budget < 5:
            out.append([])
            continue
        out.append(solve_instance(n, d, seed, budget))
    
    while len(out) < 3:
        out.append([])
    
    return out
# EVOLVE-BLOCK-END
