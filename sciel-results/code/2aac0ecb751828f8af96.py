import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def generate_instance(n, d, seed):
        """Generate the specific bounded degree graph."""
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edges:
                continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, n):
        """
        Build conflict masks for each edge.
        mask[i] is an integer where bit j is set if edge i and edge j conflict.
        Conflict: share a vertex OR are joined by an edge in G.
        """
        m = len(edges)
        if m == 0:
            return []

        # Map vertex to list of incident edge indices
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)

        # Adjacency sets for original graph
        orig_adj = [set() for _ in range(n)]
        for u, v in edges:
            orig_adj[u].add(v)
            orig_adj[v].add(u)

        masks = [0] * m

        for i in range(m):
            u1, v1 = edges[i]
            mask = 0
            # 1. Edges sharing a vertex
            for j in vertex_to_edges[u1]:
                if j != i:
                    mask |= (1 << j)
            for j in vertex_to_edges[v1]:
                if j != i:
                    mask |= (1 << j)

            # 2. Edges joined by an edge
            # For each neighbor of u1 and v1, check incident edges
            # We need to check if there is an edge connecting {u1, v1} to {u2, v2}
            # This is equivalent to checking if any neighbor of u1 or v1 is incident to edge j
            # But we must be careful not to double count or include self.
            
            # Let's iterate over neighbors of u1 and v1
            # If w is neighbor of u1, then any edge incident to w conflicts with i (unless it shares vertex with i, already handled)
            # Actually, the definition is: edges e1=(u,v) and e2=(x,y) conflict if they share a vertex OR there is an edge in G connecting a vertex of e1 to a vertex of e2.
            # So if w is adjacent to u1, and edge j is incident to w, then e1 and ej are joined by edge (u1, w).
            
            for w in orig_adj[u1]:
                for j in vertex_to_edges[w]:
                    if j != i:
                        mask |= (1 << j)
            for w in orig_adj[v1]:
                for j in vertex_to_edges[w]:
                    if j != i:
                        mask |= (1 << j)
            
            masks[i] = mask

        return masks

    def dsatur_greedy(masks, m):
        """
        DSATUR heuristic using bitmasks.
        """
        if m == 0:
            return [], 0

        color = [-1] * m
        sat_mask = [0] * m  # bitmask of colors used by neighbors
        num_colors = 0
        degrees = [bin(masks[i]).count('1') for i in range(m)]

        for _ in range(m):
            best_v = -1
            best_sat = -1
            best_deg = -1

            for i in range(m):
                if color[i] == -1:
                    sat = bin(sat_mask[i]).count('1')
                    if sat > best_sat or (sat == best_sat and degrees[i] > best_deg):
                        best_sat = sat
                        best_deg = degrees[i]
                        best_v = i

            if best_v == -1:
                break

            used = sat_mask[best_v]
            c = 0
            while (used >> c) & 1:
                c += 1

            color[best_v] = c
            if c >= num_colors:
                num_colors = c + 1

            # Update neighbors
            mask = masks[best_v]
            # Iterate set bits in mask
            # Optimization: precompute list of neighbors?
            # For now, simple loop
            for j in range(m):
                if (mask >> j) & 1:
                    if color[j] == -1:
                        sat_mask[j] |= (1 << c)

        return color, num_colors

    def try_remove_color(color, masks, target_color, m):
        """
        Try to eliminate target_color by reassigning its edges to other colors < target_color.
        Returns True if successful, mutating color.
        """
        target_edges = [i for i in range(m) if color[i] == target_color]
        if not target_edges:
            return True

        # Sort target edges by number of available colors (excluding target_color)
        def count_avail(i):
            used = set()
            mask = masks[i]
            for j in range(m):
                if (mask >> j) & 1:
                    if color[j] != -1 and color[j] != target_color:
                        used.add(color[j])
            return len([c for c in range(target_color) if c not in used])

        target_edges.sort(key=count_avail)

        for i in target_edges:
            used = set()
            mask = masks[i]
            for j in range(m):
                if (mask >> j) & 1:
                    if color[j] != -1 and color[j] != target_color:
                        used.add(color[j])
            
            avail = [c for c in range(target_color) if c not in used]
            if not avail:
                return False
            color[i] = random.choice(avail)

        return True

    def solve_instance(n, d, seed, time_budget):
        edges = generate_instance(n, d, seed)
        m = len(edges)

        if m == 0:
            return []

        masks = build_conflict_bitmasks(edges, n)
        
        # Initial coloring
        color, num_colors = dsatur_greedy(masks, m)
        
        best_color = color[:]
        best_nc = num_colors
        
        # Local Search: Simulated Annealing / Hill Climbing hybrid
        # Objective: Minimize number of colors.
        
        T = 1.0
        cooling_rate = 0.99
        min_T = 0.01
        rng = random.Random(seed)
        
        # Precompute neighbor lists for faster iteration in local search
        # Bitmask iteration is slow in Python loops. List is better.
        neighbors = [list((i for i in range(m) if (masks[j] >> i) & 1)) for j in range(m)]
        
        # Maintain color counts
        color_counts = [0] * best_nc
        for c in color:
            color_counts[c] += 1
            
        end_time = time.time() + time_budget
        no_improve_steps = 0
        max_no_improve = 5000
        
        while time.time() < end_time:
            if T < min_T:
                T = min_T
            
            # Pick a random vertex
            i = rng.randrange(m)
            old_c = color[i]
            
            # Find available colors
            used = set(color[j] for j in neighbors[i])
            avail = [c for c in range(best_nc) if c not in used]
            
            if not avail:
                # Cannot move, try swap? Or just skip
                # For simplicity, skip
                continue
                
            if old_c in avail:
                if len(avail) == 1:
                    continue
                avail.remove(old_c)
                
            new_c = rng.choice(avail)
            
            # Apply move
            color[i] = new_c
            color_counts[old_c] -= 1
            color_counts[new_c] += 1
            
            # Check if we freed a color
            old_nc = best_nc
            # If old_c was the highest color and its count becomes 0, we might reduce nc
            # But we only reduce nc if we can re-color it? No, nc is the number of colors used.
            # If color_counts[old_c] == 0 and old_c == best_nc - 1, then effective nc is best_nc - 1.
            # But we don't immediately reduce best_nc, we just note it.
            # Actually, let's define current_nc as max index with count > 0.
            
            # To keep it simple, we only accept moves that don't increase the max color used,
            # and we try to remove the top color periodically.
            
            # Calculate delta in "sum of colors" or just "max color"?
            # The objective is to minimize the number of colors.
            # A move that reduces the count of the highest color is good.
            # A move that increases the count of a lower color is neutral/good if it allows reduction later.
            
            # Simple heuristic: Accept if new_c < old_c (reduces sum, likely reduces max eventually)
            # Or accept with probability based on T if new_c > old_c.
            
            if new_c < old_c:
                # Improvement in sum, likely good
                pass
            elif new_c > old_c:
                # Worse in sum
                # Accept with prob exp(-(new_c - old_c) / T)
                delta = new_c - old_c
                if rng.random() > (1.0 / (1.0 + delta * T)): # Simplified acceptance
                    # Revert
                    color[i] = old_c
                    color_counts[old_c] += 1
                    color_counts[new_c] -= 1
                    T *= cooling_rate
                    no_improve_steps += 1
                    continue
            else:
                # Same color index? Impossible since we removed old_c from avail
                pass
            
            # Update best if we reduced the number of distinct colors used?
            # Let's check if we can reduce best_nc.
            # If the highest color (best_nc - 1) has count 0, we can reduce best_nc.
            # But we need to ensure no vertex has color >= best_nc.
            
            # Check if we can eliminate the current top color
            # This is expensive. Do it periodically.
            
            if no_improve_steps > max_no_improve:
                # Try to eliminate top color
                trial = color[:]
                trial_counts = color_counts[:]
                # We need to simulate removing best_nc - 1
                # But color array has indices up to best_nc - 1.
                # If we remove it, we re-color those vertices to < best_nc - 1.
                
                if try_remove_color(trial, masks, best_nc - 1, m):
                    color = trial
                    best_nc -= 1
                    # Recalculate counts
                    color_counts = [0] * best_nc
                    for c in color:
                        color_counts[c] += 1
                    best_color = color[:]
                    T = max(T, 0.5)
                    no_improve_steps = 0
                else:
                    no_improve_steps = 0 # Reset counter to prevent infinite loop of failed removals?
                    # Actually, if it fails, we are stuck. Maybe randomize more?
            
            # Periodically try to remove top color even if not stuck
            if rng.random() < 0.05:
                trial = color[:]
                if try_remove_color(trial, masks, best_nc - 1, m):
                    color = trial
                    best_nc -= 1
                    color_counts = [0] * best_nc
                    for c in color:
                        color_counts[c] += 1
                    best_color = color[:]
                    T = max(T, 0.5)
                    no_improve_steps = 0
            
            T *= cooling_rate
            no_improve_steps += 1
            
        # Final polish
        for _ in range(50):
            if time.time() > end_time - 1:
                break
            trial = color[:]
            if try_remove_color(trial, masks, best_nc - 1, m):
                color = trial
                best_nc -= 1
                best_color = color[:]
            else:
                break
                
        return best_color

    # Allocation
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    # Weights: more edges = harder
    weights = [40*4, 60*5, 80*6]
    total_w = sum(weights)
    
    out = []
    for i, (n, d, seed) in enumerate(specs):
        remaining = deadline - time.time() - 2
        if remaining < 5:
            out.append([])
            continue
            
        # Allocate time proportional to weight, but ensure minimum time for each
        # Last instance gets all remaining
        if i == len(specs) - 1:
            budget = remaining
        else:
            budget = min(total_budget * weights[i] / total_w, remaining - 5)
            # Ensure at least 5s for remaining instances
            min_budget = 5 * (len(specs) - i - 1)
            budget = min(budget, remaining - min_budget)
            budget = max(budget, 1)
            
        res = solve_instance(n, d, seed, budget)
        out.append(res)
        
    while len(out) < 3:
        out.append([])
        
    return out

# EVOLVE-BLOCK-END
