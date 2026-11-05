"""Strong edge colourings of three fixed bounded-degree graphs using DSATUR + local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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

    def build_conflict(edges, adj):
        """Build conflict bitmask for each edge: edge i conflicts with edge j if they share a vertex
        or are joined by an edge (i.e., one endpoint of i is adjacent to one endpoint of j)."""
        n_edges = len(edges)
        conflict = [0] * n_edges
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                if ui == uj or ui == vj or vi == uj or vi == vj:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                elif ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        return conflict

    def dsatur_color(edges, conflict, n_edges):
        """DSATUR heuristic for strong edge coloring.
        At each step, pick the uncolored edge with the highest saturation (number of different
        colors used by its colored neighbors), breaking ties by degree (number of colored neighbors).
        Assign the smallest available color."""
        colour = [-1] * n_edges
        # For each edge, track which colors are used by its colored neighbors
        colour_used = [set() for _ in range(n_edges)]
        
        for step in range(n_edges):
            # Find the uncolored edge with highest saturation, then highest degree
            best_idx = -1
            best_sat = -1
            best_deg = -1
            for i in range(n_edges):
                if colour[i] == -1:
                    sat = len(colour_used[i])
                    deg = bin(conflict[i] & 0).count('1')  # placeholder
                    # Actually count colored neighbors
                    colored_neighbours = 0
                    for j in range(n_edges):
                        if colour[j] != -1 and ((conflict[i] >> j) & 1):
                            colored_neighbours += 1
                    if sat > best_sat or (sat == best_sat and colored_neighbours > best_deg):
                        best_sat = sat
                        best_deg = colored_neighbours
                        best_idx = i
            
            if best_idx == -1:
                break
            
            # Assign smallest available color
            c = 0
            while c in colour_used[best_idx]:
                c += 1
            colour[best_idx] = c
            
            # Update colour_used for all uncolored neighbors
            for j in range(n_edges):
                if colour[j] == -1 and ((conflict[best_idx] >> j) & 1):
                    colour_used[j].add(c)
        
        max_color = max(colour) if colour else 0
        return colour, max_color + 1

    def local_search_improve(edges, conflict, colour, num_colors, n_edges, time_budget, rng):
        """Local search: try to reduce num_colors by moving edges from the highest color class."""
        start = time.time()
        current_max = num_colors - 1
        improved = True
        
        while improved and time.time() - start < time_budget:
            improved = False
            if current_max <= 0:
                break
            
            high_edges = [i for i in range(n_edges) if colour[i] == current_max]
            if not high_edges:
                current_max -= 1
                continue
            
            rng.shuffle(high_edges)
            
            for e_idx in high_edges:
                # Try to move e_idx to a lower color
                taken = set()
                for j in range(n_edges):
                    if j != e_idx and ((conflict[e_idx] >> j) & 1):
                        if colour[j] < current_max:
                            taken.add(colour[j])
                
                new_c = 0
                while new_c < current_max and new_c in taken:
                    new_c += 1
                
                if new_c < current_max:
                    colour[e_idx] = new_c
                    improved = True
                    # Check if current_max is still used
                    if not any(colour[i] == current_max for i in range(n_edges)):
                        current_max -= 1
                    break
        
        return colour, current_max + 1

    def try_color_reduction(edges, conflict, colour, num_colors, n_edges, time_budget, rng):
        """Try to reduce the number of colors by one through systematic reassignment."""
        start = time.time()
        if num_colors <= 1:
            return colour, num_colors
        
        high_color = num_colors - 1
        high_edges = [i for i in range(n_edges) if colour[i] == high_color]
        
        if not high_edges:
            return colour, num_colors - 1
        
        # Try to reassign all high_color edges to lower colors
        for target_order in range(num_colors):
            if time.time() - start > time_budget:
                break
            
            new_colour = colour[:]
            success = True
            shuffled_high = high_edges[:]
            rng.shuffle(shuffled_high)
            
            for e_idx in shuffled_high:
                taken = set()
                for j in range(n_edges):
                    if j != e_idx and ((conflict[e_idx] >> j) & 1):
                        if new_colour[j] < high_color:
                            taken.add(new_colour[j])
                
                new_c = 0
                while new_c < high_color and new_c in taken:
                    new_c += 1
                
                if new_c < high_color:
                    new_colour[e_idx] = new_c
                else:
                    success = False
                    break
            
            if success:
                return new_colour, high_color
        
        # Try swap-based moves
        for target_color in range(high_color):
            if time.time() - start > time_budget:
                break
            other_edges = [i for i in range(n_edges) if colour[i] == target_color]
            if not other_edges:
                continue
            
            rng.shuffle(other_edges)
            for h_idx in high_edges:
                if time.time() - start > time_budget:
                    break
                for o_idx in other_edges:
                    # Check if h_idx can go to target_color (ignoring o_idx which will leave)
                    taken_h = set()
                    for j in range(n_edges):
                        if j != h_idx and j != o_idx and ((conflict[h_idx] >> j) & 1):
                            if colour[j] == target_color:
                                taken_h.add(target_color)
                    
                    # Check if o_idx can go to high_color (ignoring h_idx which will leave)
                    taken_o = set()
                    for j in range(n_edges):
                        if j != o_idx and j != h_idx and ((conflict[o_idx] >> j) & 1):
                            if colour[j] == high_color:
                                taken_o.add(high_color)
                    
                    if target_color not in taken_h and high_color not in taken_o:
                        new_colour = colour[:]
                        new_colour[h_idx] = target_color
                        new_colour[o_idx] = high_color
                        return new_colour, num_colors
        
        return colour, num_colors

    def solve_instance(edges, adj, time_budget):
        n_edges = len(edges)
        conflict = build_conflict(edges, adj)
        
        # DSATUR initial coloring
        colour, num_colors = dsatur_color(edges, conflict, n_edges)
        
        rng = random.Random(42)
        remaining = time_budget - (time.time() - start_time)
        
        # Local search improvement
        if remaining > 5:
            budget = min(remaining * 0.4, 30.0)
            colour, num_colors = local_search_improve(edges, conflict, colour, num_colors, n_edges, budget, rng)
        
        # Try to reduce colors
        remaining = time_budget - (time.time() - start_time)
        if remaining > 3:
            budget = min(remaining * 0.5, 40.0)
            while time.time() - start_time < time_budget - 3:
                old = num_colors
                colour, num_colors = try_color_reduction(edges, conflict, colour, num_colors, n_edges, 
                                                          min(10.0, time_budget - (time.time() - start_time)), rng)
                if num_colors < old:
                    colour, num_colors = local_search_improve(edges, conflict, colour, num_colors, n_edges, 5.0, rng)
                else:
                    break
        
        return colour, num_colors

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    # Allocate time proportionally to problem size
    total_budget = deadline - time.time() - 5
    weights = [40, 60, 80]
    total_w = sum(weights)
    budgets = [total_budget * w / total_w for w in weights]
    
    for idx, spec in enumerate(specs):
        if time.time() > deadline - 3:
            break
        
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining = deadline - time.time() - 3
        budget = min(budgets[idx], remaining)
        
        if budget > 2:
            colour, num_colors = solve_instance(edges, adj, budget)
        else:
            # Fallback: simple greedy
            conflict = build_conflict(edges, adj)
            n_edges = len(edges)
            colour = [0] * n_edges
            for i in range(n_edges):
                taken = set()
                for j in range(i):
                    if ((conflict[i] >> j) & 1):
                        taken.add(colour[j])
                c = 0
                while c in taken:
                    c += 1
                colour[i] = c
        
        out.append(colour)
    
    # Fallback for any missing instances
    while len(out) < 3:
        spec = specs[len(out)]
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict(edges, adj)
        n_edges = len(edges)
        colour = [0] * n_edges
        for i in range(n_edges):
            taken = set()
            for j in range(i):
                if ((conflict[i] >> j) & 1):
                    taken.add(colour[j])
            c = 0
            while c in taken:
                c += 1
            colour[i] = c
        out.append(colour)
    
    return out
# EVOLVE-BLOCK-END
