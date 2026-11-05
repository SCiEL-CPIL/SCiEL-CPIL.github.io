"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

    def build_conflict_bitmasks(edges, adj):
        """Build bitmask conflict representation for fast color checking."""
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

    def greedy_color_bitmask(edges, conflict, order=None):
        """Greedy coloring using bitmask conflict representation."""
        n_edges = len(edges)
        if order is None:
            order = list(range(n_edges))
        
        colour = [0] * n_edges
        color_edges = {}
        max_color = -1
        
        for idx in order:
            taken_mask = 0
            for j in range(n_edges):
                if (conflict[idx] >> j) & 1:
                    taken_mask |= (1 << colour[j])
            
            c = 0
            while (taken_mask >> c) & 1:
                c += 1
            
            colour[idx] = c
            if c not in color_edges:
                color_edges[c] = 0
            color_edges[c] |= (1 << idx)
            
            if c > max_color:
                max_color = c
        
        return colour, max_color + 1

    def local_improve_bitmask(edges, conflict, colour, max_colors, max_iter=10000, rng=None):
        """Try to reduce max color by reassigning edges using bitmasks."""
        if rng is None:
            rng = random.Random()
        
        n_edges = len(edges)
        current_max = max_colors - 1
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            if current_max <= 0:
                break
            
            high_edges = [i for i in range(n_edges) if colour[i] == current_max]
            if not high_edges:
                current_max -= 1
                continue
            
            rng.shuffle(high_edges)
            
            for e_idx in high_edges:
                taken_mask = 0
                for j in range(n_edges):
                    if j != e_idx and ((conflict[e_idx] >> j) & 1):
                        taken_mask |= (1 << colour[j])
                
                new_c = 0
                while (taken_mask >> new_c) & 1:
                    new_c += 1
                
                if new_c < colour[e_idx]:
                    colour[e_idx] = new_c
                    if current_max == max(colour):
                        current_max = max(colour)
                    improved = True
                    break
        
        return colour, current_max + 1

    def max_independent_set_bnb(candidates, conflict, time_budget=2.0):
        """Branch and bound for maximum independent set in conflict graph.
        candidates: list of edge indices that are mutually compatible (no conflicts among them)
        conflict: bitmask conflict array
        Returns a subset of candidates that forms a maximum independent set."""
        if not candidates:
            return []
        
        n = len(candidates)
        # Build local conflict: for each candidate i, which other candidates conflict
        local_conflict = [0] * n
        for i in range(n):
            ci = candidates[i]
            for j in range(i + 1, n):
                cj = candidates[j]
                if (conflict[ci] >> cj) & 1:
                    local_conflict[i] |= (1 << j)
                    local_conflict[j] |= (1 << i)
        
        best_set = [0] * n
        best_size = 0
        
        def bnb(idx, current_set, current_size):
            nonlocal best_set, best_size
            if time.time() > time_budget:
                return
            if idx == n:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
                return
            
            # Upper bound: current_size + (n - idx)
            if current_size + (n - idx) <= best_size:
                return
            
            # Option 1: exclude candidate idx
            bnb(idx + 1, current_set, current_size)
            
            # Option 2: include candidate idx if compatible
            if not (current_set[idx]):
                compatible = True
                for j in range(idx):
                    if current_set[j] and (local_conflict[idx] >> j) & 1:
                        compatible = False
                        break
                if compatible:
                    current_set[idx] = 1
                    bnb(idx + 1, current_set, current_size + 1)
                    current_set[idx] = 0
        
        bnb(0, [0] * n, 0)
        
        return [candidates[i] for i in range(n) if best_set[i]]

    def try_merge_colors(edges, conflict, colour, max_colors, rng=None):
        """Try to merge two color classes by reassigning edges, with swap moves."""
        if rng is None:
            rng = random.Random()
        
        n_edges = len(edges)
        if max_colors <= 1:
            return colour, max_colors
        
        high_color = max_colors - 1
        high_edges = [i for i in range(n_edges) if colour[i] == high_color]
        
        if not high_edges:
            return colour, max_colors - 1
        
        # Try simple reassignment first
        for target_color in range(high_color):
            success = True
            new_colour = colour[:]
            
            for e_idx in high_edges:
                taken_mask = 0
                for j in range(n_edges):
                    if j != e_idx and ((conflict[e_idx] >> j) & 1):
                        taken_mask |= (1 << new_colour[j])
                
                new_c = 0
                while new_c < high_color and (taken_mask >> new_c) & 1:
                    new_c += 1
                
                if new_c < high_color:
                    new_colour[e_idx] = new_c
                else:
                    success = False
                    break
            
            if success:
                return new_colour, high_color
        
        # Try swap-based moves: swap an edge from high_color with an edge from another color
        for target_color in range(high_color):
            other_edges = [i for i in range(n_edges) if colour[i] == target_color]
            if not other_edges:
                continue
            
            for h_idx in high_edges:
                for o_idx in other_edges:
                    # Check if swapping h_idx to target_color and o_idx to high_color works
                    # First, check h_idx can go to target_color
                    taken_h = 0
                    for j in range(n_edges):
                        if j != h_idx and j != o_idx and ((conflict[h_idx] >> j) & 1):
                            if colour[j] == target_color:
                                taken_h |= (1 << target_color)
                        elif j == o_idx:
                            # o_idx will move to high_color, so it doesn't block target_color
                            pass
                    
                    # Check o_idx can go to high_color
                    taken_o = 0
                    for j in range(n_edges):
                        if j != o_idx and j != h_idx and ((conflict[o_idx] >> j) & 1):
                            if colour[j] == high_color:
                                taken_o |= (1 << high_color)
                    
                    # If both can swap, do it
                    if not ((taken_h >> target_color) & 1) and not ((taken_o >> high_color) & 1):
                        new_colour = colour[:]
                        new_colour[h_idx] = target_color
                        new_colour[o_idx] = high_color
                        return new_colour, max_colors
        
        return colour, max_colors

    def multi_order_greedy(edges, adj, n_trials=20, time_budget=10.0):
        """Try multiple random orderings and keep the best result."""
        n_edges = len(edges)
        conflict = build_conflict_bitmasks(edges, adj)
        
        edge_constraints = [0] * n_edges
        for i in range(n_edges):
            edge_constraints[i] = bin(conflict[i]).count('1')
        
        best_colour = None
        best_num_colors = float('inf')
        
        order = sorted(range(n_edges), key=lambda i: -edge_constraints[i])
        colour, num_colors = greedy_color_bitmask(edges, conflict, order)
        best_colour = colour[:]
        best_num_colors = num_colors
        
        rng = random.Random(42)
        trial_time = time.time()
        
        for trial in range(n_trials):
            if time.time() - trial_time > time_budget:
                break
            
            order = list(range(n_edges))
            for i in range(len(order) - 1, 0, -1):
                j = rng.randint(0, i)
                if edge_constraints[order[i]] > edge_constraints[order[j]] and rng.random() < 0.7:
                    order[i], order[j] = order[j], order[i]
            
            colour, num_colors = greedy_color_bitmask(edges, conflict, order)
            
            if num_colors < best_num_colors:
                best_colour = colour[:]
                best_num_colors = num_colors
        
        return best_colour, best_num_colors, conflict

    def polish(edges, conflict, colour, max_colors, time_budget=20.0):
        """Polish the coloring with local improvement and color merging."""
        rng = random.Random(123)
        start_polish = time.time()
        
        colour, max_colors = local_improve_bitmask(edges, conflict, colour, max_colors, 
                                                    max_iter=5000, rng=rng)
        
        while time.time() - start_polish < time_budget and max_colors > 1:
            old_max = max_colors
            colour, max_colors = try_merge_colors(edges, conflict, colour, max_colors, rng=rng)
            
            if max_colors < old_max:
                colour, max_colors = local_improve_bitmask(edges, conflict, colour, max_colors,
                                                            max_iter=2000, rng=rng)
        
        # Final pass: try to find a maximum independent set for the highest color class
        # and see if we can redistribute edges more optimally
        if time.time() - start_polish < time_budget - 5 and max_colors > 1:
            high_color = max_colors - 1
            high_edges = [i for i in range(len(edges)) if colour[i] == high_color]
            
            if len(high_edges) <= 20:  # Only try B&B for small sets
                # Find all edges that could potentially be moved to lower colors
                movable = []
                for e_idx in high_edges:
                    taken_mask = 0
                    for j in range(len(edges)):
                        if j != e_idx and ((conflict[e_idx] >> j) & 1):
                            if colour[j] < high_color:
                                taken_mask |= (1 << colour[j])
                    
                    # Count available colors
                    available = 0
                    for c in range(high_color):
                        if not ((taken_mask >> c) & 1):
                            available += 1
                    
                    if available > 0:
                        movable.append(e_idx)
                
                if len(movable) < len(high_edges):
                    # Some edges can be moved; try to find optimal subset to keep in high color
                    # This is complex, so just do another round of local improvement
                    colour, max_colors = local_improve_bitmask(edges, conflict, colour, max_colors,
                                                                max_iter=3000, rng=rng)
        
        return colour, max_colors

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    for spec in specs:
        if time.time() > deadline - 15:
            break
        
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        n_edges = len(edges)
        
        colour, num_colors, conflict = multi_order_greedy(edges, adj, n_trials=40, time_budget=18.0)
        
        if time.time() < deadline - 20:
            colour, num_colors = polish(edges, conflict, colour, num_colors, time_budget=30.0)
        
        result = [colour[i] for i in range(n_edges)]
        out.append(result)
    
    while len(out) < 3:
        spec = specs[len(out)]
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict_bitmasks(edges, adj)
        colour, _ = greedy_color_bitmask(edges, conflict)
        result = [colour[i] for i in range(len(edges))]
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
