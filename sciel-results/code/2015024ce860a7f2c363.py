import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def generate_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, n):
        """
        Build conflict structure using bitmasks for fast operations.
        Returns:
            - m: number of edges
            - conflict: list of int bitmasks, conflict[i] has bit j set if edge i conflicts with edge j
            - edge_to_vertices: list of (u, v) for each edge
        """
        m = len(edges)
        edge_to_vertices = edges  # already sorted list of (u, v)
        
        # Build adjacency for original graph
        orig_adj = [set() for _ in range(n)]
        for u, v in edges:
            orig_adj[u].add(v)
            orig_adj[v].add(u)
        
        # Build conflict bitmasks
        conflict = [0] * m
        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                x, y = edges[j]
                # Check conflict
                if u == x or u == y or v == x or v == y:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                else:
                    # Disjoint but connected by an edge
                    if (u in orig_adj[x] or u in orig_adj[y] or 
                        v in orig_adj[x] or v in orig_adj[y]):
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
                        
        return m, conflict, edge_to_vertices

    def greedy_first_fit(m, conflict, order=None):
        """
        Greedy first-fit coloring.
        Returns: list of colors, num_colors
        """
        if order is None:
            order = list(range(m))
        
        colors = [-1] * m
        # For each color, maintain a bitmask of vertices that have that color
        color_masks = []  # color_masks[c] = bitmask of vertices with color c
        
        for v in order:
            # Find smallest color c such that no neighbor of v has color c
            # i.e., conflict[v] & color_masks[c] == 0
            assigned = False
            for c, mask in enumerate(color_masks):
                if (conflict[v] & mask) == 0:
                    colors[v] = c
                    color_masks[c] |= (1 << v)
                    assigned = True
                    break
            if not assigned:
                c = len(color_masks)
                colors[v] = c
                color_masks.append(1 << v)
        
        num_colors = len(color_masks)
        return colors, num_colors

    def local_search(m, conflict, colors, time_limit=5.0):
        """
        Local search to reduce number of colors.
        Tries to recolor vertices to smaller color indices.
        Also tries swaps between vertices of different colors.
        """
        start = time.time()
        num_colors = max(colors) + 1 if m > 0 else 0
        
        # Build color masks
        color_masks = [0] * num_colors
        for v in range(m):
            color_masks[colors[v]] |= (1 << v)
        
        # Precompute degree (number of conflicts)
        degrees = [bin(conflict[v]).count('1') for v in range(m)]
        
        changed = True
        while changed and (time.time() - start) < time_limit:
            changed = False
            # Try to recolor vertices with high colors to lower colors
            for v in range(m - 1, -1, -1):
                if (time.time() - start) > time_limit:
                    break
                cur_color = colors[v]
                if cur_color == 0:
                    continue
                # Try all colors smaller than cur_color
                for c in range(cur_color):
                    if (conflict[v] & color_masks[c]) == 0:
                        # Can recolor v to c
                        color_masks[cur_color] &= ~(1 << v)
                        color_masks[c] |= (1 << v)
                        colors[v] = c
                        changed = True
                        break
        
        # Now try to reduce the number of colors by checking if the highest color can be eliminated
        # Repeat until no more reduction
        while (time.time() - start) < time_limit:
            num_colors = max(colors) + 1 if m > 0 else 0
            if num_colors <= 1:
                break
            
            # Try to eliminate the highest color (num_colors - 1)
            highest = num_colors - 1
            vertices_in_highest = [v for v in range(m) if colors[v] == highest]
            
            if not vertices_in_highest:
                break
            
            success = False
            # Try to move each vertex in highest color to a lower color
            for v in vertices_in_highest:
                if (time.time() - start) > time_limit:
                    break
                for c in range(highest):
                    if (conflict[v] & color_masks[c]) == 0:
                        # Move v to color c
                        color_masks[highest] &= ~(1 << v)
                        color_masks[c] |= (1 << v)
                        colors[v] = c
                        success = True
                        break
                if success:
                    break
            
            if not success:
                # Try swap-based moves
                # For each pair of vertices in highest color, try to swap with a vertex in another color
                # This is more complex; for now, just break
                break
            
            # Check if highest color is now empty
            if color_masks[highest] == 0:
                # Remove the highest color
                color_masks.pop()
                # Renumber colors
                for v in range(m):
                    if colors[v] > highest - 1:
                        colors[v] -= 1  # This shouldn't happen if we removed the top
                # Actually, we need to renumber properly
                # Let's just rebuild
                pass
        
        # Final pass: renumber colors to be contiguous
        if m > 0:
            color_map = {}
            next_color = 0
            for v in range(m):
                if colors[v] not in color_map:
                    color_map[colors[v]] = next_color
                    next_color += 1
                colors[v] = color_map[colors[v]]
            num_colors = next_color
        else:
            num_colors = 0
        
        return colors, num_colors

    def simulated_annealing(m, conflict, colors, num_colors, time_limit=10.0):
        """
        Simulated annealing to further reduce colors.
        Moves: recolor a vertex to a random color (possibly higher).
        Objective: minimize total colors used.
        """
        start = time.time()
        rng = random.Random(42)
        
        # Build color masks
        color_masks = [0] * num_colors
        for v in range(m):
            color_masks[colors[v]] |= (1 << v)
        
        best_colors = colors[:]
        best_num_colors = num_colors
        
        def get_num_colors(col, cmasks):
            return max(col) + 1 if m > 0 else 0
        
        # Initial temperature
        T = 1.0
        cooling = 0.9995
        
        while (time.time() - start) < time_limit:
            T *= cooling
            if T < 0.001:
                break
            
            # Pick a random vertex
            v = rng.randrange(m)
            cur_color = colors[v]
            
            # Pick a new color (random, possibly different)
            new_color = rng.randrange(num_colors + 1)  # +1 to allow new color
            
            # Check if move is valid
            if new_color == cur_color:
                continue
            
            # Check validity: no neighbor of v can have new_color
            if new_color < num_colors:
                if (conflict[v] & color_masks[new_color]) != 0:
                    continue
            # If new_color == num_colors, it's a new color, always valid
            
            # Compute delta in number of colors
            old_num = num_colors
            # Update masks temporarily
            color_masks[cur_color] &= ~(1 << v)
            if new_color < num_colors:
                color_masks[new_color] |= (1 << v)
                new_num = old_num
            else:
                color_masks.append(1 << v)
                new_num = old_num + 1
            
            # Check if cur_color became empty
            if color_masks[cur_color] == 0 and cur_color == old_num - 1:
                new_num = old_num - 1
            
            delta = new_num - old_num
            
            # Accept or reject
            if delta <= 0 or rng.random() < (1.0 if delta == 1 else 0.1):
                # Accept
                colors[v] = new_color
                num_colors = new_num
                if new_color >= num_colors:
                    # Shrink if needed
                    while num_colors > 0 and color_masks[num_colors - 1] == 0:
                        color_masks.pop()
                        num_colors -= 1
                
                if num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_colors = colors[:]
            else:
                # Reject
                if new_color < num_colors:
                    color_masks[new_color] &= ~(1 << v)
                else:
                    color_masks.pop()
                color_masks[cur_color] |= (1 << v)
        
        # Final renumber
        if m > 0:
            color_map = {}
            next_color = 0
            for v in range(m):
                if colors[v] not in color_map:
                    color_map[colors[v]] = next_color
                    next_color += 1
                colors[v] = color_map[colors[v]]
            num_colors = next_color
        
        # Return best
        if best_num_colors < num_colors:
            # Renumber best_colors
            color_map = {}
            next_color = 0
            for v in range(m):
                if best_colors[v] not in color_map:
                    color_map[best_colors[v]] = next_color
                    next_color += 1
                best_colors[v] = color_map[best_colors[v]]
            return best_colors, best_num_colors
        else:
            return colors, num_colors

    def multi_start(m, conflict, time_limit=30.0):
        """
        Multiple restarts with different orderings.
        Returns the best coloring found.
        """
        start = time.time()
        best_colors = None
        best_num = float('inf')
        
        rng = random.Random(123)
        
        # Strategy 1: DSATUR-like ordering (by degree descending)
        degrees = [bin(conflict[v]).count('1') for v in range(m)]
        order1 = sorted(range(m), key=lambda x: -degrees[x])
        
        # Strategy 2: Random order
        order2 = list(range(m))
        rng.shuffle(order2)
        
        # Strategy 3: Another random
        order3 = list(range(m))
        rng.shuffle(order3)
        
        orders = [order1, order2, order3]
        
        for attempt, order in enumerate(orders):
            if (time.time() - start) > time_limit:
                break
            
            colors, num_colors = greedy_first_fit(m, conflict, order)
            
            # Local search
            colors, num_colors = local_search(m, conflict, colors, time_limit=min(5.0, max(0.5, (time_limit - (time.time() - start)) / 2)))
            
            # Simulated annealing
            remaining = max(0.5, time_limit - (time.time() - start) - 2.0)
            if remaining > 1.0:
                colors, num_colors = simulated_annealing(m, conflict, colors, num_colors, time_limit=remaining)
            
            if num_colors < best_num:
                best_num = num_colors
                best_colors = colors[:]
        
        if best_colors is None:
            best_colors, best_num = greedy_first_fit(m, conflict, order1)
        
        return best_colors, best_num

    results = []
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    
    # Allocate time proportionally to graph size
    total_time = deadline - time.time()
    time_budgets = [total_time * 0.25, total_time * 0.35, total_time * 0.40]
    
    for idx, (n, d, seed) in enumerate(specs):
        if time.time() > deadline - 5:
            # Fallback: quick greedy
            edges = generate_instance(n, d, seed)
            m = len(edges)
            if m == 0:
                results.append([])
                continue
            _, conflict, _ = build_conflict_bitmasks(edges, n)
            colors, _ = greedy_first_fit(m, conflict)
            results.append(colors)
            continue
        
        edges = generate_instance(n, d, seed)
        m = len(edges)
        
        if m == 0:
            results.append([])
            continue
        
        m, conflict, _ = build_conflict_bitmasks(edges, n)
        
        # Multi-start optimization
        colors, num_colors = multi_start(m, conflict, time_limit=time_budgets[idx])
        
        results.append(colors)
    
    return results

# EVOLVE-BLOCK-END
