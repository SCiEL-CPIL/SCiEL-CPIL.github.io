"""Strong edge colourings (induced matching partition) via branch-and-bound."""

def construct():
    import time
    import sys
    sys.setrecursionlimit(1000000)

    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def instance(n, d, seed):
        import random
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

    def solve_instance(n, d, seed):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        # Build conflict structure: two edges conflict if they share a vertex or are joined by an edge
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        edge_index = {e: i for i, e in enumerate(edges)}

        conflict = [set() for _ in range(m)]
        # Precompute a set of normalized edges for O(1) lookup
        # edge_index is a dict of (min, max) -> idx. 
        # We need a set for faster "in" check if we normalize, or just use the dict.
        # Using dict is fine.
        
        def has_edge(a, b):
            if a > b:
                a, b = b, a
            return (a, b) in edge_index

        for i in range(m):
            u, v = edges[i]
            for j in range(i + 1, m):
                w, x = edges[j]
                # Check if they share a vertex
                if u == w or u == x or v == w or v == x:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if joined by an edge: is there an edge connecting {u,v} to {w,x}?
                if has_edge(u, w) or has_edge(u, x) or has_edge(v, w) or has_edge(v, x):
                    conflict[i].add(j)
                    conflict[j].add(i)

        # Convert to sorted lists for faster membership
        conflict = [sorted(s) for s in conflict]

        # Greedy initialization: order edges by degree in conflict graph (descending), assign smallest available color
        deg_conflict = [len(c) for c in conflict]
        order = sorted(range(m), key=lambda i: -deg_conflict[i])

        # Greedy coloring
        color = [-1] * m
        color_used = set()
        for i in order:
            taken = set()
            for j in conflict[i]:
                if color[j] != -1:
                    taken.add(color[j])
            c = 0
            while c in taken:
                c += 1
            color[i] = c
            color_used.add(c)
        best = max(color_used) + 1 if color_used else 0

        if elapsed() > DEADLINE:
            return [color[i] for i in range(m)]

        # Branch and bound to improve
        # We'll try to find a coloring with fewer colors
        # Use a smarter approach: try to find a valid coloring with K colors for decreasing K
        # Start from the greedy bound and try to reduce

        # Precompute for each edge the list of conflicting edges
        # For B&B, we'll use the following:
        # State: current partial coloring, next edge to color
        # We'll use a recursive search with pruning

        # Better approach: use a maximum independent set style
        # For each color class, we need an induced matching (independent set in the conflict graph)
        # So the problem is: partition V of conflict graph into minimum number of independent sets = chromatic number

        # Let's use a proper B&B for graph coloring
        # Order vertices by degree descending
        vertices_order = order  # already sorted by conflict degree descending

        # Map from original index to position in order
        pos_in_order = [0] * m
        for idx, v in enumerate(vertices_order):
            pos_in_order[v] = idx

        # For B&B, we process vertices in order. For each vertex, try colors 0..K-1.
        # Pruning: if a vertex has all K colors taken by its already-colored neighbors, prune.
        # Also: if the number of colors used so far + remaining lower bound > best, prune.

        # Lower bound: for the remaining uncolored vertices, at least ceil(max_degree_remaining / (K-1))... 
        # Simpler: use the greedy bound on remaining subgraph as a lower bound on additional colors needed
        # Actually, since we're trying to fit in K total colors, the pruning is:
        # if current vertex has all K colors blocked, prune.

        # Let's try K from best-1 down to 1, and see if we can find a valid K-coloring
        # For each K, do B&B.

        # To speed up, use bitmasks for conflict sets
        # m can be up to ~240 (for n=80, d=6, max edges = 80*6/2 = 240)
        # So we can use Python integers as bitmasks

        conflict_mask = [0] * m
        for i in range(m):
            mask = 0
            for j in conflict[i]:
                mask |= (1 << j)
            conflict_mask[i] = mask

        # B&B with bitmask tracking which colors are used by neighbors
        # For each vertex, we track which colors are blocked by already-colored neighbors

        # Let's implement a recursive B&B that tries to color all vertices with K colors
        # We'll use the order defined above

        # For efficiency, precompute for each vertex in order, the list of neighbors that come earlier in the order
        # (since we color in order, only earlier-colored neighbors matter)
        earlier_neighbors = [set() for _ in range(m)]
        for i in range(m):
            for j in conflict[i]:
                if pos_in_order[j] < pos_in_order[i]:
                    earlier_neighbors[i].add(j)

        # Convert to lists
        earlier_neighbors = [sorted(s, key=lambda x: pos_in_order[x]) for s in earlier_neighbors]

        # Now B&B
        # We'll use an iterative deepening approach: try K = best-1, best-2, ...
        # But that might be slow. Instead, let's do a single B&B that minimizes colors.

        # Actually, let's use a different strategy: local search / simulated annealing on the coloring
        # to reduce the number of colors.

        # Simulated annealing: start from greedy coloring, randomly move an edge to a different color,
        # accept if valid or with some probability. Track best (fewest colors).

        import random
        rng = random.Random(42)

        # SA approach
        def is_valid_coloring(col, K):
            # Check that each color class is an independent set
            for i in range(m):
                for j in conflict[i]:
                    if j > i and col[i] == col[j]:
                        return False
            return True

        def count_colors(col):
            return len(set(col))

        # SA: try to reduce number of colors
        current_col = color[:]
        current_cost = count_colors(current_col)
        best_col = current_col[:]
        best_cost = current_cost

        # SA parameters
        T = 10.0
        cooling = 0.995
        max_iter = 200000
        iter_count = 0

        while T > 0.01 and iter_count < max_iter:
            if elapsed() > DEADLINE:
                break
            # Pick a random vertex
            i = rng.randrange(m)
            old_color = current_col[i]
            # Try to change to a random other color
            new_color = rng.randrange(current_cost + 1)
            if new_color == old_color:
                new_color = (new_color + 1) % (current_cost + 1)
            # Check if new_color is valid for vertex i
            valid = True
            for j in conflict[i]:
                if current_col[j] == new_color:
                    valid = False
                    break
            if valid:
                current_col[i] = new_color
                new_cost = count_colors(current_col)
                delta = new_cost - current_cost
                if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_col = current_col[:]
                else:
                    current_col[i] = old_color
            else:
                # Can't change to new_color, try old_color (no change)
                pass
            T *= cooling
            iter_count += 1

        if best_cost < best:
            best = best_cost
            color = best_col[:]

        # Attempt to reduce the number of colors further using a simple recoloring heuristic
        # Try to eliminate each color class one by one
        current_colors = set(color)
        for c in sorted(current_colors, reverse=True):
            if elapsed() > DEADLINE:
                break
            # Try to recolor all vertices of color c to other colors
            # We do this by temporarily removing color c and trying to assign new colors
            # to the vertices that had color c.
            
            # List of vertices to recolor
            vertices_to_recolor = [i for i in range(m) if color[i] == c]
            
            # Save old colors
            old_colors = {i: color[i] for i in vertices_to_recolor}
            
            # Set them to -1 (uncolored)
            for i in vertices_to_recolor:
                color[i] = -1
                
            # Try to color them using available colors (excluding c, but we can use any < c or others)
            # We'll use a greedy approach for these vertices
            success = True
            for i in vertices_to_recolor:
                taken = set()
                for j in conflict[i]:
                    if color[j] != -1:
                        taken.add(color[j])
                
                # Find smallest available color
                new_c = 0
                while new_c in taken:
                    new_c += 1
                
                # If new_c == c, it means all other colors are blocked, or c is the only one?
                # Actually, if we are trying to remove color c, we should prefer colors != c.
                # But if new_c is c, it's not a problem for validity, but it means we failed to remove c
                # for this vertex. We can still assign c, but then we haven't removed the color class.
                # To be strict, if we want to remove c, we must assign something else.
                # If no other color is available, we fail.
                
                # Let's check if new_c is c. If so, try next available.
                if new_c == c:
                    new_c += 1
                    while new_c in taken:
                        new_c += 1
                    # If we end up with a new color, great.
                    # But wait, if new_c was c, it means c was NOT in taken.
                    # So we can assign c. But we want to remove c.
                    # So we MUST assign something else.
                    # If the only available color is c, we cannot remove c for this vertex.
                    # So we fail.
                    if new_c in taken: # This logic is slightly off.
                        pass
                        
                # Simpler: just find the smallest color not in taken.
                # If that color is c, we failed to remove c for this vertex.
                # If we fail for any vertex, we revert.
                
                # Let's just do: assign smallest available.
                # If any vertex gets assigned c, we didn't remove c.
                # We'll check at the end if c is still used.
                
                color[i] = new_c
            
            # Check if color c is still used
            if c in set(color):
                # Revert
                for i in vertices_to_recolor:
                    color[i] = old_colors[i]
            else:
                # Success, color c is removed
                # Update best
                new_best = max(color) + 1
                if new_best < best:
                    best = new_best
                    best_col = color[:]
                    current_colors.discard(c)

        if elapsed() > DEADLINE:
            return [color[i] for i in range(m)]

        # Use best_col if it's better
        if best_cost < best: # This check is redundant if we updated best, but safe
             pass
             
        # Ensure we return the best found
        if best_col is not None and best < max(color)+1:
            return best_col
        else:
            return color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        if elapsed() > DEADLINE:
            # Return a trivial coloring if we're out of time
            edges = instance(n, d, seed)
            out.append(list(range(len(edges))))
        else:
            out.append(solve_instance(n, d, seed))

    return out
