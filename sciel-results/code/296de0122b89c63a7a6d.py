"""Strong edge colourings via greedy + local search (DSATUR-inspired)."""
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = t0 + 145  # safety margin before 150s kill

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

    def build_structure(edges, n):
        """Build adjacency and edge conflict graph."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each edge, the set of edges that conflict with it (share a vertex or are adjacent)
        # Two edges (u,v) and (x,y) conflict if:
        #   - they share a vertex, OR
        #   - there is an edge between an endpoint of one and an endpoint of the other
        # i.e., the union of endpoints {u,v} and {x,y} must have no edge between them
        # Actually: induced matching means no two edges share a vertex AND no edge connects
        # an endpoint of one to an endpoint of the other.
        
        # Precompute for each vertex, the set of edges incident to it
        incident = [set() for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            incident[u].add(idx)
            incident[v].add(idx)
        
        # For each edge idx, compute the set of conflicting edge indices
        m = len(edges)
        conflict = [set() for _ in range(m)]
        for idx, (u, v) in enumerate(edges):
            # Edges sharing u or v
            for j in incident[u]:
                if j != idx:
                    conflict[idx].add(j)
            for j in incident[v]:
                if j != idx:
                    conflict[idx].add(j)
            # Edges that have an endpoint connected to u or v (but not sharing u or v)
            for w in adj[u]:
                for j in incident[w]:
                    if j != idx:
                        conflict[idx].add(j)
            for w in adj[v]:
                for j in incident[w]:
                    if j != idx:
                        conflict[idx].add(j)
        
        return adj, incident, conflict

    def greedy_color(conflict, m, order=None):
        """Greedy colouring in the given order (or natural order)."""
        if order is None:
            order = list(range(m))
        colour = [-1] * m
        used = [set() for _ in range(m)]  # used[i] = set of colours used by neighbours of i
        for i in order:
            # Find smallest available colour
            c = 0
            while c in used[i]:
                c += 1
            colour[i] = c
            # Update neighbours
            for j in conflict[i]:
                used[j].add(c)
        return colour

    def is_valid(colour, conflict, m):
        """Check if colouring is valid (no two adjacent edges share a colour)."""
        for i in range(m):
            for j in conflict[i]:
                if j > i and colour[i] == colour[j]:
                    return False
        return True

    def local_search(colour, conflict, m, max_iters=5000, seed=42):
        """Try to reduce number of colours via recoloring."""
        rng = random.Random(seed)
        current_colour = colour[:]
        current_ncols = max(current_colour) + 1 if m > 0 else 0
        
        # Build colour classes
        def get_colour_classes(col):
            classes = defaultdict(list)
            for i in range(m):
                classes[col[i]].append(i)
            return classes
        
        best_colour = current_colour[:]
        best_ncols = current_ncols
        
        for iteration in range(max_iters):
            if time.time() > deadline - 2:
                break
            nc = max(current_colour) + 1 if m > 0 else 0
            if nc < best_ncols:
                best_ncols = nc
                best_colour = current_colour[:]
            
            # Try to eliminate the highest colour
            target = nc - 1
            classes = get_colour_classes(current_colour)
            if target not in classes or not classes[target]:
                continue
            
            # Try to recolor each edge in the target class to a lower colour
            improved = False
            for edge_idx in classes[target]:
                # Try colours 0..target-1
                for new_c in range(target):
                    # Check if new_c is available for edge_idx
                    ok = True
                    for nb in conflict[edge_idx]:
                        if current_colour[nb] == new_c:
                            ok = False
                            break
                    if ok:
                        current_colour[edge_idx] = new_c
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try swapping colours of two edges in the target class
                # Or try a random recoloring of one edge in target class
                if len(classes[target]) >= 2:
                    e1, e2 = rng.sample(classes[target], 2)
                    # Swap: try to recolor e1 to e2's colour and vice versa? No, same colour.
                    # Instead, try to recolor e1 to some other colour, then e2 to target's original... 
                    # This is getting complex. Let's just try random recoloring.
                    pass
            
            # If we reduced the number of colours, continue
            # Occasionally do a random perturbation
            if iteration % 100 == 99 and not improved:
                # Randomly recolor one edge from a high colour class
                if nc > 1:
                    high_class = classes.get(nc - 1, [])
                    if high_class:
                        edge_idx = rng.choice(high_class)
                        # Try to give it a random lower colour
                        for _ in range(5):
                            new_c = rng.randint(0, nc - 2)
                            ok = True
                            for nb in conflict[edge_idx]:
                                if current_colour[nb] == new_c:
                                    ok = False
                                    break
                            if ok:
                                current_colour[edge_idx] = new_c
                                break
        
        return best_colour, best_ncols

    def dsatur_coloring(conflict, m, n, adj, edges):
        """DSATUR-style greedy: pick edge with most saturated colour neighbourhood."""
        colour = [-1] * m
        # For each edge, track the set of colours used by its conflicting neighbours
        sat = [set() for _ in range(m)]
        uncoloured = set(range(m))
        
        # Initial ordering: pick edges with highest degree in conflict graph first
        deg = [len(conflict[i]) for i in range(m)]
        
        for _ in range(m):
            if not uncoloured:
                break
            # Pick edge with max saturation, then max degree
            best = max(uncoloured, key=lambda i: (len(sat[i]), deg[i]))
            # Find smallest available colour
            c = 0
            while c in sat[best]:
                c += 1
            colour[best] = c
            uncoloured.discard(best)
            # Update saturation of neighbours
            for nb in conflict[best]:
                if nb in uncoloured:
                    sat[nb].add(c)
        
        return colour

    results = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    for spec in specs:
        if time.time() > deadline - 5:
            # Not enough time, use greedy for remaining
            edges = instance(*spec)
            n = spec[0]
            m = len(edges)
            adj, incident, conflict = build_structure(edges, n)
            colour = dsatur_coloring(conflict, m, n, adj, edges)
            results.append(colour)
            continue
        
        edges = instance(*spec)
        n = spec[0]
        m = len(edges)
        adj, incident, conflict = build_structure(edges, n)
        
        # Start with DSATUR
        colour = dsatur_coloring(conflict, m, n, adj, edges)
        ncols = max(colour) + 1 if m > 0 else 0
        
        # Try multiple local search runs with different seeds
        best_colour = colour[:]
        best_ncols = ncols
        
        for seed in range(20):
            if time.time() > deadline - 3:
                break
            lc, ln = local_search(colour, conflict, m, max_iters=3000, seed=seed)
            if ln < best_ncols:
                best_ncols = ln
                best_colour = lc[:]
        
        results.append(best_colour)
    
    return results
