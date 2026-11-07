"""Rewritten program with a stronger algorithm for list colouring planar bipartite quadrangulations."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random as _random
    from random import Random
    from collections import deque

    DEADLINE = 130.0  # wall-clock seconds (evaluator kills at 150s)
    START = time.time()

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def count_conflicts_fast(colour, adj, n):
        """Count number of conflicting edges (u,v) with u<v and colour[u]==colour[v]."""
        conflicts = 0
        for v in range(n):
            cv = colour[v]
            for u in adj[v]:
                if u < v and colour[u] == cv:
                    conflicts += 1
        return conflicts

    def try_greedy(n, adj, lists, rng, order=None):
        """Attempt greedy colouring in given order. Returns list of ints or None."""
        if order is None:
            order = list(range(n))
            rng.shuffle(order)
        colour = [None] * n
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            available = [c for c in lists[v] if c not in taken]
            if available:
                colour[v] = rng.choice(available)
            else:
                return None
        # Verify
        for v in range(n):
            for u in adj[v]:
                if u < v and colour[v] == colour[u]:
                    return None
        return colour

    def local_search(n, adj, edges, lists, colour, rng, max_iters=100000):
        """Hill-climbing local search. Returns list of ints if solved, else None."""
        best_colour = colour[:]
        best_conflicts = count_conflicts_fast(best_colour, adj, n)
        if best_conflicts == 0:
            return [int(c) for c in best_colour]
        
        current_colour = colour[:]
        current_conflicts = best_conflicts
        
        # Build edge list for sampling
        edge_list = edges
        
        it = 0
        while it < max_iters:
            it += 1
            if time.time() - START > DEADLINE - 3:
                break
            
            # Pick a random edge
            u, v = edge_list[rng.randrange(len(edge_list))]
            if current_colour[u] != current_colour[v]:
                continue
            
            # Try changing u or v to reduce conflicts
            # Try changing u
            improved = False
            for c in lists[u]:
                if c == current_colour[u]:
                    continue
                old = current_colour[u]
                current_colour[u] = c
                # Only count changes involving u
                delta = 0
                for w in adj[u]:
                    if w == u:
                        continue
                    # Before: conflict if old == colour[w]
                    if old == current_colour[w]:
                        delta -= 1
                    # After: conflict if c == colour[w]
                    if c == current_colour[w]:
                        delta += 1
                new_conflicts = current_conflicts + delta
                if new_conflicts < current_conflicts:
                    current_conflicts = new_conflicts
                    if new_conflicts < best_conflicts:
                        best_conflicts = new_conflicts
                        best_colour = current_colour[:]
                        if best_conflicts == 0:
                            return [int(c) for c in best_colour]
                    improved = True
                    break
                else:
                    current_colour[u] = old
            
            if improved:
                continue
            
            # Try changing v
            for c in lists[v]:
                if c == current_colour[v]:
                    continue
                old = current_colour[v]
                current_colour[v] = c
                delta = 0
                for w in adj[v]:
                    if w == v:
                        continue
                    if old == current_colour[w]:
                        delta -= 1
                    if c == current_colour[w]:
                        delta += 1
                new_conflicts = current_conflicts + delta
                if new_conflicts < current_conflicts:
                    current_conflicts = new_conflicts
                    if new_conflicts < best_conflicts:
                        best_conflicts = new_conflicts
                        best_colour = current_colour[:]
                        if best_conflicts == 0:
                            return [int(c) for c in best_colour]
                    improved = True
                    break
                else:
                    current_colour[v] = old
            
            if not improved:
                # No improvement found; try a random move to escape local minimum
                # Pick a random vertex and change its colour
                vtx = rng.randrange(n)
                old = current_colour[vtx]
                other = [c for c in lists[vtx] if c != old]
                if other:
                    new_c = rng.choice(other)
                    delta = 0
                    for w in adj[vtx]:
                        if w == vtx:
                            continue
                        if old == current_colour[w]:
                            delta -= 1
                        if new_c == current_colour[w]:
                            delta += 1
                    current_colour[vtx] = new_c
                    current_conflicts += delta
                    if current_conflicts < best_conflicts:
                        best_conflicts = current_conflicts
                        best_colour = current_colour[:]
                        if best_conflicts == 0:
                            return [int(c) for c in best_colour]
        
        if best_conflicts == 0:
            return [int(c) for c in best_colour]
        return None

    def solve_instance(n, edges, lists, rng):
        adj = build_adj(n, edges)
        
        # Strategy 1: Multiple greedy attempts with different orderings
        # Use degree-based ordering as a good heuristic
        degrees = [len(adj[v]) for v in range(n)]
        
        # Try greedy with degree-descending order
        order_deg = sorted(range(n), key=lambda v: -degrees[v])
        result = try_greedy(n, adj, lists, rng, order_deg)
        if result is not None:
            return result
        
        # Try greedy with degree-ascending order
        order_deg_asc = sorted(range(n), key=lambda v: degrees[v])
        result = try_greedy(n, adj, lists, rng, order_deg_asc)
        if result is not None:
            return result
        
        # Try several random greedy attempts
        for attempt in range(50):
            if time.time() - START > DEADLINE - 3:
                break
            result = try_greedy(n, adj, lists, rng)
            if result is not None:
                return result
        
        # Strategy 2: Initialize with first available colour, then local search
        colour = [lists[v][0] for v in range(n)]
        result = local_search(n, adj, edges, lists, colour, rng, max_iters=50000)
        if result is not None:
            return result
        
        # Strategy 3: Try different initialisations
        for init_seed in range(5):
            if time.time() - START > DEADLINE - 3:
                break
            rng2 = Random(seed * 1000003 + splits + init_seed * 7919)
            colour = [rng2.choice(lists[v]) for v in range(n)]
            result = local_search(n, adj, edges, lists, rng, max_iters=30000)
            if result is not None:
                return result
        
        # Strategy 4: More aggressive local search with restarts
        for restart in range(10):
            if time.time() - START > DEADLINE - 3:
                break
            colour = [lists[v][0] for v in range(n)]
            result = local_search(n, adj, edges, lists, rng, max_iters=20000)
            if result is not None:
                return result
        
        # Last resort: return first available colour (may be invalid but keeps structure)
        return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() - START > DEADLINE - 10:
            # Not enough time; still need to produce something
            n, edges, lists = instance(splits, seed)
            out.append([lists[v][0] for v in range(n)])
            continue
        n, edges, lists = instance(splits, seed)
        rng = Random(seed * 1000003 + splits)  # deterministic per instance
        result = solve_instance(n, edges, lists, rng)
        if result is not None:
            out.append(result)
        else:
            out.append([lists[v][0] for v in range(n)])
    
    # Final check: ensure all are Python ints and lists are of type list
    final_out = []
    for inst in out:
        final_out.append([int(c) for c in inst])
    return final_out
# EVOLVE-BLOCK-END
