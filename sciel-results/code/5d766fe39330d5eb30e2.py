"""Strong edge colourings via Local Search on Conflict Graph."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict
from math import log, exp, sqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Generate random graph with max degree d
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, n):
        """
        Build a conflict graph where nodes are edge indices.
        Two nodes are connected if the corresponding edges in the original graph
        share a vertex or are joined by an edge (i.e., they cannot have the same color).
        """
        m = len(edges)
        # Map vertex to list of incident edge indices
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)
        
        # Build adjacency list for conflict graph
        conflict_adj = [set() for _ in range(m)]
        
        # 1. Edges sharing a vertex conflict
        for u in range(n):
            inc_edges = vertex_to_edges[u]
            for i in range(len(inc_edges)):
                for j in range(i + 1, len(inc_edges)):
                    e_i = inc_edges[i]
                    e_j = inc_edges[j]
                    conflict_adj[e_i].add(e_j)
                    conflict_adj[e_j].add(e_i)
        
        # 2. Edges joined by an edge conflict
        vertex_adj = [set() for _ in range(n)]
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)
        
        for i, (u, v) in enumerate(edges):
            for nu in vertex_adj[u]:
                if nu == v:
                    continue
                for j in vertex_to_edges[nu]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
            for nv in vertex_adj[v]:
                if nv == u:
                    continue
                for j in vertex_to_edges[nv]:
                    if j != i:
                        conflict_adj[i].add(j)
                        conflict_adj[j].add(i)
                        
        return conflict_adj

    def greedy_color_with_order(order, conflict_adj, m):
        """Color edges in the given order, using first-fit."""
        colours = [0] * m
        for idx in order:
            blocked = set()
            for nb in conflict_adj[idx]:
                if nb in order_set:  # Only check colored edges
                    blocked.add(colours[nb])
            c = 0
            while c in blocked:
                c += 1
            colours[idx] = c
        return colours

    def try_remove_color(colours, c_remove, conflict_adj, m, rng):
        """
        Try to reassign all edges with color c_remove to colors < c_remove.
        Returns (success, new_colours).
        """
        edges_with_c = [i for i in range(m) if colours[i] == c_remove]
        if not edges_with_c:
            return True, colours[:]
        
        old_colours = colours[:]
        rng.shuffle(edges_with_c)
        
        for idx in edges_with_c:
            available = []
            for c_new in range(c_remove):
                valid = True
                for nb in conflict_adj[idx]:
                    if colours[nb] == c_new:
                        valid = False
                        break
                if valid:
                    available.append(c_new)
            if not available:
                colours = old_colours
                return False, old_colours
            colours[idx] = rng.choice(available)
        
        return True, colours

    def local_search_improve(colours, conflict_adj, m, rng, time_limit, max_iters=200):
        """
        Iteratively try to remove the highest color.
        Returns improved colours.
        """
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < time_limit:
            improved = False
            iters += 1
            max_c = max(colours) if m > 0 else -1
            for c_remove in range(max_c, 0, -1):
                if time.time() > time_limit:
                    break
                edges_with_c = [i for i in range(m) if colours[i] == c_remove]
                if not edges_with_c:
                    continue
                success, new_colours = try_remove_color(colours, c_remove, conflict_adj, m, rng)
                if success:
                    colours = new_colours
                    improved = True
                    break
        return colours

    def simulated_annealing(colours, conflict_adj, m, rng, time_limit, T_start=10.0, T_end=0.1, cooling=0.995):
        """
        Simulated annealing to minimize the number of colors.
        We use a cost function that penalizes the number of distinct colors used,
        and also penalizes conflicts (though we should maintain validity).
        
        Actually, for strong edge coloring, we must maintain validity at all times.
        So we use a cost = number of distinct colors + lambda * (some measure of how close to optimal).
        
        A simpler approach: we only move to valid states.
        Cost = max(colours) + 1 (number of colors).
        Moves: pick an edge, change its color to a valid one.
        Accept if better, or with probability exp(-delta/T).
        
        But minimizing max is hard. Let's use:
        cost = sum(1 for c in range(max_c+1) if c in colours) + 0.1 * max(colours)
        
        Or just: cost = number of distinct colors.
        
        Let's stick to the removal-based approach but with more aggressive moves.
        
        Actually, a better SA move:
        1. Pick a random edge e.
        2. Pick a random valid color c' for e.
        3. If c' == current color, no change.
        4. If the set of colors used decreases (i.e., c was the last edge of its color and c' is not new), accept.
        5. If the set of colors used increases, reject (or accept with low prob).
        6. If the set is the same, accept with prob exp(-0/T) = 1? No, we want to minimize max color index too?
        
        Let's define cost as:
        cost(colours) = (number of distinct colors, max color)
        Lexicographic minimization.
        
        In SA, we need a scalar cost.
        Let cost = 10 * (number of distinct colors) + max(colours)
        
        Moves:
        - Pick edge e, new color c_new (valid).
        - Calculate delta_cost.
        - Accept if delta_cost < 0 or with prob exp(-delta_cost / T).
        """
        current_colours = colours[:]
        T = T_start
        
        # Precompute initial cost
        def calc_cost(cols):
            if m == 0:
                return 0
            used = set(cols)
            return 10 * len(used) + max(cols)
        
        current_cost = calc_cost(current_colours)
        best_colours = current_colours[:]
        best_cost = current_cost
        
        while T > T_end and time.time() < time_limit:
            # Pick a random edge
            e = rng.randrange(m)
            current_c = current_colours[e]
            
            # Find all valid colors for e
            blocked = set()
            for nb in conflict_adj[e]:
                blocked.add(current_colours[nb])
            
            # Try a few random valid colors, or just one
            # To speed up, we can try the current color and one random other
            # Or pick a random color from 0 to current_max+1
            max_c = max(current_colours) if m > 0 else 0
            # Let's try a random color in [0, max_c + 1]
            # If it's valid, we propose it.
            # But we need to be efficient.
            
            # Alternative move: try to change color to a random valid one.
            # We need to generate valid colors.
            # Since max degree is small, the number of blocked colors is small.
            # Let's pick a random integer c_new in [0, max_c + 2]
            # and check if it's valid.
            
            c_new = rng.randint(0, max_c + 2)
            if c_new in blocked:
                # Try another
                c_new = rng.randint(0, max_c + 2)
                if c_new in blocked:
                    T *= cooling
                    continue
            
            if c_new == current_c:
                T *= cooling
                continue
            
            # Check validity
            # We already checked against blocked (neighbors' colors)
            # But we need to ensure c_new is not in blocked.
            # The blocked set was computed from neighbors' CURRENT colors.
            # If we change e to c_new, we need c_new not in {colours[nb] for nb in neighbors}.
            # That's exactly the blocked set.
            if c_new in blocked:
                T *= cooling
                continue
            
            # Calculate new cost
            # This is expensive if we recompute from scratch.
            # Let's maintain the count of each color.
            # For now, just recompute for simplicity (m is small).
            new_colours = current_colours[:]
            new_colours[e] = c_new
            new_cost = calc_cost(new_colours)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < exp(-delta / T):
                current_colours = new_colours
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colours = current_colours[:]
            
            T *= cooling
        
        return best_colours

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflict_adj = build_conflict_graph(edges, n)
        
        best_colours = None
        best_count = float('inf')
        
        rng = random.Random(seed)
        
        # Precompute a set of all edge indices for fast lookup in greedy
        # Actually, in greedy we process in order, so we can just track which are colored.
        # But the greedy function above uses order_set which is not defined. Let's fix.
        
        def greedy_color(order, conflict_adj, m):
            coloured_set = set()
            colours = [0] * m
            for idx in order:
                blocked = set()
                for nb in conflict_adj[idx]:
                    if nb in coloured_set:
                        blocked.add(colours[nb])
                c = 0
                while c in blocked:
                    c += 1
                colours[idx] = c
                coloured_set.add(idx)
            return colours
        
        max_restarts = 100
        restart_count = 0
        
        while restart_count < max_restarts and time.time() < time_limit:
            # 1. Greedy with random order
            order = list(range(m))
            rng.shuffle(order)
            
            colours = greedy_color(order, conflict_adj, m)
            current_count = max(colours) + 1 if m > 0 else 0
            
            # 2. Local search: try to remove colors
            colours = local_search_improve(colours, conflict_adj, m, rng, time_limit)
            current_count = max(colours) + 1 if m > 0 else 0
            
            # 3. Simulated annealing for further improvement
            if time.time() < time_limit - 2.0:  # Leave 2s for finishing
                sa_colours = simulated_annealing(colours, conflict_adj, m, rng, time.time() + 5.0)
                sa_count = max(sa_colours) + 1 if m > 0 else 0
                if sa_count < current_count:
                    colours = sa_colours
                    current_count = sa_count
            
            if current_count < best_count:
                best_count = current_count
                best_colours = colours[:]
            
            restart_count += 1
            
            # If we've hit the theoretical lower bound (5d^2/4), stop early?
            # Lower bound for strong edge coloring is not exactly 5d^2/4, that's the conjectured upper bound.
            # Actually, the conjecture says chi_s(G) <= 5d^2/4.
            # So if we reach that, we can stop.
            lb = 5 * d * d // 4
            if current_count <= lb:
                break
        
        if best_colours is None:
            # Fallback
            order = list(range(m))
            best_colours = greedy_color(order, conflict_adj, m)
        
        return best_colours

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        inst_start = time.time()
        inst_limit = min(inst_start + 45.0, deadline)
        
        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)
        
        if time.time() > deadline:
            break

    return results
# EVOLVE-BLOCK-END
