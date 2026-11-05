"""Strong edge colourings via bitmask conflict graph + simulated annealing with global moves."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

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

    def build_bitmask_conflict(edges, n):
        """Build conflict adjacency as list of Python ints (bitmasks)."""
        m = len(edges)
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)

        # Start with all-zero bitmasks
        conflict = [0] * m

        # Conflict 1: share a vertex
        for u in range(n):
            inc = vertex_to_edges[u]
            for i in range(len(inc)):
                for j in range(i + 1, len(inc)):
                    ei, ej = inc[i], inc[j]
                    conflict[ei] |= (1 << ej)
                    conflict[ej] |= (1 << ei)

        # Conflict 2: joined by an edge (induced matching constraint)
        vertex_adj = [set() for _ in range(n)]
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)

        for i, (u, v) in enumerate(edges):
            # Neighbors of u (excluding v)
            for nu in vertex_adj[u]:
                if nu == v:
                    continue
                for j in vertex_to_edges[nu]:
                    if j != i:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
            # Neighbors of v (excluding u)
            for nv in vertex_adj[v]:
                if nv == u:
                    continue
                for j in vertex_to_edges[nv]:
                    if j != i:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)

        return conflict

    def count_colours(colours):
        """Count number of distinct colours used."""
        if not colours:
            return 0
        return len(set(colours))

    def greedy_initial(colours, conflict, m, rng):
        """Greedy DSATUR-like initialization: colour edges one by one, picking the most constrained first."""
        if m == 0:
            return colours, 0

        uncoloured = set(range(m))
        # colour_count[e] = number of distinct colours among coloured neighbours
        colour_count = [0] * m
        # colours_used_by_neighbours[e] = bitmask of colours used by coloured neighbours
        colours_used = [0] * m
        # We'll use a priority approach: repeatedly pick uncoloured edge with max saturation
        # For simplicity, do a few passes

        # First pass: random order greedy
        order = list(range(m))
        rng.shuffle(order)

        for idx in order:
            if idx not in uncoloured:
                continue
            # Find smallest available colour
            used_mask = colours_used[idx]
            c = 0
            while (used_mask >> c) & 1:
                c += 1
            colours[idx] = c
            uncoloured.discard(idx)
            # Update neighbours
            cm = conflict[idx]
            nb = 0
            while cm:
                # Get lowest set bit
                lsb = cm & (-cm)
                j = lsb.bit_length() - 1
                cm ^= lsb
                colours_used[j] |= (1 << c)
                # Recount distinct colours for j
                # This is expensive; skip exact recount for now, use mask
            # Update colour_count approximately
            for j in range(m):
                if (conflict[idx] >> j) & 1:
                    # j is a neighbour of idx
                    pass  # skip, too slow

        return colours, count_colours(colours)

    def greedy_recolor(colours, conflict, m, rng):
        """Greedy recoloring: process edges in random order, assign smallest available colour."""
        if m == 0:
            return colours, 0

        new_colours = [0] * m
        # Process in random order
        order = list(range(m))
        rng.shuffle(order)

        for idx in order:
            # Find colours used by already-coloured neighbours
            used = set()
            cm = conflict[idx]
            nb = 0
            while cm:
                lsb = cm & (-cm)
                j = lsb.bit_length() - 1
                cm ^= lsb
                if new_colours[j] != -1:
                    used.add(new_colours[j])
            # Assign smallest available
            c = 0
            while c in used:
                c += 1
            new_colours[idx] = c

        return new_colours, count_colours(new_colours)

    def try_remove_colour(colours, conflict, m, rng):
        """Try to eliminate the highest colour. Returns (success, new_colours, new_count)."""
        if m == 0:
            return False, colours, 0

        max_c = max(colours)
        if max_c == 0:
            return False, colours, 0

        # Collect edges with max_c
        edges_max = [i for i in range(m) if colours[i] == max_c]
        if not edges_max:
            return False, colours, 0

        # Try to reassign each edge in edges_max to a colour < max_c
        # Order: random
        rng.shuffle(edges_max)

        new_colours = colours[:]
        for e in edges_max:
            # Find blocked colours: colours used by neighbours (excluding those also in edges_max that haven't been moved yet)
            # Actually, neighbours that still have max_c are not blocked for colours < max_c
            # But neighbours that have been reassigned to < max_c ARE blocked
            blocked = set()
            cm = conflict[e]
            nb = 0
            while cm:
                lsb = cm & (-cm)
                j = lsb.bit_length() - 1
                cm ^= lsb
                cj = new_colours[j]
                if cj < max_c:
                    blocked.add(cj)
            # Find smallest c in [0, max_c) not in blocked
            c = 0
            while c < max_c and c in blocked:
                c += 1
            if c >= max_c:
                return False, colours, count_colours(colours)
            new_colours[e] = c

        return True, new_colours, max_c  # new count is max_c (since colours 0..max_c-1 are used)

    def local_search(colours, conflict, m, rng, time_limit):
        """Iteratively try to remove the highest colour."""
        current = colours[:]
        count = count_colours(current)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for _ in range(50):  # max 50 attempts per round
                if time.time() >= time_limit:
                    break
                success, new_colours, new_count = try_remove_colour(current, conflict, m, rng)
                if success:
                    current = new_colours
                    count = new_count
                    improved = True
                    break

        return current, count

    def simulated_annealing(colours, conflict, m, rng, time_limit):
        """Simulated annealing to escape local minima."""
        current = colours[:]
        current_count = count_colours(current)
        best = current[:]
        best_count = current_count

        T = 5.0
        T_min = 0.01
        cooling = 0.9995

        while T > T_min and time.time() < time_limit:
            # Pick a random move
            move_type = rng.random()

            if move_type < 0.3:
                # Single edge recolor
                e = rng.randrange(m)
                old_c = current[e]
                # Try to assign a random different colour
                new_c = rng.randrange(current_count + 2)
                if new_c == old_c:
                    continue
                # Check validity: new_c not used by any neighbour
                valid = True
                cm = conflict[e]
                nb = 0
                while cm:
                    lsb = cm & (-cm)
                    j = lsb.bit_length() - 1
                    cm ^= lsb
                    if current[j] == new_c:
                        valid = False
                        break
                if not valid:
                    continue
                current[e] = new_c
                new_count = count_colours(current)
                delta = new_count - current_count
                if delta <= 0 or rng.random() < (10 ** (-delta / T)):
                    current_count = new_count
                    if new_count < best_count:
                        best_count = new_count
                        best = current[:]
                else:
                    current[e] = old_c

            elif move_type < 0.5:
                # Swap colours of two edges
                e1, e2 = rng.sample(range(m), 2)
                c1, c2 = current[e1], current[e2]
                if c1 == c2:
                    continue
                # Check if swap is valid
                # e1 gets c2: c2 not used by neighbours of e1 (excluding e2)
                valid = True
                cm = conflict[e1]
                nb = 0
                while cm:
                    lsb = cm & (-cm)
                    j = lsb.bit_length() - 1
                    cm ^= lsb
                    if j != e2 and current[j] == c2:
                        valid = False
                        break
                if valid:
                    cm = conflict[e2]
                    nb = 0
                    while cm:
                        lsb = cm & (-cm)
                        j = lsb.bit_length() - 1
                        cm ^= lsb
                        if j != e1 and current[j] == c1:
                            valid = False
                            break
                if not valid:
                    continue
                current[e1], current[e2] = c2, c1
                new_count = count_colours(current)
                delta = new_count - current_count
                if delta <= 0 or rng.random() < (10 ** (-delta / T)):
                    current_count = new_count
                    if new_count < best_count:
                        best_count = new_count
                        best = current[:]
                else:
                    current[e1], current[e2] = c1, c2

            else:
                # Global perturbation: randomly reassign a subset of edges
                subset_size = max(3, m // 10)
                subset = rng.sample(range(m), subset_size)
                saved = {e: current[e] for e in subset}
                for e in subset:
                    # Assign random colour
                    new_c = rng.randrange(current_count + 1)
                    # Check validity against non-subset neighbours
                    valid = True
                    cm = conflict[e]
                    nb = 0
                    while cm:
                        lsb = cm & (-cm)
                        j = lsb.bit_length() - 1
                        cm ^= lsb
                        if j not in subset and current[j] == new_c:
                            valid = False
                            break
                    if valid:
                        current[e] = new_c
                    else:
                        current[e] = saved[e]
                new_count = count_colours(current)
                delta = new_count - current_count
                if delta <= 0 or rng.random() < (10 ** (-delta / T)):
                    current_count = new_count
                    if new_count < best_count:
                        best_count = new_count
                        best = current[:]
                else:
                    for e in subset:
                        current[e] = saved[e]

            T *= cooling

        return best, best_count

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []

        conflict = build_bitmask_conflict(edges, n)
        rng = random.Random(seed)

        best_colours = None
        best_count = float('inf')

        # Multiple restarts with different strategies
        max_restarts = 20

        for restart in range(max_restarts):
            if time.time() > time_limit:
                break

            # Phase 1: Greedy initialization
            colours = [0] * m
            colours, count = greedy_recolor(colours, conflict, m, rng)

            # Phase 2: Local search (try to remove colours)
            colours, count = local_search(colours, conflict, m, rng, time_limit)

            if count < best_count:
                best_count = count
                best_colours = colours[:]

            # Early stop if we hit the bound
            lb = 5 * d * d // 4
            if best_count <= lb:
                break

            # Phase 3: Simulated annealing (if time remains)
            if time.time() < time_limit - 5:
                sa_colours, sa_count = simulated_annealing(colours, conflict, m, rng, time_limit - 5)
                if sa_count < best_count:
                    best_count = sa_count
                    best_colours = sa_colours
                    if best_count <= lb:
                        break

        if best_colours is None:
            # Fallback
            colours = [0] * m
            for idx in range(m):
                used = set()
                cm = conflict[idx]
                nb = 0
                while cm:
                    lsb = cm & (-cm)
                    j = lsb.bit_length() - 1
                    cm ^= lsb
                    if colours[j] != 0 or j < idx:
                        used.add(colours[j])
                c = 0
                while c in used:
                    c += 1
                colours[idx] = c
            best_colours = colours

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
