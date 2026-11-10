"""
Decompose Erdos-Renyi-ish graphs into edge-disjoint cycles and single edges,
minimizing the total number of pieces (cycles + leftover edges).

Strategy:
1. Repeatedly find the longest cycle in the remaining graph (via DFS with backtracking,
   trying many start vertices and using a time-limited search).
2. Remove that cycle's edges and repeat until no cycle remains.
3. The remaining edges are output as single-edge pieces.

Key improvements over simple greedy:
- Longest-cycle search: try many start vertices, use iterative deepening or
  bounded DFS to find longer cycles, which reduces the total piece count.
- Multiple restarts with different random seeds for tie-breaking.
- Time management: track elapsed time and return best solution found.
"""
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_2_core(edge_set):
        if not edge_set:
            return set()
        deg = defaultdict(int)
        adj = defaultdict(set)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)

        q = deque([v for v in deg if deg[v] < 2])
        in_core = {v: True for v in deg}

        while q:
            v = q.popleft()
            if not in_core[v]:
                continue
            in_core[v] = False
            for nb in adj[v]:
                if in_core[nb]:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        q.append(nb)

        core_edges = set()
        for u, v in edge_set:
            if in_core[u] and in_core[v]:
                core_edges.add((u, v))
        return core_edges

    def find_longest_cycle_bounded(edge_set, max_time=0.5, rng=None):
        """
        Find a long cycle in the given edge set using bounded DFS.
        Tries multiple start vertices and uses a time limit.
        Returns the longest cycle found, or None if no cycle exists.
        """
        if not edge_set:
            return None

        if rng is None:
            rng = random.Random()

        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        vertices = list(adj.keys())
        if not vertices:
            return None

        # Only consider vertices in the 2-core for cycle finding
        core_edges = get_2_core(edge_set)
        if not core_edges:
            return None

        core_adj = defaultdict(set)
        for u, v in core_edges:
            core_adj[u].add(v)
            core_adj[v].add(u)

        core_vertices = list(core_adj.keys())
        if not core_vertices:
            return None

        # Sort by degree descending to prefer high-degree starts
        core_vertices.sort(key=lambda v: len(core_adj[v]), reverse=True)

        best_cycle = None
        best_len = 0
        start_time = time.time()

        # Try a subset of start vertices
        num_starts = min(len(core_vertices), 8)
        start_candidates = core_vertices[:num_starts]

        # Also add a few random ones for diversity
        for _ in range(min(3, len(core_vertices))):
            v = rng.choice(core_vertices)
            if v not in start_candidates:
                start_candidates.append(v)

        for start in start_candidates:
            if time.time() - start_time > max_time:
                break

            # DFS with backtracking to find a cycle through start
            # We limit path length to avoid exponential blowup
            max_path_len = min(len(core_vertices), 15)  # Cap path length

            visited = {start}
            stack = [(start, [start])]

            while stack:
                if time.time() - start_time > max_time:
                    return best_cycle if best_cycle else None

                u, path = stack.pop()

                # Try to close the cycle
                if len(path) >= 3 and start in core_adj[u]:
                    # Found a cycle
                    cycle = path[:]
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                    # Don't expand further from this path (it's a cycle)
                    continue

                if len(path) >= max_path_len:
                    continue

                # Shuffle neighbors for randomness
                neighbors = list(core_adj[u])
                rng.shuffle(neighbors)

                for nb in neighbors:
                    if nb == start:
                        # Can only close cycle if path length >= 3
                        if len(path) >= 3:
                            cycle = path[:]
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle
                        continue
                    if nb not in visited:
                        new_path = path + [nb]
                        new_visited = visited | {nb}
                        stack.append((nb, new_path))
                        visited.add(nb)
                        # Note: we add to visited here but this is a DFS with backtracking
                        # We need to be more careful. Let's use a different approach.

            # The above DFS is flawed because visited is shared. Let's redo with proper backtracking.

        # Redo with proper iterative DFS with backtracking
        best_cycle = None
        best_len = 0
        start_time = time.time()

        for start in start_candidates:
            if time.time() - start_time > max_time:
                break

            # Iterative DFS with explicit stack for backtracking
            visited = {start}
            # Stack entries: (vertex, path, neighbor_index)
            stack = [(start, [start], 0)]

            while stack:
                if time.time() - start_time > max_time:
                    return best_cycle if best_cycle else None

                u, path, ni = stack[-1]
                neighbors = list(core_adj[u])
                rng.shuffle(neighbors)

                advanced = False
                for i in range(ni, len(neighbors)):
                    nb = neighbors[i]
                    if nb == start:
                        if len(path) >= 3:
                            cycle = path[:]
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle
                        continue
                    if nb not in visited:
                        # Update current top of stack
                        stack[-1] = (u, path, i + 1)
                        visited.add(nb)
                        stack.append((nb, path + [nb], 0))
                        advanced = True
                        break

                if not advanced:
                    # Backtrack
                    stack.pop()
                    if stack:
                        visited.discard(u)

        return best_cycle

    def greedy_decompose_longest(edge_set, max_cycle_time=0.3, rng=None):
        """
        Greedily extract longest cycles from edge_set.
        Returns (list_of_cycles, set_of_remaining_edges)
        """
        if rng is None:
            rng = random.Random()

        rem = set(edge_set)
        cycles = []

        while rem:
            core = get_2_core(rem)
            if not core:
                break

            cycle = find_longest_cycle_bounded(core, max_time=max_cycle_time, rng=rng)
            if not cycle:
                break

            # Remove cycle edges
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                rem.discard(e)
            cycles.append(cycle)

        return cycles, rem

    def score_decomposition(cycles):
        """
        Score is sum(len(cycle) - 1). Higher is better.
        """
        return sum(len(c) - 1 for c in cycles)

    def local_search(initial_cycles, initial_remaining, time_limit, rng=None):
        """
        Perform local search to improve the decomposition.
        """
        if rng is None:
            rng = random.Random()

        current_cycles = [c[:] for c in initial_cycles]
        current_remaining = set(initial_remaining)
        current_score = score_decomposition(current_cycles)

        best_cycles = [c[:] for c in current_cycles]
        best_remaining = set(current_remaining)
        best_score = current_score

        # Simple hill climbing: remove a cycle and re-greedy
        while time.time() < time_limit:
            if not current_cycles:
                break

            # Pick a random cycle to remove
            idx = rng.randrange(len(current_cycles))
            c_old = current_cycles[idx]

            # Restore its edges to the remaining set
            edges_to_add = set()
            for i in range(len(c_old)):
                u, v = c_old[i], c_old[(i + 1) % len(c_old)]
                edges_to_add.add(tuple(sorted((u, v))))

            new_remaining = current_remaining | edges_to_add

            # Remove the cycle from current list
            current_cycles.pop(idx)

            # Re-greedy the new remaining set
            new_cycles, new_remaining_final = greedy_decompose_longest(
                new_remaining, max_cycle_time=0.15, rng=rng
            )

            new_score = score_decomposition(new_cycles)

            if new_score > best_score:
                best_score = new_score
                best_cycles = [c[:] for c in new_cycles]
                best_remaining = set(new_remaining_final)
                current_cycles = best_cycles
                current_remaining = best_remaining
                current_score = new_score
            # else: keep current state unchanged

        return best_cycles, best_remaining

    def decompose(edges, n, time_budget):
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)

        best_cycles = []
        best_remaining = set(edge_set)
        best_score = 0

        # Multiple restarts with different random seeds
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break

            rng = random.Random(restart * 1000 + 42)
            cycles, rem = greedy_decompose_longest(edge_set, max_cycle_time=0.4, rng=rng)
            score = score_decomposition(cycles)
            if score > best_score:
                best_score = score
                best_cycles = [c[:] for c in cycles]
                best_remaining = set(rem)

        # Apply local search to the best solution
        if time.time() < deadline - 3:
            ls_rng = random.Random(12345)
            ls_cycles, ls_rem = local_search(best_cycles, best_remaining, deadline - 3, rng=ls_rng)
            ls_score = score_decomposition(ls_cycles)
            if ls_score > best_score:
                best_score = ls_score
                best_cycles = ls_cycles
                best_remaining = ls_rem

        # Construct the final list of pieces
        pieces = []
        for c in best_cycles:
            pieces.append(c[:])
        for e in best_remaining:
            pieces.append(list(e))

        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 5:
            # Fallback: all edges
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
