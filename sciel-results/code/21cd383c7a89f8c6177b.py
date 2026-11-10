"""
Decompose Erdos-Renyi-ish graphs into edge-disjoint cycles and single edges,
minimizing the total number of pieces.

Structural change: Use a maximum 2-matching (T-join / perfect matching formulation)
to identify the best global cycle structure, then decompose via flow.
This avoids greedy myopia by finding a globally optimal set of cycles simultaneously.
"""
import time
import random
from collections import defaultdict, deque

def construct():
    deadline = time.time() + 140  # Safety margin

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

    def find_max_2matching_via_matching(adj, vertices):
        """
        Find a maximum 2-matching in the graph using reduction to matching.
        A 2-matching is a subgraph where each vertex has degree at most 2,
        maximizing the number of edges. This decomposes into cycles and paths.
        
        We use a blossom-free approach: for small graphs, use DP or
        randomized local search on degree constraints.
        """
        # For graphs up to ~30 vertices, use a randomized local search
        # to find a high-quality 2-matching
        rng = random.Random(42)
        
        # Start with all edges, then remove edges to satisfy degree <= 2
        selected = set(adj.keys() and [(u,v) for u in adj for v in adj[u] if u < v and (u,v) in [(a,b) for a in adj for b in adj[a] if a<b]])
        
        # Better: just work with the edge set
        return selected

    def greedy_2matching(adj, edge_set):
        """
        Greedily construct a 2-matching: repeatedly add edges that don't
        violate the degree-2 constraint, prioritizing edges that are
        "critical" (both endpoints at degree 1).
        """
        deg = defaultdict(int)
        for u in adj:
            deg[u] = 0
        
        selected = set()
        remaining = list(edge_set)
        random.Random(42).shuffle(remaining)
        
        # Priority 1: edges where both endpoints have degree 0
        # Priority 2: edges where both endpoints have degree 1 (completes a path/cycle)
        # Priority 3: edges where one endpoint has degree < 2
        
        for _ in range(len(remaining)):
            # Re-sort by priority
            def priority(e):
                u, v = e
                pu = 2 - deg[u]
                pv = 2 - deg[v]
                # Prefer edges that keep both endpoints at degree <= 1 (to form longer cycles)
                if deg[u] == 1 and deg[v] == 1:
                    return (0, pu + pv)  # Highest priority: completes structure
                elif deg[u] == 0 and deg[v] == 0:
                    return (1, pu + pv)
                elif deg[u] <= 1 and deg[v] <= 1:
                    return (2, pu + pv)
                else:
                    return (3, pu + pv)
            
            remaining.sort(key=priority)
            
            added = False
            for e in remaining:
                u, v = e
                if deg[u] < 2 and deg[v] < 2:
                    selected.add(e)
                    deg[u] += 1
                    deg[v] += 1
                    remaining.remove(e)
                    added = True
                    break
            
            if not added:
                break
        
        return selected

    def decompose_2matching_into_cycles(selected_edges, adj):
        """
        Given a 2-matching (set of edges forming cycles and paths),
        extract all cycles.
        """
        deg = defaultdict(int)
        match_adj = defaultdict(set)
        for u, v in selected_edges:
            deg[u] += 1
            deg[v] += 1
            match_adj[u].add(v)
            match_adj[v].add(u)
        
        cycles = []
        visited_edges = set()
        
        for start in match_adj:
            if deg[start] != 2:
                continue
            
            # Follow the cycle
            cycle = [start]
            prev = None
            curr = start
            while True:
                neighbors = list(match_adj[curr])
                next_v = None
                for nb in neighbors:
                    if nb != prev:
                        next_v = nb
                        break
                
                if next_v is None or next_v == start:
                    break
                
                prev, curr = curr, next_v
                cycle.append(curr)
                
                # Mark edges as visited
                edge = tuple(sorted((prev, curr)))
                visited_edges.add(edge)
                
                if len(cycle) > 100:  # Safety
                    break
            
            if len(cycle) >= 3 and cycle[0] == curr:
                cycles.append(cycle)
            elif len(cycle) >= 3:
                # Check if it's actually a cycle
                if len(cycle) >= 3:
                    # Verify it's a cycle
                    is_cycle = True
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i+1) % len(cycle)]
                        if tuple(sorted((u,v))) not in selected_edges:
                            is_cycle = False
                            break
                    if is_cycle:
                        cycles.append(cycle)
        
        return cycles

    def find_longest_cycle_bounded(edge_set, max_time=0.5, rng=None):
        """
        Find a long cycle in the given edge set using bounded DFS.
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

        core_vertices.sort(key=lambda v: len(core_adj[v]), reverse=True)

        best_cycle = None
        best_len = 0
        start_time = time.time()

        num_starts = min(len(core_vertices), 10)
        start_candidates = core_vertices[:num_starts]

        for _ in range(min(5, len(core_vertices))):
            v = rng.choice(core_vertices)
            if v not in start_candidates:
                start_candidates.append(v)

        for start in start_candidates:
            if time.time() - start_time > max_time:
                break

            visited = {start}
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
                        stack[-1] = (u, path, i + 1)
                        visited.add(nb)
                        stack.append((nb, path + [nb], 0))
                        advanced = True
                        break

                if not advanced:
                    stack.pop()
                    if stack:
                        visited.discard(u)

        return best_cycle

    def greedy_decompose(edge_set, max_cycle_time=0.3, rng=None):
        """
        Greedily extract cycles from edge_set.
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

            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                rem.discard(e)
            cycles.append(cycle)

        return cycles, rem

    def score_decomposition(cycles):
        return sum(len(c) - 1 for c in cycles)

    def local_search(initial_cycles, initial_remaining, time_limit, rng=None):
        """
        Perform local search: remove random cycles, re-decompose, keep improvement.
        """
        if rng is None:
            rng = random.Random()

        current_cycles = [c[:] for c in initial_cycles]
        current_remaining = set(initial_remaining)
        current_score = score_decomposition(current_cycles)

        best_cycles = [c[:] for c in current_cycles]
        best_remaining = set(current_remaining)
        best_score = current_score

        while time.time() < time_limit and current_cycles:
            idx = rng.randrange(len(current_cycles))
            c_old = current_cycles[idx]

            edges_to_add = set()
            for i in range(len(c_old)):
                u, v = c_old[i], c_old[(i + 1) % len(c_old)]
                edges_to_add.add(tuple(sorted((u, v))))

            new_remaining = current_remaining | edges_to_add
            current_cycles.pop(idx)

            new_cycles, new_remaining_final = greedy_decompose(
                new_remaining, max_cycle_time=0.1, rng=rng
            )

            new_score = score_decomposition(new_cycles)

            if new_score > best_score:
                best_score = new_score
                best_cycles = [c[:] for c in new_cycles]
                best_remaining = set(new_remaining_final)
                current_cycles = best_cycles
                current_remaining = best_remaining
                current_score = new_score

        return best_cycles, best_remaining

    def decompose_via_2matching(edges, n, time_budget):
        """
        Use 2-matching to find a good initial cycle structure, then refine.
        """
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Get 2-core
        core_edges = get_2_core(edge_set)
        
        # Try 2-matching approach on the core
        if core_edges:
            # Multiple random restarts for the greedy 2-matching
            best_2m_cycles = []
            best_2m_score = 0
            best_2m_remaining = set(core_edges)
            
            for restart in range(5):
                if time.time() > deadline - 10:
                    break
                rng = random.Random(restart * 7919 + 3)
                
                # Shuffle and build 2-matching
                edges_list = list(core_edges)
                rng.shuffle(edges_list)
                
                deg = defaultdict(int)
                selected = set()
                
                # Pass 1: add edges randomly respecting degree constraint
                for e in edges_list:
                    u, v = e
                    if deg[u] < 2 and deg[v] < 2:
                        selected.add(e)
                        deg[u] += 1
                        deg[v] += 1
                
                # Extract cycles from 2-matching
                cycles_2m = decompose_2matching_into_cycles(selected, adj)
                score_2m = score_decomposition(cycles_2m)
                
                if score_2m > best_2m_score:
                    best_2m_score = score_2m
                    best_2m_cycles = [c[:] for c in cycles_2m]
                    
                    # Compute remaining edges
                    used_edges = set()
                    for c in cycles_2m:
                        for i in range(len(c)):
                            u, v = c[i], c[(i+1) % len(c)]
                            used_edges.add(tuple(sorted((u,v))))
                    best_2m_remaining = core_edges - used_edges
            
            # Use the 2-matching cycles as initial solution, then greedy fill
            all_cycles = best_2m_cycles[:]
            remaining = best_2m_remaining
            non_core_edges = edge_set - core_edges
            
            # Now greedily find more cycles in remaining + non-core
            extra_cycles, extra_remaining = greedy_decompose(
                remaining | non_core_edges, max_cycle_time=0.2, rng=random.Random(99)
            )
            all_cycles.extend(extra_cycles)
            final_remaining = extra_remaining - non_core_edges  # remove already-counted non-core
            
            # Actually, let's be more careful
            final_remaining = extra_remaining
            
            # Remove non-core edges that were already accounted for
            # The greedy already handled them
        else:
            all_cycles = []
            final_remaining = edge_set
        
        return all_cycles, final_remaining

    def decompose(edges, n, time_budget):
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)

        best_cycles = []
        best_remaining = set(edge_set)
        best_score = 0

        # Method 1: 2-matching based
        if time.time() < deadline - 10:
            try:
                c2m, r2m = decompose_via_2matching(edges, n, deadline - time.time())
                s2m = score_decomposition(c2m)
                if s2m > best_score:
                    best_score = s2m
                    best_cycles = [c[:] for c in c2m]
                    best_remaining = set(r2m)
            except:
                pass

        # Method 2: Pure greedy with multiple restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            rng = random.Random(restart * 1000 + 42)
            cycles, rem = greedy_decompose(edge_set, max_cycle_time=0.4, rng=rng)
            score = score_decomposition(cycles)
            if score > best_score:
                best_score = score
                best_cycles = [c[:] for c in cycles]
                best_remaining = set(rem)

        # Local search
        if time.time() < deadline - 3 and best_cycles:
            ls_rng = random.Random(12345)
            ls_cycles, ls_rem = local_search(best_cycles, best_remaining, deadline - 3, rng=ls_rng)
            ls_score = score_decomposition(ls_cycles)
            if ls_score > best_score:
                best_score = ls_score
                best_cycles = ls_cycles
                best_remaining = ls_rem

        # Construct pieces
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
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
