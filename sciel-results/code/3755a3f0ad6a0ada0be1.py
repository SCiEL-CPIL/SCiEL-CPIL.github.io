"""Rewritten program: decompose each instance graph into edge-disjoint cycles and edges, minimizing pieces."""
# EVOLVE-BLOCK-START
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

    def find_longest_cycle_in_core(core_edges):
        """Find a cycle in the 2-core, preferring longer cycles."""
        if not core_edges:
            return None
        adj = defaultdict(set)
        for u, v in core_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        vertices = list(adj.keys())
        if not vertices:
            return None
        
        best_cycle = None
        best_len = 0
        
        # Try from multiple starting points
        for start in vertices[:min(len(vertices), 5)]:
            # BFS/DFS to find cycles through start
            visited = {start}
            stack = [(start, [start])]
            
            while stack:
                u, path = stack.pop()
                nbrs = list(adj[u])
                random.shuffle(nbrs)
                
                for nb in nbrs:
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    elif nb not in visited:
                        visited.add(nb)
                        stack.append((nb, path + [nb]))
                        # Limit depth to avoid explosion
                        if len(path) > 25:
                            break
        
        if best_cycle:
            return best_cycle
        
        # Fallback: simple DFS for any cycle
        start = random.choice(vertices)
        visited = {start}
        stack = [start]
        
        while stack:
            u = stack[-1]
            nbrs = list(adj[u])
            random.shuffle(nbrs)
            
            advanced = False
            for nb in nbrs:
                if nb == start and len(stack) >= 3:
                    return stack[:]
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
                    advanced = True
                    break
                    
            if not advanced:
                stack.pop()
                
        return None

    def greedy_decompose(edge_set):
        """
        Greedily extract cycles from edge_set, preferring longer cycles.
        Returns (list_of_cycles, set_of_remaining_edges)
        """
        rem = set(edge_set)
        cycles = []
        
        while rem:
            core = get_2_core(rem)
            if not core:
                break
                
            cycle = find_longest_cycle_in_core(core)
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

    def local_search(initial_cycles, initial_remaining, time_limit):
        """
        Perform local search to improve the decomposition.
        Tries: (1) remove one cycle and re-greedy, (2) swap edges between cycles.
        """
        current_cycles = [c[:] for c in initial_cycles]
        current_remaining = set(initial_remaining)
        current_score = score_decomposition(current_cycles)
        
        best_cycles = [c[:] for c in current_cycles]
        best_remaining = set(current_remaining)
        best_score = current_score
        
        # Build edge-to-cycle mapping for quick lookup
        def build_edge_map(cycles):
            edge_map = {}
            for ci, c in enumerate(cycles):
                for i in range(len(c)):
                    u, v = c[i], c[(i + 1) % len(c)]
                    e = tuple(sorted((u, v)))
                    edge_map[e] = ci
            return edge_map
        
        while time.time() < time_limit:
            if not current_cycles:
                break
            
            # Strategy 1: Remove a random cycle and re-greedy
            idx = random.randrange(len(current_cycles))
            c_old = current_cycles[idx]
            
            edges_to_add = set()
            for i in range(len(c_old)):
                u, v = c_old[i], c_old[(i + 1) % len(c_old)]
                edges_to_add.add(tuple(sorted((u, v))))
                
            new_remaining = current_remaining | edges_to_add
            new_cycles = [c[:] for c in current_cycles]
            new_cycles.pop(idx)
            
            # Re-greedy on the combined set
            re_cycles, re_rem = greedy_decompose(new_remaining)
            re_score = score_decomposition(re_cycles)
            
            if re_score > best_score:
                best_score = re_score
                best_cycles = [c[:] for c in re_cycles]
                best_remaining = set(re_rem)
                current_cycles = best_cycles
                current_remaining = best_remaining
                current_score = re_score
            else:
                # Try strategy 2: edge swap between two cycles
                if len(current_cycles) >= 2:
                    i, j = random.sample(range(len(current_cycles)), 2)
                    ci = current_cycles[i]
                    cj = current_cycles[j]
                    
                    # Try to find an edge in ci and an edge in cj that can be swapped
                    # to create two new cycles with more total edges
                    ci_edges = []
                    for k in range(len(ci)):
                        u, v = ci[k], ci[(k + 1) % len(ci)]
                        ci_edges.append(tuple(sorted((u, v))))
                    
                    cj_edges = []
                    for k in range(len(cj)):
                        u, v = cj[k], cj[(k + 1) % len(cj)]
                        cj_edges.append(tuple(sorted((u, v))))
                    
                    # Try swapping: remove one edge from each, add cross edges
                    for ei in range(len(ci_edges)):
                        for ej in range(len(cj_edges)):
                            e1 = ci_edges[ei]
                            e2 = cj_edges[ej]
                            
                            # Get the endpoints
                            u1, v1 = e1
                            u2, v2 = e2
                            
                            # Try two possible cross connections
                            for (a, b, c, d) in [(u1, u2, v1, v2), (u1, v2, v1, u2)]:
                                # Check if cross edges exist in the graph (they must be in remaining or in the cycles)
                                # For now, just check if we can form valid cycles
                                new_ci = ci[:]
                                new_cj = cj[:]
                                
                                # Remove edge ei from ci: split into path
                                # This is complex; skip for now and just use re-greedy approach
                                pass
                    
                    # If no swap found, just continue
                    pass
        
        return best_cycles, best_remaining

    def decompose(edges, n, time_budget):
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        
        best_cycles = []
        best_remaining = set(edge_set)
        best_score = 0
        
        # Multiple restarts of greedy with longest-cycle preference
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            cycles, rem = greedy_decompose(edge_set)
            score = score_decomposition(cycles)
            if score > best_score:
                best_score = score
                best_cycles = [c[:] for c in cycles]
                best_remaining = set(rem)
        
        # Apply local search on the best solution with remaining time
        if time.time() < deadline - 3 and best_cycles:
            ls_time = min(deadline - 3, time.time() + 30)
            ls_cycles, ls_rem = local_search(best_cycles, best_remaining, ls_time)
            ls_score = score_decomposition(ls_cycles)
            if ls_score > best_score:
                best_score = ls_score
                best_cycles = ls_cycles
                best_remaining = ls_rem
        
        # Final polish: try to merge any remaining edges into cycles
        if time.time() < deadline - 1 and best_remaining:
            # Try one more greedy pass on remaining edges
            extra_cycles, final_rem = greedy_decompose(best_remaining)
            if extra_cycles:
                best_cycles.extend(extra_cycles)
                best_remaining = final_rem
        
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
# EVOLVE-BLOCK-END
