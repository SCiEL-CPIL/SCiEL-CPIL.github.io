import time
import random
import sys

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adj(edge_set, n):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_longest_cycle(rem_edges, n):
        """Find a cycle in rem_edges using DFS, preferring longer cycles."""
        if not rem_edges:
            return None
        adj = build_adj(rem_edges, n)
        # Try to find a cycle starting from each vertex
        best_cycle = None
        for start in range(n):
            if not adj[start]:
                continue
            # DFS to find a cycle back to start
            stack = [(start, [start], {start})]
            found = None
            while stack:
                u, path, visited = stack.pop()
                for v in sorted(adj[u], key=lambda x: -len(adj[x])):  # prefer high-degree
                    if v == start and len(path) >= 3:
                        cyc = path + [start]
                        if best_cycle is None or len(cyc) > len(best_cycle):
                            best_cycle = cyc
                        found = True
                    elif v not in visited:
                        stack.append((v, path + [v], visited | {v}))
            if found:
                break  # Found a cycle, move on
        return best_cycle

    def greedy_decompose(edges, n, rng, time_limit=0.5):
        """Multi-start greedy: extract longest cycles repeatedly."""
        rem = set((min(u,v), max(u,v)) for u, v in edges)
        cycles = []
        t0 = time.time()
        
        while rem and time.time() - t0 < time_limit:
            adj = build_adj(rem, n)
            # Find a cycle via DFS
            cycle = None
            for start in range(n):
                if not adj[start]:
                    continue
                # BFS/DFS to find cycle
                visited = set()
                parent = {}
                stack = [start]
                visited.add(start)
                parent[start] = None
                found = False
                while stack and not found:
                    u = stack.pop()
                    for v in sorted(adj[u]):
                        if v == start and len(visited) >= 3:
                            # Reconstruct cycle
                            cyc = [v]
                            cur = u
                            while cur != v:
                                cyc.append(cur)
                                cur = parent[cur]
                            cyc.reverse()
                            cycle = cyc
                            found = True
                            break
                        elif v not in visited:
                            visited.add(v)
                            parent[v] = u
                            stack.append(v)
                if found:
                    break
            
            if not cycle:
                break
            
            # Remove cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                rem.discard((min(u, v), max(u, v)))
            cycles.append(cycle)
        
        return cycles, rem

    def local_search(cycles, rem, n, time_limit=2.0):
        """Local search: try to absorb edges into cycles or merge cycles."""
        t0 = time.time()
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            
            # Try to absorb a single edge into an existing cycle
            for e in list(rem):
                u, v = e
                for ci, cyc in enumerate(cycles):
                    # Check if adding edge (u,v) to cycle creates a valid longer cycle
                    # This is complex; skip for now
                    pass
            
            # Try to find a new cycle in remaining edges
            if rem:
                cycle = find_longest_cycle(rem, n)
                if cycle:
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i+1) % len(cycle)]
                        rem.discard((min(u, v), max(u, v)))
                    cycles.append(cycle)
                    improved = True
        
        return cycles, rem

    def decompose_graph(edges, n, time_budget):
        """Combine multi-start greedy with local search."""
        t_start = time.time()
        best_cost = float('inf')
        best_cycles = None
        best_rem = None
        
        # Multiple random restarts
        num_restarts = max(1, int(time_budget / 1.0))
        for restart in range(num_restarts):
            if time.time() - t_start > time_budget - 0.5:
                break
            
            rng = Random(42 + restart)
            # Shuffle edge order for diversity
            shuffled_edges = edges[:]
            rng.shuffle(shuffled_edges)
            
            cycles, rem = greedy_decompose(shuffled_edges, n, rng, time_limit=0.3)
            cycles, rem = local_search(cycles, rem, n, time_limit=0.5)
            
            cost = len(cycles) + len(rem)
            if cost < best_cost:
                best_cost = cost
                best_cycles = [c[:] for c in cycles]
                best_rem = set(rem)
        
        if best_cycles is None:
            # Fallback: all edges as singletons
            return [list(e) for e in edges]
        
        pieces = []
        for cyc in best_cycles:
            pieces.append(list(cyc))
        for e in best_rem:
            pieces.append(list(e))
        return pieces

    t_start = time.time()
    total_budget = 140  # safety margin from 150s
    results = []
    
    for i, spec in enumerate(((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))):
        n = spec[0]
        edges = instance(*spec)
        if i < 2:
            budget = (total_budget / 3.0)
        else:
            budget = total_budget - (time.time() - t_start)
        pieces = decompose_graph(edges, n, budget)
        results.append(pieces)
    
    return results

if __name__ == "__main__":
    construct()
