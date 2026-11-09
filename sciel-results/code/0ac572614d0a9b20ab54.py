import time
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)
import random


def construct():
    start_time = time.time()
    DEADLINE = 110.0  # safety margin before the 150 s kill

    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_any_cycle(adj):
        """Guaranteed cycle finder (undirected DFS back-edge). Returns vertex list or None."""
        color = {}
        parent = {}

        def dfs(u):
            color[u] = 1
            for w in adj[u]:
                if w == parent.get(u):
                    continue
                if color.get(w, 0) == 0:
                    parent[w] = u
                    res = dfs(w)
                    if res is not None:
                        return res
                elif color.get(w, 0) == 1:
                    cyc = []
                    cur = u
                    while cur != w:
                        cyc.append(cur)
                        cur = parent[cur]
                    cyc.append(w)
                    cyc.reverse()
                    if len(cyc) >= 3:
                        return cyc
            color[u] = 2
            return None

        for v in list(adj.keys()):
            if color.get(v, 0) == 0:
                res = dfs(v)
                if res is not None:
                    return res
        return None

    def find_cycle_with_randomization(adj, rng, budget=10000):
        """
        Finds a cycle by starting from a random edge and trying to close it.
        Uses a randomized DFS to find a path from v to u that doesn't use the edge (u,v).
        """
        if not adj:
            return None
        
        # Pick a random edge
        all_edges = []
        for u in adj:
            for v in adj[u]:
                if u < v:
                    all_edges.append((u, v))
        
        if not all_edges:
            return None
            
        # Try a few random edges
        for _ in range(10):
            u, v = rng.choice(all_edges)
            # We want a cycle containing (u,v).
            # This is equivalent to finding a path from u to v in the graph without using edge (u,v).
            # If such a path exists, path + (u,v) is a cycle.
            
            # Temporarily remove edge (u,v)
            adj[u].discard(v)
            adj[v].discard(u)
            
            # BFS/DFS from u to v
            visited = {u}
            parent = {u: None}
            queue = [u]
            found = False
            steps = 0
            max_steps = budget // 10
            
            while queue and steps < max_steps:
                steps += 1
                curr = queue.pop(0)
                if curr == v:
                    found = True
                    break
                for w in adj[curr]:
                    if w not in visited:
                        visited.add(w)
                        parent[w] = curr
                        queue.append(w)
            
            # Restore edge
            adj[u].add(v)
            adj[v].add(u)
            
            if found:
                # Reconstruct path
                path = []
                curr = v
                while curr is not None:
                    path.append(curr)
                    curr = parent[curr]
                path.reverse()
                # path is u...v
                # Cycle is path + closing edge (v,u) which is already in graph
                # The cycle vertices are path
                if len(path) >= 3:
                    return path
        return None

    def decompose(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(edges)
        pieces = []
        
        if not remaining:
            return pieces

        # 1. Generate a pool of candidate cycles
        # We want diverse cycles.
        pool = []
        rng = Random(42) # Fixed seed for reproducibility within the run, or use time-based? Fixed is better for stability.
        
        # Time budget for pool generation
        pool_deadline = start_time + 5.0 # 5 seconds for pool generation
        
        # Strategy:
        # a) Try to find long cycles using randomized DFS from high-degree nodes.
        # b) Try to find cycles containing specific random edges.
        
        # Helper to add a cycle to pool if it's valid and disjoint from others? 
        # No, pool can have overlapping cycles. We select disjoint ones later.
        
        # Add cycles found by randomization
        while time.time() < pool_deadline and len(pool) < 200:
            # Randomize neighbor order for diversity
            # We can't easily randomize adj in place without copy, so we simulate by shuffling in the finder?
            # Let's just call find_cycle_with_randomization which uses random edges.
            cyc = find_cycle_with_randomization(adj, rng, budget=20000)
            if cyc:
                # Check if this cycle is already in pool (by edge set)
                edge_set = frozenset((min(cyc[i], cyc[(i+1)%len(cyc)]), max(cyc[i], cyc[(i+1)%len(cyc)])) for i in range(len(cyc)))
                is_dup = False
                for p in pool:
                    if frozenset((min(p[i], p[(i+1)%len(p)]), max(p[i], p[(i+1)%len(p)])) for i in range(len(p))) == edge_set:
                        is_dup = True
                        break
                if not is_dup:
                    pool.append(cyc)
            
            # Also try a standard DFS with randomized start
            if adj:
                start_node = rng.choice(list(adj.keys()))
                # Simple DFS for any cycle starting near start_node?
                # The random edge method is more targeted.
                
        # Also add some "greedy" long cycles from the original deterministic finder to ensure we have good ones
        # Re-run find_long_cycle a few times with different random perturbations?
        # The current find_long_cycle is deterministic. 
        # Let's just add the result of find_long_cycle once.
        # But we need to be careful: find_long_cycle modifies nothing, just reads.
        
        # Let's add a few cycles from a simple "random walk" heuristic
        for _ in range(50):
             if time.time() > pool_deadline: break
             # Pick random start
             if not adj: break
             start = rng.choice(list(adj.keys()))
             # Do a random walk trying to return to start
             path = [start]
             on_path = {start}
             for _ in range(50): # Max path length
                 if not adj[start]: break
                 neighbors = list(adj[start])
                 if not neighbors: break
                 next_node = rng.choice(neighbors)
                 if next_node == path[0] and len(path) >= 3:
                     pool.append(list(path))
                     break
                 elif next_node not in on_path:
                     path.append(next_node)
                     on_path.add(next_node)
                     start = next_node
                 else:
                     # Stuck or looped without closing to start?
                     # Try to close to any node in path?
                     # This is getting complicated. The random edge method is more robust.
                     break

        # 2. Greedy Set Cover Selection
        # Sort pool by length descending
        pool.sort(key=len, reverse=True)
        
        selected_cycles = []
        used_edges = set()
        
        for cyc in pool:
            # Check if all edges in cyc are in remaining
            valid = True
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i+1)%len(cyc)]
                e = (min(a, b), max(a, b))
                if e not in remaining:
                    valid = False
                    break
            if valid:
                # Select this cycle
                selected_cycles.append(cyc)
                for i in range(len(cyc)):
                    a, b = cyc[i], cyc[(i+1)%len(cyc)]
                    e = (min(a, b), max(a, b))
                    used_edges.add(e)
                    remaining.discard(e)
                    
        # 3. Cover remaining edges with singletons
        for e in remaining:
            pieces.append([e[0], e[1]])
            
        # Add the selected cycles
        for cyc in selected_cycles:
            pieces.append(list(cyc))
            
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        results.append(decompose(edges))
    return results
