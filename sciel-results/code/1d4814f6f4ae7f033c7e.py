import time
import random
from random import Random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(edges_set):
        if not edges_set:
            return None
        
        # Build adjacency list
        adj = {i: set() for i in range(max(edges_set[0][0], edges_set[-1][1]) + 1) if edges_set}
        # Need to handle isolated nodes if any, but edges_set implies connected components
        # Let's just build adj for nodes present in edges
        nodes = set()
        for u, v in edges_set:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
            nodes.add(u)
            nodes.add(v)
            
        if not nodes:
            return None

        best_cycle = []
        best_len = 0
        
        # Try a few random edges as the "closing" edge
        edge_list = list(edges_set)
        random.shuffle(edge_list)
        
        # Limit number of trials to keep it fast
        trials = min(20, len(edge_list))
        
        for _ in range(trials):
            if not edge_list:
                break
            u, v = edge_list.pop()
            if (u, v) not in edges_set:
                continue
            if v not in adj.get(u, set()):
                continue
            
            # Find longest path from v to u in G - (u, v)
            # We use a DFS with a visited set
            # To find the *longest* path, we need to explore all possibilities?
            # That's exponential.
            # Instead, let's use a greedy heuristic for path finding.
            # Or, since N is small, we can use a simple DFS that backtracks and keeps the longest path found.
            # But worst case is still bad.
            # Let's use a randomized greedy path finder:
            # Start at v, try to walk to u, avoiding visited nodes.
            # Prefer neighbors with higher degree.
            
            path = [v]
            visited = {v}
            current = v
            
            # We want to reach u
            # Let's try a simple DFS that looks for *any* path first, then maybe extend?
            # No, let's stick to a heuristic that finds a reasonably long path.
            
            # Actually, for small N, we can use a DP-like approach or just a bounded DFS.
            # Let's try a DFS that explores neighbors in order of decreasing degree.
            # And we stop if we hit u.
            # If we don't hit u, we backtrack.
            # This finds *a* path, not necessarily the longest.
            
            # To get a longer path, we can try to find a path that doesn't get stuck early.
            # Let's use a simple random walk with backtracking (DFS) but with a limit on depth?
            # No, N is small.
            
            # Let's try a different approach:
            # Find a cycle by starting at a random node and doing a DFS, keeping the path.
            # If we hit a node already in the path, we found a cycle.
            # We want the longest one.
            
            # Let's use a simple heuristic:
            # 1. Pick a random start node s.
            # 2. Do a DFS from s, always choosing the neighbor with the highest degree that hasn't been visited.
            # 3. If we get stuck, backtrack.
            # 4. If we hit a node in the current path, extract the cycle.
            # 5. Keep the longest cycle found.
            
            # This is a standard heuristic for finding long cycles.
            
            def dfs_heuristic(start, adj, edges_set):
                best = []
                path = []
                visited = set()
                
                def dfs(node):
                    nonlocal best
                    path.append(node)
                    visited.add(node)
                    
                    # Check for cycle
                    # If current node has a neighbor in path (other than previous), we have a cycle
                    # Actually, we just need to check if we can close the cycle later.
                    # When we backtrack, we can check if the current node is connected to the start of the path?
                    # No, the cycle can start anywhere in the path.
                    
                    # Standard approach:
                    # When we visit a node, if it's already in the path, we found a cycle.
                    # But we are using a visited set for the whole search, so we don't revisit.
                    # We need to detect when we hit a node that is in the *current* path.
                    
                    # Let's restructure:
                    # We maintain a path list.
                    # When we move to a neighbor, if the neighbor is in the path, we found a cycle.
                    # The cycle is from the first occurrence of the neighbor to the end of the path.
                    
                    for neighbor in sorted(adj[node], key=lambda x: len(adj[x]), reverse=True):
                        if neighbor in visited:
                            if neighbor in path:
                                # Found a cycle
                                idx = path.index(neighbor)
                                cycle = path[idx:]
                                if len(cycle) > len(best):
                                    best = cycle[:]
                            continue
                        # Recurse
                        dfs(neighbor)
                        if len(best) >= len(path): # Pruning?
                            pass
                        path.pop()
                        visited.remove(node)
                    return
                
                dfs(start)
                return best

            # This DFS might be too slow if it explores too much.
            # But N=30, and we prune by not revisiting nodes in the *global* visited set?
            # No, for finding a single cycle, we should allow backtracking within the path.
            # The `visited` set above is global to the DFS call, which prevents revisiting nodes in different branches.
            # This is correct for finding *a* simple path/cycle.
            # But it might miss longer cycles if it gets stuck in a dead end early.
            # However, with degree-based ordering, it's a decent heuristic.
            
            # Let's try this for a few random starts.
            for _ in range(10):
                if not nodes:
                    break
                start = random.choice(list(nodes))
                cycle = dfs_heuristic(start, adj, edges_set)
                if len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle

        return best_cycle

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 10:
            # Fallback to edges only
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        edges_set = set(tuple(e) for e in edges)
        pieces = []
        
        while edges_set:
            if time.time() > deadline - 5:
                # Fallback
                for e in edges_set:
                    pieces.append(list(e))
                break
                
            cycle = find_longest_cycle(edges_set)
            if cycle:
                # Convert cycle to pieces and remove edges
                # Cycle is a list of vertices [v0, v1, ..., vk-1]
                # Edges are (v0, v1), (v1, v2), ..., (vk-1, v0)
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    if edge in edges_set:
                        edges_set.remove(edge)
            else:
                # No cycle found, take an edge
                e = next(iter(edges_set))
                pieces.append(list(e))
                edges_set.remove(e)
                
        results.append(pieces)
        
    return results
