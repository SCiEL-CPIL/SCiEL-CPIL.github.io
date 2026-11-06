"""Decompose graphs into minimum number of cycles and single edges."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict, deque
import heapq

def construct():
    from random import Random
    import math

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_min_weight_perfect_matching(dist, nodes):
        """
        Find min weight perfect matching on a complete graph with 'nodes'.
        dist is a dict of (i, j) -> weight, where i, j are indices in nodes.
        Returns list of pairs of node indices (indices into 'nodes').
        """
        k = len(nodes)
        if k == 0:
            return []
        # DP over subsets
        # dp[mask] = min cost to match nodes in mask
        # mask is a bitmask of which nodes are already matched
        # We iterate through masks.
        # To save memory, we can use a dict or array of size 2^k.
        # k <= 30 is too big for 2^k. But k is number of odd vertices.
        # In random graphs, number of odd vertices is usually small?
        # Sum of degrees is 2|E|. Number of odd vertices is even.
        # For n=30, p=0.3, E ~ 135. Odd vertices could be up to 30.
        # 2^30 is 1e9. Too big.
        # We need a heuristic or a better algorithm.
        # Since graph is small, maybe use a simple greedy or local search?
        # Or, since we just need to minimize edges removed, and edges removed are paths,
        # maybe we can just pair them greedily by shortest path?
        
        # Let's try a simple heuristic: Greedy pairing by shortest distance.
        # This is not guaranteed optimal but likely good enough.
        # Or, use a simple local search: start with random matching, swap to improve.
        
        # Given the constraint of 120s, and small N, let's try a simple DP if k <= 20, else heuristic.
        if k <= 20:
            size = 1 << k
            dp = [float('inf')] * size
            dp[0] = 0
            parent = [None] * size # store (prev_mask, node) to reconstruct
            for mask in range(size):
                if dp[mask] == float('inf'):
                    continue
                # Find first unmatched node
                first = -1
                for i in range(k):
                    if not (mask & (1 << i)):
                        first = i
                        break
                if first == -1:
                    continue
                # Try to match first with any other unmatched node
                for j in range(k):
                    if i == j or (mask & (1 << j)):
                        continue
                    new_mask = mask | (1 << i) | (1 << j)
                    cost = dp[mask] + dist.get((i, j), dist.get((j, i), float('inf')))
                    if cost < dp[new_mask]:
                        dp[new_mask] = cost
                        parent[new_mask] = (mask, j)
                        
            full_mask = (1 << k) - 1
            if dp[full_mask] == float('inf'):
                return []
                
            # Reconstruct
            pairs = []
            mask = full_mask
            while mask != 0:
                prev_mask, j = parent[mask]
                # Find i in mask that is not in prev_mask
                diff = mask ^ prev_mask
                # diff has two bits set
                i1 = (diff & -diff).bit_length() - 1
                i2 = (diff // (1 << i1) & 1)
                # Actually simpler:
                bits = []
                temp = diff
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    temp ^= lsb
                i1, i2 = bits
                pairs.append((i1, i2))
                mask = prev_mask
            return pairs
        else:
            # Heuristic: Greedy nearest neighbor
            # Create list of all pairs sorted by distance
            all_pairs = []
            for i in range(k):
                for j in range(i+1, k):
                    w = dist.get((i, j), dist.get((j, i), float('inf')))
                    all_pairs.append((w, i, j))
            all_pairs.sort()
            
            used = [False] * k
            pairs = []
            for w, i, j in all_pairs:
                if not used[i] and not used[j]:
                    pairs.append((i, j))
                    used[i] = True
                    used[j] = True
            return pairs

    def decompose(edges, n):
        # Build adjacency
        adj = defaultdict(list)
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u,v), max(u,v)))
        
        # 1. Find odd degree vertices
        odd_vertices = [u for u in range(n) if len(adj[u]) % 2 == 1]
        
        removed_edges = set()
        
        if odd_vertices:
            # Map odd vertices to indices
            odd_to_idx = {v: i for i, v in enumerate(odd_vertices)}
            k = len(odd_vertices)
            
            # Compute shortest paths between all odd vertices
            # Use Dijkstra from each odd vertex
            dist = {}
            for i, start in enumerate(odd_vertices):
                # Dijkstra
                dist_to = {v: float('inf') for v in range(n)}
                prev = {v: -1 for v in range(n)}
                dist_to[start] = 0
                pq = [(0, start)]
                while pq:
                    d, u = heapq.heappop(pq)
                    if d > dist_to[u]:
                        continue
                    for v in adj[u]:
                        nd = d + 1
                        if nd < dist_to[v]:
                            dist_to[v] = nd
                            prev[v] = u
                            heapq.heappush(pq, (nd, v))
                
                for j, end in enumerate(odd_vertices):
                    if i < j:
                        # Reconstruct path to get edges
                        if dist_to[end] < float('inf'):
                            path = []
                            cur = end
                            while cur != -1:
                                path.append(cur)
                                if cur == start:
                                    break
                                cur = prev[cur]
                            path.reverse()
                            for x in range(len(path)-1):
                                u, v = path[x], path[x+1]
                                removed_edges.add((min(u,v), max(u,v)))
                            dist[(i, j)] = dist_to[end]
                            dist[(j, i)] = dist_to[end]
                        else:
                            dist[(i, j)] = float('inf')
                            dist[(j, i)] = float('inf')
            
            # Solve matching
            pairs = solve_min_weight_perfect_matching(dist, odd_vertices)
            
            # Add edges from matched pairs to removed_edges
            # Wait, the dist calculation already added edges to removed_edges?
            # No, I calculated dist, but I need to explicitly remove the edges corresponding to the chosen matching paths.
            # The `dist` dict only has costs. I need the actual paths.
            # Let's re-calculate paths for the chosen pairs.
            
            # Reset removed_edges to just the paths from the matching
            removed_edges = set()
            
            for i, j in pairs:
                u = odd_vertices[i]
                v = odd_vertices[j]
                # Find shortest path between u and v
                # We can re-run Dijkstra or store paths. Re-running is fine for small N.
                dist_to = {v: float('inf') for v in range(n)}
                prev = {v: -1 for v in range(n)}
                dist_to[u] = 0
                pq = [(0, u)]
                while pq:
                    d, node = heapq.heappop(pq)
                    if d > dist_to[node]:
                        continue
                    for nb in adj[node]:
                        nd = d + 1
                        if nd < dist_to[nb]:
                            dist_to[nb] = nd
                            prev[nb] = node
                            heapq.heappush(pq, (nd, nb))
                
                # Reconstruct path
                if dist_to[v] < float('inf'):
                    path = []
                    cur = v
                    while cur != -1:
                        path.append(cur)
                        if cur == u:
                            break
                        cur = prev[cur]
                    path.reverse()
                    for x in range(len(path)-1):
                        a, b = path[x], path[x+1]
                        removed_edges.add((min(a,b), max(a,b)))
        
        # 2. Remove edges from graph
        remaining_edges = set()
        for e in edge_set:
            if e not in removed_edges:
                remaining_edges.add(e)
        
        # 3. Decompose remaining graph into cycles
        pieces = []
        # Add removed edges as single pieces
        for e in removed_edges:
            pieces.append([e[0], e[1]])
            
        # Now decompose remaining_edges
        rem_adj = defaultdict(list)
        for u, v in remaining_edges:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
            
        # Helper to find a cycle
        def find_cycle(adj):
            visited = set()
            for start in adj:
                if start in visited:
                    continue
                stack = [(start, -1, [start])]
                visited.add(start)
                while stack:
                    node, par, path = stack.pop()
                    for nb in adj[node]:
                        if nb == par:
                            continue
                        if nb in visited:
                            if nb in path:
                                idx = path.index(nb)
                                cycle = path[idx:]
                                if len(cycle) >= 3:
                                    return cycle
                        else:
                            visited.add(nb)
                            stack.append((nb, node, path + [nb]))
            return None

        # Iteratively extract cycles
        while remaining_edges:
            cycle = find_cycle(rem_adj)
            if not cycle:
                # Should not happen if all degrees are even, but just in case
                # If no cycle found, take any edge
                # This implies the "even" logic failed or graph is disconnected with isolated edges?
                # If all degrees are even, every component has a cycle (unless it's a single vertex with 0 degree).
                # If there are edges, there must be a cycle.
                # So if find_cycle returns None, something is wrong.
                # Fallback: take an edge
                u, v = next(iter(remaining_edges))
                pieces.append([u, v])
                remaining_edges.discard((min(u,v), max(u,v)))
                if v in rem_adj[u]: rem_adj[u].remove(v)
                if u in rem_adj[v]: rem_adj[v].remove(u)
                continue
            
            pieces.append(cycle)
            # Remove cycle edges
            for i in range(len(cycle)-1):
                u, v = cycle[i], cycle[i+1]
                e = (min(u,v), max(u,v))
                remaining_edges.discard(e)
                if v in rem_adj[u]: rem_adj[u].remove(v)
                if u in rem_adj[v]: rem_adj[v].remove(u)
                
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
