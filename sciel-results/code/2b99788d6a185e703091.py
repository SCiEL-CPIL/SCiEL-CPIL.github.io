# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import heapq
    from collections import deque

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

    def ac3(domains, adj):
        """
        Arc Consistency 3 algorithm.
        domains: list of sets
        adj: list of lists of neighbors
        Returns True if consistent, False if a domain becomes empty.
        """
        queue = deque()
        n = len(domains)
        for i in range(n):
            for j in adj[i]:
                queue.append((i, j))
        
        while queue:
            i, j = queue.popleft()
            if revise(domains, i, j):
                if not domains[i]:
                    return False
                for k in adj[i]:
                    if k != j:
                        queue.append((k, i))
        return True

    def revise(domains, i, j):
        """
        Removes values from domains[i] that have no support in domains[j].
        Returns True if domains[i] was revised.
        """
        revised = False
        for x in list(domains[i]):
            if not any(x != y for y in domains[j]):
                domains[i].discard(x)
                revised = True
        return revised

    def solve(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        domains = [set(lst) for lst in lists]
        
        # Preprocess with AC-3
        if not ac3(domains, adj):
            # Should not happen for valid instances, but handle gracefully
            return [lists[v][0] for v in range(n)]

        # Backtracking search
        def backtrack(assigned):
            # Find unassigned vertex with minimum remaining values (MRV)
            best_v = -1
            best_count = None
            for v in range(n):
                if v not in assigned:
                    cnt = len(domains[v])
                    if cnt == 0:
                        return None
                    if best_count is None or cnt < best_count:
                        best_v = v
                        best_count = cnt
            
            if best_v == -1:
                return {v: c for v, c in assigned.items()}
            
            # Try each color in domain of best_v
            for c in sorted(domains[best_v]):
                # Check compatibility with assigned neighbors
                conflict = False
                for nb in adj[best_v]:
                    if nb in assigned and assigned[nb] == c:
                        conflict = True
                        break
                if conflict:
                    continue
                
                # Forward Checking
                # Save state for backtracking
                saved_domains = [d.copy() for d in domains]
                new_assigned = dict(assigned)
                new_assigned[best_v] = c
                
                ok = True
                for nb in adj[best_v]:
                    if nb not in new_assigned:
                        if c in domains[nb]:
                            domains[nb].discard(c)
                            if len(domains[nb]) == 0:
                                ok = False
                                break
                
                if not ok:
                    # Restore domains
                    for i in range(n):
                        domains[i] = saved_domains[i]
                    continue
                
                result = backtrack(new_assigned)
                if result is not None:
                    return result
                
                # Restore domains for next iteration
                for i in range(n):
                    domains[i] = saved_domains[i]
            
            return None
        
        result = backtrack({})
        if result is None:
            return [lists[v][0] for v in range(n)]
        return [result[v] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve(n, edges, lists)
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    return out
# EVOLVE-BLOCK-END
