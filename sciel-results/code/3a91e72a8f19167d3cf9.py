<<<<<<< SEARCH
    def has_short_cycle(adj, max_girth):
        # Check if there is a cycle of length < max_girth
        # max_girth is g, so we check for cycles of length 3..g-1
        n = len(adj)
        # BFS from each vertex up to depth g//2
        # If we find a cycle, return True
        # To be fast, we can just check up to depth 7 for g=16
        # A cycle of length L exists if there is a path of length L-1 between two adjacent vertices
        # Or simpler: BFS from u, if we reach a neighbor of u at depth d, cycle length d+1.
        # We need to ensure no cycle length < g.
        
        # Optimized: Check girth >= g
        # This is equivalent to: for all u, the ball of radius floor((g-1)/2) is a tree.
        # For g=16, radius 7.
        radius = (g - 1) // 2
        
        for u in range(n):
            visited = {u: 0}
            queue = [(u, 0)]
            while queue:
                node, dist = queue.pop(0)
                if dist == radius:
                    continue
                for nei in adj[node]:
                    if nei in visited:
                        # Found a cycle
                        # Cycle length is dist + 1 + visited[nei]? No.
                        # If we revisit a node, it's a cycle.
                        # In a tree, we only visit each node once.
                        # If we see a node already visited, it's a cycle.
                        # The length of the cycle is not simply determined, but existence is enough.
                        return True
                    else:
                        visited[nei] = dist + 1
                        queue.append((nei, dist + 1))
        return False
=======
    def has_short_cycle_fast(adj, max_girth):
        """
        Correct girth check using non-backtracking BFS.
        For each vertex u, do a BFS that doesn't backtrack.
        If we reach any vertex at distance >= radius, we have a cycle of length < max_girth.
        Actually: if the ball of radius floor((g-1)/2) around any vertex is not a tree,
        there's a cycle shorter than g.
        
        Better: BFS from u with parent tracking. If we encounter a node that was visited
        by a different path (not the parent), we found a cycle.
        Cycle length = dist_current + dist_visited + 1.
        We check if this is < max_girth.
        """
        n = len(adj)
        radius = (max_girth - 1) // 2
        
        for u in range(n):
            # BFS with parent tracking
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = [u]
            qi = 0
            found_cycle = False
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                if dist[node] >= radius:
                    continue
                for nei in adj[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        parent[nei] = node
                        queue.append(nei)
                    elif nei != parent[node]:
                        # Found a cycle
                        cycle_len = dist[node] + dist[nei] + 1
                        if cycle_len < max_girth:
                            return True
        return False
>>>>>>> REPLACE

