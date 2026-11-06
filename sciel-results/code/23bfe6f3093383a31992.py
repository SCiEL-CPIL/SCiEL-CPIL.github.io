import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    graphs = [instance(*spec) for spec in specs]

    def max_weight_matching(weights, n):
        """
        Maximum weight perfect matching on a complete graph with n vertices.
        weights[i][j] = weight of edge (i,j), or None if edge doesn't exist.
        Uses Hungarian algorithm (assignment problem) for even n.
        Returns list of (i, j) pairs and total weight.
        """
        if n == 0:
            return [], 0
        if n % 2 != 0:
            # Can't have perfect matching on odd number of vertices
            # This shouldn't happen in our use case
            return [], float('-inf')
        
        # Hungarian algorithm for maximum weight assignment
        # We want to assign each vertex to exactly one other vertex, symmetrically
        # This is equivalent to finding a perfect matching in a complete graph
        
        # For maximum weight perfect matching, we can use the Hungarian algorithm
        # on a 2n x 2n matrix where we duplicate vertices
        
        # Actually, for a perfect matching on n vertices (n even), we can use
        # a simpler approach: the problem is equivalent to finding an assignment
        # where each vertex is matched to exactly one other.
        
        # Let's use a DP approach for small n, or Hungarian for larger
        if n <= 20:
            # DP over subsets
            INF = float('inf')
            dp = [INF] * (1 << n)
            dp[0] = 0
            parent = [None] * (1 << n)
            
            for mask in range(1 << n):
                if dp[mask] == INF:
                    continue
                # Find first unmatched vertex
                first = -1
                for i in range(n):
                    if not (mask >> i) & 1:
                        first = i
                        break
                if first == -1:
                    continue
                for j in range(first + 1, n):
                    if not (mask >> j) & 1:
                        if weights[first][j] is None:
                            continue
                        new_mask = mask | (1 << first) | (1 << j)
                        new_val = dp[mask] + weights[first][j]
                        if new_val > dp[new_mask]:
                            dp[new_mask] = new_val
                            parent[new_mask] = (first, j)
            
            total = dp[(1 << n) - 1]
            if total == INF:
                return [], float('-inf')
            
            # Reconstruct
            pairs = []
            mask = (1 << n) - 1
            while mask:
                i, j = parent[mask]
                pairs.append((i, j))
                mask ^= (1 << i)
                mask ^= (1 << j)
            return pairs, total
        else:
            # For larger n, use greedy + local search
            # Greedy: repeatedly pick highest weight edge
            used = [False] * n
            pairs = []
            total = 0
            # Create list of all edges sorted by weight descending
            edge_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if weights[i][j] is not None:
                        edge_list.append((weights[i][j], i, j))
            edge_list.sort(reverse=True)
            
            for w, i, j in edge_list:
                if not used[i] and not used[j]:
                    used[i] = used[j] = True
                    pairs.append((i, j))
                    total += w
                    if len(pairs) == n // 2:
                        break
            
            # Local search: try to improve by swapping
            improved = True
            while improved:
                improved = False
                # Try all pairs of edges and see if swapping improves
                for a in range(len(pairs)):
                    for b in range(a + 1, len(pairs)):
                        i1, j1 = pairs[a]
                        i2, j2 = pairs[b]
                        # Try (i1,i2) and (j1,j2)
                        if weights[i1][i2] is not None and weights[j1][j2] is not None:
                            new_w = weights[i1][i2] + weights[j1][j2]
                            old_w = weights[i1][j1] + weights[i2][j2]
                            if new_w > old_w:
                                pairs[a] = (i1, i2)
                                pairs[b] = (j1, j2)
                                total += new_w - old_w
                                improved = True
                        # Try (i1,j2) and (j1,i2)
                        if weights[i1][j2] is not None and weights[j1][i2] is not None:
                            new_w = weights[i1][j2] + weights[j1][i2]
                            old_w = weights[i1][j1] + weights[i2][j2]
                            if new_w > old_w:
                                pairs[a] = (i1, j2)
                                pairs[b] = (j1, i2)
                                total += new_w - old_w
                                improved = True
            
            return pairs, total

    def decompose(edges, time_limit):
        """
        Decompose edges into cycles and single edges using maximum weight matching
        on the directed line graph.
        """
        if not edges:
            return []

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Each directed edge becomes a node in our matching graph
        # Node representation: (u, v) means the directed edge from u to v
        directed_edges = []
        edge_to_node = {}
        for u, v in edges:
            directed_edges.append((u, v))
            directed_edges.append((v, u))
            edge_to_node[(u, v)] = len(directed_edges) - 2
            edge_to_node[(v, u)] = len(directed_edges) - 1

        n_nodes = len(directed_edges)
        if n_nodes == 0:
            return []

        # Build weight matrix for matching
        # A match between node i and node j means we follow directed edge i then directed edge j
        # Node i = (u_i, v_i), node j = (u_j, v_j)
        # They can be matched if v_i == u_j (the end of i is the start of j)
        # Weight: 1 if this continuation helps form a longer path/cycle, 0 otherwise
        # Actually, we want to maximize the number of continuations, which minimizes pieces
        
        # The weight for matching i to j is 1 if v_i == u_j, else -inf (or very negative)
        # But we need a perfect matching, so we need to handle the structure carefully
        
        # Actually, for a perfect matching on the line graph, each directed edge must be
        # matched to exactly one other directed edge. A match (i, j) means edge i is followed by edge j.
        # This forms paths and cycles in the original graph.
        
        # The number of pieces = number of paths + number of cycles
        # = n_directed_edges / 2 - number_of_matches_that_continue_paths
        # Wait, let me think again...
        
        # If we have a perfect matching on 2m nodes (m = number of undirected edges),
        # each match pairs two directed edges. If they form a valid continuation (v_i == u_j),
        # they're "connected". The number of pieces is the number of connected components
        # in the resulting graph of directed edges.
        
        # Actually, the standard approach: each directed edge has one outgoing match and one incoming match.
        # A perfect matching pairs each directed edge with exactly one other.
        # If directed edge (u,v) is matched with (v,w), that means we continue from (u,v) to (v,w).
        # The number of paths/cycles equals the number of "starts" which equals the number of matches
        # where the two directed edges are NOT continuations of each other.
        
        # So we want to maximize the number of "good" matches (where v_i == u_j).
        
        # Build the weight matrix
        weights = [[None] * n_nodes for _ in range(n_nodes)]
        
        for i in range(n_nodes):
            u_i, v_i = directed_edges[i]
            for j in range(i + 1, n_nodes):
                u_j, v_j = directed_edges[j]
                # Can i be followed by j? (v_i == u_j)
                # Can j be followed by i? (v_j == u_i)
                # We can only match i and j if at least one direction works
                if v_i == u_j or v_j == u_i:
                    # Weight: 1 if they form a continuation in at least one direction
                    # But we need to be careful about which direction
                    weights[i][j] = 1
                    weights[j][i] = 1
                else:
                    # They can't form a continuation, but we still need to match them
                    # Give them a very low weight
                    weights[i][j] = -1000
                    weights[j][i] = -1000
        
        # Solve maximum weight perfect matching
        if n_nodes <= 40:
            match_pairs, total_weight = max_weight_matching(weights, n_nodes)
        else:
            # For larger graphs, use a simpler heuristic
            # Greedy: match high-weight edges first
            used = [False] * n_nodes
            match_pairs = []
            total_weight = 0
            
            # Create list of all potential matches
            potential = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if weights[i][j] is not None:
                        potential.append((weights[i][j], i, j))
            potential.sort(reverse=True)
            
            for w, i, j in potential:
                if not used[i] and not used[j]:
                    used[i] = used[j] = True
                    match_pairs.append((i, j))
                    total_weight += w
                    if len(match_pairs) == n_nodes // 2:
                        break
            
            # If we didn't match all, add remaining
            remaining = [i for i in range(n_nodes) if not used[i]]
            for k in range(0, len(remaining) - 1, 2):
                i, j = remaining[k], remaining[k + 1]
                match_pairs.append((i, j))
                total_weight += weights[i][j] if weights[i][j] is not None else 0

        # Extract paths and cycles from the matching
        # Build next-edge mapping
        next_edge = [None] * n_nodes
        for i, j in match_pairs:
            # Determine direction: does i lead to j or j lead to i?
            u_i, v_i = directed_edges[i]
            u_j, v_j = directed_edges[j]
            if v_i == u_j:
                next_edge[i] = j
                next_edge[j] = i  # This is wrong, let me fix
            elif v_j == u_i:
                next_edge[j] = i
                next_edge[i] = j  # This is wrong too
            else:
                # They don't form a continuation, so they're separate
                next_edge[i] = -1
                next_edge[j] = -1
        
        # Actually, the matching pairs two directed edges. The pair represents a "link" between them.
        # To form paths, we need to follow the links.
        # Let me reconsider: each match (i, j) means directed edge i and directed edge j are adjacent in a path/cycle.
        # If v_i == u_j, then i comes before j in the path.
        # If v_j == u_i, then j comes before i in the path.
        
        # Build adjacency for the path/cycle graph
        path_adj = defaultdict(set)
        for i, j in match_pairs:
            u_i, v_i = directed_edges[i]
            u_j, v_j = directed_edges[j]
            if v_i == u_j:
                # i -> j
                path_adj[i].add(j)
                path_adj[j].add(i)
            elif v_j == u_i:
                # j -> i
                path_adj[i].add(j)
                path_adj[j].add(i)
            else:
                # No valid continuation, they're separate pieces
                pass
        
        # Find all paths and cycles
        visited_nodes = set()
        pieces = []
        
        for start in range(n_nodes):
            if start in visited_nodes:
                continue
            
            # Follow the path
            path = [start]
            visited_nodes.add(start)
            current = start
            
            while True:
                # Find next node
                next_node = None
                for neighbor in path_adj.get(current, []):
                    if neighbor not in visited_nodes:
                        next_node = neighbor
                        break
                
                if next_node is None:
                    # Check if we can close a cycle
                    # The last node in path should connect back to the first
                    last = path[-1]
                    u_last, v_last = directed_edges[last]
                    u_first, v_first = directed_edges[path[0]]
                    if v_last == u_first:
                        # Cycle!
                        pieces.append([directed_edges[x][0] for x in path])
                    else:
                        # Path
                        pieces.append([directed_edges[x][0] for x in path])
                    break
                else:
                    path.append(next_node)
                    visited_nodes.add(next_node)
                    current = next_node
        
        # Convert to proper pieces format
        final_pieces = []
        for piece in pieces:
            if len(piece) >= 3:
                # Check if it's a valid cycle
                # The piece is a list of vertices, but we need to verify it's a cycle
                # Actually, we just need to output the vertices in order
                # Remove duplicates if any
                if len(set(piece)) == len(piece):
                    final_pieces.append(piece)
                else:
                    # Fallback to edges
                    for i in range(len(piece) - 1):
                        u, v = piece[i], piece[i + 1]
                        final_pieces.append([min(u, v), max(u, v)])
            else:
                # Single edge
                if len(piece) == 2:
                    final_pieces.append([min(piece[0], piece[1]), max(piece[0], piece[1])])
        
        # Verify and fix: ensure all edges are covered exactly once
        edge_count = defaultdict(int)
        for piece in final_pieces:
            if len(piece) == 2:
                u, v = piece
                edge_count[(min(u, v), max(u, v))] += 1
            else:
                for i in range(len(piece)):
                    u = piece[i]
                    v = piece[(i + 1) % len(piece)]
                    edge_count[(min(u, v), max(u, v))] += 1
        
        # Check coverage
        original_edges = set((min(u, v), max(u, v)) for u, v in edges)
        for e in original_edges:
            if edge_count.get(e, 0) != 1:
                # Fallback to simple decomposition
                return decompose_simple(edges)
        
        return final_pieces

    def decompose_simple(edges):
        """Simple fallback decomposition."""
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining = set((min(u, v), max(u, v)) for u, v in edges)
        pieces = []
        
        while remaining:
            # Try to find a cycle
            cycle = None
            for start in list(adj.keys()):
                if len(adj[start]) < 2:
                    continue
                visited = {start}
                path = [start]
                
                def dfs(node):
                    for neighbor in sorted(adj[node]):
                        if neighbor == start and len(path) >= 3:
                            return path[:]
                        if neighbor not in visited:
                            visited.add(neighbor)
                            path.append(neighbor)
                            result = dfs(neighbor)
                            if result:
                                return result
                            path.pop()
                            visited.discard(neighbor)
                    return None
                
                cycle = dfs(start)
                if cycle:
                    break
            
            if cycle:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
            else:
                # Take a single edge
                u, v = next(iter(remaining))
                pieces.append([u, v])
                remaining.discard((u, v))
                adj[u].discard(v)
                adj[v].discard(u)
        
        return pieces

    results = []
    for i, edges in enumerate(graphs):
        remaining_time = deadline - time.time()
        time_per_graph = remaining_time / (len(graphs) - i)
        pieces = decompose(edges, time.time() + time_per_graph)
        results.append(pieces)

    return results
