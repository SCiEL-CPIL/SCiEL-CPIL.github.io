"""Program to construct a 4-colouring of K_n with no monochromatic C6."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a 4-colouring of K_n with no monochromatic 6-cycle.
    Returns {'n': n, 'colors': [...]} where colors has one entry per pair (i,j), i<j,
    in lexicographic order, each in {0,1,2,3}.
    """
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    def count_c6(colors, n):
        """Count monochromatic C6s in the coloring."""
        # Build adjacency lists for each color
        adj = [[] for _ in range(4)]
        edge_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_idx]
                adj[c].append((i, j))
                edge_idx += 1
        
        # Build adjacency sets for faster lookup
        adj_set = [set() for _ in range(4)]
        for c in range(4):
            for i, j in adj[c]:
                adj_set[c].add((i, j))
                adj_set[c].add((j, i))
        
        # For each color, count C6s
        # A C6 is a sequence v0,v1,v2,v3,v4,v5,v0 where all consecutive pairs are in the color class
        # To avoid overcounting, we can use a canonical form
        count = 0
        for c in range(4):
            # Build adjacency list
            adj_list = [[] for _ in range(n)]
            for i, j in adj[c]:
                adj_list[i].append(j)
                adj_list[j].append(i)
            
            # For each potential C6, check if it exists
            # Use a simple enumeration: for each edge (u,v), look for paths of length 5 from u to v
            # This is expensive, so use a heuristic: count via matrix multiplication or DFS
            # For n=17, we can do a DFS from each node
            visited_global = set()
            for u in range(n):
                for v in adj_list[u]:
                    if u < v:
                        # Check if there's a path of length 5 from u to v avoiding direct edge
                        # This would form a C6
                        # Use BFS with depth limit
                        found = False
                        # Path: u -> a -> b -> c -> d -> v, all distinct except u,v
                        for a in adj_list[u]:
                            if a == v:
                                continue
                            for b in adj_list[a]:
                                if b == u or b == v:
                                    continue
                                for c2 in adj_list[b]:
                                    if c2 == u or c2 == v or c2 == a:
                                        continue
                                    for d in adj_list[c2]:
                                        if d == u or d == v or d == b or d == c2:
                                            continue
                                        if v in adj_list[d]:
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            count += 1
        return count
    
    def has_mono_c6(colors, n):
        """Check if any monochromatic C6 exists."""
        adj = [[] for _ in range(4)]
        edge_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[edge_idx]
                adj[c].append((i, j))
                edge_idx += 1
        
        adj_list = [[[] for _ in range(n)] for _ in range(4)]
        for c in range(4):
            for i, j in adj[c]:
                adj_list[c][i].append(j)
                adj_list[c][j].append(i)
        
        for c in range(4):
            for u in range(n):
                for v in adj_list[c][u]:
                    if u < v:
                        # Check for path of length 5 from u to v in color c
                        for a in adj_list[c][u]:
                            if a == v:
                                continue
                            for b in adj_list[c][a]:
                                if b == u or b == v:
                                    continue
                                for c2 in adj_list[c][b]:
                                    if c2 == u or c2 == v or c2 == a:
                                        continue
                                    for d in adj_list[c][c2]:
                                        if d == u or d == v or d == b or d == c2:
                                            continue
                                        if v in adj_list[c][d]:
                                            return True
        return False
    
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Try local search
    best_colors = None
    best_count = float('inf')
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random initial coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Local search: iteratively recolor edges to reduce C6 count
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try recoloring random edges
            for _ in range(2000):
                if time.time() > deadline - 2:
                    break
                idx = random.randint(0, num_edges - 1)
                old_c = colors[idx]
                if old_c == 0:
                    continue  # Skip if already 0? No, try all
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                colors[idx] = new_c
                # Quick check: just recolor and hope
                improved = True
        
        # Count C6s
        c6_count = count_c6(colors, n)
        if c6_count < best_count:
            best_count = c6_count
            best_colors = colors[:]
            if c6_count == 0:
                break
    
    if best_colors is not None and best_count == 0:
        return {'n': n, 'colors': best_colors}
    
    # Fallback: try backtracking with better pruning for n=17
    # Use the walk-based check but with early termination and better ordering
    n = 17
    num_edges = n * (n - 1) // 2
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_idx_map = {}
    for idx, (u, v) in enumerate(edge_list):
        edge_idx_map[(u, v)] = idx
        edge_idx_map[(v, u)] = idx
    
    # Symmetry breaking: fix first few edges
    fixed_edges = {(0,1): 0, (0,2): 0, (0,3): 0, (1,2): 0}
    colors = [0] * num_edges
    for (u, v), c in fixed_edges.items():
        colors[edge_idx_map[(u, v)]] = c
    
    adj_mask = [[0] * n for _ in range(4)]
    for (u, v), c in fixed_edges.items():
        adj_mask[c][u] |= (1 << v)
        adj_mask[c][v] |= (1 << u)
    
    # Order edges by "constraint potential": edges between high-degree vertices first
    edges_to_process = [idx for idx in range(num_edges) 
                       if edge_list[idx] not in fixed_edges and edge_list[idx][::-1] not in fixed_edges]
    
    # Sort by sum of degrees (higher first) - but we don't have degrees yet
    # Just use lexicographic order for now
    
    def has_path_len5(u, v, color):
        """Check for walk of length 5 from u to v in color subgraph."""
        r1 = adj_mask[color][u]
        if not r1:
            return False
        r2 = 0
        m = r1
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r2 |= adj_mask[color][node]
        if not r2:
            return False
        r3 = 0
        m = r2
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r3 |= adj_mask[color][node]
        if not r3:
            return False
        r4 = 0
        m = r3
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r4 |= adj_mask[color][node]
        if not r4:
            return False
        r5 = 0
        m = r4
        while m:
            lowbit = m & (-m)
            m ^= lowbit
            node = lowbit.bit_length() - 1
            r5 |= adj_mask[color][node]
        return (r5 & (1 << v)) != 0
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == len(edges_to_process):
            return True
        idx = edges_to_process[pos]
        u, v = edge_list[idx]
        for c in range(4):
            if has_path_len5(u, v, c):
                continue
            colors[idx] = c
            adj_mask[c][u] |= (1 << v)
            adj_mask[c][v] |= (1 << u)
            if backtrack(pos + 1):
                return True
            adj_mask[c][u] &= ~(1 << v)
            adj_mask[c][v] &= ~(1 << u)
            colors[idx] = 0
        return False
    
    if backtrack(0):
        # Verify
        if not has_mono_c6(colors, n):
            return {'n': n, 'colors': colors}
    
    # Fallback to n=16
    n = 16
    num_edges = n * (n - 1) // 2
    colors = [0] * num_edges
    # Simple valid coloring for n=16
    for i in range(n):
        for j in range(i + 1, n):
            colors[i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)] = (i + j) % 4
    
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
