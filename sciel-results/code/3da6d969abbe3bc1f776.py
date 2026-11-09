import time
import random
import sys
from itertools import combinations

def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_small_subgraph(edges_list, n_vertices):
        """
        Finds the minimum number of pieces (cycles and edges) to cover edges_list.
        edges_list: list of (u, v) tuples.
        Uses DP over subsets of edges if len(edges_list) <= 16.
        Returns list of pieces (cycles as lists of vertices, edges as [u,v]).
        """
        m = len(edges_list)
        if m == 0:
            return []
        
        # Map edges to indices
        edge_to_idx = {e: i for i, e in enumerate(edges_list)}
        
        # Precompute all valid cycles and edges as bitmasks
        # A piece is either a single edge (mask with 1 bit) or a cycle (mask with k bits)
        # We need to check if a set of edges forms a valid cycle.
        
        valid_pieces = []
        
        # 1. Single edges are always valid pieces
        for i in range(m):
            valid_pieces.append((1 << i, [edges_list[i]])) # Mask, representation (list of edges for now, will convert)
            
        # 2. Find all simple cycles in the subgraph
        # Build adjacency for subgraph
        adj = {i: set() for i in range(n_vertices)}
        for u, v in edges_list:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find cycles of length 3 to min(m, 6?) 
        # Actually, any subset of edges that forms a cycle.
        # Iterate over all subsets? No, iterate over combinations of edges that could form a cycle.
        # A cycle of length k uses k edges.
        # We can just check all subsets of size >= 3?
        # For m <= 16, 2^16 is 65536. We can iterate all masks and check if it's a cycle.
        
        def is_cycle(mask):
            # Check if edges in mask form a single simple cycle
            # 1. Number of edges k = popcount(mask)
            # 2. If k < 3, return False
            # 3. Each vertex in the induced subgraph must have degree 2
            # 4. The subgraph must be connected
            
            k = bin(mask).count('1')
            if k < 3:
                return False
            
            # Get involved vertices
            verts = set()
            deg = {}
            for i in range(m):
                if mask & (1 << i):
                    u, v = edges_list[i]
                    verts.add(u)
                    verts.add(v)
                    deg[u] = deg.get(u, 0) + 1
                    deg[v] = deg.get(v, 0) + 1
            
            # Check degree 2 for all involved vertices
            for v in verts:
                if deg.get(v, 0) != 2:
                    return False
            
            # Check connectivity (simple DFS/BFS on the subgraph)
            if not verts:
                return False
            start = next(iter(verts))
            visited = {start}
            stack = [start]
            # Build local adj for these edges only
            local_adj = {v: [] for v in verts}
            for i in range(m):
                if mask & (1 << i):
                    u, v = edges_list[i]
                    local_adj[u].append(v)
                    local_adj[v].append(u)
            
            while stack:
                u = stack.pop()
                for v in local_adj[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            
            if len(visited) != len(verts):
                return False
            
            return True

        # Precompute masks for all valid cycles
        # Optimization: Only check masks with popcount >= 3
        # For m=16, iterating all 65536 masks is fast.
        for mask in range(1, 1 << m):
            if bin(mask).count('1') >= 3:
                if is_cycle(mask):
                    # Reconstruct the cycle vertices
                    # We need the order of vertices.
                    # Find a start vertex, walk the cycle.
                    verts_in_mask = set()
                    for i in range(m):
                        if mask & (1 << i):
                            u, v = edges_list[i]
                            verts_in_mask.add(u)
                            verts_in_mask.add(v)
                    
                    # Pick a start
                    start = next(iter(verts_in_mask))
                    cycle_verts = [start]
                    # Find neighbor of start in the cycle
                    neighbors = []
                    for i in range(m):
                        if mask & (1 << i):
                            u, v = edges_list[i]
                            if u == start:
                                neighbors.append(v)
                            elif v == start:
                                neighbors.append(u)
                    
                    # There should be exactly 2 neighbors
                    if len(neighbors) != 2:
                        continue # Should not happen if is_cycle passed
                    
                    # Walk
                    prev = start
                    curr = neighbors[0]
                    while curr != start:
                        cycle_verts.append(curr)
                        # Find next neighbor of curr, not prev
                        nexts = []
                        for i in range(m):
                            if mask & (1 << i):
                                u, v = edges_list[i]
                                if u == curr:
                                    nexts.append(v)
                                elif v == curr:
                                    nexts.append(u)
                        # One is prev, one is next
                        next_curr = None
                        for nb in nexts:
                            if nb != prev:
                                next_curr = nb
                                break
                        prev, curr = curr, next_curr
                        if curr is None:
                            break
                    
                    valid_pieces.append((mask, cycle_verts))

        # DP over subsets
        # dp[mask] = min pieces to cover edges in mask
        # best_prev[mask] = previous mask
        # best_piece[mask] = the piece added
        
        INF = float('inf')
        dp = [INF] * (1 << m)
        dp[0] = 0
        best_prev = [-1] * (1 << m)
        best_piece = [None] * (1 << m)
        
        # Iterate masks
        for mask in range(1, 1 << m):
            # Try adding each valid piece to a submask
            # This is O(2^m * #pieces). #pieces can be large.
            # Optimization: Iterate valid pieces, and for each, iterate masks that contain it?
            # Or: For each mask, find the lowest set bit, and try all pieces containing that bit?
            
            # Let's stick to simple: For each mask, find min over all pieces p such that p is subset of mask?
            # No, dp[mask] = min(dp[mask ^ p] + 1) for all p subset of mask.
            # This is slow if #pieces is large.
            
            # Better: Iterate through valid_pieces.
            # For each piece p, iterate through all masks that are supersets of p?
            # That's also slow.
            
            # Standard subset DP:
            # For each mask, we can iterate through all submasks? No.
            
            # Let's optimize the inner loop.
            # We can precompute for each mask, the list of valid pieces that are subsets of it?
            # Too much memory.
            
            # Alternative: Since m is small (<=16), we can just do:
            # for mask in 1..2^m:
            #   for p in valid_pieces:
            #      if (mask & p) == p: # p is subset of mask
            #          if dp[mask ^ p] + 1 < dp[mask]:
            #             ...
            
            # #valid_pieces is at most 2^m. So O(4^m). 4^16 = 4e9. Too slow.
            
            # We need a faster DP.
            # Note: A valid piece is a cycle or an edge.
            # We can generate pieces on the fly?
            
            # Let's restrict the search space for m > 10.
            # If m > 12, we might skip exact DP and use greedy for the local move.
            # For now, let's assume m is small enough for the specific moves we generate.
            # We will only call this for m <= 12 to be safe.
            
            if m > 12:
                # Fallback: Greedy
                rem = set(edges_list)
                pieces = []
                while rem:
                    # Find longest cycle
                    best_cyc = None
                    # Just pick any cycle or edge
                    # Simple greedy: pick edge, try to extend to cycle
                    # For fallback, just return edges as pieces (worst case)
                    # Or a quick greedy cycle finder
                    found = False
                    for e in list(rem):
                        # Try to find a cycle containing e
                        # Simplified: just add e as a piece for now in fallback
                        pieces.append(list(e))
                        rem.discard(e)
                    return pieces

            # Exact DP for m <= 12
            # O(2^m * #pieces). #pieces <= 2^m. 4^12 = 16M. Acceptable.
            
            for mask in range(1, 1 << m):
                # Find the lowest set bit to avoid symmetry? No, pieces are distinct.
                # Just iterate all valid pieces
                for p_mask, p_rep in valid_pieces:
                    if (mask & p_mask) == p_mask:
                        prev = mask ^ p_mask
                        if dp[prev] + 1 < dp[mask]:
                            dp[mask] = dp[prev] + 1
                            best_prev[mask] = prev
                            best_piece[mask] = p_rep
                            
        # Reconstruct
        if dp[(1 << m) - 1] == INF:
            # Should not happen
            return [list(e) for e in edges_list]
            
        pieces = []
        mask = (1 << m) - 1
        while mask:
            p_rep = best_piece[mask]
            pieces.append(p_rep)
            mask = best_prev[mask]
            
        return pieces

    def decompose_local(edges, n):
        # 1. Greedy Initial Solution
        rem_edges = set((min(u,v), max(u,v)) for u,v in edges)
        cycles = []
        
        def find_longest_cycle(rem):
            if not rem:
                return None
            adj = {i: [] for i in range(n)}
            for u, v in rem:
                adj[u].append(v)
                adj[v].append(u)
            
            best_cyc = None
            # Try DFS from each node to find a cycle
            # To find longest, we'd need DP or exhaustive.
            # For greedy init, just find ANY cycle, preferably long.
            # We'll use a simple DFS that stops at first found, but we can bias towards high degree.
            
            # For speed, just find one.
            visited = set()
            parent = {}
            
            def dfs(u, start):
                visited.add(u)
                for v in adj[u]:
                    if v == start and len(visited) > 2:
                        cyc = [v]
                        cur = u
                        while cur != v:
                            cyc.append(cur)
                            cur = parent[cur]
                        cyc.reverse()
                        return cyc
                    if v not in visited:
                        parent[v] = u
                        res = dfs(v, start)
                        if res:
                            return res
                return None

            for u in range(n):
                if adj[u]:
                    visited = set()
                    parent = {}
                    res = dfs(u, u)
                    if res:
                        # Check if longer than best
                        if not best_cyc or len(res) > len(best_cyc):
                            best_cyc = res
            return best_cyc

        while True:
            if not rem_edges:
                break
            cyc = find_longest_cycle(rem_edges)
            if not cyc:
                break
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i+1)%len(cyc)]
                rem_edges.discard((min(u,v), max(u,v)))
            cycles.append(cyc)
            
        # Current state: cycles, rem_edges
        # Local Search
        rng = Random(123)
        start_time = time.time()
        deadline = start_time + 10 # 10s per graph? We have 150s total. 3 graphs. 45s each.
        
        # We will use the local DP solver to improve pairs of cycles.
        
        while time.time() < deadline:
            improved = False
            # Pick two cycles
            if len(cycles) >= 2:
                i, j = rng.sample(range(len(cycles)), 2)
                c1 = cycles[i]
                c2 = cycles[j]
                
                # Extract edges
                e1 = set()
                for k in range(len(c1)):
                    u, v = c1[k], c1[(k+1)%len(c1)]
                    e1.add((min(u,v), max(u,v)))
                e2 = set()
                for k in range(len(c2)):
                    u, v = c2[k], c2[(k+1)%len(c2)]
                    e2.add((min(u,v), max(u,v)))
                
                combined_edges = list(e1 | e2)
                # Solve local
                local_pieces = solve_small_subgraph(combined_edges, n)
                
                # Count pieces
                # local_pieces contains cycles (len>2) and edges (len=2)
                new_count = len(local_pieces)
                old_count = 2
                
                if new_count < old_count:
                    # Apply
                    cycles[i] = None # Mark for removal
                    cycles[j] = None
                    # Add new pieces
                    # We need to distinguish cycles from edges in the global list
                    # Let's represent global state as list of "pieces" where piece is list of verts.
                    # If len(piece) == 2, it's an edge. If > 2, it's a cycle.
                    
                    # Remove old
                    cycles = [c for c in cycles if c is not None]
                    # Add new
                    for p in local_pieces:
                        if len(p) == 2:
                            # Edge
                            cycles.append(p)
                        else:
                            # Cycle
                            cycles.append(p)
                    improved = True
                    
            # Also try merging a cycle and an edge
            elif len(cycles) >= 1 and rem_edges:
                i = rng.randint(0, len(cycles)-1)
                c1 = cycles[i]
                e_edge = rng.choice(list(rem_edges))
                
                e1 = set()
                for k in range(len(c1)):
                    u, v = c1[k], c1[(k+1)%len(c1)]
                    e1.add((min(u,v), max(u,v)))
                
                combined_edges = list(e1 | {e_edge})
                local_pieces = solve_small_subgraph(combined_edges, n)
                
                new_count = len(local_pieces)
                old_count = 2 # 1 cycle + 1 edge
                
                if new_count < old_count:
                    # Apply
                    cycles.pop(i)
                    rem_edges.discard(e_edge)
                    for p in local_pieces:
                        cycles.append(p)
                    improved = True

            # If no improvement in a while, we could try restarting or changing strategy
            # But for now, just continue.
            
            # To avoid infinite loops on same bad moves, we rely on the fact that cost only decreases.
            # Since cost is integer and bounded, it will terminate.
            # However, we might get stuck in a local minimum.
            # We can add a "kick" move: randomly break a cycle into edges.
            if not improved and rng.random() < 0.05:
                # Break a random cycle
                if cycles:
                    i = rng.randint(0, len(cycles)-1)
                    c = cycles[i]
                    if len(c) > 2:
                        # Convert to edges
                        cycles.pop(i)
                        for k in range(len(c)):
                            u, v = c[k], c[(k+1)%len(c)]
                            rem_edges.add((min(u,v), max(u,v)))
                        improved = True # Force accept kick to escape

        # Final conversion: rem_edges are single pieces
        final_pieces = []
        for c in cycles:
            final_pieces.append(c)
        for e in rem_edges:
            final_pieces.append(list(e))
            
        return final_pieces

    results = []
    t_start = time.time()
    total_budget = 145 # Safety margin
    
    for i, spec in enumerate(((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))):
        n = spec[0]
        edges = instance(*spec)
        
        # Allocate time
        if i < 2:
            dl = t_start + (i + 1) * (total_budget / 3.0)
        else:
            dl = t_start + total_budget
        
        pieces = decompose_local(edges, n)
        results.append(pieces)
        
    return results

if __name__ == "__main__":
    construct()
