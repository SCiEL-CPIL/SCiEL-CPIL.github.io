import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # --- C6 Detection using Bitsets ---
    def _has_c6_bitset(adj, n):
        """
        Check if the graph defined by adjacency masks 'adj' contains a 6-cycle.
        adj[i] is an integer bitmask where bit j is 1 if edge (i,j) exists.
        n is the number of vertices.
        """
        for v0 in range(n):
            # v1 in N(v0)
            mask_v1 = adj[v0]
            while mask_v1:
                v1 = (mask_v1 & -mask_v1).bit_length() - 1
                mask_v1 &= mask_v1 - 1
                
                # v2 in N(v1) \ {v0}
                mask_v2 = adj[v1] & ~(1 << v0)
                while mask_v2:
                    v2 = (mask_v2 & -mask_v2).bit_length() - 1
                    mask_v2 &= mask_v2 - 1
                    
                    # v3 in N(v2) \ {v0, v1}
                    mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                    while mask_v3:
                        v3 = (mask_v3 & -mask_v3).bit_length() - 1
                        mask_v3 &= mask_v3 - 1
                        
                        # v4 in N(v3) \ {v0, v1, v2}
                        mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while mask_v4:
                            v4 = (mask_v4 & -mask_v4).bit_length() - 1
                            mask_v4 &= mask_v4 - 1
                            
                            # Check if there exists v5 such that:
                            # v5 in N(v4) AND v5 in N(v0)
                            # v5 not in {v0, v1, v2, v3, v4}
                            
                            # Candidates for v5
                            mask_v5 = adj[v4] & adj[v0]
                            # Exclude used vertices
                            used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                            mask_v5 &= ~used_mask
                            
                            if mask_v5:
                                return True
        return False

    def _count_c6s(adj, n):
        """Count total number of monochromatic C6s in the coloring."""
        total = 0
        for c in range(4):
            # Count C6s in color class c
            for v0 in range(n):
                mask_v1 = adj[c][v0]
                while mask_v1:
                    v1 = (mask_v1 & -mask_v1).bit_length() - 1
                    mask_v1 &= mask_v1 - 1
                    
                    mask_v2 = adj[c][v1] & ~(1 << v0)
                    while mask_v2:
                        v2 = (mask_v2 & -mask_v2).bit_length() - 1
                        mask_v2 &= mask_v2 - 1
                        
                        mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                        while mask_v3:
                            v3 = (mask_v3 & -mask_v3).bit_length() - 1
                            mask_v3 &= mask_v3 - 1
                            
                            mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while mask_v4:
                                v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                mask_v4 &= mask_v4 - 1
                                
                                mask_v5 = adj[c][v4] & adj[c][v0]
                                used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                mask_v5 &= ~used_mask
                                if mask_v5:
                                    total += 1
        return total

    def _is_valid(colors, n):
        """Check if the coloring is valid (no mono C6)."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
            
        for c in range(4):
            if _has_c6_bitset(adj[c], n):
                return False
        return True

    def _local_search(n, time_limit=10.0):
        """
        Find a valid 4-coloring of K_n using local search with C6-counting heuristic.
        Returns colors list if found, else None.
        """
        # Use a fixed seed for reproducibility within the run, or random?
        # Random is better for diversity, but we want to be fast.
        random.seed(int(time.time() * 1000) % 10000)
        
        deadline = time.time() + time_limit
        num_edges = n * (n - 1) // 2
        
        def count_c6s(colors):
            adj = [[0] * n for _ in range(4)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            total = 0
            for c in range(4):
                # Count C6s in color class c
                for v0 in range(n):
                    mask_v1 = adj[c][v0]
                    while mask_v1:
                        v1 = (mask_v1 & -mask_v1).bit_length() - 1
                        mask_v1 &= mask_v1 - 1
                        
                        mask_v2 = adj[c][v1] & ~(1 << v0)
                        while mask_v2:
                            v2 = (mask_v2 & -mask_v2).bit_length() - 1
                            mask_v2 &= mask_v2 - 1
                            
                            mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                            while mask_v3:
                                v3 = (mask_v3 & -mask_v3).bit_length() - 1
                                mask_v3 &= mask_v3 - 1
                                
                                mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                                while mask_v4:
                                    v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                    mask_v4 &= mask_v4 - 1
                                    
                                    mask_v5 = adj[c][v4] & adj[c][v0]
                                    used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                    mask_v5 &= ~used_mask
                                    if mask_v5:
                                        total += 1
            return total

        # Start with random coloring
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        current_c6s = count_c6s(colors)
        
        if current_c6s == 0:
            return colors
        
        best_colors = colors[:]
        best_c6s = current_c6s
        
        # Local search: flip random edges, keep if it reduces C6 count
        # Use simulated annealing or simple hill climbing?
        # Simple hill climbing with random restarts if stuck.
        
        for iteration in range(50000):
            if time.time() > deadline:
                break
            if best_c6s == 0:
                return best_colors
                
            # Pick a random edge to flip
            edge_idx = random.randint(0, num_edges - 1)
            old_color = colors[edge_idx]
            new_color = (old_color + 1 + random.randint(0, 2)) % 4
            
            colors[edge_idx] = new_color
            new_c6s = count_c6s(colors)
            
            if new_c6s <= best_c6s:
                best_c6s = new_c6s
                best_colors = colors[:]
            else:
                colors[edge_idx] = old_color
            
            # If stuck for a while, maybe restart?
            # For now, just continue.
            
        if best_c6s == 0:
            return best_colors
        return None

    def _backtrack_search(n, time_limit=5.0):
        """
        Backtracking search with strong pruning.
        Returns colors list if found, else None.
        """
        deadline = time.time() + time_limit
        E = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Shuffle edge order for better pruning?
        # Usually, ordering by vertex degree or just random helps.
        random.shuffle(edges)
        
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * E
        edge_map = {e: i for i, e in enumerate(edges)}
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(edges):
                return True
                
            u, v = edges[idx]
            
            # Try colors 0-3
            for c in range(4):
                # Check if adding (u,v) to color c creates a C6
                # We can check this by temporarily adding the edge and running _has_c6_bitset
                # But _has_c6_bitset is O(n^5) or so.
                # Better: Check if there is a path of length 5 between u and v in adj[c]
                
                # Let's use the path check logic from the current program but optimized with bitsets?
                # Or just use the full check if n is small enough.
                # For n=10, full check is fast.
                
                # Temporarily add edge
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                
                # Check for C6
                if _has_c6_bitset(adj[c], n):
                    # Remove edge
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    continue
                else:
                    # Valid move
                    colors[idx] = c
                    if backtrack(idx + 1):
                        return True
                    # Undo
                    colors[idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    
            return False
        
        if backtrack(0):
            # Reconstruct colors in lex order
            res_colors = [0] * E
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    # Find the index in edges
                    # This is slow. Let's store the original index in the edges list?
                    # No, we shuffled.
                    # We need to map back.
                    # Better approach: store the original index in the edge tuple?
                    # edges = [(i, j, orig_idx) ...]
                    pass
            # This reconstruction is messy. Let's redo the search to store in a matrix.
            return None # Placeholder, we'll use local search primarily
        
        return None

    # Strategy:
    # 1. Try to find a solution for n=17 using local search.
    # 2. If successful, try to improve to n=18 or 19? (Unlikely, but worth a quick try if time permits)
    # 3. If n=17 fails, try n=16, 15, etc.
    
    best_n = 0
    best_colors = None
    
    # Try n from 17 down to 1
    for n in range(17, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Try local search for this n
        # Allocate more time for larger n?
        # n=17 is the target.
        time_alloc = 10.0 if n <= 15 else 15.0
        
        sol = _local_search(n, time_limit=time_alloc)
        if sol is not None:
            # Verify
            if _is_valid(sol, n):
                best_n = n
                best_colors = sol
                break # Found the max n we tried, which is the highest possible we attempted
        
        # If local search failed, try a few more times with different seeds?
        # Or try backtracking for small n?
        # For n > 10, backtracking is likely too slow.
        
    # If we didn't find anything (unlikely), fallback to a known small solution
    if best_n == 0:
        # n=4 is always valid
        n = 4
        best_colors = [0, 1, 2, 3, 0, 1] # Random
        best_n = n
        # Let's just do a random search for n=4
        for _ in range(100):
            colors = [random.randint(0, 3) for _ in range(6)]
            if _is_valid(colors, 4):
                best_colors = colors
                break

    return {'n': best_n, 'colors': best_colors}
