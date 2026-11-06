import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 115.0

    def verify(n, colors):
        """Verify that no color class contains a C6."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[c][i] >> j) & 1:
                        continue
                    # Check for path of length 5 from i to j avoiding direct edge
                    # i-a-b-d-e-j
                    found = False
                    mask_i = adj[c][i] & ~(1 << j)
                    while mask_i and not found:
                        lsb_a = mask_i & -mask_i
                        a = lsb_a.bit_length() - 1
                        mask_i -= lsb_a
                        mask_a = adj[c][a] & ~(1 << i) & ~(1 << j)
                        while mask_a and not found:
                            lsb_b = mask_a & -mask_a
                            b = lsb_b.bit_length() - 1
                            mask_a -= lsb_b
                            mask_b = adj[c][b] & ~(1 << i) & ~(1 << a) & ~(1 << j)
                            while mask_b and not found:
                                lsb_d = mask_b & -mask_b
                                d = lsb_d.bit_length() - 1
                                mask_b -= lsb_d
                                mask_d = adj[c][d] & ~(1 << i) & ~(1 << a) & ~(1 << b) & ~(1 << j)
                                while mask_d and not found:
                                    lsb_e = mask_d & -mask_d
                                    e = lsb_e.bit_length() - 1
                                    mask_d -= lsb_e
                                    if (adj[c][e] >> j) & 1:
                                        found = True
                                        break
                    if found:
                        return False
        return True

    def has_c6_in_color(c, adj, u, v):
        """Check if adding edge (u,v) to color c creates a C6."""
        mask_a = adj[c][u] & ~(1 << v)
        while mask_a:
            lsb_a = mask_a & -mask_a
            a = lsb_a.bit_length() - 1
            mask_a -= lsb_a
            mask_b = adj[c][a] & ~(1 << u) & ~(1 << v)
            while mask_b:
                lsb_b = mask_b & -mask_b
                b = lsb_b.bit_length() - 1
                mask_b -= lsb_b
                mask_d = adj[c][b] & ~(1 << u) & ~(1 << a) & ~(1 << v)
                while mask_d:
                    lsb_d = mask_d & -mask_d
                    d = lsb_d.bit_length() - 1
                    mask_d -= lsb_d
                    mask_e = adj[c][d] & ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                    while mask_e:
                        lsb_e = mask_e & -mask_e
                        e = lsb_e.bit_length() - 1
                        mask_e -= lsb_e
                        if (adj[c][e] >> v) & 1:
                            return True
        return False

    def greedy_color(n):
        """
        Greedy heuristic: assign colors to minimize immediate C6 creation.
        Returns (colors, conflict_edges)
        """
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        colors = [-1] * len(edges)
        adj = [[0] * n for _ in range(4)]
        
        # Order edges: maybe random or standard
        # Standard lexicographic is fine for greedy
        for idx, (u, v) in enumerate(edges):
            best_c = -1
            min_conflicts = 5 # > 4
            # Try all 4 colors
            for c in range(4):
                # Check if adding (u,v) to c creates a C6
                if has_c6_in_color(c, adj, u, v):
                    continue # This color creates a C6, skip if possible
                # If no C6 created, this is a good color
                # We can pick the first one that doesn't create a C6
                # Or we can count "near misses" (paths of length 4) to be safer
                # For now, just pick the first valid one
                best_c = c
                break
                
            if best_c == -1:
                # All colors create a C6. Pick one that creates the "fewest" new C6s?
                # Or just pick random?
                # For simplicity, pick 0. This edge is in a conflict.
                best_c = random.randint(0, 3)
            
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
            
        # Identify conflict edges: edges that are part of any C6 in the final coloring
        # We need to find all C6s.
        conflict_set = set()
        # Re-use verify logic to find C6s, but collect edges
        # Actually, verify returns bool. We need a modified version.
        # For now, we can just assume edges that failed the "no C6" check in greedy are conflicts?
        # No, an edge might be valid when added, but become part of a C6 later.
        # So we must scan the final graph.
        
        # Efficiently find C6s and mark edges
        # This is expensive if done naively.
        # But n is small.
        for c in range(4):
            # Build adj list for this color
            # Check all pairs
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[c][i] >> j) & 1:
                        continue
                    # Check if (i,j) is in a C6
                    # Path i-a-b-d-e-j
                    mask_i = adj[c][i] & ~(1 << j)
                    found = False
                    while mask_i and not found:
                        lsb_a = mask_i & -mask_i
                        a = lsb_a.bit_length() - 1
                        mask_i -= lsb_a
                        mask_a = adj[c][a] & ~(1 << i) & ~(1 << j)
                        while mask_a and not found:
                            lsb_b = mask_a & -mask_a
                            b = lsb_b.bit_length() - 1
                            mask_a -= lsb_b
                            mask_b = adj[c][b] & ~(1 << i) & ~(1 << a) & ~(1 << j)
                            while mask_b and not found:
                                lsb_d = mask_b & -mask_b
                                d = lsb_d.bit_length() - 1
                                mask_b -= lsb_d
                                mask_d = adj[c][d] & ~(1 << i) & ~(1 << a) & ~(1 << b) & ~(1 << j)
                                while mask_d and not found:
                                    lsb_e = mask_d & -mask_d
                                    e = lsb_e.bit_length() - 1
                                    mask_d -= lsb_e
                                    if (adj[c][e] >> j) & 1:
                                        # C6 found: i-a-b-d-e-j
                                        # Mark all edges in this C6 as conflict
                                        conflict_set.add((i, a))
                                        conflict_set.add((a, b))
                                        conflict_set.add((b, d))
                                        conflict_set.add((d, e))
                                        conflict_set.add((e, j))
                                        conflict_set.add((i, j))
                                        found = True
                                        break
                    if found:
                        continue # One C6 is enough to mark as conflict? 
                        # Actually, we should mark all edges in ALL C6s?
                        # Or just break? Breaking saves time.
                        # If we break, we might miss some conflict edges, 
                        # but the backtracking will handle it.
        
        return colors, conflict_set

    def informed_backtrack(n, initial_colors, conflict_edges):
        """
        Backtracking with informed edge ordering.
        """
        local_deadline = time.time() + 10.0 # Short budget per attempt
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        num_edges = len(edges)
        
        # Map edge to index
        edge_to_idx = {}
        for idx, e in enumerate(edges):
            edge_to_idx[e] = idx
            
        # Order edges: Non-conflict first, then conflict
        # For non-conflict, try initial color first.
        # For conflict, try all colors.
        
        # Create a list of (edge_idx, preferred_color)
        # We sort by: is_conflict (False=0, True=1), then edge_idx
        edge_order = []
        for idx, e in enumerate(edges):
            is_conf = e in conflict_edges
            edge_order.append((is_conf, idx))
            
        edge_order.sort()
        
        # Unpack ordered indices
        ordered_edge_indices = [idx for _, idx in edge_order]
        
        colors = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        
        # Pre-compute preferred colors
        # For non-conflict edges, prefer initial_colors[idx]
        # For conflict edges, prefer initial_colors[idx] too? Or random?
        # Let's prefer initial_colors for all, but allow all 4.
        
        stack = [(0, 0)] # (position_in_order, tried_mask)
        
        while stack:
            pos, tried_mask = stack[-1]
            
            if time.time() > local_deadline:
                return None
                
            if pos == num_edges:
                return colors[:]
                
            idx = ordered_edge_indices[pos]
            u, v = edges[idx]
            
            found = False
            # Try colors in order: preferred first, then others
            preferred = initial_colors[idx]
            for c in [preferred] + [x for x in range(4) if x != preferred]:
                cbit = 1 << c
                if tried_mask & cbit:
                    continue
                if not has_c6_in_color(c, adj, u, v):
                    colors[idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    stack[-1] = (pos, tried_mask | cbit)
                    stack.append((pos + 1, 0))
                    found = True
                    break
                    
            if not found:
                stack.pop()
                if pos > 0:
                    prev_pos = pos - 1
                    prev_idx = ordered_edge_indices[prev_pos]
                    prev_u, prev_v = edges[prev_idx]
                    prev_c = colors[prev_idx]
                    colors[prev_idx] = -1
                    adj[prev_c][prev_u] &= ~(1 << prev_v)
                    adj[prev_c][prev_v] &= ~(1 << prev_u)
                    
        return None

    # Try n=17
    for n in [17, 16, 15, 14]:
        if time.time() > hard_deadline - 20:
            break
            
        # 1. Greedy
        init_colors, conflicts = greedy_color(n)
        
        # 2. Check if greedy was perfect
        if not conflicts:
            if verify(n, init_colors):
                return {'n': n, 'colors': init_colors}
                
        # 3. Informed Backtrack
        # Budget: 10s per n
        res = informed_backtrack(n, init_colors, conflicts)
        if res:
            if verify(n, res):
                return {'n': n, 'colors': res}
                
    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
