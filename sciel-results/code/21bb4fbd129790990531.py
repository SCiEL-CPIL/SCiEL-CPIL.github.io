"""
Combined approach: iterative backtracking with local C6 checking and symmetry breaking.
Structural change from recursive to iterative (stack-based) DFS to avoid recursion overhead.
"""
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def edge_pos_to_pair(pos, n):
        """Convert lexicographic edge index to (i, j) pair."""
        i = 0
        remaining = pos
        while remaining >= (n - i - 1):
            remaining -= (n - i - 1)
            i += 1
        j = i + 1 + remaining
        return (i, j)

    def verify_coloring(colors, n):
        """Verify that no color class contains a C6."""
        for c in range(4):
            adj = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj[i].add(j)
                        adj[j].add(i)
                    idx += 1
            # Check for C6 in this adjacency
            for a0 in range(n):
                for a1 in adj[a0]:
                    if a1 <= a0:
                        continue
                    for a2 in adj[a1]:
                        if a2 == a0:
                            continue
                        for a3 in adj[a2]:
                            if a3 in (a0, a1):
                                continue
                            for a4 in adj[a3]:
                                if a4 in (a0, a1, a2):
                                    continue
                                for a5 in adj[a4]:
                                    if a5 in (a0, a1, a2, a3):
                                        continue
                                    if a0 in adj[a5]:
                                        return False
        return True

    def would_create_c6(adj_c, c, u, v, n):
        """Check if adding edge (u,v) to color-c graph creates a C6.
        Need a path of length 4 from u to v: u-a-b-d-e-v with 6 distinct vertices."""
        # BFS/DFS of depth 4 from u avoiding v
        # Level 1: neighbors of u (excluding v)
        level1 = set()
        for a in adj_c[c][u]:
            if a != v:
                level1.add(a)
        if not level1:
            return False
        # Level 2
        visited = {u, v} | level1
        level2 = set()
        for a in level1:
            for b in adj_c[c][a]:
                if b not in visited:
                    level2.add(b)
                    visited.add(b)
        if not level2:
            return False
        # Level 3
        level3 = set()
        for b in level2:
            for d in adj_c[c][b]:
                if d not in visited:
                    level3.add(d)
                    visited.add(d)
        if not level3:
            return False
        # Level 4: check if any node at level 4 is adjacent to v
        for d in level3:
            for e in adj_c[c][d]:
                if e not in visited:
                    if e == v:
                        return True
                    if v in adj_c[c][e]:
                        return True
        return False

    def backtrack_iterative(n, time_limit):
        """Iterative backtracking search for valid 4-coloring of K_n."""
        num_edges = n * (n - 1) // 2
        assignment = [-1] * num_edges
        adj_c = [[set() for _ in range(n)] for _ in range(4)]

        # Precompute edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))

        # Symmetry breaking: force first edge to color 0
        # Edge (0,1) is position 0
        assignment[0] = 0
        adj_c[0][0].add(1)
        adj_c[0][1].add(0)

        # Use iterative DFS with explicit stack
        # Stack entries: (pos, color_to_try)
        # We process edges in order, trying colors 0-3 for each
        
        # For efficiency, use a stack-based approach
        pos = 1  # start after first edge
        color_stack = [0]  # colors to try for current position
        
        # Actually, let's use a cleaner iterative approach
        # Maintain: current position, and for each position, which color we're trying
        
        # Better: use a proper iterative DFS
        # State: (pos, next_color_to_try)
        # When we backtrack, we restore state
        
        # Let's use a simpler but effective approach:
        # Recursive-like iterative with explicit state management
        
        # Actually, for n up to 17, let's just use recursion with sys.setrecursionlimit
        # but the key insight from the observation is that iterative is faster.
        
        # Let me implement a proper iterative backtracking:
        
        # We'll maintain:
        # - pos: current edge position
        # - For each position, the next color to try (0-3, or 4 if all tried)
        # - When we go deeper, save current state
        # - When we backtrack, restore state
        
        # Simpler: just use a stack of (pos, color) pairs
        # But we need to undo assignments when backtracking
        
        # Cleanest approach for iterative:
        # Stack contains (pos, color) where color is the next color to try
        # When pushing a new level, we set assignment[pos] = color and add to adj
        # When popping (all colors tried), we undo the last assignment
        
        # Let's just do it the straightforward way with a stack
        
        stack = [(1, 0)]  # (position, next_color_to_try)
        
        while stack:
            if time.time() > time_limit:
                return None
            
            pos, next_color = stack.pop()
            
            if pos == num_edges:
                # Found a solution!
                return assignment[:]
            
            u, v = edges[pos]
            found_valid = False
            
            for c in range(next_color, 4):
                # Check if this color would create a C6
                if would_create_c6(adj_c, c, u, v, n):
                    continue
                
                # Assign this color
                assignment[pos] = c
                adj_c[c][u].add(v)
                adj_c[c][v].add(u)
                found_valid = True
                
                # Push remaining colors for this position (to try if we backtrack)
                if c + 1 < 4:
                    stack.append((pos, c + 1))
                
                # Push next position
                if pos + 1 < num_edges:
                    stack.append((pos + 1, 0))
                else:
                    # This is the last edge, check if complete
                    # Actually, if pos+1 == num_edges, we've assigned all edges
                    # But we need to verify... actually the local check ensures no C6
                    return assignment[:]
                
                # If we pushed next position, the loop will process it
                # If not, we'll continue with remaining colors for this pos
                break
            
            if not found_valid:
                # No valid color for this position, backtrack
                # The assignment at pos is already -1 (or we never set it)
                # Need to undo the previous assignment
                if pos > 0:
                    prev_pos = pos - 1
                    prev_color = assignment[prev_pos]
                    if prev_color >= 0:
                        pu, pv = edges[prev_pos]
                        adj_c[prev_color][pu].discard(pv)
                        adj_c[prev_color][pv].discard(pu)
                        assignment[prev_pos] = -1
        
        return None

    def backtrack_recursive(n, time_limit):
        """Recursive backtracking with local C6 check."""
        num_edges = n * (n - 1) // 2
        assignment = [-1] * num_edges
        adj_c = [[set() for _ in range(n)] for _ in range(4)]
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        def solve(pos):
            if time.time() > time_limit:
                return None
            if pos == num_edges:
                return assignment[:]
            
            u, v = edges[pos]
            
            for c in range(4):
                if would_create_c6(adj_c, c, u, v, n):
                    continue
                
                assignment[pos] = c
                adj_c[c][u].add(v)
                adj_c[c][v].add(u)
                
                result = solve(pos + 1)
                if result is not None:
                    return result
                
                assignment[pos] = -1
                adj_c[c][u].discard(v)
                adj_c[c][v].discard(u)
            
            return None
        
        # Symmetry breaking
        assignment[0] = 0
        adj_c[0][0].add(1)
        adj_c[0][1].add(0)
        
        return solve(1)

    def try_cyclic(n, num_trials=10000):
        """Try cyclic 4-coloring of K_n where n is odd."""
        if n % 2 == 0:
            return None
        
        m = (n - 1) // 2
        elements = list(range(1, n))
        
        for trial in range(num_trials):
            random.shuffle(elements)
            sets = [frozenset(elements[i*m:(i+1)*m]) for i in range(4)]
            
            valid = True
            for c in range(4):
                S = sets[c]
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        j1 = (i + s) % n
                        j2 = (i - s) % n
                        adj[i].add(j1)
                        adj[i].add(j2)
                # Check for C6
                found = False
                for a0 in range(n):
                    if found:
                        break
                    for a1 in adj[a0]:
                        if found or a1 <= a0:
                            continue
                        for a2 in adj[a1]:
                            if found or a2 == a0:
                                continue
                            for a3 in adj[a2]:
                                if found or a3 in (a0, a1):
                                    continue
                                for a4 in adj[a3]:
                                    if found or a4 in (a0, a1, a2):
                                        continue
                                    for a5 in adj[a4]:
                                        if a5 in (a0, a1, a2, a3):
                                            continue
                                        if a0 in adj[a5]:
                                            found = True
                                            break
                if found:
                    valid = False
                    break
            
            if valid:
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = (j - i) % n
                        for c in range(4):
                            if d in sets[c] or (n - d) in sets[c]:
                                colors.append(c)
                                break
                return colors
        return None

    # Strategy: try n=18, then n=17, then smaller
    # Use iterative backtracking for speed
    
    # Try n=18 with limited time
    if time.time() < deadline - 80:
        colors = backtrack_iterative(18, deadline - 80)
        if colors is not None and verify_coloring(colors, 18):
            return {'n': 18, 'colors': colors}

    # Try n=17
    if time.time() < deadline - 40:
        # First try cyclic (fast)
        colors = try_cyclic(17, num_trials=5000)
        if colors is not None and verify_coloring(colors, 17):
            return {'n': 17, 'colors': colors}
        
        # Then try backtracking
        colors = backtrack_iterative(17, deadline - 20)
        if colors is not None and verify_coloring(colors, 17):
            return {'n': 17, 'colors': colors}

    # Try n=16
    if time.time() < deadline - 15:
        colors = backtrack_iterative(16, deadline - 10)
        if colors is not None and verify_coloring(colors, 16):
            return {'n': 16, 'colors': colors}

    # Try n=15 cyclic
    if time.time() < deadline - 5:
        colors = try_cyclic(15, num_trials=5000)
        if colors is not None and verify_coloring(colors, 15):
            return {'n': 15, 'colors': colors}

    # Fallback: K9
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
