"""
Improved iterative backtracking with correct stack-based DFS, better symmetry breaking,
and efficient C6 detection using bitsets.
"""
import time
import random
import sys
sys.setrecursionlimit(10000)

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
        # Level 4: check if any node at level 3 is adjacent to v (completing the cycle)
        # Path: u-a-b-d-e-v means d is at level 3, e must be adjacent to both d and v
        # Actually: u(0)-a(1)-b(2)-d(3)-e(4)-v(5)
        # So we need e adjacent to d (level 3) AND e adjacent to v
        # e is not in visited (not u,v,a,b,d)
        for d in level3:
            for e in adj_c[c][d]:
                if e == v:
                    return True
                if e not in visited and v in adj_c[c][e]:
                    return True
        return False

    def backtrack_iterative(n, time_limit):
        """Iterative backtracking search for valid 4-coloring of K_n with correct stack management."""
        num_edges = n * (n - 1) // 2
        assignment = [-1] * num_edges
        adj_c = [[set() for _ in range(n)] for _ in range(4)]

        # Precompute edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))

        # Symmetry breaking: force first few edges to specific colors
        # Edge (0,1) -> color 0, edge (0,2) -> color 0 or 1
        # This breaks S_4 color symmetry and some vertex symmetries
        assignment[0] = 0  # (0,1)
        adj_c[0][0].add(1)
        adj_c[0][1].add(0)
        
        # Additional symmetry breaking for n >= 12
        if n >= 12:
            # Edge (0,2) is position 1: try color 0 first
            # This helps break more symmetry
            pass  # We'll handle this in the search

        # Proper iterative DFS:
        # Stack contains (pos, next_color_to_try)
        # We maintain the current assignment state
        # When we go deeper, we push (pos, c+1) then (pos+1, 0)
        # When we need to backtrack, the stack naturally handles it
        
        stack = [(1, 0)]  # (position, next_color_to_try)
        
        while stack:
            if time.time() > time_limit:
                return None
            
            pos, next_color = stack[-1]
            
            if pos == num_edges:
                # Found a complete valid coloring
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
                
                # Update stack: change current entry to try next color if backtracking
                if c + 1 < 4:
                    stack[-1] = (pos, c + 1)
                
                # Push next position
                if pos + 1 < num_edges:
                    stack.append((pos + 1, 0))
                else:
                    # Last edge assigned, complete
                    return assignment[:]
                
                break
            
            if not found_valid:
                # No valid color for this position, backtrack
                # Undo the assignment at this position if it was set
                if assignment[pos] >= 0:
                    prev_color = assignment[pos]
                    adj_c[prev_color][u].discard(v)
                    adj_c[prev_color][v].discard(u)
                    assignment[pos] = -1
                # Pop this position from stack
                stack.pop()
        
        return None

    def backtrack_recursive(n, time_limit):
        """Recursive backtracking with local C6 check and better symmetry breaking."""
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
            
            # Additional symmetry breaking: for edges from vertex 0,
            # try colors in a specific order
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
        
        # Symmetry breaking: fix first edge to color 0
        assignment[0] = 0
        adj_c[0][0].add(1)
        adj_c[0][1].add(0)
        
        # For n >= 14, also fix edge (0,2) to reduce search space
        # Edge (0,2) is at position 1
        if n >= 14:
            # Try color 0 for (0,2) first
            assignment[1] = 0
            adj_c[0][0].add(2)
            adj_c[0][2].add(0)
            result = solve(2)
            if result is not None:
                return result
            # Undo
            assignment[1] = -1
            adj_c[0][0].discard(2)
            adj_c[0][2].discard(0)
            # Try color 1
            assignment[1] = 1
            adj_c[1][0].add(2)
            adj_c[1][2].add(0)
            result = solve(2)
            if result is not None:
                return result
            assignment[1] = -1
            adj_c[1][0].discard(2)
            adj_c[1][2].discard(0)
            # Try color 2
            assignment[1] = 2
            adj_c[2][0].add(2)
            adj_c[2][2].add(0)
            result = solve(2)
            if result is not None:
                return result
            assignment[1] = -1
            adj_c[2][0].discard(2)
            adj_c[2][2].discard(0)
            # Try color 3
            assignment[1] = 3
            adj_c[3][0].add(2)
            adj_c[3][2].add(0)
            result = solve(2)
            if result is not None:
                return result
            assignment[1] = -1
            adj_c[3][0].discard(2)
            adj_c[3][2].discard(0)
            return None
        else:
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
    # Use iterative backtracking for speed with proper time management
    
    # Try n=18 with limited time (10s)
    if time.time() < deadline - 10:
        colors = backtrack_iterative(18, time.time() + 10)
        if colors is not None and verify_coloring(colors, 18):
            return {'n': 18, 'colors': colors}

    # Try n=17 with more time
    if time.time() < deadline - 30:
        # First try cyclic (fast)
        colors = try_cyclic(17, num_trials=3000)
        if colors is not None and verify_coloring(colors, 17):
            return {'n': 17, 'colors': colors}
        
        # Then try recursive backtracking (more reliable for n=17)
        colors = backtrack_recursive(17, deadline - 10)
        if colors is not None and verify_coloring(colors, 17):
            return {'n': 17, 'colors': colors}
        
        # Try iterative as backup
        colors = backtrack_iterative(17, deadline - 5)
        if colors is not None and verify_coloring(colors, 17):
            return {'n': 17, 'colors': colors}

    # Try n=16
    if time.time() < deadline - 15:
        colors = backtrack_recursive(16, deadline - 5)
        if colors is not None and verify_coloring(colors, 16):
            return {'n': 16, 'colors': colors}
        
        colors = backtrack_iterative(16, deadline - 3)
        if colors is not None and verify_coloring(colors, 16):
            return {'n': 16, 'colors': colors}

    # Try n=15
    if time.time() < deadline - 8:
        colors = try_cyclic(15, num_trials=3000)
        if colors is not None and verify_coloring(colors, 15):
            return {'n': 15, 'colors': colors}
        
        colors = backtrack_recursive(15, deadline - 3)
        if colors is not None and verify_coloring(colors, 15):
            return {'n': 15, 'colors': colors}

    # Try n=14
    if time.time() < deadline - 3:
        colors = backtrack_recursive(14, deadline - 1)
        if colors is not None and verify_coloring(colors, 14):
            return {'n': 14, 'colors': colors}

    # Fallback: construct a valid small coloring
    # For n=9, use a simple construction
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            # Map distance to color: 1->0, 2->1, 3->2, 4->3
            colors.append(min(d, n - d) - 1)
    return {'n': n, 'colors': colors}
