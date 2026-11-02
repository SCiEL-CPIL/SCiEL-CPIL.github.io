import random
import time
import sys
from collections import deque

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer

    # Fallback: LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    best_n, best_edges = luw_construct(k, g)

    # Try cyclic Z_n-regular construction for better (smaller) graphs
    if time.time() < deadline - 30:
        # For 4-regular cyclic graph: vertices 0..n-1, edges (i, i+a) and (i, i+b) mod n
        # Girth >= g means no non-trivial sum of ±a, ±b of length < g equals 0 mod n
        
        def check_girth_cyclic(n, a, b, g):
            # Check that no non-backtracking walk of length < g returns to start
            # Non-backtracking: can't immediately reverse direction
            # Walks are sequences of ±a, ±b where consecutive steps aren't reverses
            # A cycle of length L < g exists if there's a closed non-backtracking walk of length L
            # For cyclic graph, this means sum of L terms (each ±a or ±b) = 0 mod n
            # with the constraint that no two consecutive terms are negatives of each other
            # (i.e., we can't have a followed by -a, or b followed by -b)
            
            # Check all sequences of length 3 to g-1
            # For length 3: a+a+a=0, a+a-b=0, a-b+a=0, a-b-b=0, b+b+b=0, b+b-a=0, b-a+b=0, b-a-a=0, -a-a-a=0, etc.
            # But non-backtracking means step[i] != -step[i-1]
            
            # More efficient: BFS in the "non-backtracking" state space
            # State: (current_position, last_step_direction)
            # last_step_direction: 0=+a, 1=-a, 2=+b, 3=-b, or None for start
            # From state (pos, last_dir), we can go to:
            #   pos+a (dir=0) if last_dir != 1
            #   pos-a (dir=1) if last_dir != 0
            #   pos+b (dir=2) if last_dir != 3
            #   pos-b (dir=3) if last_dir != 2
            
            # Check if any walk of length < g returns to start with a valid last_dir
            # Start: at position 0, no last direction
            # After first step: at ±a or ±b, with corresponding direction
            
            # BFS by depth
            # visited[(pos, dir)] = min depth reached
            visited = {}
            # Initialize: 4 possible first steps
            for step_val, dir_idx in [(a, 0), (-a, 1), (b, 2), (-b, 3)]:
                pos = step_val % n
                if pos == 0:
                    return False  # cycle of length 1 (shouldn't happen for valid a,b)
                visited[(pos, dir_idx)] = 1
            
            queue = deque(list(visited.keys()))
            found_cycle = False
            while queue:
                pos, last_dir = queue.popleft()
                depth = visited[(pos, last_dir)]
                if depth >= g - 1:
                    continue  # we only care about cycles of length < g, so max depth to expand is g-2
                # Try all 4 next steps
                for next_val, next_dir in [(a, 0), (-a, 1), (b, 2), (-b, 3)]:
                    # Non-backtracking: next_dir != (last_dir ^ 1) [since 0<->1, 2<->3]
                    if next_dir == (last_dir ^ 1):
                        continue
                    new_pos = (pos + next_val) % n
                    if new_pos == 0:
                        # Found a cycle of length depth+1
                        if depth + 1 < g:
                            return False
                    else:
                        if (new_pos, next_dir) not in visited:
                            visited[(new_pos, next_dir)] = depth + 1
                            queue.append((new_pos, next_dir))
            return True
        
        # Try to find a,b for various n starting from 320
        found = False
        for n in range(320, min(600, best_n)):
            if time.time() > deadline - 5:
                break
            for a in range(1, n // 2 + 1):
                if time.time() > deadline - 5:
                    break
                for b in range(a + 1, n // 2 + 1):
                    if time.time() > deadline - 5:
                        break
                    # Basic checks: a != b, a != n-b, b != n-a (to avoid multiedges)
                    if a == b or a + b == n or a == n - b:
                        continue
                    if check_girth_cyclic(n, a, b, g):
                        # Valid! Build edges
                        edges = []
                        for i in range(n):
                            edges.append((i, (i + a) % n))
                            edges.append((i, (i + b) % n))
                        # Remove duplicates (each edge appears twice)
                        edge_set = set()
                        for u, v in edges:
                            edge_set.add((min(u, v), max(u, v)))
                        best_n, best_edges = n, list(edge_set)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    return best_n, best_edges
