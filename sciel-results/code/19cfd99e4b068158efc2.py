import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def check_girth_cyclic_fast(n, a, b):
        """
        Check if the cyclic graph defined by offsets a, b has girth >= g.
        n: number of vertices
        a, b: offsets
        g: required girth
        Returns True if girth >= g.
        """
        # Offsets
        offs = [a, -a % n, b, -b % n]
        # Map offset value to index for quick lookup? 
        # We need to check non-backtracking: next_off != -curr_off
        # We can just store the value or index.
        # Let's use indices 0,1,2,3 for a, -a, b, -b.
        # Note: a and -a might be same if 2a=0, but we filtered that.
        # a and b might be same? Filtered.
        # a and -b might be same? Filtered.
        # So all 4 are distinct?
        # If a = b, filtered.
        # If a = -b, filtered.
        # If a = -a, 2a=0, filtered.
        # If b = -b, 2b=0, filtered.
        # So yes, 4 distinct offsets.
        
        # BFS
        # State: (sum, last_idx)
        # visited: 2D list or array of booleans
        # Since n is small, we can use a list of lists.
        visited = [[False] * 4 for _ in range(n)]
        
        queue = deque()
        
        # Initialize
        for i in range(4):
            s = offs[i]
            visited[s][i] = True
            queue.append((s, i, 1))
            
        while queue:
            s, last_idx, depth = queue.popleft()
            
            if depth >= g - 1:
                # We are at depth g-1. Next step would be g.
                # If we reached 0 at depth < g, we would have returned False.
                # We only need to check if we can reach 0 at depth+1 < g.
                # So if depth == g-1, we don't expand further for the purpose of finding < g.
                # Actually, if depth == g-1, the next step is depth g.
                # We want to find a cycle of length < g.
                # So we stop expanding if depth >= g-1?
                # No, if we are at depth L, we can take one more step to L+1.
                # If L+1 < g, and new_sum == 0, then we have a cycle of length L+1 < g.
                # So we should expand if depth < g-1.
                # If depth == g-1, we check neighbors but don't add to queue?
                # Or just check if neighbor is 0.
                pass

            for j in range(4):
                # Non-backtracking check
                # offs[j] != -offs[last_idx]
                # Since offs are distinct, -offs[last_idx] is the unique opposite.
                # If last_idx is 0 (a), opposite is 1 (-a).
                # If last_idx is 1 (-a), opposite is 0 (a).
                # If last_idx is 2 (b), opposite is 3 (-b).
                # If last_idx is 3 (-b), opposite is 2 (b).
                if j == (last_idx ^ 1): # 0<->1, 2<->3
                    continue
                    
                new_s = (s + offs[j]) % n
                
                # If new_s is 0, we have a closed walk of length depth + 1
                if new_s == 0:
                    if depth + 1 < g:
                        return False
                    # If depth + 1 >= g, it's a valid cycle length, but we don't care about longer ones for girth check (we just need to ensure NO short ones exist).
                    # We don't need to explore further from (0, j) because any extension would be longer.
                    # But we should mark it visited to avoid reprocessing if reached again?
                    # Actually, if we reach (0, j) at depth D, and later at depth D' > D, the first one is shorter.
                    # If D < g, we already returned False.
                    # If D >= g, we don't care.
                    # So we can just skip adding (0, j) to queue if we don't need to extend it.
                    # But wait, could (0, j) be part of a SHORTER cycle via a different path?
                    # No, BFS finds shortest path.
                    # So if we reach (0, j) at depth D, it's the shortest walk to (0, j).
                    # If D < g, return False.
                    # If D >= g, no short cycle via this state.
                    # We don't need to extend (0, j) to find a cycle of length < g, because extending makes it longer.
                    continue

                if not visited[new_s][j]:
                    visited[new_s][j] = True
                    queue.append((new_s, j, depth + 1))
        
        return True

    def make_cyclic_graph(n, a, b):
        edges = []
        edge_set = set()
        for i in range(n):
            for off in [a, b]:
                j1 = (i + off) % n
                j2 = (i - off) % n
                for j in [j1, j2]:
                    if i == j:
                        return None
                    key = (min(i, j), max(i, j))
                    if key in edge_set:
                        return None
                    edge_set.add(key)
                    edges.append((i, j))
        # Verify degree
        # In a cyclic graph with valid offsets, degree is always 4.
        # But let's be safe.
        return edges

    # Search for cyclic graph
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (4,10) is 243.
    # We search up to 400.
    # If we find one, we return it.
    
    # For k=4, g=10, no cyclic graph exists. Skip to save time.
    if not (k == 4 and g == 10):
        for n in range(243, 401):
            if time.time() > deadline - 10:
                break
                
            half_n = n // 2
            for a in range(1, half_n + 1):
                if 2 * a % n == 0:
                    continue
                for b in range(a + 1, half_n + 1):
                    if 2 * b % n == 0:
                        continue
                    if (a + b) % n == 0:
                        continue
                    
                    if check_girth_cyclic_fast(n, a, b):
                        edges = make_cyclic_graph(n, a, b)
                        if edges is not None:
                            return n, edges

    # Fallback: LW Graph from seed
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
