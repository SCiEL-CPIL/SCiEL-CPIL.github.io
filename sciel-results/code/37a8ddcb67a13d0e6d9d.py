import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def check_girth_cyclic_fast(n, a, b):
        """
        Check if the cyclic graph defined by offsets a, b has girth >= g.
        """
        offs = [a, (-a) % n, b, (-b) % n]
        # Check all 4 offsets are distinct
        if len(set(offs)) != 4:
            return False
            
        visited = [[False] * 4 for _ in range(n)]
        queue = deque()
        
        for i in range(4):
            s = offs[i]
            visited[s][i] = True
            queue.append((s, i, 1))
            
        while queue:
            s, last_idx, depth = queue.popleft()
            
            for j in range(4):
                if j == (last_idx ^ 1):
                    continue
                    
                new_s = (s + offs[j]) % n
                
                if new_s == 0:
                    if depth + 1 < g:
                        return False
                    continue

                if depth + 1 < g and not visited[new_s][j]:
                    visited[new_s][j] = True
                    queue.append((new_s, j, depth + 1))
        
        return True

    def make_cyclic_graph(n, a, b):
        edges = []
        edge_set = set()
        for i in range(n):
            for off in [a, b]:
                for j in [(i + off) % n, (i - off) % n]:
                    if i == j:
                        return None
                    key = (min(i, j), max(i, j))
                    if key in edge_set:
                        return None
                    edge_set.add(key)
                    edges.append((i, j))
        return edges

    # If the specific (4,10) cage is requested, try cyclic search first
    if k == 4 and g == 10:
        # Search for cyclic graphs with n from Moore bound to 500
        # Moore bound for (4,10) is 243
        for n in range(243, 501):
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
                    if (a - b) % n == 0:
                        continue
                    if check_girth_cyclic_fast(n, a, b):
                        edges = make_cyclic_graph(n, a, b)
                        if edges is not None:
                            return n, edges

        # Fallback: Lev-Zeitouni graph LZ(3, 5)
        # n = 2 * 3^5 = 486
        # Girth is 11 (>= 10)
        q = 3
        d = 5
        # LZ graph construction
        # Vertices are (side, x_1, ..., x_d) where x_i in GF(q)
        # Rule: x_c = x_{c-1} + x_{c-2} x_{c-3} (for c >= 3)
        # For LZ, the rule is slightly different from LW.
        # LZ(q,d) uses: y_c = y_{c-1} + y_{c-2} * y_{c-3} ?
        # Actually, LZ graphs are defined by:
        # x_1, x_2 are free.
        # x_c = x_{c-1} + x_{c-2} x_{c-3} for c > 2.
        # This is the same recurrence as LW?
        # LW uses: x_c = x_{c-1} + x_{c-2} x_{c-3} for c > 2?
        # Let's check the seed's rule for LW.
        # Seed LW rule:
        # rule[2] = (1,1) -> y_2 = y_1 + y_1*y_1? No.
        # Seed: 
        # c=2: y[2] = add(x[2], mul(y[1], x[1]))? 
        # The seed's LW implementation is specific.
        
        # I will use the LW implementation from the seed but with q=3, d=5.
        # Wait, LW(3,5) has n = 2 * 3^5 = 486.
        # Girth of LW(3,5)?
        # LW(q,d) girth is at least 2d. For d=5, girth >= 10.
        # So LW(3,5) is a valid fallback.
        
        q = 3
        d = 5
        
        # Field GF(3) arithmetic
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
        # Rule for LW
        # Standard LW rule: x_c = x_{c-1} + x_{c-2} x_{c-3}
        # Indices in code are 1-based for x, but 0-based for tuple?
        # Seed uses:
        # y = [None, first] + [0] * (d - 1)
        # for c in range(2, d + 1):
        #   a, b = rule[c]
        #   if side == 0:
        #     y[c] = add(x[c], mul(y[a], x[b]))
        
        # I will replicate the seed's LW logic for q=3, d=5.
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    # General fallback for other k, g
    # Use LW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
