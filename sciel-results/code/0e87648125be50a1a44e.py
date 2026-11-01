import time
import random
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Fallback: L-U-W graph
    def luw_construct():
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

    if k != 3 or g != 16:
        return luw_construct()

    # Try to find a small Cayley graph
    def try_group(n_perm, is_alternating=False):
        if time.time() > deadline:
            return None
            
        # Generate permutations
        perms = list(itertools.permutations(range(n_perm)))
        if is_alternating:
            # Filter even permutations
            def sign(p):
                invs = 0
                for i in range(n_perm):
                    for j in range(i+1, n_perm):
                        if p[i] > p[j]:
                            invs += 1
                return 1 if invs % 2 == 0 else -1
            perms = [p for p in perms if sign(p) == 1]
            
        n = len(perms)
        index = {p: i for i, p in enumerate(perms)}
        
        # Build multiplication table
        mul = [[0]*n for _ in range(n)]
        for i, p in enumerate(perms):
            for j, q in enumerate(perms):
                r = tuple(p[q[k]] for k in range(n_perm))
                mul[i][j] = index[r]
                
        # Find involutions
        invs = []
        for i, p in enumerate(perms):
            if i == 0: continue # skip identity
            is_inv = True
            for k in range(n_perm):
                if p[p[k]] != k:
                    is_inv = False
                    break
            if is_inv:
                invs.append(i)
                
        if len(invs) < 3:
            return None

        # Try random sets of 3 involutions
        max_attempts = 2000
        for _ in range(max_attempts):
            if time.time() > deadline:
                return None
            s_set = random.sample(invs, 3)
            
            # Check connectivity
            visited = [False] * n
            queue = [0]
            visited[0] = True
            count = 1
            while queue:
                u = queue.pop(0)
                for s in s_set:
                    v = mul[u][s]
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            if count != n:
                continue
                
            # Check girth
            # DFS to find cycle < 16
            def has_short_cycle():
                # We can use the mul table
                # Start from 0
                # Path is list of nodes
                # We need to check all paths of length 3 to 15
                # To speed up, we can just do BFS/DFS
                # Since n is small, we can even just check all words of length L
                
                # Optimized check:
                # For L in 3..15:
                #   For each word of length L (no backtracking):
                #      if product == 0 and all prefixes distinct: return True
                
                # We can generate words recursively
                def dfs(current, depth, visited, path):
                    if depth == 15:
                        return False # We are looking for < 16, so max depth 15
                    # Actually, if we close at depth d, d < 16
                    # So we check if current == 0 at any depth >= 3
                    
                    # Wait, the cycle length is the depth.
                    # If we return to 0 at depth d, cycle length is d.
                    # We want d < 16.
                    
                    for s in s_set:
                        next_node = mul[current][s]
                        if next_node in visited:
                            continue
                        if next_node == 0:
                            if depth + 1 < 16:
                                return True
                            else:
                                continue
                        visited.add(next_node)
                        if dfs(next_node, depth + 1, visited, path + [next_node]):
                            return True
                        visited.remove(next_node)
                    return False

                for s in s_set:
                    first = mul[0][s]
                    if first == 0: continue
                    visited = {0, first}
                    if dfs(first, 1, visited, [first]):
                        return True
                return False

            if not has_short_cycle():
                # Construct edges
                edges = []
                for u in range(n):
                    for s in s_set:
                        v = mul[u][s]
                        if u < v:
                            edges.append((u, v))
                return n, edges

        return None

    # Try groups in order of size
    # S5 (120)
    res = try_group(5, False)
    if res: return res
    
    # A6 (360)
    res = try_group(6, True)
    if res: return res
    
    # S6 (720)
    res = try_group(6, False)
    if res: return res
    
    # Fallback
    return luw_construct()
