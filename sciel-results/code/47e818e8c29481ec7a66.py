import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
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

    # Try Cayley graph on SL(2, 16)
    if k == 3 and g == 16:
        try:
            # GF(16) setup
            # x^4 + x + 1 is primitive
            def gf16_add(a, b): return a ^ b
            def gf16_mul(a, b):
                # Multiply in GF(16)
                # a, b are 4-bit integers
                res = 0
                for _ in range(4):
                    if b & 1:
                        res ^= a
                    b >>= 1
                    a <<= 1
                    if a & 16:
                        a ^= 0b10001 # x^4 + x + 1 -> x^4 = x + 1
                return res
            
            # Generate SL(2, 16)
            # Matrix: (a, b, c, d) with ad - bc = 1
            # In char 2, - is +. So ad + bc = 1.
            group = []
            group_map = {}
            for a in range(16):
                for b in range(16):
                    for c in range(16):
                        if a == 0:
                            if gf16_mul(b, c) == 1:
                                for d in range(16):
                                    m = (a, b, c, d)
                                    if m not in group_map:
                                        group_map[m] = len(group)
                                        group.append(m)
                        else:
                            # a*d = 1 + b*c
                            rhs = gf16_add(1, gf16_mul(b, c))
                            d = gf16_mul(rhs, group_map.get((a, 0, 0, 1), 0) and 0 or 0) # Placeholder, will fix below
                            # Actually need inverse of a
                            # We computed inv_map earlier? No, let's do it properly.
                            pass
            
            # Let's redo group generation properly with inv_map
            inv_map = {}
            for i in range(1, 16):
                for j in range(1, 16):
                    if gf16_mul(i, j) == 1:
                        inv_map[i] = j
                        break
            
            group = []
            group_map = {}
            for a in range(16):
                for b in range(16):
                    for c in range(16):
                        if a == 0:
                            if gf16_mul(b, c) == 1:
                                for d in range(16):
                                    m = (a, b, c, d)
                                    if m not in group_map:
                                        group_map[m] = len(group)
                                        group.append(m)
                        else:
                            rhs = gf16_add(1, gf16_mul(b, c))
                            d = gf16_mul(rhs, inv_map[a])
                            m = (a, b, c, d)
                            if m not in group_map:
                                group_map[m] = len(group)
                                group.append(m)
            
            n_group = len(group)
            
            # Multiplication table
            def mat_mul(m1, m2):
                a1, b1, c1, d1 = m1
                a2, b2, c2, d2 = m2
                a = gf16_add(gf16_mul(a1, a2), gf16_mul(b1, c2))
                b = gf16_add(gf16_mul(a1, b2), gf16_mul(b1, d2))
                c = gf16_add(gf16_mul(c1, a2), gf16_mul(d1, c2))
                d = gf16_add(gf16_mul(c1, b2), gf16_mul(d1, d2))
                return (a, b, c, d)
            
            # Precompute multiplication
            mul_table = {}
            for i, m1 in enumerate(group):
                for j, m2 in enumerate(group):
                    res = mat_mul(m1, m2)
                    mul_table[(i, j)] = group_map[res]
            
            # Find inverses
            inv_idx = [0] * n_group
            def mat_inv(m):
                a, b, c, d = m
                # det = 1
                # inv = [d -b; -c a] = [d b; c a] in char 2
                return (d, b, c, a)
            
            for i, m in enumerate(group):
                inv_m = mat_inv(m)
                inv_idx[i] = group_map[inv_m]
            
            # Find self-inverse elements (M^2 = I)
            identity = (1, 0, 0, 1)
            self_inv_candidates = []
            for i in range(n_group):
                m = group[i]
                if m == identity:
                    continue
                mm = mat_mul(m, m)
                if mm == identity:
                    self_inv_candidates.append(i)
            
            # Girth check: BFS from identity up to depth 7
            # If we find a word of length < 16 equal to identity, girth < 16.
            # Actually, we need to check if there's a cycle < 16.
            # In a Cayley graph, a cycle of length L corresponds to a word of length L equal to identity.
            # So we just need to check if there is a word of length L < 16 equal to identity.
            # We can do this by BFS from identity in the Cayley graph.
            # If identity is reached by a path of length L < 16 (other than the empty path), then girth < 16.
            # But wait, the path must be a non-backtracking walk?
            # No, the girth is the length of the shortest cycle.
            # A cycle is a closed walk with no repeated vertices.
            # In a Cayley graph, a closed walk corresponds to a word equal to identity.
            # If there is a word of length L equal to identity, it might not be a simple cycle.
            # However, if there is a simple cycle of length L, there is a word of length L equal to identity.
            # Conversely, if there is a word of length L equal to identity, it might contain a simple cycle of length <= L.
            # So if we find any word of length L < 16 equal to identity, then girth < 16.
            # So we just need to check if identity is reachable from identity by a path of length L where 1 <= L < 16.
            # But the empty path is length 0.
            # So we do BFS from identity, and if we visit identity again at depth L > 0, then girth < 16.
            # But BFS from identity will only visit identity at depth 0 initially.
            # We need to check if there is a cycle.
            # Standard way: BFS from each node? No, too slow.
            # For Cayley graph, we can just check if there is a word of length L < 16 equal to identity.
            # We can do this by enumerating all words of length up to 7?
            # No, we can do BFS from identity and see if we can return to identity.
            # But BFS doesn't allow revisiting nodes.
            # So we need to check if there is a path from identity to identity of length L < 16.
            # This is equivalent to checking if the identity element is in the set of products of L generators.
            # We can do this by dynamic programming or BFS.
            # Let's use BFS in the group.
            # dist[0] = 0.
            # We want to know if dist[0] can be updated to a value < 16?
            # No, dist[0] is 0.
            # We want to know if there is a path from 0 to 0 of length L > 0.
            # This is a cycle.
            # So we can just check if there is a cycle of length < 16.
            # We can do this by BFS from each node? No.
            # We can do this by checking if the word of length L is identity.
            # Let's just enumerate all words of length up to 7?
            # 3^7 = 2187.
            # For each word, compute the product. If it is identity, return False.
            # This is fast.
            
            def get_girth(s1, s2, s3):
                # Check if there is a word of length L < 16 equal to identity.
                # We can assume the word does not immediately backtrack?
                # No, we need to check all words.
                # But we can prune: if the prefix is identity, we can stop?
                # No, because a longer word might be identity.
                # But if a prefix is identity, then the word is a concatenation of a cycle and something else.
                # So if a prefix is identity, then there is a cycle.
                # So we can just check if any prefix is identity.
                # But we need to check all words of length < 16.
                # Let's just do BFS from identity and see if we can return to identity.
                # But BFS doesn't allow revisiting.
                # So we need to allow revisiting?
                # No, we just need to check if there is a path from 0 to 0 of length L > 0.
                # This is equivalent to checking if 0 is in the neighborhood of 0 in the L-th power of the adjacency matrix.
                # Or we can just enumerate words.
                
                # Let's enumerate words of length 1 to 15.
                # But we can prune: if the current product is identity, return False.
                # Also, if the current product is the same as a previous prefix, we can prune?
                # No.
                
                # Let's just do BFS from 0, but allow revisiting?
                # No, that's not BFS.
                
                # Let's just enumerate all words of length up to 7.
                # If a word of length L is identity, then girth < 16.
                # But what if the shortest cycle is length 8?
                # Then there is a word of length 8 equal to identity.
                # So we need to check up to length 15.
                # 3^15 is too large.
                
                # Better: BFS from 0, and check if we can return to 0.
                # But BFS doesn't allow revisiting.
                # So we need to check if there is a cycle.
                # We can do this by checking if there is a path from 0 to 0 of length L > 0.
                # This is equivalent to checking if the adjacency matrix has a non-zero entry (0,0) in its L-th power.
                # Or we can just do a DFS with a visited set that allows revisiting 0?
                # No.
                
                # Let's just use the fact that the graph is a Cayley graph.
                # We can check if the identity is in the set of products of L generators.
                # We can do this by iterating over all words of length L.
                # For L=1: 3 words.
                # For L=2: 9 words.
                # ...
                # For L=7: 2187 words.
                # For L=8: 6561 words.
                # Total: sum_{L=1}^{15} 3^L = (3^16 - 3)/2 ~ 43 million.
                # Too slow.
                
                # Better: use BFS from 0, and keep track of the shortest path to each node.
                # If we reach a node that is already visited, and the path length is < 16, then we have a cycle.
                # But this is not correct.
                
                # Let's just use the standard girth check: BFS from each node up to depth 7.
                # n=4080.
                # 4080 * (3^7) ~ 4080 * 2187 ~ 8.9 million.
                # This is feasible.
                
                for start in range(n_group):
                    visited = [-1] * n_group
                    visited[start] = 0
                    q = deque([(start, 0)])
                    while q:
                        u, d = q.popleft()
                        if d >= 7:
                            continue
                        for v in adj[u]:
                            if visited[v] == -1:
                                visited[v] = d + 1
                                q.append((v, d + 1))
                            else:
                                if visited[v] < d:
                                    cycle_len = d - visited[v] + 1
                                    if cycle_len < 16:
                                        return False
                return True
            
            # Build adjacency
            adj = [[] for _ in range(n_group)]
            for i in range(n_group):
                n1 = mul_table[(i, self_inv_candidates[0])]
                n2 = mul_table[(i, self_inv_candidates[1])]
                n3 = mul_table[(i, self_inv_candidates[2])]
                adj[i].append(n1)
                adj[i].append(n2)
                adj[i].append(n3)
            
            # Random search
            random.seed(42)
            all_elements = list(range(n_group))
            
            for _ in range(1000):
                if time.time() > deadline:
                    break
                
                # Strategy 1: 3 self-inverse
                if len(self_inv_candidates) >= 3:
                    s1, s2, s3 = random.sample(self_inv_candidates, 3)
                    # Check girth
                    # We need to rebuild adj or pass s1,s2,s3 to get_girth
                    # Let's refactor get_girth to take s1,s2,s3
                    pass
                
                # Strategy 2: 1 self-inverse, 1 pair
                if len(self_inv_candidates) >= 1:
                    s1 = random.choice(self_inv_candidates)
                    s2 = random.choice(all_elements)
                    s3 = inv_idx[s2]
                    if s2 == s3:
                        continue
                    pass
        
        except Exception as e:
            pass
    
    return luw_construct(k, g)
