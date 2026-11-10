import random
import sys
from collections import deque

# GF(q) arithmetic for q=2^m
class GF:
    def __init__(self, m, poly):
        self.m = m
        self.q = 1 << m
        self.poly = poly
        self.mul_table = [[0]*self.q for _ in range(self.q)]
        self.inv_table = [0]*self.q
        
        for i in range(self.q):
            for j in range(self.q):
                res = 0
                a = i
                b = j
                while b:
                    if b & 1:
                        res ^= a
                    a <<= 1
                    if a >= self.q:
                        a ^= self.poly
                    b >>= 1
                self.mul_table[i][j] = res
        self.inv_table[1] = 1
        for i in range(2, self.q):
            for j in range(2, self.q):
                if self.mul_table[i][j] == 1:
                    self.inv_table[i] = j
                    self.inv_table[j] = i
                    break

    def add(self, a, b):
        return a ^ b

    def sub(self, a, b):
        return a ^ b

    def mul(self, a, b):
        return self.mul_table[a][b]

    def inv(self, a):
        return self.inv_table[a]

    def div(self, a, b):
        return self.mul_table[a][self.inv_table[b]]

def construct(k, g):
    if k != 3:
        raise ValueError("Only k=3 supported in this implementation")

    # Try smaller fields first to get smaller graphs
    # q=9 is not a power of 2, so GF class needs extension or we use q=8, 16
    # GF(8) size 504. GF(16) size 4080.
    # Let's try q=8 first. If it fails, try q=16.
    
    def try_field(m, poly, max_trials=200):
        field = GF(m, poly)
        q = field.q
        elements = []
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    if a == 0:
                        if field.mul(b, c) == 1:
                            for d in range(q):
                                elements.append((a,b,c,d))
                    else:
                        bc = field.mul(b, c)
                        rhs = field.add(1, bc)
                        d = field.div(rhs, a)
                        elements.append((a,b,c,d))
                        
        n = len(elements)
        elem_to_idx = {e: i for i, e in enumerate(elements)}
        
        involutions = []
        for e in elements:
            a,b,c,d = e
            if field.add(a,d) == 0 and not (a==1 and b==0 and c==0 and d==1):
                involutions.append(e)
                
        random.seed(42)
        
        def get_girth(adj, limit):
            n_nodes = len(adj)
            min_girth = float('inf')
            for start in range(n_nodes):
                dist = [-1] * n_nodes
                parent = [-1] * n_nodes
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                            if min_girth < limit:
                                return min_girth
            return min_girth

        for _ in range(max_trials):
            if len(involutions) < 3:
                return None
            S = random.sample(involutions, 3)
            adj = [[] for _ in range(n)]
            for i, e in enumerate(elements):
                for s in S:
                    a1,b1,c1,d1 = e
                    a2,b2,c2,d2 = s
                    aa = field.add(field.mul(a1,a2), field.mul(b1,c2))
                    ab = field.add(field.mul(a1,b2), field.mul(b1,d2))
                    ba = field.add(field.mul(c1,a2), field.mul(d1,c2))
                    bb = field.add(field.mul(c1,b2), field.mul(d1,d2))
                    f = (aa, ab, ba, bb)
                    j = elem_to_idx[f]
                    adj[i].append(j)
                    
            visited = [False]*n
            q_d = deque([0])
            visited[0] = True
            count = 1
            while q_d:
                u = q_d.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        q_d.append(v)
            if count != n:
                continue
                
            girth = get_girth(adj, g)
            if girth >= g:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        return None

    # Try q=8 (n=504)
    res = try_field(3, 0b1011, max_trials=50) # x^3+x+1=11
    if res:
        return res

    # Try q=16 (n=4080)
    res = try_field(4, 0b10011, max_trials=100) # x^4+x+1=19
    if res:
        return res

    raise RuntimeError("Failed to find valid graph")
