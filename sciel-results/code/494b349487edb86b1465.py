import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    if k == 4 and g == 10:
        def get_walks(length):
            walks = []
            for start in range(8):
                stack = [(start, -1, [start])]
                while stack:
                    curr, prev, path = stack.pop()
                    if len(path) == length + 1:
                        if path[-1] == start:
                            valid = all(path[i] != path[i-2] for i in range(2, len(path)))
                            if valid:
                                sig = []
                                for i in range(len(path)-1):
                                    u, v = path[i], path[i+1]
                                    if u < 4:
                                        sig.append((1, u, v-4))
                                    else:
                                        sig.append((-1, v-4, u))
                                walks.append(sig)
                        continue
                    for next_node in range(8):
                        if ((curr < 4) == (next_node < 4)) or next_node == prev:
                            continue
                        stack.append((next_node, curr, path + [next_node]))
            return walks
        
        walks_4 = get_walks(4)
        walks_6 = get_walks(6)
        walks_8 = get_walks(8)
        
        def check_voltages(x, n):
            for sig in walks_4:
                if sum(s * x[u][v] for s, u, v in sig) % n == 0:
                    return False
            for sig in walks_6:
                if sum(s * x[u][v] for s, u, v in sig) % n == 0:
                    return False
            for sig in walks_8:
                if sum(s * x[u][v] for s, u, v in sig) % n == 0:
                    return False
            return True
        
        def construct_graph(x, n):
            edges = []
            for u in range(4):
                for v in range(4):
                    volt = x[u][v]
                    for i in range(n):
                        edges.append((u * n + i, 4 * n + v * n + ((i + volt) % n)))
            return 8 * n, edges
        
        def verify_girth(n_graph, edges, g):
            adj = [[] for _ in range(n_graph)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            for start in range(n_graph):
                dist = [-1] * n_graph
                parent = [-1] * n_graph
                q = deque([start])
                dist[start] = 0
                while q:
                    curr = q.popleft()
                    for nei in adj[curr]:
                        if dist[nei] == -1:
                            dist[nei] = dist[curr] + 1
                            parent[nei] = curr
                            q.append(nei)
                        elif nei != parent[curr]:
                            if dist[curr] + dist[nei] + 1 < g:
                                return False
            return True
        
        for n in range(40, 30, -1):
            if time.time() > deadline - 5:
                break
            # Structured initializations
            for mult in [11, 13, 17, 19, 23]:
                x0 = [[(i * j * mult) % n for j in range(4)] for i in range(4)]
                if check_voltages(x0, n):
                    ng, edges = construct_graph(x0, n)
                    if verify_girth(ng, edges, g):
                        return ng, edges
            # Random search
            for _ in range(50000):
                if time.time() > deadline - 5:
                    break
                x = [[random.randint(0, n-1) for _ in range(4)] for _ in range(4)]
                if check_voltages(x, n):
                    ng, edges = construct_graph(x, n)
                    if verify_girth(ng, edges, g):
                        return ng, edges
    
    # LUW fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = lambda x,y: x^y, lambda x,y: x^y, lambda x,y: table[x][y]
    else:
        add, sub, mul = lambda x,y: (x+y)%q, lambda x,y: (x-y)%q, lambda x,y: x*y%q
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
