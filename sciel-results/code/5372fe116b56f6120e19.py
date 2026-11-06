import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    subsets = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        for p in range(m+1, n):
                            for q in range(p+1, n):
                                for r in range(q+1, n):
                                    subsets.append((i, j, k, l, m, p, q, r))
                                    
    adj = [0] * n
    best_edges = 0
    best_adj = [0] * n

    def is_q3(subset):
        for u in subset:
            deg = 0
            for v in subset:
                if u != v and (adj[u] >> v) & 1:
                    deg += 1
            if deg != 3:
                return False
        
        color = {}
        for u in subset:
            if u not in color:
                color[u] = 0
                stack = [u]
                while stack:
                    x = stack.pop()
                    for y in subset:
                        if x != y and (adj[x] >> y) & 1:
                            if y not in color:
                                color[y] = 1 - color[x]
                                stack.append(y)
                            elif color[y] == color[x]:
                                return False
        
        count_4_cycles = 0
        for i in range(8):
            for j in range(i+1, 8):
                u, w = subset[i], subset[j]
                common = 0
                for k in range(8):
                    v = subset[k]
                    if v != u and v != w and (adj[u] >> v) & 1 and (adj[w] >> v) & 1:
                        common += 1
                if common == 2:
                    count_4_cycles += 1
        if count_4_cycles != 12:
            return False
        return True

    def find_q3():
        for subset in subsets:
            if is_q3(subset):
                return subset
        return None

    def run_search(seed):
        nonlocal best_edges, best_adj
        random.seed(seed)
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.6:
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u

        while time.time() < deadline - 5:
            q3 = find_q3()
            if q3 is None:
                break
            for i in range(8):
                for j in range(i+1, 8):
                    u, v = q3[i], q3[j]
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        break
                else:
                    continue
                break

        for u in range(n):
            for v in range(u+1, n):
                if time.time() > deadline - 5:
                    break
                if (adj[u] >> v) & 1:
                    continue
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                if find_q3() is None:
                    pass
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        current_edges = sum(bin(adj[i]).count('1') for i in range(n)) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    run_search(42)
    
    while time.time() < deadline - 10:
        run_search(random.randint(0, 10000))

    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
