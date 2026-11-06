import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def creates_k4(u, v, adj, n):
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        c = common
        while c:
            lsb = c & -c
            x = lsb.bit_length() - 1
            c ^= lsb
            if (adj[x] & common) & ~(1 << x):
                return True
        return False

    def delta_i5(u, v, adj, n):
        is_edge = bool((adj[u] >> v) & 1)
        mask = (1 << n) - 1
        s_mask = mask & ~(adj[u] | adj[v] | (1 << u) | (1 << v))
        verts = []
        m = s_mask
        while m:
            lsb = m & -m
            verts.append(lsb.bit_length() - 1)
            m ^= lsb
        
        k = len(verts)
        i3_count = 0
        if k >= 3:
            for i in range(k):
                vi = verts[i]
                for j in range(i + 1, k):
                    vj = verts[j]
                    if not ((adj[vi] >> vj) & 1):
                        for l in range(j + 1, k):
                            vl = verts[l]
                            if not ((adj[vi] >> vl) & 1) and not ((adj[vj] >> vl) & 1):
                                i3_count += 1
        
        if is_edge:
            return i3_count
        else:
            return -i3_count

    def has_i6(adj, n):
        def bt(candidates, depth):
            if depth == 6:
                return True
            if len(candidates) < depth:
                return False
            for idx in range(len(candidates)):
                v = candidates[idx]
                new_cands = []
                for u in candidates[idx + 1:]:
                    if not ((adj[v] >> u) & 1):
                        new_cands.append(u)
                if bt(new_cands, depth + 1):
                    return True
            return False
        return bt(list(range(n)), 0)

    def is_k4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    if creates_k4(u, v, adj, n):
                        return False
        return True

    for target_n in [35, 34, 33, 32, 31, 30, 28, 25, 20, 17]:
        if time.time() > deadline - 5:
            break

        adj = [0] * target_n
        for u in range(target_n):
            for v in range(u + 1, target_n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

        # Enforce K4-free
        for _ in range(500):
            changed = False
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if (adj[u] >> v) & 1:
                        if creates_k4(u, v, adj, target_n):
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            changed = True
            if not changed:
                break

        # SA
        T = 1.0
        cooling = 0.999
        min_T = 0.1
        current_cost = 0
        
        for step in range(50000):
            if time.time() > deadline - 10:
                break
            
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v:
                continue
            
            is_edge = bool((adj[u] >> v) & 1)
            
            if not is_edge:
                if creates_k4(u, v, adj, target_n):
                    continue
            
            d_cost = delta_i5(u, v, adj, target_n)
            
            if d_cost < 0 or random.random() < math.exp(-d_cost / T):
                if is_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                current_cost += d_cost
                
                if step % 50 == 0:
                    if not has_i6(adj, target_n) and is_k4_free(adj, target_n):
                        return {'n': target_n, 'edges': get_edges(adj, target_n)}
            T *= cooling
            if T < min_T:
                break

    squares = {x * x % 17 for x in range(1, 17)}
    adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return {'n': 17, 'edges': get_edges(adj, 17)}
