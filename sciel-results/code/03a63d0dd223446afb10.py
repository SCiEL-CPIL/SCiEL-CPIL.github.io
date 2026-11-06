import time
import random

def construct():
    n = 16
    target_edges = 65
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety

    def get_score(adj, deg):
        max_s = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        if w > v:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_s:
                                max_s = s
                        common ^= lsb
        return max_s

    def init_graph():
        adj = [0] * n
        deg = [0] * n
        edges = []
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_possible)
        for u, v in all_possible[:target_edges]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            edges.append((u, v))
        return adj, deg, edges

    best_score = float('inf')
    best_adj = None
    best_deg = None

    current_adj, current_deg, current_edges = init_graph()
    current_score = get_score(current_adj, current_deg)
    
    if current_score < best_score:
        best_score = current_score
        best_adj = current_adj[:]
        best_deg = current_deg[:]

    stagnation = 0
    max_stagnation = 5000

    while time.time() < deadline:
        if stagnation > max_stagnation:
            current_adj, current_deg, current_edges = init_graph()
            current_score = get_score(current_adj, current_deg)
            stagnation = 0
            max_stagnation = 5000 + random.randint(0, 10000)
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
                best_deg = current_deg[:]
            continue

        rem_idx = random.randint(0, len(current_edges) - 1)
        rem_edge = current_edges[rem_idx]
        u1, v1 = rem_edge

        while True:
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if not (current_adj[u2] & (1 << v2)):
                add_edge = (u2, v2)
                break
        
        if rem_edge == add_edge:
            continue

        common = current_adj[u2] & current_adj[v2]
        max_new_sum = 0
        if common:
            new_deg_u2 = current_deg[u2] + 1
            new_deg_v2 = current_deg[v2] + 1
            while common:
                lsb = common & -common
                w = lsb.bit_length() - 1
                s = new_deg_u2 + new_deg_v2 + current_deg[w]
                if s > max_new_sum:
                    max_new_sum = s
                common ^= lsb
        
        accept = False
        if max_new_sum <= current_score:
            accept = True
        else:
            # Increase acceptance of bad moves if stuck
            prob = 0.05 + 0.1 * (stagnation / max_stagnation)
            if random.random() < prob:
                accept = True

        if not accept:
            stagnation += 1
            continue

        current_adj[u1] &= ~(1 << v1)
        current_adj[v1] &= ~(1 << u1)
        current_deg[u1] -= 1
        current_deg[v1] -= 1
        current_edges[rem_idx] = add_edge
        
        current_adj[u2] |= (1 << v2)
        current_adj[v2] |= (1 << u2)
        current_deg[u2] += 1
        current_deg[v2] += 1

        new_score = get_score(current_adj, current_deg)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = current_adj[:]
            best_deg = current_deg[:]
            stagnation = 0
        else:
            stagnation += 1

        current_score = new_score

    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                final_edges.append([u, v])
    
    return {'n': n, 'edges': final_edges}
