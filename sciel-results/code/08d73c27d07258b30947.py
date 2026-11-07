import time
import random

def construct():
    n = 15
    start = time.time()
    deadline = start + 85

    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def check_valid(adj_mask):
        deg = [bin(adj_mask[v]).count('1') for v in range(n)]
        max_deg = max(deg)
        deg_group = [0] * (max_deg + 1)
        for v in range(n):
            deg_group[deg[v]] |= (1 << v)
        for u in range(n):
            mask_u = adj_mask[u]
            for v in range(u + 1, n):
                if not (mask_u >> v) & 1:
                    continue
                mask_v = adj_mask[v]
                s_mask = mask_u & ~(1 << v)
                t_mask_base = mask_v & ~(1 << u)
                while s_mask:
                    lb = s_mask & (-s_mask)
                    s = lb.bit_length() - 1
                    s_mask ^= lb
                    t_mask = t_mask_base & ~(1 << s)
                    if deg[s] <= max_deg and (t_mask & deg_group[deg[s]]):
                        return False
        return True

    def count_edges(adj_mask):
        total = 0
        for v in range(n):
            total += bin(adj_mask[v]).count('1')
        return total // 2

    def make_bipartite(adj_mask, split):
        for v in range(n):
            adj_mask[v] = 0
        for i in range(split):
            for j in range(split, n):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    def try_add_edges_greedy(adj_mask, deadline, order=None):
        if order is None:
            order = list(range(len(edge_list)))
            random.shuffle(order)
        for idx in order:
            if time.time() > deadline:
                break
            i, j = edge_list[idx]
            if (adj_mask[i] >> j) & 1:
                continue
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
            if not check_valid(adj_mask):
                adj_mask[i] ^= (1 << j)
                adj_mask[j] ^= (1 << i)
        return adj_mask

    def local_search(adj_mask, deadline, max_rounds=300):
        improved = True
        rounds = 0
        while improved and time.time() < deadline - 3 and rounds < max_rounds:
            rounds += 1
            improved = False
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))
            random.shuffle(edges)
            random.shuffle(non_edges)

            for e2 in non_edges:
                if time.time() > deadline - 3:
                    break
                if improved:
                    break
                i2, j2 = e2
                adj_mask[i2] |= (1 << j2)
                adj_mask[j2] |= (1 << i2)
                if check_valid(adj_mask):
                    improved = True
                    break
                else:
                    adj_mask[i2] ^= (1 << j2)
                    adj_mask[j2] ^= (1 << i2)

            if improved:
                continue

            for e1 in edges:
                if time.time() > deadline - 3:
                    break
                if improved:
                    break
                i1, j1 = e1
                adj_mask[i1] ^= (1 << j1)
                adj_mask[j1] ^= (1 << i1)
                for e2 in non_edges:
                    if time.time() > deadline - 3:
                        break
                    if improved:
                        break
                    i2, j2 = e2
                    adj_mask[i2] |= (1 << j2)
                    adj_mask[j2] |= (1 << i2)
                    if check_valid(adj_mask):
                        improved = True
                        break
                    else:
                        adj_mask[i2] ^= (1 << j2)
                        adj_mask[j2] ^= (1 << i2)
                if not improved:
                    adj_mask[i1] |= (1 << j1)
                    adj_mask[j1] |= (1 << i1)
        return adj_mask

    best_adj = [0] * n
    best_count = 0

    # Try bipartite splits with multiple greedy trials
    for split in range(3, 13):
        if time.time() > deadline - 25:
            break
        for trial in range(5):
            if time.time() > deadline - 25:
                break
            adj_mask = [0] * n
            make_bipartite(adj_mask, split)
            adj_mask = try_add_edges_greedy(adj_mask, deadline)
            cnt = count_edges(adj_mask)
            if cnt > best_count:
                best_count = cnt
                best_adj = list(adj_mask)

    # Randomized greedy from empty
    for trial in range(15):
        if time.time() > deadline - 20:
            break
        adj_mask = [0] * n
        adj_mask = try_add_edges_greedy(adj_mask, deadline)
        cnt = count_edges(adj_mask)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Local search
    best_adj = local_search(best_adj, deadline)
    best_count = count_edges(best_adj)

    # 2-swap search
    if time.time() < deadline - 20:
        adj_mask = list(best_adj)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))
            random.shuffle(edges)
            random.shuffle(non_edges)
            max_e = min(len(edges), 35)
            max_ne = min(len(non_edges), 35)
            for idx1 in range(max_e):
                if time.time() > deadline - 5:
                    break
                if improved:
                    break
                for idx2 in range(idx1 + 1, max_e):
                    if time.time() > deadline - 5:
                        break
                    if improved:
                        break
                    e1 = edges[idx1]
                    e2 = edges[idx2]
                    for idx3 in range(max_ne):
                        if time.time() > deadline - 5:
                            break
                        if improved:
                            break
                        for idx4 in range(idx3 + 1, max_ne):
                            if time.time() > deadline - 5:
                                break
                            if improved:
                                break
                            f1 = non_edges[idx3]
                            f2 = non_edges[idx4]
                            for (a, b) in [e1, e2]:
                                adj_mask[a] ^= (1 << b)
                                adj_mask[b] ^= (1 << a)
                            for (a, b) in [f1, f2]:
                                adj_mask[a] |= (1 << b)
                                adj_mask[b] |= (1 << a)
                            if check_valid(adj_mask):
                                improved = True
                                break
                            else:
                                for (a, b) in [f1, f2]:
                                    adj_mask[a] ^= (1 << b)
                                    adj_mask[b] ^= (1 << a)
                                for (a, b) in [e1, e2]:
                                    adj_mask[a] |= (1 << b)
                                    adj_mask[b] |= (1 << a)
        cnt = count_edges(adj_mask)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Final greedy
    if time.time() < deadline - 2:
        adj_mask = list(best_adj)
        adj_mask = try_add_edges_greedy(adj_mask, deadline)
        cnt = count_edges(adj_mask)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([int(i), int(j)])
    return result
