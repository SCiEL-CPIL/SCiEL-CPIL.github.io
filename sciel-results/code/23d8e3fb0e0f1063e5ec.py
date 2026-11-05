import time
import random

N = 18
DEADLINE = 140.0

def _build_adj(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_k4_free(adj):
    for u in range(N):
        nu = adj[u]
        v = nu
        while v:
            lb = v & (-v)
            i = lb.bit_length() - 1
            v -= lb
            ni_in_nu = adj[i] & nu
            w = ni_in_nu
            while w:
                lb2 = w & (-w)
                j = lb2.bit_length() - 1
                w -= lb2
                if adj[j] & ni_in_nu:
                    return False
    return True

def _max_triangle_free(adj):
    full = (1 << N) - 1
    best = [0]

    def bb(current_mask, remaining_mask):
        cur_size = current_mask.bit_count()
        rem_size = remaining_mask.bit_count()
        if cur_size + rem_size <= best[0]:
            return
        if remaining_mask == 0:
            if cur_size > best[0]:
                best[0] = cur_size
            return
        lb = remaining_mask & (-remaining_mask)
        v = lb.bit_length() - 1
        new_rem = remaining_mask - lb
        bb(current_mask, new_rem)
        nv_in_cur = adj[v] & current_mask
        ok = True
        nn = nv_in_cur
        while nn:
            lb2 = nn & (-nn)
            u = lb2.bit_length() - 1
            nn -= lb2
            if adj[u] & nv_in_cur:
                ok = False
                break
        if ok:
            bb(current_mask | lb, new_rem)

    bb(0, full)
    return best[0]

def _edges_from_adj(adj):
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u] & (1 << v):
                edges.append([u, v])
    return edges

def construct():
    start_time = time.time()
    random.seed(42)

    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    init_edges = [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    best_adj = _build_adj(init_edges)
    best_score = _max_triangle_free(best_adj)

    for it in range(300000):
        if time.time() - start_time > DEADLINE:
            break

        adj = [0] * N
        p = 0.3 + 0.4 * random.random()
        for u in range(N):
            for v in range(u + 1, N):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

        for _ in range(100):
            changed = False
            for u in range(N):
                nu = adj[u]
                v = nu
                found = False
                while v and not found:
                    lb = v & (-v)
                    i = lb.bit_length() - 1
                    v -= lb
                    ni_in_nu = adj[i] & nu
                    w = ni_in_nu
                    while w and not found:
                        lb2 = w & (-w)
                        j = lb2.bit_length() - 1
                        w -= lb2
                        if adj[j] & ni_in_nu:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                            changed = True
                            found = True
                            break
                if found:
                    break
            if not changed:
                break

        if not _is_k4_free(adj):
            continue

        score = _max_triangle_free(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

        for _ in range(80):
            if time.time() - start_time > DEADLINE:
                break
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            has_edge = bool(adj[u] & (1 << v))
            if has_edge:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_adj = adj[:]
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _is_k4_free(adj):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    continue
                new_adj = adj[:]
            new_score = _max_triangle_free(new_adj)
            if new_score <= score:
                score = new_score
                adj = new_adj
                if score < best_score:
                    best_score = score
                    best_adj = adj[:]
            else:
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

    return _edges_from_adj(best_adj)
