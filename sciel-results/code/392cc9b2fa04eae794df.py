import time
import sys
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    def check_girth(n, edges, min_girth, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                d = dist[node]
                if d >= min_girth // 2:
                    continue
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < min_girth:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
        return min_cycle if min_cycle != float('inf') else 999

    def voltage_lift(base_edges, base_n, m, time_limit):
        edge_index = {}
        for i, (u, v) in enumerate(base_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(base_n)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)

        max_L = g - 1
        constraints = set()

        for start_v in range(base_n):
            stack = [(start_v, -1, 0, [0] * len(base_edges))]
            while stack:
                current, prev, length, counts_list = stack.pop()
                if length > max_L:
                    continue
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = base_edges[e_idx]
                    d = 1 if current == u else -1
                    new_counts = counts_list.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    if next_v == start_v and new_l >= 5:
                        c_tuple = tuple(new_counts)
                        if any(x != 0 for x in c_tuple):
                            first_nz = -1
                            for i in range(len(c_tuple)):
                                if c_tuple[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1 and c_tuple[first_nz] < 0:
                                c_tuple = tuple(-x for x in c_tuple)
                            constraints.add(c_tuple)
                    if new_l < max_L:
                        stack.append((next_v, current, new_l, new_counts))

        if not constraints:
            return None

        C_list = list(constraints)
        C = np.array(C_list, dtype=np.int64)
        
        def cost_func(v):
            r = np.mod(C @ v, m)
            return int(np.sum(r == 0))

        best_v = None
        best_cost = float('inf')
        
        for restart in range(20):
            if time.time() > time_limit:
                break
            v = np.random.randint(1, m, size=len(base_edges)).astype(np.int64)
            curr_cost = cost_func(v)
            
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                for j in range(len(base_edges)):
                    old = v[j]
                    best_j = old
                    best_jc = curr_cost
                    cands = set()
                    cands.add(random.randint(1, m - 1))
                    cands.add((old + 1) % m or 1)
                    cands.add((old - 1) % m or 1)
                    for cand in cands:
                        if cand == old or cand < 1:
                            continue
                        v[j] = cand
                        c = cost_func(v)
                        if c < best_jc:
                            best_jc = c
                            best_j = cand
                    v[j] = best_j
                    if best_jc < curr_cost:
                        curr_cost = best_jc
                        improved = True
                        if curr_cost == 0:
                            break
                if curr_cost == 0:
                    break
            
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_v = v.copy()
                if best_cost == 0:
                    break
        
        if best_cost != 0:
            return None

        edges = []
        for u, v_base in base_edges:
            e_idx = edge_index[(u, v_base)]
            vol = best_v[e_idx]
            for layer in range(m):
                u_lift = u * m + layer
                v_lift = v_base * m + (layer + vol) % m
                if u_lift < v_lift:
                    edges.append((u_lift, v_lift))
                else:
                    edges.append((v_lift, u_lift))
        
        return (base_n * m, edges)

    # Heawood Graph
    heawood_edges = [
        (0, 7), (1, 7), (2, 7),
        (0, 8), (3, 8), (4, 8),
        (0, 9), (5, 9), (6, 9),
        (1, 10), (3, 10), (5, 10),
        (1, 11), (4, 11), (6, 11),
        (2, 12), (3, 12), (6, 12),
        (2, 13), (4, 13), (5, 13)
    ]
    heawood_n = 14

    # Petersen Graph
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    pet_n = 10

    if k == 3 and g == 16:
        # Try Heawood first
        primes_heawood = [37, 41, 43, 47, 53, 59, 61]
        for m in primes_heawood:
            if remaining() < 10:
                break
            n_try = 14 * m
            if n_try >= 530: # Skip if worse than known Petersen bound
                continue
            res = voltage_lift(heawood_edges, heawood_n, m, time.time() + 15.0)
            if res:
                n, edges = res
                gc = check_girth(n, edges, g, time.time() + 5.0)
                if gc >= g:
                    return (n, edges)
        
        # Try Petersen
        primes_pet = [53, 59, 61, 67, 71, 73, 79]
        for m in primes_pet:
            if remaining() < 10:
                break
            n_try = 10 * m
            res = voltage_lift(pet_edges, pet_n, m, time.time() + 15.0)
            if res:
                n, edges = res
                gc = check_girth(n, edges, g, time.time() + 5.0)
                if gc >= g:
                    return (n, edges)
        
        return (0, [])
    else:
        return (0, [])
