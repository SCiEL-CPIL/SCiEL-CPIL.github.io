import time
from random import Random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []

        # Bitmask adjacency
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

        def popcount(x):
            return bin(x).count('1')

        def is_clique(vertices):
            if len(vertices) < 2:
                return False
            v0 = vertices[0]
            for v in vertices[1:]:
                if not (adj[v0] & (1 << v)):
                    return False
            # Check all pairs
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    if not (adj[vertices[i]] & (1 << vertices[j])):
                        return False
            return True

        def get_clique_edges(clique):
            return set((min(u, v), max(u, v)) for u, v in combinations(clique, 2))

        def max_clique(adj, n):
            best = []
            best_size = 0
            node_count = 0
            deadline_local = deadline - 10

            def expand(R, P, X):
                nonlocal best, best_size, node_count
                if time.time() > deadline_local:
                    return
                node_count += 1
                if node_count % 1024 == 0 and time.time() > deadline_local:
                    return
                if P == 0:
                    if X == 0 and len(R) > best_size:
                        best_size = len(R)
                        best = list(R)
                    return
                if len(R) + popcount(P) <= best_size:
                    return
                union = P | X
                if union == 0:
                    return
                best_pivot = -1
                best_pivot_deg = -1
                tmp = union
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    d = popcount(adj[v] & P)
                    if d > best_pivot_deg:
                        best_pivot_deg = d
                        best_pivot = v
                    tmp ^= lsb
                candidates = P & ~(adj[best_pivot] | (1 << best_pivot))
                while candidates:
                    lsb = candidates & (-candidates)
                    v = lsb.bit_length() - 1
                    candidates ^= lsb
                    R.append(v)
                    expand(R, P & adj[v], X & adj[v])
                    R.pop()
                    P ^= lsb
                    X |= lsb
                    if time.time() > deadline_local:
                        return

            expand([], (1 << n) - 1, 0)
            return best if len(best) >= 2 else None

        def greedy_clique(adj, n):
            active = [v for v in range(n) if adj[v] != 0]
            if not active:
                return None
            active.sort(key=lambda v: bin(adj[v]).count('1'), reverse=True)
            best = None
            best_size = 0
            for start in active[:20]:
                clique = [start]
                P = adj[start] & ~(1 << start)
                while P:
                    best_v = -1
                    best_d = -1
                    tmp = P
                    while tmp:
                        lsb = tmp & (-tmp)
                        v = lsb.bit_length() - 1
                        d = bin(adj[v] & P).count('1')
                        if d > best_d:
                            best_d = d
                            best_v = v
                        tmp ^= lsb
                    if best_v < 0:
                        break
                    clique.append(best_v)
                    P = (P & adj[best_v]) & ~(1 << best_v)
                if len(clique) > best_size:
                    best_size = len(clique)
                    best = clique
            return best if best and len(best) >= 2 else None

        # Track remaining edges
        remaining = set((min(u, v), max(u, v)) for u, v in edges)
        cliques = []

        # Fast greedy estimation for lookahead
        def estimate_remaining_cliques(rem_edges, rem_adj, n, steps=3):
            if not rem_edges:
                return 0
            count = 0
            # Copy adj for simulation? No, just simulate removal on a set
            sim_adj = rem_adj[:]
            sim_rem = set(rem_edges)
            for _ in range(steps):
                if not sim_rem:
                    break
                c = max_clique(sim_adj, n)
                if c is None:
                    c = greedy_clique(sim_adj, n)
                if c is None:
                    # Fallback: count edges / 2
                    return count + len(sim_rem) // 2 + 1
                cset = set(c)
                to_remove = set()
                for u, v in sim_rem:
                    if u in cset and v in cset:
                        to_remove.add((u, v))
                if not to_remove:
                    # Should not happen if c is a clique in sim_adj
                    to_remove = {next(iter(sim_rem))}
                sim_rem.difference_update(to_remove)
                for u, v in to_remove:
                    sim_adj[u] &= ~(1 << v)
                    sim_adj[v] &= ~(1 << u)
                count += 1
            # Estimate rest
            if sim_rem:
                # Rough bound: each clique covers at most max_clique_size^2 edges
                # Or just linear
                count += len(sim_rem) // 10 + 1
            return count

        while remaining:
            if time.time() > deadline - 20:
                for e in list(remaining):
                    cliques.append(list(e))
                break

            # Find candidate cliques
            candidates = []
            # Try to find a few good cliques
            # 1. Max clique
            c1 = max_clique(adj, n)
            if c1:
                candidates.append(c1)
            
            # 2. Greedy clique
            c2 = greedy_clique(adj, n)
            if c2 and c2 != c1:
                candidates.append(c2)
            
            # 3. Randomized greedy
            if len(candidates) < 3:
                active = [v for v in range(n) if adj[v] != 0]
                if active:
                    start = Random().choice(active)
                    c3 = [start]
                    P = adj[start] & ~(1 << start)
                    while P:
                        best_v = -1
                        best_d = -1
                        tmp = P
                        while tmp:
                            lsb = tmp & (-tmp)
                            v = lsb.bit_length() - 1
                            d = bin(adj[v] & P).count('1')
                            if d > best_d:
                                best_d = d
                                best_v = v
                            tmp ^= lsb
                        if best_v < 0:
                            break
                        c3.append(best_v)
                        P = (P & adj[best_v]) & ~(1 << best_v)
                    if len(c3) >= 2 and c3 not in candidates:
                        candidates.append(c3)

            if not candidates:
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])
                continue

            # Evaluate candidates with lookahead
            best_cand = None
            best_score = float('inf')
            
            for c in candidates:
                cset = set(c)
                to_remove = set()
                for u, v in remaining:
                    if u in cset and v in cset:
                        to_remove.add((u, v))
                
                if not to_remove:
                    continue

                # Simulate
                sim_rem = remaining - to_remove
                sim_adj = adj[:]
                for u, v in to_remove:
                    sim_adj[u] &= ~(1 << v)
                    sim_adj[v] &= ~(1 << u)
                
                # Estimate
                est = estimate_remaining_cliques(sim_rem, sim_adj, n, steps=2)
                total_est = 1 + est
                
                if total_est < best_score:
                    best_score = total_est
                    best_cand = c
                    best_to_remove = to_remove

            if best_cand is None:
                e = next(iter(remaining))
                cliques.append(list(e))
                remaining.discard(e)
                adj[e[0]] &= ~(1 << e[1])
                adj[e[1]] &= ~(1 << e[0])
            else:
                cliques.append(best_cand)
                remaining.difference_update(best_to_remove)
                for u, v in best_to_remove:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)

        # Local Search: Merge cliques if possible
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(cliques):
                j = i + 1
                while j < len(cliques):
                    c1 = cliques[i]
                    c2 = cliques[j]
                    # Check if union is a clique
                    union_is_clique = True
                    for u in c1:
                        for v in c2:
                            if not (adj[u] & (1 << v)):
                                union_is_clique = False
                                break
                        if not union_is_clique:
                            break
                    if union_is_clique:
                        # Merge c2 into c1
                        cliques[i] = c1 + c2
                        cliques.pop(j)
                        changed = True
                        break
                    j += 1
                if changed:
                    break
                i += 1

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
