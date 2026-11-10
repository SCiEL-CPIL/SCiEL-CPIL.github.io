import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin (evaluator kills at 150s)

    n = 20
    adj = [0] * n

    def popcount(x):
        return bin(x).count('1')

    def can_add(u, v):
        # Check if adding edge (u,v) keeps graph K2,3-free
        if (adj[u] >> v) & 1:
            return False
        # 1. Pair (u,v) common neighbours must be <= 2
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # 2. For all w in N(v)\{u}, pair (u,w) gains v -> need common(u,w) <= 1
        nv = adj[v] & ~(1 << u)
        while nv:
            lsb = nv & -nv
            w = lsb.bit_length() - 1
            nv ^= lsb
            if popcount(adj[u] & adj[w]) > 1:
                return False
        # 3. For all w in N(u)\{v}, pair (v,w) gains u -> need common(v,w) <= 1
        nu = adj[u] & ~(1 << v)
        while nu:
            lsb = nu & -nu
            w = lsb.bit_length() - 1
            nu ^= lsb
            if popcount(adj[v] & adj[w]) > 1:
                return False
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count():
        s = 0
        for i in range(n):
            s += popcount(adj[i])
        return s // 2

    def get_neighbors(i):
        res = []
        t = adj[i]
        while t:
            lsb = t & -t
            res.append(lsb.bit_length() - 1)
            t ^= lsb
        return res

    def rand_neighbor(u):
        # Fast unbiased pick of a random neighbour of u via bit-sampling
        mask = adj[u]
        while mask:
            k = random.randrange(n)
            if (mask >> k) & 1:
                return k
        return -1

    def greedy_completion():
        # Add every possible edge in a random order; returns new count
        nonlocal adj
        cnt = edge_count()
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            if (adj[u] >> v) & 1:
                continue
            if can_add(u, v):
                add_edge(u, v)
                cnt += 1
        return cnt

    def local_search(initial_adj, initial_count, time_limit):
        nonlocal adj
        adj = initial_adj[:]
        cnt = initial_count
        t0 = time.time()

        # Phase 1: greedy completion (fast, deterministic improvement)
        cnt = greedy_completion()

        # Phase 2: annealed search with 2-opt structural moves
        T = 6.0
        cooling = 0.9997
        min_T = 0.05

        while time.time() < t0 + time_limit:
            if T < min_T:
                T = min_T
            r = random.random()

            if r < 0.30:
                # Add a single edge (net +1, always accept if valid)
                u = random.randrange(n)
                v = random.randrange(n)
                if u == v or ((adj[u] >> v) & 1):
                    continue
                if can_add(u, v):
                    add_edge(u, v)
                    cnt += 1

            elif r < 0.50:
                # 2-opt: remove one edge, add two (net +1). Key escape move.
                u1 = random.randrange(n)
                v1 = rand_neighbor(u1)
                if v1 < 0:
                    continue
                # pick a non-edge (a,b) with a != b, not incident to the removed edge
                for _ in range(30):
                    a = random.randrange(n)
                    b = random.randrange(n)
                    if a == b:
                        continue
                    if (a == u1 and b == v1) or (a == v1 and b == u1):
                        continue
                    if (adj[a] >> b) & 1:
                        continue
                    break
                else:
                    continue
                # try remove (u1,v1) then add (a,b) and (b,a)
                remove_edge(u1, v1)
                if can_add(a, b):
                    add_edge(a, b)
                    if can_add(b, a):
                        add_edge(b, a)
                        cnt += 1
                    else:
                        remove_edge(a, b)
                        add_edge(u1, v1)
                else:
                    add_edge(u1, v1)

            elif r < 0.68:
                # Remove a single edge (net -1, annealed acceptance)
                u = random.randrange(n)
                v = rand_neighbor(u)
                if v < 0:
                    continue
                remove_edge(u, v)
                if random.random() < math.exp(-1.0 / T):
                    cnt -= 1
                else:
                    add_edge(u, v)

            elif r < 0.88:
                # Swap: remove one edge, add one (net 0, exploration)
                u1 = random.randrange(n)
                v1 = rand_neighbor(u1)
                if v1 < 0:
                    continue
                for _ in range(30):
                    u2 = random.randrange(n)
                    v2 = random.randrange(n)
                    if u2 == v2:
                        continue
                    if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                        continue
                    if (adj[u2] >> v2) & 1:
                        continue
                    break
                else:
                    continue
                remove_edge(u1, v1)
                if can_add(u2, v2):
                    add_edge(u2, v2)
                    # net 0, always accept for exploration
                else:
                    add_edge(u1, v1)

            else:
                # Double swap: remove two edges, add two (net 0, deeper exploration)
                u1 = random.randrange(n)
                v1 = rand_neighbor(u1)
                if v1 < 0:
                    continue
                u2 = random.randrange(n)
                v2 = rand_neighbor(u2)
                if v2 < 0:
                    continue
                if (u2 == u1 and v2 == v1) or (u2 == v1 and v2 == u1):
                    continue
                remove_edge(u1, v1)
                remove_edge(u2, v2)
                ok = True
                for _ in range(30):
                    a = random.randrange(n)
                    b = random.randrange(n)
                    if a == b or (adj[a] >> b) & 1:
                        continue
                    if (a == u1 and b == v1) or (a == v1 and b == u1):
                        continue
                    if (a == u2 and b == v2) or (a == v2 and b == u2):
                        continue
                    break
                else:
                    add_edge(u1, v1)
                    add_edge(u2, v2)
                    continue
                add_edge(a, b)
                for _ in range(30):
                    c = random.randrange(n)
                    d = random.randrange(n)
                    if c == d or (adj[c] >> d) & 1:
                        continue
                    if (c == a and d == b) or (c == b and d == a):
                        continue
                    if (c == u1 and d == v1) or (c == v1 and d == u1):
                        continue
                    if (c == u2 and d == v2) or (c == v2 and d == u2):
                        continue
                    break
                else:
                    remove_edge(a, b)
                    add_edge(u1, v1)
                    add_edge(u2, v2)
                    continue
                if can_add(c, d):
                    add_edge(c, d)
                    # net 0, accept
                else:
                    remove_edge(a, b)
                    add_edge(u1, v1)
                    add_edge(u2, v2)

            T *= cooling

        return adj[:], cnt

    # === Multi-start with diverse initialization ===
    best_adj = [0] * n
    best_count = 0

    n_restarts = 10
    restart_time = (deadline - start_time - 4.0) / n_restarts

    for restart in range(n_restarts):
        if time.time() > deadline - 4.0:
            break

        if restart == 0:
            # Random edge order greedy
            adj = [0] * n
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges:
                if can_add(u, v):
                    add_edge(u, v)
            count = edge_count()
        elif restart == 1:
            # Sparse random seed, then fill
            adj = [0] * n
            for _ in range(20):
                u = random.randrange(n)
                v = random.randrange(n)
                if u != v and not ((adj[u] >> v) & 1):
                    add_edge(u, v)
            count = edge_count()
        elif restart == 2:
            # Start from best, no perturbation (refine in place)
            adj = best_adj[:]
            count = best_count
        else:
            # Perturb best solution: remove some random edges to escape basin
            adj = best_adj[:]
            count = best_count
            for _ in range(random.randint(4, 10)):
                if count == 0:
                    break
                u = random.randrange(n)
                v = rand_neighbor(u)
                if v < 0:
                    continue
                remove_edge(u, v)
                count -= 1

        time_left = deadline - start_time - 4.0
        if time_left <= 0:
            break
        ls_time = min(restart_time, time_left * 0.6)
        if ls_time < 1.0:
            ls_time = max(0.5, time_left * 0.3)

        new_adj, new_count = local_search(adj[:], count, ls_time)
        if new_count > best_count:
            best_count = new_count
            best_adj = new_adj

    # Final greedy completion on the best solution
    if time.time() < deadline - 2.0:
        adj = best_adj[:]
        cnt = greedy_completion()
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # Convert best_adj back to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])

    return edges
