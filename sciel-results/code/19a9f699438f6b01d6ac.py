import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    def remaining():
        return deadline - time.time()

    # ---------- arithmetic helpers ----------
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def is_prime_power(n):
        if n < 2:
            return False
        if is_prime(n):
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if n % p == 0:
                t = n
                while t % p == 0:
                    t //= p
                return t == 1
        return False

    def is_quadratic_residue(a, q):
        a = a % q
        if a == 0:
            return True
        if not is_prime_power(q):
            return True
        p = None
        for cand in range(2, int(math.isqrt(q)) + 1):
            if q % cand == 0:
                t = q
                while t % cand == 0:
                    t //= cand
                if t == 1:
                    p = cand
                    break
        if p is None:
            p = q
        a_p = a % p
        if a_p == 0:
            return True
        return pow(a_p, (p - 1) // 2, p) == 1

    def modinv(a, m):
        """Extended Euclidean modular inverse, works for any modulus."""
        if m <= 1:
            return None
        a = a % m
        if a == 0:
            return None
        t, new_t = 0, 1
        r, new_r = m, a
        while new_r != 0:
            quotient = r // new_r
            t, new_t = new_t, t - quotient * new_t
            r, new_r = new_r, r - quotient * new_r
        if r != 1:
            return None
        return t % m

    def moore_bound(k, g):
        if k <= 1:
            return 0
        if g == 1:
            return 1
        r = g // 2
        if g % 2 == 1:
            s = 1
            term = k
            for _ in range(r - 1):
                s += term
                term *= (k - 1)
            return s
        else:
            s = 0
            term = 1
            for _ in range(r):
                s += term
                term *= (k - 1)
            return 2 * s

    n_min = moore_bound(k, g)

    # ---------- fast girth check ----------
    def check_girth_fast(adj, n, g):
        """Check that girth >= g using non-backtracking BFS from every vertex.
        A cycle of length L < g exists iff some non-backtracking walk of length L
        returns to its start. We detect this by BFS from each vertex tracking
        (current, parent) states; if we reach a state where current == start at
        depth L < g, we have a short cycle."""
        for u in range(n):
            # BFS state: (node, parent_node), depth = distance from u via non-backtracking
            # We track visited as set of (node, parent) to avoid revisiting
            visited = set()
            queue = deque()
            for v in adj[u]:
                queue.append((v, u))
                visited.add((v, u))
            depth = 1
            while queue and depth < g:
                new_queue = deque()
                for (cur, prev) in queue:
                    for nxt in adj[cur]:
                        if nxt == prev:
                            continue
                        if nxt == u:
                            return False  # cycle of length depth+1 < g
                        state = (nxt, cur)
                        if state not in visited:
                            visited.add(state)
                            new_queue.append(state)
                queue = new_queue
                depth += 1
        return True

    def is_connected(adj, n):
        visited = [False] * n
        visited[0] = True
        queue = deque([0])
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return count == n

    def validate(adj, n, k, g):
        for i in range(n):
            if len(adj[i]) != k:
                return False
        if not is_connected(adj, n):
            return False
        if not check_girth_fast(adj, n, g):
            return False
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Lazebnik-Ustimenko-Woldar baseline ----------
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # ---------- PGL(2, q) machinery ----------
    def action_on_point(x, a, b, c, d, q):
        if x == q:
            if c == 0:
                return q
            inv_c = modinv(c, q)
            if inv_c is None:
                return None
            return (a * inv_c) % q
        num = (a * x + b) % q
        den = (c * x + d) % q
        if den == 0:
            return q
        inv_den = modinv(den, q)
        if inv_den is None:
            return None
        return (num * inv_den) % q

    def inv_matrix(a, b, c, d, q):
        det = (a * d - b * c) % q
        det_inv = modinv(det, q)
        if det_inv is None:
            return None
        return (d * det_inv % q, -b * det_inv % q, -c * det_inv % q, a * det_inv % q)

    def is_derangement(a, b, c, d, q):
        a, b, c, d = a % q, b % q, c % q, d % q
        if c == 0:
            return False
        disc = ((d - a) ** 2 + 4 * b * c) % q
        if disc == 0:
            return False
        if is_quadratic_residue(disc, q):
            return False
        return True

    def build_from_gens(gens, q, k, g):
        n = q + 1
        adj = [set() for _ in range(n)]
        for i in range(n):
            x = i if i < q else q
            for gen in gens:
                y = action_on_point(x, gen[0], gen[1], gen[2], gen[3], q)
                if y is None or y == i:
                    return None
                j = y if y < q else q
                adj[i].add(j)
                adj[j].add(i)
        for i in range(n):
            if len(adj[i]) != k:
                return None
        adj_list = [list(s) for s in adj]
        return validate(adj_list, n, k, g)

    # ---------- Fast word-search pre-filter for PGL generators ----------
    def fast_word_search(gens, q, max_len):
        """
        Quick check: does any non-trivial word of length < max_len in the generators
        fix any point in P^1(F_q)? If yes, the Cayley graph has a short cycle.
        Uses BFS from EVERY point, tracking (point, last_gen_index) to avoid backtracking.
        Returns True if no short cycle detected (i.e., girth >= max_len is plausible).
        """
        n_pts = q + 1
        # Precompute generator actions on all points
        gen_actions = []
        for gen in gens:
            a, b, c, d = gen
            acts = [None] * n_pts
            for x in range(n_pts):
                px = x if x < q else q
                acts[x] = action_on_point(px, a, b, c, d, q)
            gen_actions.append(acts)

        max_depth = max_len - 1
        # For each starting point, do BFS
        for start_pt in range(n_pts):
            visited = set()
            queue = deque()
            for gi in range(len(gens)):
                y = gen_actions[gi][start_pt]
                if y is None:
                    continue
                if y == start_pt:
                    return False  # generator fixes start_pt -> self-loop
                state = (y, gi)
                if state not in visited:
                    visited.add(state)
                    queue.append((y, gi))

            depth = 1
            while queue and depth < max_depth:
                new_queue = deque()
                for (pt, last_gi) in queue:
                    for gi in range(len(gens)):
                        if gi == last_gi:
                            continue
                        y = gen_actions[gi][pt]
                        if y is None:
                            continue
                        if y == start_pt:
                            return False  # cycle back to start of length depth+1
                        state = (y, gi)
                        if state not in visited:
                            visited.add(state)
                            new_queue.append((y, gi))
                queue = new_queue
                depth += 1
        return True

    # ---------- Deterministic structured PGL ----------
    def try_structured_pgl(q, k, g):
        if q % 2 == 0 or not is_prime_power(q):
            return None
        n = q + 1
        if n < n_min:
            return None

        def is_nonsquare(a):
            a %= q
            if a == 0:
                return False
            return not is_quadratic_residue(a, q)

        cands = []
        for cc in range(1, q):
            if is_nonsquare(1 + 4 * cc):
                cands.append(cc)
            if len(cands) >= 12:
                break
        if len(cands) < 2:
            return None

        h_ok = is_nonsquare(8)
        h2_cands = []
        for cc in range(1, q):
            if is_nonsquare(1 - cc * cc):
                h2_cands.append(cc)
            if len(h2_cands) >= 10:
                break

        c1, c2 = cands[0], cands[1]
        g1 = (1, c1, 1, 0)
        g1i = inv_matrix(*g1, q)
        g2 = (1, c2, 1, 0)
        g2i = inv_matrix(*g2, q)
        if g1i is None or g2i is None:
            return None
        h = (1, 1, 1, q - 1)

        if h_ok:
            res = build_from_gens([g1, g1i, g2, g2i, h], q, k, g)
            if res:
                return res
            if len(cands) >= 3:
                g2b = (1, cands[2], 1, 0)
                g2bi = inv_matrix(*g2b, q)
                if g2bi is not None:
                    res = build_from_gens([g1, g1i, g2b, g2bi, h], q, k, g)
                    if res:
                        return res
            if len(cands) >= 4:
                g3b = (1, cands[3], 1, 0)
                g3bi = inv_matrix(*g3b, q)
                if g3bi is not None:
                    res = build_from_gens([g1, g1i, g3b, g3bi, h], q, k, g)
                    if res:
                        return res

        for cc in h2_cands[:6]:
            g3 = (1, q - cc, 1, cc)
            res = build_from_gens([g1, g1i, g2, g2i, g3], q, k, g)
            if res:
                return res
        return None

    # ---------- Random PGL search with fast word-search pre-filter ----------
    def random_search_pgl(q, k, g, budget):
        n = q + 1
        if n < n_min or n >= best_n[0]:
            return None
        if q % 2 == 0 or not is_prime_power(q):
            return None

        def random_gen():
            while True:
                a = random.randrange(q)
                b = random.randrange(q)
                c = random.randrange(q)
                d = random.randrange(q)
                if (a * d - b * c) % q != 0:
                    return (a, b, c, d)

        def random_gens():
            out = []
            attempts = 0
            max_attempts = k * 50
            while len(out) < k and attempts < max_attempts:
                attempts += 1
                m = random_gen()
                mi = inv_matrix(*m, q)
                if mi is None:
                    continue
                if not is_derangement(*m, q):
                    continue
                a2 = (m[0]*m[0] + m[1]*m[2]) % q
                b2 = (m[1]*(m[0]+m[3])) % q
                c2 = (m[2]*(m[0]+m[3])) % q
                d2 = (m[3]*m[3] + m[1]*m[2]) % q
                is_involution = (a2 == 1 and b2 == 0 and c2 == 0 and d2 == 1)
                if is_involution:
                    out.append(m)
                elif random.random() < 0.5:
                    out.append(m)
                    if mi != m and len(out) < k:
                        out.append(mi)
            while len(out) < k:
                m = random_gen()
                if is_derangement(*m, q):
                    out.append(m)
            return out[:k]

        seen = set()
        while time.time() < budget:
            gens = random_gens()
            key = tuple(gens)
            if key in seen:
                continue
            seen.add(key)
            if not all(is_derangement(*gen, q) for gen in gens):
                continue
            # Fast pre-filter: check for short relations via word search
            if not fast_word_search(gens, q, g):
                continue
            res = build_from_gens(gens, q, k, g)
            if res:
                return res
        return None

    # ---------- Cage-filling: build k-regular graph with girth maximization ----------
    def cage_fill_heuristic(n, k, g, budget):
        """
        Build a k-regular graph on n vertices:
        1. Start with a random spanning tree.
        2. Add remaining edges greedily, preferring pairs with high degree deficit.
        3. Validate girth.
        """
        if n * k % 2 != 0:
            return None
        total_edges = n * k // 2
        if total_edges < n - 1:
            return None

        # Build a random spanning tree
        adj = [set() for _ in range(n)]
        perm = list(range(n))
        random.shuffle(perm)
        visited = [False] * n
        visited[perm[0]] = True
        queue = deque([perm[0]])
        edge_count = 0
        while queue and edge_count < n - 1:
            u = queue.popleft()
            for v in perm:
                if visited[v]:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                visited[v] = True
                queue.append(v)
                edge_count += 1
                break

        # Fix connectivity if needed
        if edge_count < n - 1:
            unvisited = [i for i in range(n) if not visited[i]]
            if unvisited:
                for i in range(len(unvisited) - 1):
                    adj[unvisited[i]].add(unvisited[i+1])
                    adj[unvisited[i+1]].add(unvisited[i])
                    edge_count += 1
                adj[perm[0]].add(unvisited[0])
                adj[unvisited[0]].add(perm[0])
                edge_count += 1

        # Add remaining edges
        remaining_deg = [k - len(adj[i]) for i in range(n)]
        if any(d < 0 for d in remaining_deg):
            return None

        edges_to_add = sum(remaining_deg) // 2
        attempts = 0
        max_attempts = edges_to_add * 30 + 500

        while any(d > 0 for d in remaining_deg) and attempts < max_attempts:
            attempts += 1
            if time.time() > budget:
                break
            max_deg = max(remaining_deg)
            if max_deg <= 0:
                break
            candidates_u = [i for i in range(n) if remaining_deg[i] == max_deg]
            u = random.choice(candidates_u)
            v_candidates = [v for v in range(n) if v != u and remaining_deg[v] > 0 and v not in adj[u]]
            if not v_candidates:
                found = False
                for i in range(n):
                    if remaining_deg[i] <= 0:
                        continue
                    for j in range(i+1, n):
                        if remaining_deg[j] <= 0:
                            continue
                        if j not in adj[i]:
                            adj[i].add(j)
                            adj[j].add(i)
                            remaining_deg[i] -= 1
                            remaining_deg[j] -= 1
                            found = True
                            break
                    if found:
                        break
                if not found:
                    break
                continue
            v = random.choice(v_candidates)
            adj[u].add(v)
            adj[v].add(u)
            remaining_deg[u] -= 1
            remaining_deg[v] -= 1

        if any(d != 0 for d in remaining_deg):
            return None
        adj_list = [list(s) for s in adj]
        for i in range(n):
            if len(adj_list[i]) != k:
                return None
        if not is_connected(adj_list, n):
            return None
        if not check_girth_fast(adj_list, n, g):
            return None
        edges = []
        for i in range(n):
            for j in adj_list[i]:
                if i < j:
                    edges.append((i, j))
        return (n, edges)

    # ---------- Random regular graph via configuration model ----------
    def random_regular_graph(n, k, g, budget):
        if n * k % 2 != 0:
            return None
        if n < n_min:
            return None
        while time.time() < budget:
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if not valid:
                continue
            deg_ok = True
            for i in range(n):
                if len(adj[i]) != k:
                    deg_ok = False
                    break
            if not deg_ok:
                continue
            if not is_connected(adj, n):
                continue
            if not check_girth_fast(adj, n, g):
                continue
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return (n, edges)
        return None

    # ---------- main ----------
    n_luw, edges_luw = luw_construct(k, g)
    best_n = [n_luw]
    best_result = (n_luw, edges_luw)

    # 2. Deterministic structured PGL
    q_max_struct = min(best_n[0] - 2, 300)
    for curr_q in range(max(3, n_min - 1), q_max_struct + 1):
        if time.time() > deadline - 40.0:
            break
        res = try_structured_pgl(curr_q, k, g)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            for curr_q2 in range(curr_q + 1, min(curr_q + 8, best_n[0] - 1)):
                if time.time() > deadline - 40.0:
                    break
                res2 = try_structured_pgl(curr_q2, k, g)
                if res2 and res2[0] < best_n[0]:
                    best_n[0] = res2[0]
                    best_result = res2
            return best_result

    # 3. Random PGL search with fast word-search pre-filter
    q_cursor = max(3, n_min - 1)
    while q_cursor + 1 < best_n[0] and time.time() < deadline - 15.0:
        time_left = deadline - 15.0 - time.time()
        if time_left <= 0:
            break
        q_range = best_n[0] - 1 - q_cursor
        if q_range > 0:
            budget = min(time_left, max(0.3, 2.0 * (2.0 / (1.0 + q_range))))
        else:
            budget = time_left
        res = random_search_pgl(q_cursor, k, g, budget)
        if res and res[0] < best_n[0]:
            best_n[0] = res[0]
            best_result = res
            for _ in range(3):
                if time.time() > deadline - 15.0:
                    break
                q_cursor += 1
                if q_cursor + 1 >= best_n[0]:
                    break
                b2 = min(deadline - 15.0 - time.time(), 1.5)
                if b2 <= 0:
                    break
                res2 = random_search_pgl(q_cursor, k, g, b2)
                if res2 and res2[0] < best_n[0]:
                    best_n[0] = res2[0]
                    best_result = res2
            break
        q_cursor += 1

    # 4. Cage-filling heuristic
    if time.time() < deadline - 5.0:
        for trial_n in range(n_min, min(best_n[0], n_min + 30)):
            if time.time() > deadline - 5.0:
                break
            budget = min(deadline - 5.0 - time.time(), 2.0)
            if budget <= 0:
                break
            res = cage_fill_heuristic(trial_n, k, g, budget)
            if res and res[0] < best_n[0]:
                best_n[0] = res[0]
                best_result = res
                break

    # 5. Random regular graph search
    if time.time() < deadline - 2.0:
        for trial_n in range(n_min, min(best_n[0], n_min + 50)):
            if time.time() > deadline - 2.0:
                break
            budget = min(deadline - 2.0 - time.time(), 0.5)
            if budget <= 0:
                break
            res = random_regular_graph(trial_n, k, g, budget)
            if res and res[0] < best_n[0]:
                best_n[0] = res[0]
                best_result = res
                break

    return best_result

# EVOLVE-BLOCK-END
