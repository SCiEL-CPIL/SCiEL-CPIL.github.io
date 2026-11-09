import time
import random

def construct():
    start = time.time()
    deadline = start + 138.0

    def time_left():
        return deadline - time.time()

    def count_ind6(adj, n):
        """Exact count of independent 6-sets."""
        total = 0
        for a in range(n - 5):
            ca = adj[a]
            for b in range(a + 1, n - 4):
                if (ca >> b) & 1:
                    continue
                cb = adj[b]
                for c in range(b + 1, n - 3):
                    if (ca >> c) & 1 or (cb >> c) & 1:
                        continue
                cc = adj[c]
                for d in range(c + 1, n - 2):
                    if (ca >> d) & 1 or (cb >> d) & 1 or (cc >> d) & 1:
                        continue
                cd = adj[d]
                for e in range(d + 1, n - 1):
                    if (ca >> e) & 1 or (cb >> e) & 1 or (cc >> e) & 1 or (cd >> e) & 1:
                        continue
                    ce = adj[e]
                    for f in range(e + 1, n):
                        if (ca >> f) & 1 or (cb >> f) & 1 or (cc >> f) & 1 or (cd >> f) & 1 or (ce >> f) & 1:
                            continue
                        total += 1
        return total

    def count_k4(adj, n):
        """Exact count of K4s."""
        total = 0
        for a in range(n - 3):
            ca = adj[a]
            for b in range(a + 1, n - 2):
                if not ((ca >> b) & 1):
                    continue
                cb = adj[b]
                for c in range(b + 1, n - 1):
                    if not ((ca >> c) & 1) or not ((cb >> c) & 1):
                        continue
                    cc = adj[c]
                    for d in range(c + 1, n):
                        if ((ca >> d) & 1) and ((cb >> d) & 1) and ((cc >> d) & 1):
                            total += 1
        return total

    def is_valid(adj, n):
        if n <= 1:
            return True
        if count_k4(adj, n) > 0:
            return False
        if count_ind6(adj, n) > 0:
            return False
        return True

    def edges_from_masks(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def random_adj(n, p):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def random_valid_small(n, tries=300):
        """Randomly find a valid graph of size n (small n is easy)."""
        for _ in range(tries):
            p = random.uniform(0.35, 0.65)
            adj = random_adj(n, p)
            if is_valid(adj, n):
                return adj
        return None

    def extract_best(adj, n):
        """Try to recover the largest valid subgraph (by vertex subset)."""
        if is_valid(adj, n):
            return n, adj
        # Try dropping single vertices
        for drop in range(n):
            m = n - 1
            sub = [0] * m
            for i in range(m):
                src = i if i < drop else i + 1
                mask = adj[src]
                newmask = 0
                for j in range(m):
                    dst = j if j < drop else j + 1
                    if (mask >> dst) & 1:
                        newmask |= (1 << j)
                sub[i] = newmask
            if is_valid(sub, m):
                return m, sub
        # Try dropping pairs
        for d1 in range(n):
            for d2 in range(d1 + 1, n):
                m = n - 2
                sub = [0] * m
                for i in range(m):
                    src = i
                    if src >= d1:
                        src += 1
                    if src >= d2:
                        src += 1
                    mask = adj[src]
                    newmask = 0
                    for j in range(m):
                        dst = j
                        if dst >= d1:
                            dst += 1
                        if dst >= d2:
                            dst += 1
                        if (mask >> dst) & 1:
                            newmask |= (1 << j)
                    sub[i] = newmask
                if is_valid(sub, m):
                    return m, sub
        return 0, None

    def sa_optimize(n, init_adj, budget):
        """Simulated annealing minimizing K4count + Ind6count. Returns best (cost, adj)."""
        adj = init_adj
        cur_k4 = count_k4(adj, n)
        cur_ind = count_ind6(adj, n)
        cur_cost = cur_k4 + cur_ind
        best_cost = cur_cost
        best_adj = adj[:]
        t0 = time.time()
        T = 1.0
        cool = 0.9995
        rng = random
        while time_left() > budget:
            T *= cool
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if (adj[i] >> j) & 1:
                # removing edge i-j
                new_k4 = cur_k4
                new_ind = cur_ind
                # K4s destroyed: triples {a,b} with a<b, a!=i,j, b!=i,j, all adjacent to both i and j
                ai = adj[i]
                aj = adj[j]
                cand = (ai & aj)
                cand &= ~((1 << i) | (1 << j))
                # count pairs (a,b) in cand that are adjacent to each other
                # K4(i,j,a,b) requires a-b edge. Count edges within cand.
                destroyed = 0
                c = cand
                while c:
                    lb = c & (-c)
                    c ^= lb
                    a = lb.bit_length() - 1
                    destroyed += (cand & adj[a]).bit_count()
                destroyed //= 2
                new_k4 -= destroyed
                # Ind6 created: 5-sets S in (non-neighbors of i) ∩ (non-neighbors of j), S excludes i,j
                non_i = ~ai & ((1 << n) - 1)
                non_j = ~aj & ((1 << n) - 1)
                pool = (non_i & non_j) & ~((1 << i) | (1 << j))
                # count independent 5-sets within pool
                created = 0
                # iterate 5-subsets of pool
                verts = []
                pp = pool
                while pp:
                    lb = pp & (-pp)
                    pp ^= lb
                    verts.append(lb.bit_length() - 1)
                k = len(verts)
                if k >= 5:
                    # count independent 5-sets
                    for a in range(k - 4):
                        va = verts[a]
                        for b in range(a + 1, k - 3):
                            vb = verts[b]
                            if (adj[va] >> vb) & 1:
                                continue
                            for c2 in range(b + 1, k - 2):
                                vc = verts[c2]
                                if (adj[va] >> vc) & 1 or (adj[vb] >> vc) & 1:
                                    continue
                                for d2 in range(c2 + 1, k - 1):
                                    vd = verts[d2]
                                    if (adj[va] >> vd) & 1 or (adj[vb] >> vd) & 1 or (adj[vc] >> vd) & 1:
                                        continue
                                    for e2 in range(d2 + 1, k):
                                        ve = verts[e2]
                                        if (adj[va] >> ve) & 1 or (adj[vb] >> ve) & 1 or (adj[vc] >> ve) & 1 or (adj[vd] >> ve) & 1:
                                            continue
                                        created += 1
                new_ind += created
            else:
                # adding edge i-j
                new_k4 = cur_k4
                new_ind = cur_ind
                # Ind6 destroyed: 5-sets S in (non-neighbors of i) ∩ (non-neighbors of j)
                ai = adj[i]
                aj = adj[j]
                non_i = ~ai & ((1 << n) - 1)
                non_j = ~aj & ((1 << n) - 1)
                pool = (non_i & non_j) & ~((1 << i) | (1 << j))
                destroyed_ind = 0
                verts = []
                pp = pool
                while pp:
                    lb = pp & (-pp)
                    pp ^= lb
                    verts.append(lb.bit_length() - 1)
                k = len(verts)
                if k >= 5:
                    for a in range(k - 4):
                        va = verts[a]
                        for b in range(a + 1, k - 3):
                            vb = verts[b]
                            if (adj[va] >> vb) & 1:
                                continue
                            for c2 in range(b + 1, k - 2):
                                vc = verts[c2]
                                if (adj[va] >> vc) & 1 or (adj[vb] >> vc) & 1:
                                    continue
                                for d2 in range(c2 + 1, k - 1):
                                    vd = verts[d2]
                                    if (adj[va] >> vd) & 1 or (adj[vb] >> vd) & 1 or (adj[vc] >> vd) & 1:
                                        continue
                                    for e2 in range(d2 + 1, k):
                                        ve = verts[e2]
                                        if (adj[va] >> ve) & 1 or (adj[vb] >> ve) & 1 or (adj[vc] >> ve) & 1 or (adj[vd] >> ve) & 1:
                                            continue
                                        destroyed_ind += 1
                new_ind -= destroyed_ind
                # K4s created: triples {a,b} with a<b, all adjacent to both i and j, and a-b edge
                cand = (ai & aj) & ~((1 << i) | (1 << j))
                created_k4 = 0
                c = cand
                while c:
                    lb = c & (-c)
                    c ^= lb
                    a = lb.bit_length() - 1
                    created_k4 += (cand & adj[a]).bit_count()
                created_k4 //= 2
                new_k4 += created_k4
            new_cost = new_k4 + new_ind
            if new_cost < cur_cost:
                accept = True
            else:
                delta = new_cost - cur_cost
                if delta == 0:
                    accept = True
                else:
                    import math
                    accept = rng.random() < math.exp(-delta / max(T, 1e-9))
            if accept:
                if (adj[i] >> j) & 1:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                else:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                cur_k4 = new_k4
                cur_ind = new_ind
                cur_cost = new_cost
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_adj = adj[:]
        return best_cost, best_adj

    # Build a ladder of valid graphs; use each as seed for the next larger n.
    best_n = 1
    best_adj = [0]
    current_seed = None  # adj list for seeding

    # Start from small valid graph
    for base_n in range(2, 8):
        if time_left() < 30:
            break
        adj = random_valid_small(base_n, tries=400)
        if adj is not None and is_valid(adj, base_n):
            best_n = base_n
            best_adj = adj
            current_seed = adj
            break

    # Now grow upward, seeding each new n from the previous valid graph.
    target = best_n + 1
    while target <= 36 and time_left() > 20:
        # Try to build a valid graph of size `target`
        # Seed: extend current_seed by adding (target - len) new vertices with random edges.
        seed_adj = None
        if current_seed is not None:
            prev_n = len(current_seed)
            seed_adj = current_seed[:] + [0] * (target - prev_n)
            for i in range(prev_n, target):
                for j in range(target):
                    if i == j:
                        continue
                    if random.random() < 0.5:
                        seed_adj[i] |= (1 << j)
                        seed_adj[j] |= (1 << i)
        if seed_adj is None:
            seed_adj = random_adj(target, 0.5)

        # Run SA to minimize violations, with a time budget
        budget = 18.0
        cost, sa_adj = sa_optimize(target, seed_adj, budget)
        if cost == 0:
            # Found a valid graph of size target!
            if is_valid(sa_adj, target):
                best_n = target
                best_adj = sa_adj
                current_seed = sa_adj
                target += 1
                continue
        # SA didn't reach zero; try to extract a valid subgraph
        m, sub = extract_best(sa_adj, target)
        if m >= best_n:
            # Use the extracted subgraph as a candidate
            if m > best_n:
                best_n = m
                best_adj = sub
            current_seed = sub if m >= target - 2 else current_seed
        # If we couldn't make progress, try a couple more restarts with different seeds
        improved = False
        for _ in range(2):
            if time_left() < 22:
                break
            fresh = random_adj(target, random.uniform(0.4, 0.6))
            cost2, sa_adj2 = sa_optimize(target, fresh, 10.0)
            if cost2 == 0 and is_valid(sa_adj2, target):
                best_n = target
                best_adj = sa_adj2
                current_seed = sa_adj2
                improved = True
                break
            m2, sub2 = extract_best(sa_adj2, target)
            if m2 > best_n:
                best_n = m2
                best_adj = sub2
                current_seed = sub2
                improved = True
        if improved:
            target += 1
        else:
            # Stuck: try one more push with a bigger budget before giving up on this target
            if time_left() > 40:
                fresh = random_adj(target, random.uniform(0.4, 0.6))
                cost3, sa_adj3 = sa_optimize(target, fresh, 25.0)
                if cost3 == 0 and is_valid(sa_adj3, target):
                    best_n = target
                    best_adj = sa_adj3
                    current_seed = sa_adj3
                    target += 1
                    continue
                m3, sub3 = extract_best(sa_adj3, target)
                if m3 > best_n:
                    best_n = m3
                    best_adj = sub3
                    current_seed = sub3
            # Give up on this target; stop growing.
            break

    # Final safety: ensure we have a valid graph.
    if best_adj is None or not is_valid(best_adj, best_n):
        # Fallback to a guaranteed small valid graph.
        for fb in range(7, 1, -1):
            adj = random_valid_small(fb, tries=200)
            if adj is not None and is_valid(adj, fb):
                best_n = fb
                best_adj = adj
                break

    edges = edges_from_masks(best_adj, best_n)
    return {'n': best_n, 'edges': edges}
