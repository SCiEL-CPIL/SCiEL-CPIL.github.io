import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before the 150 s kill

    # ---------- bitmask helpers ----------
    def count_edges(adj):
        return sum(b.bit_count() for b in adj) // 2

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lsb
        return edges

    def is_add_valid(u, v, adj):
        """
        K3,3 appears iff some pair {a1,a2} in N(u) shares >=2 common neighbors
        inside N(v) (or symmetrically for the swapped roles).
        After adding (u,v) the new edges are (u,v) and (a1,b),(a2,b) for
        b in N(a1)∩N(a2)∩N(v).
        """
        nv = adj[v] & ~(1 << u)
        # pairs in N(u)
        nu = adj[u] & ~(1 << v)
        t = nu
        while t:
            lb = t & -t
            a1 = lb.bit_length() - 1
            t ^= lb
            t2 = nu & ~((1 << (a1 + 1)) - 1)  # a2 > a1
            while t2:
                lb2 = t2 & -t2
                a2 = lb2.bit_length() - 1
                t2 ^= lb2
                if (adj[a1] & adj[a2] & nv).bit_count() >= 2:
                    return False
        # symmetric: pairs in N(v), common neighbors in N(u)
        nu2 = adj[u] & ~(1 << v)
        t = nv
        while t:
            lb = t & -t
            b1 = lb.bit_length() - 1
            t ^= lb
            t2 = nv & ~((1 << (b1 + 1)) - 1)
            while t2:
                lb2 = t2 & -t2
                b2 = lb2.bit_length() - 1
                t2 ^= lb2
                if (adj[b1] & adj[b2] & nu2).bit_count() >= 2:
                    return False
        return True

    def find_k33(adj):
        """Return the 9 edges of some K3,3, or None."""
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                c = ai & adj[j]
                if c.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    ck = c & adj[k]
                    if ck.bit_count() >= 3:
                        cl = []
                        m = ck
                        while m and len(cl) < 3:
                            lsb = m & -m
                            cl.append(lsb.bit_length() - 1)
                            m ^= lsb
                        edges = []
                        for a in (i, j, k):
                            for b in cl:
                                edges.append((a, b))
                        return edges
        return None

    def conflict_count(adj):
        """Number of (3-set A, 3-set B) K3,3 subgraphs."""
        total = 0
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                c = ai & adj[j]
                if c.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    ck = c & adj[k]
                    b = ck.bit_count()
                    if b >= 3:
                        total += b * (b - 1) * (b - 2) // 6
        return total

    def repair(adj):
        """Conflict-directed repair: repeatedly remove the edge in the current
        K3,3 that participates in the most K3,3 conflicts (greedy Max-2-SAT)."""
        while True:
            k33 = find_k33(adj)
            if k33 is None:
                break
            # pick the edge appearing in the most conflicts
            best_e = None
            best_c = -1
            for e in k33:
                u, v = e
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                c = conflict_count(adj)
                if c < best_c:
                    best_c = c
                    best_e = e
                # restore
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            u, v = best_e
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        return adj

    def greedy_fill(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_add_valid(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            changed = True
        return adj

    # ---------- deterministic 5x4 group construction ----------
    def build_group_construction():
        adj = [0] * n
        groups = [[4 * g + r for r in range(4)] for g in range(5)]
        for g in range(5):
            for h in range(g + 1, 5):
                for r in range(4):
                    for c in range(4):
                        if (r + c) % 4 in (0, 1):
                            u = groups[g][r]
                            v = groups[h][c]
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj

    # ---------- 2-edge-swap simulated annealing ----------
    def sa_search(adj_init, budget):
        adj = adj_init[:]
        current_edges = count_edges(adj)
        best_adj = adj[:]
        best_edges = current_edges

        T = 2.0
        cooling = 0.9997
        min_T = 0.01
        steps_since_improve = 0
        max_steps_no_improve = 8000

        while time.time() < budget and T > min_T:
            u, v = random.sample(range(n), 2)
            if adj[u] & (1 << v):
                # remove (u,v), then try to add a random non-edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges -= 1
                u2, v2 = random.sample(range(n), 2)
                if not (adj[u2] & (1 << v2)) and is_add_valid(u2, v2, adj):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    current_edges += 1
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    T *= cooling
                    continue
            else:
                if is_add_valid(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                else:
                    # 2-edge swap: drop a random existing edge, then add (u,v)
                    all_edges = []
                    for i in range(n):
                        m = adj[i]
                        while m:
                            lsb = m & -m
                            j = lsb.bit_length() - 1
                            if i < j:
                                all_edges.append((i, j))
                            m ^= lsb
                    if not all_edges:
                        T *= cooling
                        continue
                    ru, rv = random.choice(all_edges)
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    current_edges -= 1
                    if not (adj[u] & (1 << v)) and is_add_valid(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges += 1
                    else:
                        adj[ru] |= (1 << rv)
                        adj[rv] |= (1 << ru)
                        current_edges += 1
                        T *= cooling
                        continue

            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
                steps_since_improve = 0
            else:
                steps_since_improve += 1

            T *= cooling

            if steps_since_improve > 500 or random.random() < 0.03:
                greedy_fill(adj)
                ne = count_edges(adj)
                if ne > best_edges:
                    best_edges = ne
                    best_adj = adj[:]
                steps_since_improve = 0

            if steps_since_improve > max_steps_no_improve:
                break

        return best_adj, best_edges

    # ---------- main ----------
    best_global_adj = None
    best_global_edges = 0

    adj_init = build_group_construction()
    repair(adj_init)
    greedy_fill(adj_init)
    best_global_adj = adj_init[:]
    best_global_edges = count_edges(adj_init)

    # SA from the structured start
    adj_sa, e_sa = sa_search(adj_init[:], start_time + 60.0)
    if e_sa > best_global_edges:
        best_global_edges = e_sa
        best_global_adj = adj_sa

    # Random restarts
    for r in range(40):
        if time.time() > deadline - 15.0:
            break
        adj_r = [0] * n
        p = 0.3 + 0.04 * (r % 5)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj_r[i] |= (1 << j)
                    adj_r[j] |= (1 << i)
        repair(adj_r)
        greedy_fill(adj_r)
        e_r = count_edges(adj_r)
        if e_r > best_global_edges:
            best_global_edges = e_r
            best_global_adj = adj_r[:]
        adj_sa2, e_sa2 = sa_search(adj_r[:], min(deadline - 15.0, time.time() + 1.5))
        if e_sa2 > best_global_edges:
            best_global_edges = e_sa2
            best_global_adj = adj_sa2

    # Final deterministic polish on the best graph
    greedy_fill(best_global_adj)
    final_edges = count_edges(best_global_adj)
    if final_edges > best_global_edges:
        best_global_edges = final_edges

    return get_edges_from_adj(best_global_adj)
