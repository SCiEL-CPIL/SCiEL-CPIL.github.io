"""Spectral (Paley) construction + fast exact local search for K2,3-free graph on 20 vertices."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s limit

    n = 20
    FULL = (1 << n) - 1

    def popcount(x):
        return bin(x).count('1')

    def can_add(adj, u, v):
        if (adj[u] >> v) & 1:
            return False
        # Pair (u,v): common neighbors must be <= 2
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # Pairs (u, w) for w in N(v): after adding, common = (N(u)&N(w)) | {v}
        #   need popcount(N(u) & N(w)) <= 1
        m = adj[v]
        while m:
            lsb = m & (-m)
            w = lsb.bit_length() - 1
            m ^= lsb
            if w == u:
                continue
            if popcount(adj[u] & adj[w]) > 1:
                return False
        # Pairs (v, w) for w in N(u): need popcount(N(v) & N(w)) <= 1
        m = adj[u]
        while m:
            lsb = m & (-m)
            w = lsb.bit_length() - 1
            m ^= lsb
            if w == v:
                continue
            if popcount(adj[v] & adj[w]) > 1:
                return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2

    def is_valid(adj):
        for i in range(n):
            for j in range(i + 1, n):
                if popcount(adj[i] & adj[j]) > 2:
                    return False
        return True

    def build_paley_seed():
        """Paley graph on vertices 0..18 (mod 19), plus vertex 19 connected to a set."""
        p = 19
        adj = [0] * n
        for i in range(p):
            for j in range(i + 1, p):
                d = (j - i) % p
                # quadratic residue test mod 19
                if pow(d, (p - 1) // 2, p) == 1:
                    add_edge(adj, i, j)
        # Vertex 19 (index 18 is last Paley vertex). Connect 19 to a chosen subset.
        # Connect to vertices that are "central"; we'll fix via search, start with a spread.
        for j in range(0, p, 2):
            add_edge(adj, 19, j)
        return adj

    def greedy_saturation(adj):
        """Add every edge that can be added; return new edge count."""
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not ((adj[i] >> j) & 1):
                        if can_add(adj, i, j):
                            add_edge(adj, i, j)
                            changed = True
        return edge_count(adj)

    # ---- Build initial seed ----
    adj = build_paley_seed()
    # Fix any violations by removing edges (Paley has common-neighbor 3 or 4)
    # Remove edges until valid, preferring to keep dense.
    guard = 0
    while not is_valid(adj) and guard < 2000:
        guard += 1
        # find a violating pair and remove an edge between them
        removed = False
        for i in range(n):
            for j in range(i + 1, n):
                if popcount(adj[i] & adj[j]) > 2:
                    # remove edge (i,j) if present, else remove an edge incident to common neighbors
                    if (adj[i] >> j) & 1:
                        remove_edge(adj, i, j)
                        removed = True
                        break
                    # remove an edge to a common neighbor to reduce overlap
                    common = adj[i] & adj[j]
                    lsb = common & (-common)
                    w = lsb.bit_length() - 1
                    remove_edge(adj, i, w)
                    removed = True
                    break
            if removed:
                break
        if not removed:
            break

    current_edges = edge_count(adj)
    best_adj = adj[:]
    best_edges = current_edges
    greedy_saturation(adj)
    current_edges = edge_count(adj)
    if current_edges > best_edges:
        best_edges = current_edges
        best_adj = adj[:]

    # ---- Local search: add + swap ----
    T = 1.0
    cooling = 0.9995

    while time.time() < deadline:
        # Try adding a random missing edge
        if random.random() < 0.6:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if not ((adj[u] >> v) & 1):
                if can_add(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
        else:
            # Swap: remove a random edge, try to add another
            # Collect existing edges
            existing = []
            for i in range(n):
                m = adj[i]
                while m:
                    lsb = m & (-m)
                    j = lsb.bit_length() - 1
                    m ^= lsb
                    if j > i:
                        existing.append((i, j))
            if not existing:
                continue
            rem_u, rem_v = random.choice(existing)
            remove_edge(adj, rem_u, rem_v)
            current_edges -= 1
            # Try to add a new edge
            added = False
            for _ in range(15):
                au = random.randint(0, n - 1)
                av = random.randint(0, n - 1)
                if au == av:
                    continue
                if (au, av) == (rem_u, rem_v) or (au, av) == (rem_v, rem_u):
                    continue
                if (adj[au] >> av) & 1:
                    continue
                if can_add(adj, au, av):
                    add_edge(adj, au, av)
                    current_edges += 1
                    added = True
                    break
            if not added:
                # revert
                add_edge(adj, rem_u, rem_v)
                current_edges += 1
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]

        T *= cooling
        if T < 0.05:
            T = 0.05

    # Final greedy saturation on best
    work = best_adj[:]
    greedy_saturation(work)
    final_edges = edge_count(work)
    if final_edges > best_edges:
        best_adj = work
        best_edges = final_edges

    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    return edges
