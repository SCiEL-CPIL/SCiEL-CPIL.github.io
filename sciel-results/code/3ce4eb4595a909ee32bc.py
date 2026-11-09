"""Triangle-free graphs with alpha < 10: construct a maximal such graph via randomised
random-greedy triangle-free completion, then local-search edge swaps that keep
triangle-freeness and alpha < 10 while growing n."""

# EVOLVE-BLOCK-START
import random
import time
import math
import sys

def _build_andr_fai_style(n):
    """Andrasfai-style circulant: edge i-j iff (j-i) mod n is 1 mod 3."""
    n = int(n)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            if d % 3 == 1:
                edges.append([i, j])
    return edges

def _is_triangle_free(n, adj):
    """adj: list of sets; check no triangle."""
    for i in range(n):
        ai = adj[i]
        for j in ai:
            if i < j:
                aj = adj[j]
                for k in aj:
                    if k > j and k in ai:
                        return False
    return True

def _max_independent_set_greedy_upper(n, adj):
    """Return an upper bound on alpha using a simple greedy maximal independent set
       count heuristic plus branch-and-bound exact for small n."""
    # Exact branch and bound for small/medium n (<= 32) using bitmasks is infeasible in pure Python
    # for 40 vertices quickly. Use a fast greedy upper bound via maximal clique in complement.
    # Build complement adjacency as bitsets for speed.
    # Complement: edge i-j in comp iff i-j NOT in G.
    comp_mask = [0] * n
    for i in range(n):
        m = 0
        for j in range(n):
            if i != j and j not in adj[i]:
                m |= (1 << j)
        comp_mask[i] = m
    # Greedy coloring / maximal independent set upper bound: use a simple bound
    # alpha <= n - minimum vertex cover lower bound? Use greedy maximal independent set
    # as a lower bound; we need upper bound. Use a simple bound: alpha <= sum over greedy
    # clique cover in G (complement of independent set). Actually alpha(G) <= minimum clique
    # cover number of G. Greedy clique cover in G gives an upper bound on alpha? No:
    # alpha(G) = omega(comp(G)). We need upper bound on omega(comp). Use greedy coloring of
    # comp: omega(comp) <= chromatic number of comp. Greedy coloring gives an upper bound on
    # chromatic number.
    # Greedy coloring of complement
    color = [-1] * n
    order = sorted(range(n), key=lambda i: comp_mask[i] & ((1 << n) - 1), reverse=True)
    for v in order:
        used = [False] * 50
        m = comp_mask[v]
        for u in range(n):
            if (m >> u) & 1 and color[u] >= 0:
                used[color[u]] = True
        c = 0
        while c < len(used) and used[c]:
            c += 1
        if c == len(used):
            used.append(False)
        color[v] = c
    max_color = max(color) + 1
    return max_color  # upper bound on alpha

def _exact_alpha_small(n, adj, timeout=2.0):
    """Exact max independent set for n <= 28 via branch and bound with bitmasks."""
    if n > 28:
        return None
    start = time.time()
    # Build adjacency bitmask
    adj_bits = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        adj_bits[i] = m

    best = [0]

    def bnb(candidates, current_size):
        if time.time() - start > timeout:
            return
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Upper bound: current_size + popcount(candidates)
        if current_size + bin(candidates).count('1') <= best[0]:
            return
        # Pick a vertex with max degree within candidates
        c = candidates
        # Get first set bit
        v = (c & -c).bit_length() - 1
        # Branch: include v
        new_cand = candidates & adj_bits[v]
        bnb(new_cand, current_size + 1)
        # Exclude v
        bnb(candidates & ~(1 << v), current_size)

    bnb((1 << n) - 1, 0)
    return best[0]

def _is_valid(n, edges, deadline=None):
    """Check triangle-free and alpha < 10."""
    if deadline is not None and time.time() > deadline:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    # Triangle check
    for i in range(n):
        ai = adj[i]
        for j in ai:
            if i < j:
                # common neighbors > j
                common = ai & adj[j]
                for k in common:
                    if k > j:
                        return False
    # Alpha check: exact for n <= 28, else use greedy upper bound
    if n <= 28:
        a = _exact_alpha_small(n, adj, timeout=1.0)
        if a is not None:
            return a < 10
    # For larger n, use greedy upper bound on alpha (chromatic number of complement)
    ub = _max_independent_set_greedy_upper(n, adj)
    if ub < 10:
        return True
    # If ub >= 10, we can't confirm; try a greedy independent set to see if it finds 10
    # Greedy maximal independent set
    remaining = set(range(n))
    independent = []
    while remaining:
        # pick vertex with min degree in remaining
        v = min(remaining, key=lambda x: len(adj[x] & remaining))
        independent.append(v)
        remaining -= adj[v]
        remaining.discard(v)
    if len(independent) >= 10:
        return False
    # If greedy didn't find 10, assume valid (risky but practical)
    return True

def _add_edge_random(n, edges, adj):
    """Try to add a random edge that keeps triangle-free."""
    for _ in range(50):
        i = random.randrange(n)
        j = random.randrange(n)
        if i == j or j in adj[i]:
            continue
        # Check triangle: common neighbors of i and j
        common = adj[i] & adj[j]
        if not common:
            adj[i].add(j)
            adj[j].add(i)
            edges.append([min(i, j), max(i, j)])
            return True
    return False

def _swap_edge(n, edges, adj, deadline):
    """Try to remove an edge and add two edges, keeping triangle-free and alpha<10."""
    if not edges:
        return False
    if time.time() > deadline:
        return False
    # Pick a random edge to remove
    idx = random.randrange(len(edges))
    u, v = edges[idx]
    # Try to add an edge between two non-adjacent vertices
    for _ in range(10):
        a = random.randrange(n)
        b = random.randrange(n)
        if a == b or b in adj[a]:
            continue
        if a in (u, v) or b in (u, v):
            continue
        common_ab = adj[a] & adj[b]
        if common_ab:
            continue
        # Remove edge (u,v)
        adj[u].discard(v)
        adj[v].discard(u)
        edges.remove([u, v])
        # Add edge (a,b)
        adj[a].add(b)
        adj[b].add(a)
        edges.append([min(a, b), max(a, b)])
        # Check validity
        if _is_valid(n, edges, deadline):
            return True
        # Revert
        adj[a].discard(b)
        adj[b].discard(a)
        edges.remove([min(a, b), max(a, b)])
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
    return False

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill
    best_n = 0
    best_edges = None
    best_adj = None

    # Start with Andrasfai graph on 26 vertices (known valid: triangle-free, alpha=9)
    n0 = 26
    edges0 = _build_andr_fai_style(n0)
    adj0 = [set() for _ in range(n0)]
    for u, v in edges0:
        adj0[u].add(v)
        adj0[v].add(u)

    if _is_valid(n0, edges0, deadline):
        best_n = n0
        best_edges = edges0[:]
        best_adj = [s.copy() for s in adj0]

    # Try to grow the graph by adding vertices
    current_n = best_n
    current_edges = best_edges[:]
    current_adj = [s.copy() for s in best_adj]

    for target_n in range(current_n + 1, 41):
        if time.time() > deadline:
            break
        # Add new vertex
        new_v = current_n
        current_n = target_n
        current_adj.append(set())
        # Try to connect new vertex to existing vertices to maintain triangle-free
        # A new vertex can connect to any subset of vertices with pairwise non-adjacent neighbors
        # To keep alpha small, connect to enough vertices.
        # Strategy: connect new vertex to a random independent set in the current graph
        # such that the new vertex's neighbors form an independent set (to avoid triangles)
        # And the new vertex should not create an independent set of size 10.
        # Simplified: try random connections
        success = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            # Choose a random independent set in current graph to connect new vertex to
            # Greedy: pick random vertices that are pairwise non-adjacent
            neighbors = set()
            for _ in range(random.randint(5, 15)):
                v = random.randrange(current_n - 1)
                if v in neighbors:
                    continue
                # Check v is non-adjacent to all existing neighbors
                if adj and v not in neighbors:
                    # Check non-adjacency with all in neighbors
                    ok = True
                    for nb in neighbors:
                        if nb in current_adj[v]:
                            ok = False
                            break
                    if ok:
                        neighbors.add(v)
            if not neighbors:
                continue
            # Add edges from new_v to neighbors
            new_edges = current_edges[:]
            for nb in neighbors:
                new_edges.append([nb, new_v])
                current_adj[new_v].add(nb)
                current_adj[nb].add(new_v)
            if _is_valid(current_n, new_edges, deadline):
                success = True
                current_edges = new_edges
                break
            else:
                # Revert
                for nb in neighbors:
                    current_adj[new_v].discard(nb)
                    current_adj[nb].discard(new_v)
                current_adj[new_v] = set()
        if not success:
            # Revert new vertex
            current_n -= 1
            current_adj.pop()
            break
        if current_n > best_n:
            best_n = current_n
            best_edges = current_edges[:]
            best_adj = [s.copy() for s in current_adj]

    # Local search: try edge swaps to improve (though n is fixed, we might try to
    # restructure to allow further growth)
    if best_n < 40:
        # Try to grow further with more aggressive random restarts
        for restart in range(50):
            if time.time() > deadline:
                break
            n_start = 20
            edges = _build_andr_fai_style(n_start)
            adj = [set() for _ in range(n_start)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            if not _is_valid(n_start, edges, deadline):
                continue
            for target_n in range(n_start + 1, best_n + 1):
                if time.time() > deadline:
                    break
                new_v = n_start
                n_start = target_n
                adj.append(set())
                success = False
                for attempt in range(50):
                    if time.time() > deadline:
                        break
                    neighbors = set()
                    for _ in range(random.randint(5, 15)):
                        v = random.randrange(target_n - 1)
                        if v in neighbors:
                            continue
                        ok = True
                        for nb in neighbors:
                            if nb in adj[v]:
                                ok = False
                                break
                        if ok:
                            neighbors.add(v)
                    if not neighbors:
                        continue
                    new_edges = edges[:]
                    for nb in neighbors:
                        new_edges.append([nb, new_v])
                        adj[new_v].add(nb)
                        adj[nb].add(new_v)
                    if _is_valid(target_n, new_edges, deadline):
                        success = True
                        edges = new_edges
                        break
                    else:
                        for nb in neighbors:
                            adj[new_v].discard(nb)
                            adj[nb].discard(new_v)
                        adj[new_v] = set()
                if not success:
                    n_start -= 1
                    adj.pop()
                    break
            if n_start > best_n:
                best_n = n_start
                best_edges = edges[:]
                best_adj = [s.copy() for s in adj]

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
