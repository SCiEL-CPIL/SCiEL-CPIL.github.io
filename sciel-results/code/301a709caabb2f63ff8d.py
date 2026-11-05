import time
import random

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s kill

    N = 40
    FULL = (1 << N) - 1

    # adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * N

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count():
        return sum(bin(adj[i]).count('1') for i in range(N)) // 2

    def has_c6_path(u, v):
        """Return True if there exists a simple path of length exactly 5 from u to v.
        Uses bitmask-based BFS/DFS for speed."""
        if u == v:
            return False
        target = 1 << v
        # We need path u=x0-x1-x2-x3-x4-x5=v, all distinct.
        # DFS: at depth d, we're at vertex cur with visited mask vis.
        # At depth 4, check if v is a neighbor not in vis.
        # At depths 0-3, recurse to neighbors not in vis (and not v, since v only at depth 5).
        
        # Use iterative stack: (cur, depth, vis_mask)
        stack = [(u, 0, 1 << u)]
        while stack:
            cur, depth, vis = stack.pop()
            if depth == 4:
                # Check if v is reachable from cur without going through vis
                if (adj[cur] & ~vis) & target:
                    return True
                continue
            # depth 0..3: expand to neighbors
            nbrs = adj[cur] & ~vis & ~target  # exclude v (only reachable at depth 5)
            m = nbrs
            while m:
                lsb = m & (-m)
                nb = lsb.bit_length() - 1
                m ^= lsb
                stack.append((nb, depth + 1, vis | lsb))
        return False

    def can_add(u, v):
        if adj[u] & (1 << v):
            return False
        return not has_c6_path(u, v)

    # Initialize: 8 disjoint K5s
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                add_edge(base + a, base + b)

    best_edges = []
    best_count = edge_count()

    def get_edges():
        es = []
        for u in range(N):
            m = adj[u]
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                if u < v:
                    es.append([u, v])
                m ^= lsb
        return es

    def save_if_better():
        nonlocal best_edges, best_count
        c = edge_count()
        if c > best_count:
            best_count = c
            best_edges = get_edges()

    # Phase 1: Greedy add
    def greedy_add():
        improved = True
        while improved:
            improved = False
            for u in range(N):
                for v in range(u + 1, N):
                    if not (adj[u] & (1 << v)):
                        if can_add(u, v):
                            add_edge(u, v)
                            improved = True
            save_if_better()

    greedy_add()

    # Phase 2: Local search with swaps and kicks
    
    def try_add_any():
        for u in range(N):
            for v in range(u + 1, N):
                if not (adj[u] & (1 << v)):
                    if can_add(u, v):
                        add_edge(u, v)
                        return True
        return False

    def kick(k):
        es = get_edges()
        if not es:
            return
        removed = random.sample(es, min(k, len(es)))
        for e in removed:
            remove_edge(e[0], e[1])

    # Main loop
    while time.time() < deadline:
        # Try to add edges
        if try_add_any():
            save_if_better()
            continue
        
        # Stuck. Try swaps: remove one edge, see if we can add two.
        es = get_edges()
        if not es:
            break
        
        # Try a few random swaps
        found_swap = False
        for _ in range(50):
            if time.time() > deadline:
                break
            e1 = random.choice(es)
            u1, v1 = e1
            # Remove e1
            remove_edge(u1, v1)
            # Try to add as many edges as possible (want at least 2 for net gain)
            added = []
            for u in range(N):
                for v in range(u + 1, N):
                    if not (adj[u] & (1 << v)):
                        if can_add(u, v):
                            add_edge(u, v)
                            added.append((u, v))
            if len(added) >= 2:
                # Net gain of at least 1 edge
                save_if_better()
                found_swap = True
                break
            else:
                # Revert
                for u, v in added:
                    remove_edge(u, v)
                add_edge(u1, v1)
        
        if found_swap:
            continue
        
        # Kick
        if time.time() < deadline - 5:
            kick(random.randint(2, 5))
        
        # Try to add again after kick
        try_add_any()
        save_if_better()

    # Final polish: try to add any remaining edges
    try_add_any()
    save_if_better()
    
    return best_edges
