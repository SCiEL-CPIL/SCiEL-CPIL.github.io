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
        If so, adding edge (u,v) would create a C6."""
        if u == v:
            return False
        target = 1 << v
        start_mask = 1 << u
        # We need a path u = x0 - x1 - x2 - x3 - x4 - x5 = v with all distinct.
        # DFS from u, depth 0, visited = {u}. At depth 5, check if v reached.
        # Prune: if at any depth d < 5 we reach v, that's a shorter path (not C6 by itself,
        # but we need exactly 5). Actually we need a path of length 5, so v must be at depth 5.
        # If we reach v earlier, we can still continue (v not in visited until depth 5).
        # But if v is adjacent at depth 1,2,3,4 that's a shorter path, not the one we want.
        # We want SOME simple path of length 5. So we should NOT stop when reaching v early;
        # we just need to find one path of length 5.
        # However, if v is in visited before depth 5, we can't use it again.
        # So: visited starts with {u}. We explore. At depth 5, if current == v, return True.
        # We must not add v to visited until we decide to use it at depth 5.
        # Simpler: DFS where we track visited set. At each step, for each neighbor not in visited,
        # recurse. At depth 5, check if neighbor == v.
        
        # Use iterative DFS with explicit stack to avoid recursion limit and for speed.
        # Stack entry: (current_vertex, depth, visited_mask)
        # visited_mask includes u and all intermediate vertices.
        
        stack = [(u, 0, start_mask)]
        while stack:
            cur, depth, vis = stack.pop()
            if depth == 5:
                if cur == v:
                    return True
                continue
            # neighbors of cur not in vis
            nbrs = adj[cur] & ~vis
            # If depth == 4, we only care if v is a neighbor (and v not in vis)
            if depth == 4:
                if nbrs & target:
                    return True
                # else no point continuing from this branch at depth 4 unless we go deeper? 
                # No, depth 4 -> depth 5 must be v. If v not neighbor, this branch fails.
                continue
            # depth 0..3: push all valid neighbors
            m = nbrs
            while m:
                lsb = m & (-m)
                nb = lsb.bit_length() - 1
                m ^= lsb
                # If nb == v and depth < 4, we can't use v yet (need to reach at depth 5).
                # So skip v unless depth == 4 (handled above).
                if nb == v:
                    continue
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
    # 2-opt: remove edge e1, try to add two edges e2, e3 (or one) to increase count.
    # Simpler: try single edge adds (already done). Then try swaps: remove one, add one.
    # But that keeps count same. We want to increase count.
    # Better: try to add an edge. If blocked, remove a random edge, try to add two edges.
    
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
            # Try to add two edges
            added = []
            for u in range(N):
                for v in range(u + 1, N):
                    if not (adj[u] & (1 << v)):
                        if can_add(u, v):
                            add_edge(u, v)
                            added.append((u, v))
                            if len(added) >= 2:
                                break
                if len(added) >= 2:
                    break
            if len(added) >= 2:
                # Net gain of 1 edge
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

    save_if_better()
    return best_edges
