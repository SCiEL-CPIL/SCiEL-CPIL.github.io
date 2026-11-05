import random
import time


def construct():
    n = 40
    deadline = time.time() + 135.0  # 15s margin before 150s kill

    def make_adj_from_edges(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            bits = adj[i]
            while bits:
                lsb = bits & -bits
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                bits ^= lsb
        return edges

    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2

    def has_c6_between(u, v, adj):
        """
        Check if adding edge (u,v) creates a C6.
        Need a path u-a-b-c-d-v of length 5 (5 edges), all distinct.
        Equivalent: is there a path of length 4 from N(u)\{v} to N(v)\{u}
        with all intermediate vertices distinct and not u,v?
        
        Using bitsets:
        For each a in N(u), for each b in N(a), check if N(b) intersects N(N(v))
        in a way that gives a valid path.
        
        More precisely: path u-a-b-c-d-v means:
        a in N(u), b in N(a), c in N(b), d in N(c), v in N(d)
        All of {u,a,b,c,d,v} distinct.
        
        Alternative: For each pair (a, d) with a in N(u), d in N(v),
        check if there's a path a-b-c-d of length 3 with b,c distinct from 
        {u,v,a,d} and from each other.
        """
        nu = adj[u] & ~(1 << v)  # neighbors of u except v
        nv = adj[v] & ~(1 << u)  # neighbors of v except u
        
        # For each a in N(u)\{v}, for each d in N(v)\{u},
        # check if there's a path a-b-c-d with b,c not in {u,v,a,d}
        # This is equivalent to: does there exist b in N(a)\{u,a} and c in N(b)\{u,v,a,b}
        # such that d in N(c)\{u,v,a,b,c}?
        
        # Faster: for each a in N(u), compute the set of vertices reachable from a
        # in exactly 3 steps, then check intersection with N(v).
        # Reachable from a in 3 steps: union over b in N(a) of (N(b) \ {a})
        # Then from those, in 1 more step: union over c of N(c)
        
        # Let's do: for a in N(u)\{v}:
        #   for b in N(a)\{u,a}:
        #     for c in N(b)\{u,v,a,b}:
        #       if N(c) & N(v) & ~(mask of forbidden) != 0: return True
        # where forbidden = {u, v, a, b, c}
        
        bits_a = nu
        while bits_a:
            lsb_a = bits_a & -bits_a
            a = lsb_a.bit_length() - 1
            bits_a ^= lsb_a
            
            # b in N(a) \ {u, v, a}
            forbidden_ab = (1 << u) | (1 << v) | (1 << a)
            bits_b = adj[a] & ~forbidden_ab
            while bits_b:
                lsb_b = bits_b & -bits_b
                b = lsb_b.bit_length() - 1
                bits_b ^= lsb_b
                
                # c in N(b) \ {u, v, a, b}
                forbidden_bc = (1 << u) | (1 << v) | (1 << a) | (1 << b)
                bits_c = adj[b] & ~forbidden_bc
                while bits_c:
                    lsb_c = bits_c & -bits_c
                    c = lsb_c.bit_length() - 1
                    bits_c ^= lsb_c
                    
                    # d in N(c) \ {u, v, a, b, c} must be in N(v)
                    forbidden_cd = (1 << u) | (1 << v) | (1 << a) | (1 << b) | (1 << c)
                    if adj[c] & nv & ~forbidden_cd:
                        return True
        return False

    def is_valid_add(u, v, adj):
        """Check if adding edge (u,v) keeps graph C6-free."""
        if u == v:
            return False
        if adj[u] & (1 << v):
            return False
        return not has_c6_between(u, v, adj)

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_build(order, adj):
        """Try to add edges in given order if valid."""
        for u, v in order:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if is_valid_add(u, v, adj):
                    add_edge(adj, u, v)

    def local_search_add(adj):
        """Try to add any valid edge. Returns count added."""
        added = 0
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                if time.time() > deadline:
                    break
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        if is_valid_add(u, v, adj):
                            add_edge(adj, u, v)
                            added += 1
                            changed = True
        return added

    def try_swap_improve(adj):
        """
        Try to find a swap: remove edge (u,v), add edges (u,x) and (v,y) 
        or (u,x) and (w,v), etc., such that net edges increase by 1.
        Actually simpler: remove one edge, then try to add two edges.
        If we can add 2 edges after removing 1, net +1.
        """
        improved = False
        edges = adj_to_edges(adj)
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            
            # Try removing (u,v)
            remove_edge(adj, u, v)
            
            # Now try to add edges involving u or v or others
            # Try adding (u, x) for various x
            best_adds = 0
            best_u_x = None
            best_v_y = None
            
            # Try pairs of additions
            # First try: add (u, x) and (v, y) where x != v, y != u
            candidates_u = []
            for x in range(n):
                if x == u or x == v:
                    continue
                if not (adj[u] & (1 << x)):
                    if is_valid_add(u, x, adj):
                        candidates_u.append(x)
            
            candidates_v = []
            for y in range(n):
                if y == u or y == v:
                    continue
                if not (adj[v] & (1 << y)):
                    if is_valid_add(v, y, adj):
                        candidates_v.append(y)
            
            # Try combinations
            for x in candidates_u[:5]:  # limit to avoid explosion
                if time.time() > deadline:
                    break
                # Temporarily add (u,x)
                add_edge(adj, u, x)
                for y in candidates_v[:5]:
                    if y == x:
                        continue
                    if is_valid_add(v, y, adj):
                        best_adds = 2
                        best_u_x = x
                        best_v_y = y
                        break
                if best_adds >= 2:
                    break
                # Remove (u,x) to try next
                remove_edge(adj, u, x)
            
            if best_adds >= 2:
                # We already added (u, best_u_x) above; add (v, best_v_y)
                if best_u_x is not None and best_v_y is not None:
                    add_edge(adj, v, best_v_y)
                    improved = True
            else:
                # Restore (u,v)
                add_edge(adj, u, v)
            
            if improved:
                break
        
        return improved

    best_edges = 0
    best_adj = None

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Phase 1: Multiple greedy restarts with random orderings
    max_restarts = 15
    for r in range(max_restarts):
        if time.time() > deadline:
            break
        
        adj = [0] * n
        order = all_pairs[:]
        random.shuffle(order)
        
        greedy_build(order, adj)
        local_search_add(adj)
        
        cnt = count_edges(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj[:]
        
        # Phase 2: Local search with swaps
        if time.time() < deadline:
            for _ in range(3):  # a few swap rounds
                if time.time() > deadline:
                    break
                if not try_swap_improve(adj):
                    break
                cnt = count_edges(adj)
                if cnt > best_edges:
                    best_edges = cnt
                    best_adj = adj[:]

    # Final: if we have time, do more aggressive search
    if best_adj is not None and time.time() < deadline:
        adj = best_adj[:]
        # Try systematic edge additions in different orders
        for attempt in range(5):
            if time.time() > deadline:
                break
            order = all_pairs[:]
            random.shuffle(order)
            greedy_build(order, adj)
            local_search_add(adj)
            cnt = count_edges(adj)
            if cnt > best_edges:
                best_edges = cnt
                best_adj = adj[:]

    if best_adj is None:
        # Fallback: empty graph
        return []

    return adj_to_edges(best_adj)


if __name__ == "__main__":
    edges = construct()
    print(len(edges))
