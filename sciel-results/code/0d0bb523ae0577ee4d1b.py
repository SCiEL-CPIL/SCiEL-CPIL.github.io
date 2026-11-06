import time
import random

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.
    Uses fast C6 detection with precomputed 2-step reachability,
    greedy construction with multiple strategies, and local search.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s limit

    def check_c6(u, v, adj, adj2):
        """
        Check if adding edge (u,v) creates a C6.
        A C6 containing (u,v) has form u-a-x-y-b-v where all vertices distinct.
        For each a in N(u), b in N(v), a != b, a != v, b != u:
          Check if there exists x in N(a) \ {u,v,b} and y in N(b) \ {u,v,a,x}
          such that (x,y) is an edge.
        
        Equivalent: for a in N(u), b in N(v), a!=b, check if
          (N(a) \ {u,v,b}) intersects (N(b) \ {u,v,a}) via an edge.
        But we need x in N(a) and y in N(b) with (x,y) edge and x!=y, x not in {u,v,b}, y not in {u,v,a,x}.
        
        Faster approach: for each a in N(u), b in N(v) with a!=b:
          candidates_x = N(a) & ~{u,v,b}
          candidates_y = N(b) & ~{u,v,a}
          Check if any edge (x,y) with x in candidates_x, y in candidates_y, x != y.
        """
        nu = adj[u]
        nv = adj[v]
        
        # For each a in N(u)
        a_mask = nu
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            if a == v:
                continue
            
            # For each b in N(v), b != a
            b_mask = nv & ~(1 << a)
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                if b == u:
                    continue
                
                # x in N(a) excluding u, v, b
                x_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << b)
                if not x_mask:
                    continue
                
                # y in N(b) excluding u, v, a
                y_mask = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if not y_mask:
                    continue
                
                # Check if any edge (x,y) with x in x_mask, y in y_mask, x != y
                # For each x in x_mask, check if adj[x] & y_mask & ~(1<<x) is non-zero
                xm = x_mask
                while xm:
                    x_bit = xm & (-xm)
                    x = x_bit.bit_length() - 1
                    xm ^= x_bit
                    if adj[x] & (y_mask & ~(1 << x)):
                        return True
        
        return False

    def check_c6_fast(u, v, adj, adj2):
        """
        Alternative C6 check using 2-step neighborhoods.
        C6: u-a-x-y-b-v
        For a in N(u), b in N(v), a!=b, b!=u, a!=v:
          Need x in N(a)∩N(y) for some y in N(b), with proper exclusions.
        
        Actually let's use: for each a in N(u), b in N(v), a!=b:
          The middle path x-y must have x in N(a)\{u,v,b} and y in N(b)\{u,v,a}
          and (x,y) is an edge.
        
        Precompute: for each pair (a,b), store whether there's an edge between
        N(a)\{u,v,b} and N(b)\{u,v,a}. But u,v change, so this is hard to precompute.
        
        Let's stick with the direct check but optimize the inner loop.
        """
        nu = adj[u]
        nv = adj[v]
        
        a_mask = nu & ~(1 << v)  # a in N(u), a != v
        while a_mask:
            a_bit = a_mask & (-a_mask)
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit
            
            b_mask = nv & ~(1 << u) & ~(1 << a)  # b in N(v), b != u, b != a
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                # x in N(a) \ {u,v,b}, y in N(b) \ {u,v,a}, need edge (x,y), x!=y
                x_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << b)
                if not x_mask:
                    continue
                y_mask = adj[b] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if not y_mask:
                    continue
                
                # Check edges from x_mask to y_mask
                xm = x_mask
                while xm:
                    x_bit = xm & (-xm)
                    x = x_bit.bit_length() - 1
                    xm ^= x_bit
                    if adj[x] & (y_mask & ~(1 << x)):
                        return True
        
        return False

    def greedy_random(seed):
        """Greedy with random edge ordering."""
        rng = random.Random(seed)
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        rng.shuffle(all_edges)
        
        adj = [0] * N
        edge_set = set()
        
        for u, v in all_edges:
            if not check_c6_fast(u, v, adj, None):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        return edge_set, adj

    def greedy_pref_high_degree(seed):
        """Greedy that prefers edges between high-degree vertices."""
        rng = random.Random(seed)
        adj = [0] * N
        edge_set = set()
        
        # Start with some random edges
        for _ in range(20):
            u, v = rng.randint(0, N-1), rng.randint(0, N-1)
            if u != v:
                u, v = min(u,v), max(u,v)
                if (u,v) not in edge_set and not check_c6_fast(u, v, adj, None):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u,v))
        
        # Then greedily add edges, preferring those connecting high-degree vertices
        remaining = [(i, j) for i in range(N) for j in range(i+1, N) if (i,j) not in edge_set]
        rng.shuffle(remaining)
        
        for u, v in remaining:
            if not check_c6_fast(u, v, adj, None):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        return edge_set, adj

    def local_search(edge_set, adj, max_iters=5000, seed=0):
        """Local search: try to add edges, and do 1-swap moves."""
        rng = random.Random(seed)
        current_edges = set(edge_set)
        current_adj = list(adj)
        current_count = len(current_edges)
        no_improve = 0
        max_no_improve = 200
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            
            improved = False
            
            # Try to add a random non-edge
            for _ in range(30):
                u, v = rng.randint(0, N-1), rng.randint(0, N-1)
                if u == v:
                    continue
                u, v = min(u,v), max(u,v)
                if (u, v) in current_edges:
                    continue
                if not check_c6_fast(u, v, current_adj, None):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    current_count += 1
                    improved = True
                    no_improve = 0
                    break
            
            if improved:
                continue
            
            no_improve += 1
            if no_improve > max_no_improve:
                break
            
            # Try 1-swap: remove edge e1, add edge e2
            if current_count == 0:
                break
            
            # Try to find a beneficial swap
            for _ in range(50):
                e_remove = rng.choice(list(current_edges))
                u1, v1 = e_remove
                u2, v2 = rng.randint(0, N-1), rng.randint(0, N-1)
                if u2 == v2:
                    continue
                u2, v2 = min(u2,v2), max(u2,v2)
                if (u2, v2) == (u1, v1) or (u2, v2) in current_edges:
                    continue
                
                # Temporarily remove e1
                current_adj[u1] &= ~(1 << v1)
                current_adj[v1] &= ~(1 << u1)
                current_edges.discard(e_remove)
                
                if not check_c6_fast(u2, v2, current_adj, None):
                    # Add e2
                    current_adj[u2] |= (1 << v2)
                    current_adj[v2] |= (1 << u2)
                    current_edges.add((u2, v2))
                    
                    # Try to add another edge to compensate
                    added_extra = False
                    for _ in range(15):
                        u3, v3 = rng.randint(0, N-1), rng.randint(0, N-1)
                        if u3 == v3:
                            continue
                        u3, v3 = min(u3,v3), max(u3,v3)
                        if (u3, v3) in current_edges:
                            continue
                        if not check_c6_fast(u3, v3, current_adj, None):
                            current_adj[u3] |= (1 << v3)
                            current_adj[v3] |= (1 << u3)
                            current_edges.add((u3, v3))
                            current_count += 1
                            improved = True
                            added_extra = True
                            no_improve = 0
                            break
                    
                    if not added_extra:
                        # Revert: remove e2, re-add e1
                        current_adj[u2] &= ~(1 << v2)
                        current_adj[v2] &= ~(1 << u2)
                        current_edges.discard((u2, v2))
                        current_adj[u1] |= (1 << v1)
                        current_adj[v1] |= (1 << u1)
                        current_edges.add(e_remove)
                else:
                    # Revert
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
                    current_edges.add(e_remove)
                
                if improved:
                    break
        
        return current_edges, current_adj, current_count

    def structured_init():
        """Initialize with blocks and inter-block edges."""
        adj = [0] * N
        edge_set = set()
        
        # 8 groups of 5, connect within groups as K5
        for g in range(8):
            group = list(range(g*5, g*5+5))
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = group[i], group[j]
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
        
        # Add inter-group edges
        rng = random.Random(42)
        for _ in range(500):
            g1, g2 = rng.sample(range(8), 2)
            u = rng.randint(g1*5, g1*5+4)
            v = rng.randint(g2*5, g2*5+4)
            u, v = min(u,v), max(u,v)
            if (u, v) not in edge_set and not check_c6_fast(u, v, adj, None):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        
        return edge_set, adj

    # Phase 1: Multiple greedy restarts
    best_edges = []
    best_count = 0
    best_adj = [0] * N
    
    num_greedy = 80
    for seed in range(num_greedy):
        if time.time() >= deadline:
            break
        edge_set, adj = greedy_random(seed)
        count = len(edge_set)
        if count > best_count:
            best_count = count
            best_edges = list(edge_set)
            best_adj = list(adj)
    
    # Phase 2: Greedy with high-degree preference
    for seed in range(20):
        if time.time() >= deadline:
            break
        edge_set, adj = greedy_pref_high_degree(seed)
        count = len(edge_set)
        if count > best_count:
            best_count = count
            best_edges = list(edge_set)
            best_adj = list(adj)
    
    # Phase 3: Structured init + local search
    if time.time() < deadline:
        edge_set, adj = structured_init()
        count = len(edge_set)
        if count > best_count:
            best_count = count
            best_edges = list(edge_set)
            best_adj = list(adj)
        
        if time.time() < deadline:
            new_edges, new_adj, new_count = local_search(edge_set, adj, max_iters=3000, seed=12345)
            if new_count > best_count:
                best_count = new_count
                best_edges = list(new_edges)
                best_adj = list(new_adj)
    
    # Phase 4: Local search on best
    if best_edges and time.time() < deadline:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        new_edges, new_adj, new_count = local_search(best_edges, adj, max_iters=5000, seed=99999)
        if new_count > best_count:
            best_count = new_count
            best_edges = list(new_edges)
            best_adj = list(new_adj)
    
    # Phase 5: More local search with different seeds
    if best_edges and time.time() < deadline:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for ls_seed in range(10):
            if time.time() >= deadline:
                break
            new_edges, new_adj, new_count = local_search(best_edges, adj, max_iters=3000, seed=1000+ls_seed)
            if new_count > best_count:
                best_count = new_count
                best_edges = list(new_edges)
                best_adj = list(new_adj)
    
    # Final: try to add any remaining edges
    if best_edges and time.time() < deadline:
        adj = [0] * N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u in range(N):
            for v in range(u+1, N):
                if (u,v) not in best_edges and not check_c6_fast(u, v, adj, None):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    best_edges.append((u, v))
                    best_count += 1
    
    return [[u, v] for u, v in best_edges]
