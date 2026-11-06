import time
import random
import sys

def construct():
    """
    Construct a pentagon-free (C5-free) graph with no independent 9-set.
    Maximize n (max 35).
    """

    # Fast C5 checker using bitmasks
    def has_c5(n, adj_mask):
        """Check for C5 using bitmask operations. O(n^4) worst case but fast in practice."""
        for u in range(n):
            u_mask = 1 << u
            v_mask = adj_mask[u]
            while v_mask:
                lsb_v = v_mask & -v_mask
                v = lsb_v.bit_length() - 1
                v_mask -= lsb_v
                
                # w must be in N(v) \ {u}
                w_mask = adj_mask[v] & ~u_mask & ~lsb_v
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    w = lsb_w.bit_length() - 1
                    w_mask -= lsb_w
                    
                    # x must be in N(w) \ {u, v}
                    x_mask = adj_mask[w] & ~u_mask & ~lsb_v & ~lsb_w
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        x = lsb_x.bit_length() - 1
                        x_mask -= lsb_x
                        
                        # y must be in N(x) ∩ N(u) \ {v, w}
                        # adj_mask[x] & adj_mask[u] gives candidates
                        # exclude v, w (and u, x which aren't in their own adj)
                        y_cand = adj_mask[x] & adj_mask[u] & ~lsb_v & ~lsb_w
                        if y_cand:
                            return True
        return False

    # Independent set checker using bitmasks
    def has_is_k(n, adj_mask, k):
        """Check for independent set of size k using bitmask DFS with pruning."""
        all_mask = (1 << n) - 1
        
        def dfs(cand, rem):
            if rem == 0:
                return True
            if cand == 0:
                return False
            if cand.bit_count() < rem:
                return False
            
            # Pick vertex with max degree in cand (best-first for pruning)
            v = -1
            max_deg = -1
            c = cand
            while c:
                lsb = c & -c
                vi = lsb.bit_length() - 1
                deg = (adj_mask[vi] & cand).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = vi
                c -= lsb
            
            if v == -1:
                return False
                
            lsb_v = 1 << v
            
            # Include v: remove v and its neighbors from candidates
            new_cand_incl = cand & ~adj_mask[v] & ~lsb_v
            if dfs(new_cand_incl, rem - 1):
                return True
            
            # Exclude v
            new_cand_excl = cand & ~lsb_v
            if dfs(new_cand_excl, rem):
                return True
            
            return False

        return dfs(all_mask, k)

    def is_valid(n, adj_mask):
        if has_c5(n, adj_mask):
            return False
        if has_is_k(n, adj_mask, 9):
            return False
        return True

    def adj_to_edges(n, adj_mask):
        edges = []
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                mask -= lsb
        return edges

    # Base 32-vertex graph: 8 disjoint K4 blocks
    def get_base_graph():
        n = 32
        adj_mask = [0] * n
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    adj_mask[base + i] |= (1 << (base + j))
                    adj_mask[base + j] |= (1 << (base + i))
        return n, adj_mask

    # Search for a valid graph on n_target vertices
    def search_for_n(n_target, time_limit):
        start_time = time.time()
        deadline = start_time + time_limit
        
        if n_target == 33:
            # Start from base 32-vertex graph + 1 new vertex
            n, adj_mask = get_base_graph()
            adj_mask.append(0)
            # Randomly connect new vertex 32 to some vertices
            for i in range(n):
                if random.random() < 0.3:
                    adj_mask[i] |= (1 << 32)
                    adj_mask[32] |= (1 << i)
            n = 33
        elif n_target == 34:
            n, adj_mask = get_base_graph()
            adj_mask.append(0)
            adj_mask.append(0)
            for i in range(n):
                if random.random() < 0.25:
                    adj_mask[i] |= (1 << 32)
                    adj_mask[32] |= (1 << i)
                if random.random() < 0.25:
                    adj_mask[i] |= (1 << 33)
                    adj_mask[33] |= (1 << i)
            if random.random() < 0.3:
                adj_mask[32] |= (1 << 33)
                adj_mask[33] |= (1 << 32)
            n = 34
        elif n_target == 35:
            n, adj_mask = get_base_graph()
            for _ in range(3):
                adj_mask.append(0)
            for i in range(n):
                for new_v in [32, 33, 34]:
                    if random.random() < 0.2:
                        adj_mask[i] |= (1 << new_v)
                        adj_mask[new_v] |= (1 << i)
            for a in [32, 33]:
                for b in [a+1, 33, 34]:
                    if b <= 34 and a < b:
                        if random.random() < 0.25:
                            adj_mask[a] |= (1 << b)
                            adj_mask[b] |= (1 << a)
            n = 35
        else:
            # Random initialization
            adj_mask = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if random.random() < 0.4:
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
            n = n_target

        # Simulated annealing
        T = 1.0
        cooling = 0.999
        best_adj = adj_mask[:]
        best_cost = float('inf')
        current_adj = adj_mask[:]
        current_cost = None
        
        # Cost: penalize C5 heavily, IS-9 less
        def cost(adj_mask):
            c5_pen = 1000 if has_c5(n, adj_mask) else 0
            is_pen = 100 if has_is_k(n, adj_mask, 9) else 0
            return c5_pen + is_pen

        current_cost = cost(current_adj)
        if current_cost < best_cost:
            best_cost = current_cost
            best_adj = current_adj[:]

        step_count = 0
        while time.time() < deadline - 5:
            # Pick a random edge to toggle
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            mask_u = 1 << u
            mask_v = 1 << v
            
            # Toggle
            if current_adj[u] & mask_v:
                current_adj[u] &= ~mask_v
                current_adj[v] &= ~mask_u
            else:
                current_adj[u] |= mask_v
                current_adj[v] |= mask_u
            
            new_cost = cost(current_adj)
            step_count += 1
            
            if new_cost < current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = current_adj[:]
                    if best_cost == 0:
                        break
            else:
                delta = new_cost - current_cost
                if delta <= 0 or random.random() < pow(2.71828, -delta / max(T, 0.01)):
                    current_cost = new_cost
                else:
                    # Revert
                    if current_adj[u] & mask_v:
                        current_adj[u] &= ~mask_v
                        current_adj[v] &= ~mask_u
                    else:
                        current_adj[u] |= mask_v
                        current_adj[v] |= mask_u
            
            T *= cooling
            if T < 0.001:
                T = 0.001

        return best_adj

    # Try n=35, 34, 33 with time allocation
    total_time = 115.0
    time_alloc = {35: 30.0, 34: 40.0, 33: 45.0}

    for n_target in [35, 34, 33]:
        if time.time() > start_time_total if 'start_time_total' in dir() else False:
            pass
        t_start = time.time()
        adj_mask = search_for_n(n_target, time_alloc[n_target])
        if adj_mask is not None:
            if is_valid(n_target, adj_mask):
                edges = adj_to_edges(n_target, adj_mask)
                return {'n': n_target, 'edges': edges}

    # Fallback to 32
    n, adj_mask = get_base_graph()
    edges = adj_to_edges(n, adj_mask)
    return {'n': n, 'edges': edges}
