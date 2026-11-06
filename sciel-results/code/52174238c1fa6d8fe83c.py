import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin

    def compute_alpha(adj_mask, n):
        """Compute maximum independent set size using Branch and Bound with good heuristics."""
        best = [0]
        
        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            if size + candidates.bit_count() <= best[0]:
                return
            
            c = candidates
            max_deg = -1
            chosen = -1
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = (adj_mask[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            
            if chosen == -1:
                return
                
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_candidates, size + 1)
            
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, size)

        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0)
        return best[0]

    def is_c4_free(adj_mask, n):
        """Check if graph is C4-free: no two vertices have 2+ common neighbors."""
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                if common.bit_count() >= 2:
                    return False
        return True

    def polarity_graph(q):
        """
        Construct the polarity graph (C4-free) on q^2 + q + 1 vertices.
        Vertices are (x, y) for x, y in GF(q), plus a special vertex at infinity.
        Edge between (x1,y1) and (x2,y2) iff x1*x2 + y1*y2 = 0 (in GF(q)).
        Special vertex connects to (x,y) iff x^2 + y^2 = 0.
        """
        n = q * q + q + 1
        adj_mask = [0] * n
        
        def idx(x, y):
            return x * q + y  # 0 to q^2-1
        inf_idx = q * q  # special vertex
        
        # Build GF(q) tables for prime q
        if q == 2:
            add = lambda a, b: a ^ b
            mul = lambda a, b: a * b
            inv = {0: 0, 1: 1}
        elif q == 3:
            add = lambda a, b: (a + b) % q
            mul = lambda a, b: (a * b) % q
            inv = {0: 0, 1: 1, 2: 2}
        elif q == 5:
            add = lambda a, b: (a + b) % q
            mul = lambda a, b: (a * b) % q
            inv = {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
        elif q == 7:
            add = lambda a, b: (a + b) % q
            mul = lambda a, b: (a * b) % q
            inv = {0: 0, 1: 1, 2: 4, 3: 5, 4: 2, 5: 3, 6: 6}
        else:
            add = lambda a, b: (a + b) % q
            mul = lambda a, b: (a * b) % q
            inv = {0: 0}
            for i in range(1, q):
                for j in range(1, q):
                    if (i * j) % q == 1:
                        inv[i] = j
                        break
        
        # Connect regular vertices
        for x1 in range(q):
            for y1 in range(q):
                u = idx(x1, y1)
                for x2 in range(q):
                    for y2 in range(q):
                        if x1 == x2 and y1 == y2:
                            continue
                        v = idx(x2, y2)
                        if v <= u:
                            continue
                        # Check x1*x2 + y1*y2 = 0
                        val = (mul(x1, x2) + mul(y1, y2)) % q
                        if val == 0:
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
        
        # Connect infinity vertex to (x,y) where x^2 + y^2 = 0
        for x in range(q):
            for y in range(q):
                u = idx(x, y)
                val = (mul(x, x) + mul(y, y)) % q
                if val == 0:
                    adj_mask[inf_idx] |= (1 << u)
                    adj_mask[u] |= (1 << inf_idx)
        
        return adj_mask, n

    def try_extend(adj_mask, n, target_n, time_limit):
        """
        Try to extend a C4-free graph to target_n vertices by adding vertices
        one at a time, connecting to as many existing vertices as possible.
        """
        if time.time() > deadline:
            return None
        
        for new_v in range(n, target_n):
            if time.time() > deadline:
                return None
            
            # Add new vertex
            adj_mask.append(0)
            # Extend existing vertices' masks (they don't change, just the list)
            
            # Try to connect new_v to existing vertices in random order
            candidates = list(range(new_v))
            random.shuffle(candidates)
            
            for j in candidates:
                if time.time() > deadline:
                    return None
                
                can_add = True
                v_mask = adj_mask[j]
                while v_mask:
                    lsb = v_mask & (-v_mask)
                    v = lsb.bit_length() - 1
                    v_mask ^= lsb
                    if adj_mask[new_v] & adj_mask[v]:
                        can_add = False
                        break
                
                if can_add:
                    adj_mask[new_v] |= (1 << j)
                    adj_mask[j] |= (1 << new_v)
        
        # Verify C4-free
        if not is_c4_free(adj_mask, target_n):
            return None
        
        # Check alpha
        if time.time() > deadline:
            return None
        alpha = compute_alpha(adj_mask, target_n)
        if alpha < 11:
            return adj_mask
        return None

    def local_search(adj_mask, n, time_limit):
        """
        Try to improve the graph by flipping edges to reduce alpha while maintaining C4-free.
        """
        if time.time() > deadline:
            return None
        
        current_alpha = compute_alpha(adj_mask, n)
        if current_alpha < 11:
            return adj_mask
        
        # Try adding random edges that maintain C4-free
        for _ in range(200):
            if time.time() > deadline:
                return None
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or u >= v:
                continue
            
            # Check if adding edge (u,v) maintains C4-free
            # For C4-free: u and v must have at most 1 common neighbor
            common = adj_mask[u] & adj_mask[v]
            if common.bit_count() >= 2:
                continue
            
            # Also check: for each neighbor w of u, N(w) & N(v) must not include another neighbor of u
            # Actually the condition is: after adding (u,v), no pair has 2+ common neighbors.
            # The only new pairs to check are (u, x) for x in N(v) and (v, x) for x in N(u),
            # and (u, v) itself.
            
            # Check (u, v): they now have |N(u) & N(v)| common neighbors, need < 2
            # Already checked above: common.bit_count() < 2, so OK.
            
            # For each x in N(u) \ {v}: check that N(u) & N(x) doesn't gain a 2nd common neighbor via v
            # N(u) gains v. So for x in N(u), x != v: check if v in N(x) and N(u) & N(x) already had 1
            # Actually simpler: just verify full C4-free after addition, but that's expensive.
            # Let's just do a quick check:
            valid = True
            # Check pairs involving u
            for x in range(n):
                if x == u or x == v:
                    continue
                # Check if u and x have 2+ common neighbors after adding (u,v)
                # Common neighbors of u and x: N(u) & N(x), but N(u) now includes v
                # Old common: adj_mask[u] & adj_mask[x]
                # New common: (adj_mask[u] | (1<<v)) & adj_mask[x]
                new_common = (adj_mask[u] | (1 << v)) & adj_mask[x]
                if new_common.bit_count() >= 2:
                    valid = False
                    break
                # Check if v and x have 2+ common neighbors
                # Common neighbors of v and x: N(v) & N(x), but N(v) now includes u
                new_common_vx = (adj_mask[v] | (1 << u)) & adj_mask[x]
                if new_common_vx.bit_count() >= 2:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add the edge
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            # Check alpha
            if time.time() > deadline:
                return None
            new_alpha = compute_alpha(adj_mask, n)
            if new_alpha < 11:
                return adj_mask
            if new_alpha <= current_alpha:
                current_alpha = new_alpha
            # If alpha didn't improve, keep the edge anyway (it might help later)
        
        return None

    # Build known good constructions as starting points
    def get_polarity_edges(q):
        adj_mask, n = polarity_graph(q)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_mask[u] >> v) & 1:
                    edges.append([u, v])
        return n, edges

    # Fallback: 10 disjoint triangles (n=30, alpha=10)
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_n = 30
    best_edges = fallback_edges

    # Try polarity graphs for q = 5, 7 as strong starting points
    for q in [5, 7]:
        if time.time() > deadline - 30:
            break
        adj_mask, n = polarity_graph(q)
        if not is_c4_free(adj_mask, n):
            continue
        alpha = compute_alpha(adj_mask, n)
        if alpha < 11:
            if n > best_n:
                best_n = n
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj_mask[u] >> v) & 1:
                            edges.append([u, v])
                best_edges = edges
            
            # Try to extend beyond n
            for target in range(n + 1, min(n + 12, 44)):
                if time.time() > deadline - 10:
                    break
                for attempt in range(5):
                    if time.time() > deadline - 10:
                        break
                    random.seed(q * 1000 + target * 10 + attempt)
                    am_copy = [m for m in adj_mask]
                    extended = try_extend(am_copy, n, target, deadline - time.time() - 10)
                    if extended:
                        edges = []
                        for u in range(target):
                            for v in range(u + 1, target):
                                if (extended[u] >> v) & 1:
                                    edges.append([u, v])
                        return {'n': target, 'edges': edges}
        
        # Try local search on the polarity graph
        if time.time() > deadline - 15:
            break
        for attempt in range(3):
            if time.time() > deadline - 15:
                break
            random.seed(q * 7777 + attempt)
            am_copy = [m for m in adj_mask]
            improved = local_search(am_copy, n, deadline - time.time() - 15)
            if improved:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (improved[u] >> v) & 1:
                            edges.append([u, v])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                return {'n': n, 'edges': edges}

    # Try random constructions for larger n
    def try_find_graph(n):
        """
        Try to find a C4-free graph on n vertices with alpha < 11.
        Uses a greedy random construction.
        """
        if time.time() > deadline:
            return None
            
        adj_mask = [0] * n
        
        for i in range(n):
            candidates = list(range(i))
            random.shuffle(candidates)
            
            for j in candidates:
                if time.time() > deadline:
                    return None
                
                can_add = True
                v_mask = adj_mask[j]
                while v_mask:
                    lsb = v_mask & (-v_mask)
                    v = lsb.bit_length() - 1
                    v_mask ^= lsb
                    if adj_mask[i] & adj_mask[v]:
                        can_add = False
                        break
                
                if can_add:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        if time.time() > deadline:
            return None
        
        if not is_c4_free(adj_mask, n):
            return None
            
        alpha = compute_alpha(adj_mask, n)
        if alpha < 11:
            return adj_mask
        return None

    for n in range(43, best_n, -1):
        if time.time() > deadline - 5:
            break
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            random.seed(attempt * 1000 + n)
            adj_mask = try_find_graph(n)
            if adj_mask:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (adj_mask[u] >> v) & 1:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}

    return {'n': best_n, 'edges': best_edges}
