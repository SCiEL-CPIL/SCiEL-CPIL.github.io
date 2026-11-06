import time
import random
import math

def construct():
    deadline = time.time() + 115  # safety margin for 120s limit
    rng = random.Random(12345)

    def has_k4_fast(n, adj):
        """Check for K4 using bitsets. adj[i] is a bitmask of neighbors of i."""
        for a in range(n):
            na = adj[a]
            # For each pair (b,c) in na with b<c and edge(b,c), check if they share a common neighbor d > c
            b_mask = na
            while b_mask:
                lb = b_mask & -b_mask
                b = lb.bit_length() - 1
                b_mask ^= lb
                if b <= a:
                    continue
                nc = adj[b] & na  # common neighbors of a and b, restricted to na
                # nc must contain b? No, nc is neighbors of b that are also neighbors of a.
                # We need c in nc, c > b, and edge(b,c) exists (which it does since c in adj[b]).
                # Then check if any d in (nc & adj[c]) with d > c exists.
                c_mask = nc
                while c_mask:
                    lc = c_mask & -c_mask
                    c = lc.bit_length() - 1
                    c_mask ^= lc
                    if c <= b:
                        continue
                    # Check if there's a d > c adjacent to a, b, c
                    common = adj[c] & na & adj[b]
                    # common contains a, b, c? No, adj[c] doesn't contain c.
                    # We need d in common, d > c, d != a, d != b.
                    # Since a < b < c, and d > c, d is distinct from a,b.
                    # Also need d in na (adjacent to a) - already ensured by & na.
                    # And d in adj[b] - already ensured by & adj[b].
                    # So just check if common has any bit > c.
                    if common & (~((1 << (c + 1)) - 1)):
                        return True
        return False

    def find_independent_6(n, adj):
        """Find an independent set of size 6 using greedy + backtracking with bitsets."""
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~adj[i]  # non-neighbors of i (including i itself? No, adj[i] doesn't include i)
            # Actually, independent set: no edges between any two. So for vertex i, its candidates are non-neighbors.
            # But we also need to exclude i itself from the "others" when building the set.
            # Let's define: comp[i] = set of vertices j != i such that not edge(i,j).
            comp[i] = all_mask & ~adj[i] & ~(1 << i)
        
        best_size = [0]
        found_set = [None]
        
        def bk(P, R, R_size):
            if found_set[0] is not None:
                return
            if R_size >= 6:
                found_set[0] = list(R)
                return
            if not P:
                return
            if R_size + P.bit_count() < 6:
                return
            # Pivot: choose vertex with max degree in P
            # For speed, just pick the lowest bit
            u_bit = P & -P
            u = u_bit.bit_length() - 1
            candidates = P & comp[u]
            c = candidates
            while c:
                low = c & -c
                v = low.bit_length() - 1
                R.append(v)
                bk(P & comp[v], R, R_size + 1)
                R.pop()
                P &= ~low
                c &= ~low
        
        bk(all_mask, [], 0)
        return found_set[0]

    def is_valid(n, adj):
        if has_k4_fast(n, adj):
            return False
        if find_independent_6(n, adj) is not None:
            return False
        return True

    def to_edges(n, adj):
        edges = []
        for i in range(n):
            m = adj[i]
            while m:
                lb = m & -m
                j = lb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lb
        return edges

    def random_adj(n, p, rng):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def tabu_search(n, adj, deadline, rng):
        """Tabu search to find a valid graph on n vertices."""
        cur_adj = list(adj)
        cur_valid = is_valid(n, cur_adj)
        if cur_valid:
            return cur_adj
        
        best_adj = list(cur_adj)
        best_valid = cur_valid
        tabu_list = {}  # (i,j) -> time when it becomes non-tabu
        tabu_tenure = 50
        iter_count = 0
        
        while time.time() < deadline - 5:
            iter_count += 1
            # Pick a random edge to flip
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            i, j = min(i, j), max(i, j)
            
            key = (i, j)
            # Check tabu
            if key in tabu_list and tabu_list[key] > iter_count:
                continue
            
            # Flip the edge
            old_edge = (cur_adj[j] >> i) & 1
            if old_edge:
                cur_adj[i] &= ~(1 << j)
                cur_adj[j] &= ~(1 << i)
            else:
                cur_adj[i] |= (1 << j)
                cur_adj[j] |= (1 << i)
            
            new_valid = is_valid(n, cur_adj)
            
            # Accept if valid, or if we're in a "diversification" mode
            if new_valid:
                best_adj = list(cur_adj)
                best_valid = True
                return best_adj
            
            # Check if this move is tabu
            if key in tabu_list and tabu_list[key] > iter_count:
                # Revert
                if old_edge:
                    cur_adj[i] |= (1 << j)
                    cur_adj[j] |= (1 << i)
                else:
                    cur_adj[i] &= ~(1 << j)
                    cur_adj[j] &= ~(1 << i)
                continue
            
            # Keep the move, add to tabu list
            tabu_list[key] = iter_count + tabu_tenure
            
            # Periodically check if we found a valid graph
            if iter_count % 100 == 0 and best_valid:
                return best_adj
        
        return best_adj if best_valid else None

    # Try target sizes from high to low
    for target_n in range(36, 16, -1):
        if time.time() > deadline - 10:
            break
        
        found = None
        # Try multiple random starts
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Start with a random graph with appropriate density
            # For K4-free and alpha < 6, density around 0.4-0.5 is typical
            p = 0.4 + 0.1 * rng.random()
            adj = random_adj(target_n, p, rng)
            
            # Quick check: if already valid, return
            if is_valid(target_n, adj):
                found = adj
                break
            
            # Run tabu search
            result = tabu_search(target_n, adj, deadline, rng)
            if result is not None:
                found = result
                break
        
        if found is not None:
            return {'n': target_n, 'edges': to_edges(target_n, found)}
    
    # Fallback: return a small valid graph (e.g., Paley(17))
    # Paley graph of order 17
    q = 17
    adj = [0] * q
    squares = {x * x % q for x in range(1, q)}
    for i in range(q):
        for j in range(q):
            if i != j and (j - i) % q in squares:
                adj[i] |= (1 << j)
    
    return {'n': q, 'edges': to_edges(q, adj)}
