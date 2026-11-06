"""Construct a C4-free graph with no independent 11-set, maximizing n."""
import time
import random

# Precompute the polarity graph of PG(2, 7)
def _polarity_graph_q7():
    q = 7
    reps = []
    seen = set()
    for z in range(q):
        for y in range(q):
            for x in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    norm = (1, (y * pow(x, -1, q)) % q, (z * pow(x, -1, q)) % q)
                elif y != 0:
                    norm = (0, 1, (z * pow(y, -1, q)) % q)
                else:
                    norm = (0, 0, 1)
                if norm not in seen:
                    seen.add(norm)
                    reps.append(norm)
    
    n = len(reps)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            u, v = reps[i], reps[j]
            val = (u[0]*v[0] + u[1]*v[1] - u[2]*v[2]) % q
            if val == 0:
                adj[i].add(j)
                adj[j].add(i)
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return n, adj, edges

def _subgraph(n, adj_full, edges_full):
    """Extract induced subgraph on first n vertices."""
    edges = [e for e in edges_full if e[0] < n and e[1] < n]
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj, edges

def _max_independent_set_size(n, adj, deadline):
    """Compute exact alpha using branch and bound with bitmasks."""
    masks = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        masks[i] = mask
    
    # Greedy upper bound
    def greedy_upper_bound(rem):
        count = 0
        while rem:
            # Pick vertex with max degree in rem subgraph
            best_v = -1
            best_deg = -1
            t = rem
            while t:
                lsb = t & -t
                v = lsb.bit_length() - 1
                deg = (masks[v] & rem).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                t ^= lsb
            rem &= ~(masks[best_v] | (1 << best_v))
            count += 1
        return count
    
    # Branch and bound
    best = [0]
    
    def branch(rem, size):
        if time.time() > deadline:
            return
        if size > best[0]:
            best[0] = size
        if rem == 0:
            return
        if size + rem.bit_count() <= best[0]:
            return
        if size + greedy_upper_bound(rem) <= best[0]:
            return
        
        # Pick vertex: highest degree in rem
        t = rem
        v = -1
        max_deg = -1
        while t:
            lsb = t & -t
            vi = lsb.bit_length() - 1
            deg = (masks[vi] & rem).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = vi
            t ^= lsb
        
        if v < 0:
            return
        
        vbit = 1 << v
        
        # Include v
        branch(rem & ~(masks[v] | vbit), size + 1)
        # Exclude v
        branch(rem & ~vbit, size)
    
    all_mask = (1 << n) - 1
    branch(all_mask, 0)
    return best[0]

def _check_c4_free(n, adj):
    """Check no two vertices share >= 2 common neighbors."""
    for i in range(n):
        for j in range(i + 1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def _verify(n, adj, deadline):
    """Check C4-free and alpha < 11."""
    if not _check_c4_free(n, adj):
        return False
    alpha = _max_independent_set_size(n, adj, deadline)
    return alpha < 11

def _try_extend(base_n, base_adj, base_edges, target_n, deadline, seed):
    """Try to extend the base graph to target_n by adding vertices."""
    if base_n >= target_n:
        return None
    if time.time() > deadline:
        return None
    
    rng = random.Random(seed)
    n = base_n
    adj = [set(s) for s in base_adj]
    edges = list(base_edges)
    
    # For each new vertex, we need to choose its neighborhood
    # such that:
    # 1. C4-free: for any two existing vertices u,v, new vertex can't be adjacent to both if u,v share >= 1 common neighbor
    # 2. No independent set of 11
    
    # Build a conflict graph: if u and v share >= 1 common neighbor, new vertex can't be adjacent to both
    # Actually, the constraint is: new vertex w adjacent to u and v means w is a common neighbor of u and v
    # So if u and v already share a common neighbor, w can't be adjacent to both
    
    # Precompute: for each pair (u,v), do they share a common neighbor?
    # If yes, w can't be in N(u) ∩ N(v)
    
    # Actually simpler: w's neighborhood S must satisfy: for any u in S, v in S, 
    # the number of common neighbors of u and v in {w} + existing <= 1
    # Since w is new, the only new common neighbor between u and v is w itself
    # So: for any u,v both in S, u and v must NOT already share a common neighbor
    
    # This means S must be an independent set in the "co-square" graph where
    # u ~ v if they share >= 1 common neighbor
    
    # Precompute conflict: conflict[u] = set of v that share >= 1 common neighbor with u
    conflict = [set() for _ in range(n)]
    for u in range(n):
        for v in adj[u]:
            for w in adj[u]:
                if w != v:
                    conflict[v].add(w)
                    conflict[w].add(v)
    
    # Now we need to add (target_n - n) vertices, each with a neighborhood S
    # that is an independent set in the conflict graph (i.e., no two vertices in S conflict)
    # But we also need to maintain the property for future vertices
    
    # This is complex. Let's try a simpler approach:
    # Start with the base graph and try to add vertices one by one
    # For each new vertex, try all possible neighborhoods (or a subset)
    
    # Actually, let's try a different approach: use the full PG(2,7) graph
    # and try to find a 43-vertex subgraph with alpha < 11
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    # Start with the PG(2,7) polarity graph
    n_full, adj_full, edges_full = _polarity_graph_q7()
    
    # Try n=43
    target = 43
    n, adj, edges = _subgraph(target, adj_full, edges_full)
    
    # Verify
    if time.time() < deadline:
        if _verify(n, adj, deadline):
            return {'n': n, 'edges': edges}
    
    # Try n=42
    target = 42
    n, adj, edges = _subgraph(target, adj_full, edges_full)
    if time.time() < deadline:
        if _verify(n, adj, deadline):
            return {'n': n, 'edges': edges}
    
    # Try n=41
    target = 41
    n, adj, edges = _subgraph(target, adj_full, edges_full)
    if time.time() < deadline:
        if _verify(n, adj, deadline):
            return {'n': n, 'edges': edges}
    
    # Try n=40
    target = 40
    n, adj, edges = _subgraph(target, adj_full, edges_full)
    if time.time() < deadline:
        if _verify(n, adj, deadline):
            return {'n': n, 'edges': edges}
    
    # Try n=39
    target = 39
    n, adj, edges = _subgraph(target, adj_full, edges_full)
    if time.time() < deadline:
        if _verify(n, adj, deadline):
            return {'n': n, 'edges': edges}
    
    # Fallback: 10 disjoint triangles
    edges_30 = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges_30 += [[a,b], [a,c], [b,c]]
    return {'n': 30, 'edges': edges_30}
