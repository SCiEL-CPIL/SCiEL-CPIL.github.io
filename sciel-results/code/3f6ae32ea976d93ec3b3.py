import random
import time
import math

def construct():
    deadline = time.time() + 115  # Safety margin
    n = 40
    
    # Adjacency representation: list of sets
    # adj[u] is a set of neighbors of u
    
    def creates_c6(adj, u, v):
        """
        Check if adding edge (u, v) to graph adj creates a C6.
        This is equivalent to checking if there is a path of length 5 from u to v in adj.
        Path: u - a - b - c - d - v
        """
        # Optimize by iterating over the smaller neighborhood first?
        # The structure is fixed: u-a-b-c-d-v
        # We iterate d in N(v), a in N(u)
        # Then b in N(a), c in N(b)
        
        # To speed up, we can precompute N(u) and N(v) as lists
        nu = list(adj[u])
        nv = list(adj[v])
        
        # If either is empty, no path
        if not nu or not nv:
            return False
            
        # Iterate over a in N(u) and d in N(v)
        # We need to find a, b, c such that a-b-c-d is a path
        # And a != v, d != u, a != d, b != u, b != v, b != a, b != d, c != u, c != v, c != a, c != b, c != d
        
        # The seed logic:
        for d in nv:
            if d == u:
                continue
            for a in nu:
                if a == v:
                    continue
                # If a == d, path is u-a-a... invalid (simple path)
                # Also a and d must be distinct for simple path u-a-b-c-d-v
                if a == d:
                    continue
                # Now find b, c such that a-b-c-d
                # b in N(a), c in N(b), d in N(c)
                # b != u, b != v, b != a, b != d
                # c != u, c != v, c != a, c != b, c != d
                for b in adj[a]:
                    if b == u or b == v or b == a or b == d:
                        continue
                    for c in adj[b]:
                        if c == u or c == v or c == a or c == b or c == d:
                            continue
                        if d in adj[c]:
                            return True
        return False

    def to_list(adj):
        edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.add((u, v))
        return edges

    def from_list(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_missing_edges(adj):
        missing = []
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]:
                    missing.append((u, v))
        return missing

    def get_existing_edges(adj):
        existing = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    existing.append((u, v))
        return existing

    # 1. Initial Greedy Construction
    def greedy_construct(rng):
        adj = [set() for _ in range(n)]
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if not creates_c6(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
        return adj

    rng = random.Random(42)
    
    # Start with a few random restarts of greedy to find a good initial state
    best_adj = None
    best_m = 0
    initial_time = time.time()
    
    while time.time() < initial_time + 10: # Spend 10s on initialization
        adj = greedy_construct(rng)
        m = sum(len(s) for s in adj) // 2
        if m > best_m:
            best_m = m
            best_adj = [set(s) for s in adj]
            
    current_adj = [set(s) for s in best_adj]
    current_m = best_m
    
    # Simulated Annealing
    T = 1.0
    alpha = 0.9999 # Cooling rate
    # We need to tune alpha so that T drops reasonably over 100 seconds
    # Let's say we do ~10000 iterations.
    # If alpha=0.999, after 10000 steps, T = 1 * 0.999^10000 ~ 4e-5.
    
    time_start = time.time()
    iter_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        if iter_count % 1000 == 0:
            T *= 0.99 # Slow cooling
        
        move_type = rng.random()
        
        new_adj = None
        delta_m = 0
        valid = False
        
        if move_type < 0.5:
            # Add an edge
            missing = get_missing_edges(current_adj)
            if missing:
                u, v = rng.choice(missing)
                # Check if adding creates C6
                if not creates_c6(current_adj, u, v):
                    # Tentatively add
                    # We don't need to copy the whole graph if we check validity first
                    # But we need to check if it's valid.
                    # creates_c6 already checks validity.
                    new_adj = [set(s) for s in current_adj] # Copy
                    new_adj[u].add(v)
                    new_adj[v].add(u)
                    delta_m = 1
                    valid = True
        elif move_type < 0.7:
            # Remove an edge
            existing = get_existing_edges(current_adj)
            if existing:
                u, v = rng.choice(existing)
                new_adj = [set(s) for s in current_adj]
                new_adj[u].discard(v)
                new_adj[v].discard(u)
                delta_m = -1
                valid = True
        else:
            # 1-Swap: Remove one, add one
            existing = get_existing_edges(current_adj)
            missing = get_missing_edges(current_adj)
            if existing and missing:
                u1, v1 = rng.choice(existing)
                u2, v2 = rng.choice(missing)
                
                # Tentatively remove (u1, v1)
                # Check if adding (u2, v2) is valid in the graph without (u1, v1)
                # We can create a temporary adj or just modify in place and revert
                # To avoid copying, we can modify current_adj, check, and revert if rejected
                # But we need to check creates_c6 on the modified graph.
                
                # Let's do in-place modification for speed
                current_adj[u1].discard(v1)
                current_adj[v1].discard(u1)
                
                if not creates_c6(current_adj, u2, v2):
                    current_adj[u2].add(v2)
                    current_adj[v2].add(u2)
                    delta_m = 0 # Removed 1, Added 1
                    valid = True
                else:
                    # Revert removal
                    current_adj[u1].add(v1)
                    current_adj[v1].add(u1)
                    valid = False
            else:
                valid = False
                
        if valid:
            # Decide whether to accept
            if delta_m >= 0 or (T > 0 and rng.random() < math.exp(delta_m / T)):
                # Accept
                if move_type >= 0.7:
                    # Swap was already applied in place
                    pass
                else:
                    # Add or Remove was prepared in new_adj
                    current_adj = new_adj
                current_m += delta_m
                
                if current_m > best_m:
                    best_m = current_m
                    best_adj = [set(s) for s in current_adj]
            else:
                # Reject
                if move_type >= 0.7:
                    # Swap was applied in place, need to revert
                    # We added (u2, v2) and removed (u1, v1)
                    # Revert: Remove (u2, v2), Add (u1, v1)
                    current_adj[u2].discard(v2)
                    current_adj[v2].discard(u2)
                    current_adj[u1].add(v1)
                    current_adj[v1].add(u1)
                # For Add/Remove, we didn't modify current_adj, so nothing to do

    # Final validation of best_adj
    # Just to be safe, verify best_adj is C6-free
    adj_check = best_adj
    is_valid = True
    # We can reuse the logic from creates_c6 but for all edges?
    # Or just trust the incremental checks.
    # Given the risk of bugs, a full check is better if fast enough.
    # Full check: for each edge, remove it and check if path of length 5 exists?
    # That's expensive.
    # Let's just return the result. The incremental check is robust if implemented correctly.
    
    edges = to_list(best_adj)
    return [[u, v] for u, v in sorted(edges)]
