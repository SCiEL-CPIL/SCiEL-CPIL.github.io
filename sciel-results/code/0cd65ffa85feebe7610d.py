"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 15}

def _is_valid_addition(adj, n, u, v):
    """
    Check if adding edge (u, v) to the current graph 'adj' creates any violation.
    Assumes (u, v) is NOT currently in adj.
    Returns True if the addition is valid (no new violations), False otherwise.
    """
    # Current degrees before addition
    deg = [len(adj[i]) for i in range(n)]
    
    # The new edge (u, v) can be part of a path A-B-C-D in 3 ways:
    # 1. (u, v) is (A, B) -> Path u-v-C-D
    #    C must be neighbor of v (C != u)
    #    D must be neighbor of C (D != v)
    #    Violation if deg(u) == deg(D)
    for c in adj[v]:
        if c == u:
            continue
        for d in adj[c]:
            if d == v:
                continue
            # Path u-v-c-d
            # Endpoints are u and d
            if deg[u] == deg[d]:
                return False

    # 2. (u, v) is (B, C) -> Path A-u-v-D
    #    A must be neighbor of u (A != v)
    #    D must be neighbor of v (D != u)
    #    Violation if deg(A) == deg(D)
    for a in adj[u]:
        if a == v:
            continue
        for d in adj[v]:
            if d == u:
                continue
        # Wait, D is neighbor of v.
        # Path A-u-v-D
        # Endpoints are A and D
        # Check deg(A) == deg(D)
            if deg[a] == deg[d]:
                return False

    # 3. (u, v) is (C, D) -> Path A-B-v-u
    #    B must be neighbor of v (B != u)
    #    A must be neighbor of B (A != v)
    #    Violation if deg(A) == deg(u)
    for b in adj[v]:
        if b == u:
            continue
        for a in adj[b]:
            if a == v:
                continue
        # Path a-b-v-u
        # Endpoints are a and u
        # Check deg(a) == deg(u)
            if deg[a] == deg[u]:
                return False

    return True

def _full_check(adj, n):
    """Full validity check for safety."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _greedy_construct(n, rng):
    """Construct a valid graph by greedily adding edges."""
    adj = [set() for _ in range(n)]
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(edges)
    
    for u, v in edges:
        # Check if adding (u,v) is valid
        # We need to check if (u,v) is already added? No, we process each pair once.
        # But we must ensure we don't add duplicates. The list is unique.
        if _is_valid_addition(adj, n, u, v):
            adj[u].add(v)
            adj[v].add(u)
            
    return adj

def _local_search(adj, n, rng, max_steps=1000):
    """Try to add more edges or swap to improve density."""
    # Copy adj
    curr_adj = [set(s) for s in adj]
    curr_deg = [len(curr_adj[i]) for i in range(n)]
    
    # Try to add missing edges
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in curr_adj[i]]
    
    improved = True
    while improved and max_steps > 0:
        improved = False
        max_steps -= 1
        if not missing:
            break
        
        # Shuffle missing to try different orders
        rng.shuffle(missing)
        
        for u, v in missing:
            # Check if still missing
            if v in curr_adj[u]:
                continue
            
            if _is_valid_addition(curr_adj, n, u, v):
                curr_adj[u].add(v)
                curr_adj[v].add(u)
                improved = True
                # Update missing list? 
                # Removing from list while iterating is bad.
                # But we break out of inner loop? No.
                # Let's just let it be. If it's added, next check `if v in curr_adj[u]` will skip it.
                # But we should remove it from `missing` to be clean?
                # For efficiency, just checking is fine.
                
    return curr_adj

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = P['n']
    best_edges = None
    best_count = 0
    rng = random.Random(42)
    
    # Run multiple restarts
    for i in range(100):
        if time.time() > deadline:
            break
            
        # 1. Greedy Construction
        adj = _greedy_construct(n, rng)
        
        # 2. Local Search
        adj = _local_search(adj, n, rng)
        
        # 3. Full Check
        if _full_check(adj, n):
            count = sum(len(s) for s in adj) // 2
            if count > best_count:
                best_count = count
                best_edges = [[u, v] for u in range(n) for v in adj[u] if v > u]
        else:
            # If full check fails (shouldn't happen if incremental is correct), 
            # try to repair by removing edges from violations
            # But for now, just discard.
            pass
            
        # Optional: Perturbation for ILS
        # If we have a best, we could perturb it, but greedy restart is usually good enough for N=15.
        
    # Fallback
    if best_edges is None:
        # Star graph is always valid?
        # Path s-0-1-t. deg(s)=1, deg(t)=1. Violation!
        # Star graph K1,14 has center 0 deg 14, leaves deg 1.
        # Path leaf1 - 0 - leaf2 - ... wait, leaf2 has no other edges.
        # So no path of length 3 exists in a star graph.
        # So star graph is valid.
        best_edges = [[0, v] for v in range(1, n)]
        best_count = n - 1
        
    return best_edges

# EVOLVE-BLOCK-END
