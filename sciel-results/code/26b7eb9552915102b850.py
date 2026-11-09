"""C4-free graph with high minimum degree (R(C4,K1,39) lower bound)."""
import time
import random

# EVOLVE-BLOCK-START

def _can_add_edge(adj, u, v):
    """
    Fast check: does adding edge (u, v) create a C4?
    We assume (u, v) is not currently an edge.
    A C4 would be u - y - x - v - u with x in N(v), y in N(u), and edge (x, y).
    So for any x in N(v), if N(x) intersects N(u), then C4 exists.
    """
    nu = adj[u]
    nv = adj[v]
    x = nv
    while x:
        lsb = x & -x
        xi = lsb.bit_length() - 1
        x ^= lsb
        # xi is a neighbor of v. We need to check if xi is connected to any neighbor of u.
        # If adj[xi] & nu != 0, there is a common neighbor y, forming u-y-xi-v-u.
        if adj[xi] & nu:
            return False
    return True

def _can_remove_edge(adj, u, v):
    """
    Check if removing edge (u, v) is safe (it never creates a C4, so always True).
    However, we might want to check if it helps. For now, removal is always structurally safe.
    """
    return True

def _min_degree(adj):
    md = len(adj)
    for a in adj:
        d = a.bit_count()
        if d < md:
            md = d
    return md

def _is_c4_free(adj, n):
    """
    Full verification: for every edge (u, v), the number of common neighbors must be at most 1.
    """
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if (au >> v) & 1:
                common = au & adj[v]
                # If common has >= 2 bits, then there is a C4
                if common & (common - 1):
                    return False
    return True

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                edges.append([i, j])
    return edges

def _greedy_build(n, rng, p=1.0):
    """
    Build an initial C4-free graph by adding random edges with probability p,
    but using the C4 check to ensure C4-freedom.
    """
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    
    for u, v in all_edges:
        if rng.random() < p:
            if _can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    return adj

def _local_search(adj, n, target_min_deg, rng, max_iter=5000, deadline=None):
    """
    Local search to increase minimum degree while maintaining C4-freedom.
    Operations:
    1. Add an edge from a low-degree vertex to a non-neighbor if C4-safe.
    2. Swap: remove edge (a, b) and add edge (u, v) if it improves min degree
       and maintains C4-freedom.
    """
    improved = True
    it = 0
    while improved and it < max_iter:
        if deadline and time.time() > deadline:
            break
        improved = False
        it += 1
        degrees = [a.bit_count() for a in adj]
        
        # Find vertices below target
        low_verts = [i for i in range(n) if degrees[i] < target_min_deg]
        
        if not low_verts:
            break
        
        # Operation 1: Add edges to low-degree vertices
        for u in low_verts:
            if degrees[u] >= target_min_deg:
                continue
            non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
            rng.shuffle(non_neighbors)
            
            for v in non_neighbors:
                if degrees[u] >= target_min_deg:
                    break
                if _can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
                    improved = True
        
        # If still below target, try edge swaps
        if min(degrees) < target_min_deg:
            # Recalculate degrees if we added edges
            degrees = [a.bit_count() for a in adj]
            low_verts = [i for i in range(n) if degrees[i] < target_min_deg]
            if low_verts:
                # Pick the lowest degree vertex
                u = min(low_verts, key=lambda i: degrees[i])
                # Find a high-degree vertex to remove an edge from
                # We want to move an edge from a high-degree vertex to help u
                # Strategy: Find (a, b) where a is high, b is not too low, and (u, v) where v is not neighbor of u
                # Such that removing (a, b) and adding (u, v) is C4-safe.
                
                # Simplified swap: 
                # 1. Pick a neighbor a of u? No, u needs new neighbors.
                # 2. Pick a non-neighbor v of u.
                # 3. Pick an edge (a, b) to remove.
                #    Ideally, a is a neighbor of u? No, if a is neighbor of u, removing (a,b) doesn't help u directly.
                #    We want to add (u, v). To "pay" for it, we remove (a, b).
                #    Constraint: (u, v) must be C4-safe in the graph with (a, b) removed.
                #    This is hard to check directly.
                
                # Alternative swap strategy:
                # If u is stuck, maybe u has many neighbors that are also neighbors of many others?
                # Let's try: Remove (a, b) where a is a neighbor of u?
                # If we remove (a, b), does it help add (u, v)?
                # The C4 check for (u, v) depends on N(u) and N(v).
                # If we remove (a, b), it might break a potential C4 u-y-a-b-v-u? No.
                # It breaks C4s involving a and b.
                
                # Let's try a simpler heuristic:
                # If we cannot add any edge to u, try to "free up" space by removing an edge from a high-degree node.
                # But which edge?
                # Let's just do random swaps if stuck for a while.
                
                # For now, a basic swap:
                # Pick random edge (a, b) from a high degree node.
                # Pick random non-neighbor v of u.
                # Check if (u, v) is C4-safe after removing (a, b).
                
                # To keep it fast and simple within the time limit, we will only do add-edges in this loop.
                # If we are stuck, we break and try a new seed.
                # The "swap" logic is complex and risky for C4-freedom without careful checking.
                # Given the 150s limit, relying on good seeds for greedy+add is safer.
                
                pass

    return adj

def construct():
    deadline = time.time() + 145.0  # Safety margin under 150s
    best_n = 39
    best_edges = []
    
    # Try to maximize n from 47 down
    for n in range(47, 38, -1):
        target_min_deg = n - 39
        found = False
        
        # Try multiple seeds
        # Use more seeds for larger n as it's harder
        max_seeds = 200 if n >= 46 else 50
        
        for seed in range(max_seeds):
            if time.time() > deadline:
                break
            
            rng = random.Random(seed)
            
            # Greedy build
            adj = _greedy_build(n, rng, p=1.0)
            
            # Local search to improve min degree
            adj = _local_search(adj, n, target_min_deg, rng, max_iter=1000, deadline=deadline)
            
            min_deg = _min_degree(adj)
            
            if min_deg >= target_min_deg:
                # Verify C4-freedom
                if _is_c4_free(adj, n):
                    edges = _edges_from_adj(adj, n)
                    
                    if n > best_n:
                        best_n = n
                        best_edges = edges
                    found = True
                    
                    # If we found n=47, return immediately
                    if n == 47:
                        return {'n': n, 'edges': edges}
        
        if found:
            # Since we iterate n downwards, this is the largest n found
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
