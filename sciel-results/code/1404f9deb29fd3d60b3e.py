"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _check_no_c4(n, edges):
    """Check that no two vertices have two common neighbours."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        for j in range(i+1, n):
            if len(adj[i] & adj[j]) >= 2:
                return False
    return True

def _greedy_independent_set(n, adj, vertices_order=None):
    """Greedy independent set size."""
    if vertices_order is None:
        vertices_order = list(range(n))
    remaining = set(vertices_order)
    ind = 0
    while remaining:
        # Pick vertex with min degree in remaining subgraph? Or just first?
        # For upper bound estimation, greedy is fine.
        # For finding a large IS, pick min degree.
        v = min(remaining, key=lambda x: len(adj[x] & remaining))
        ind += 1
        remaining -= adj[v]
        remaining.discard(v)
    return ind

def _estimate_alpha(n, adj, timeout=1.0):
    """
    Estimate maximum independent set size.
    Returns an upper bound or a found size.
    For verification, we need to be sure alpha < 11.
    So we need a lower bound on alpha (found IS) to be < 11? No.
    We need to prove alpha < 11.
    This requires finding the MAX independent set.
    Greedy finds a LOWER bound on alpha.
    If greedy finds 11, alpha >= 11, so it's invalid.
    If greedy finds 10, alpha could be 11.
    So greedy alone is not sufficient for verification.
    We need a branch and bound or a stronger bound.
    """
    # Simple B&B for small n
    start = time.time()
    best = 0
    
    # Precompute degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    def branch(remaining, current_size):
        nonlocal best
        if time.time() - start > timeout:
            return
        if not remaining:
            if current_size > best:
                best = current_size
            return
        
        # Upper bound: current_size + len(remaining) is too loose.
        # Use greedy coloring bound on remaining subgraph.
        # For simplicity and speed, we use a heuristic bound:
        # Max IS <= current_size + (sum of 1/deg(v) for v in remaining) + 1?
        # Let's just use a simple recursive B&B with pruning by size.
        
        # Pick vertex with max degree in remaining
        v = max(remaining, key=lambda x: len(adj[x] & remaining))
        
        # Option 1: Include v
        new_remaining = set(remaining)
        new_remaining.discard(v)
        new_remaining -= adj[v]
        if current_size + len(new_remaining) > best: # Loose bound
            branch(new_remaining, current_size + 1)
            
        # Option 2: Exclude v
        remaining.discard(v)
        if current_size + len(remaining) > best:
            branch(remaining, current_size)
        remaining.add(v)
        
        if time.time() - start > timeout:
            return

    branch(set(range(n)), 0)
    return best

def _verify_graph(n, edges, timeout=5.0):
    """Verify graph has no C4 and alpha < 11."""
    if not _check_no_c4(n, edges):
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Use a more robust alpha check
    # Since n <= 43, B&B might be slow if not pruned well.
    # We'll use a time-limited B&B.
    alpha = _estimate_alpha(n, adj, timeout=timeout)
    return alpha < 11

def _can_add_edge(adj, u, v):
    """
    Check if adding edge (u,v) maintains C4-free property.
    adj is a list of sets.
    """
    # 1. u and v must not already be connected
    if v in adj[u]:
        return False
        
    # 2. Check pair (u, v)
    # Common neighbors of u and v in current graph
    common_uv = adj[u] & adj[v]
    if len(common_uv) >= 2:
        return False
    # If len == 1, adding edge creates a triangle, which is fine.
    # The condition is "no two vertices have two common neighbours".
    # After adding (u,v), u and v are neighbors.
    # Do they have two common neighbors?
    # The common neighbors are still common_uv.
    # So if len(common_uv) <= 1, the pair (u,v) is safe.
    
    # 3. Check pairs (u, w) for w in N(v)
    # When we add (u,v), v is added to N(u) and u is added to N(v).
    # For any w != u, v:
    # New N(u) = N(u) U {v}
    # New N(w) is unchanged (unless w is v, handled above, or u, handled below)
    # Common neighbors of (u, w) in new graph:
    # (N(u) U {v}) ∩ N(w) = (N(u) ∩ N(w)) ∪ ({v} ∩ N(w))
    # If v is in N(w), then v is a new common neighbor.
    # So if v in N(w), we must ensure that (N(u) ∩ N(w)) was empty.
    # Because if there was 1 common neighbor before, now there are 2.
    
    for w in adj[v]:
        if w == u:
            continue
        if v in adj[w]: # v is in N(w), so v becomes a new common neighbor of u and w
            # Check if u and w already had a common neighbor
            if adj[u] & adj[w]:
                return False
                
    # 4. Check pairs (v, w) for w in N(u)
    for w in adj[u]:
        if w == v:
            continue
        if u in adj[w]: # u is in N(w), so u becomes a new common neighbor of v and w
            # Check if v and w already had a common neighbor
            if adj[v] & adj[w]:
                return False
                
    return True

def construct():
    """
    Construct a C4-free graph with no independent 11-set.
    """
    start_time = time.time()
    deadline = start_time + 115 # 5s safety margin
    
    best_n = 30
    best_edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        best_edges += [[a,b], [a,c], [b,c]]
        
    # Try to find better n
    for target_n in range(39, 30, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random restarts
        max_restarts = 20 if target_n < 35 else 5
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            adj = [set() for _ in range(target_n)]
            edges = []
            
            # Random edge addition
            # Generate all possible edges
            possible_edges = [(i, j) for i in range(target_n) for j in range(i+1, target_n)]
            random.shuffle(possible_edges)
            
            # Add edges greedily
            for u, v in possible_edges:
                if _can_add_edge(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
            
            # Check if this graph works
            # First, quick check: does it have enough edges?
            # A graph with alpha < 11 needs to be quite dense.
            # For n=39, if alpha < 11, it's a good candidate.
            
            # Verify alpha
            # Use a faster verification if possible
            # We'll use the B&B with a short timeout
            alpha = _estimate_alpha(target_n, adj, timeout=2.0)
            
            if alpha < 11:
                # Double check with full verification
                if _verify_graph(target_n, edges, timeout=10.0):
                    best_n = target_n
                    best_edges = edges
                    found = True
                    break
        if found:
            break
            
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
