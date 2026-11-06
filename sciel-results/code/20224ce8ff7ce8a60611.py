# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n = 20
    # Use bitmasks for fast intersection
    # adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n
    
    def common_neighbors(u, v):
        return adj[u] & adj[v]
    
    def count_common(u, v):
        return bin(adj[u] & adj[v]).count('1')
    
    def is_valid_add(u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        # Adding edge (u,v) affects common neighbor counts for:
        # - Pairs involving u: for each w != u,v, common_neighbors(w, u) gains v
        # - Pairs involving v: for each w != u,v, common_neighbors(w, v) gains u
        # - The pair (u,v) itself: common_neighbors(u,v) = adj[u] & adj[v] (before adding)
        #   After adding, u and v are neighbors, so we need to check pairs that now
        #   include both u and v as potential common neighbors.
        
        # Actually, let's think more carefully.
        # A K2,3 exists if there's a pair (a,b) with |N(a)∩N(b)| >= 3.
        # When we add edge (u,v):
        # - N(u) gains v, N(v) gains u
        # - For any pair (x,y), the common neighbors change only if x or y is u or v.
        #   Specifically:
        #   - Pair (u, y) for y != v: N(u)∩N(y) gains v (if v is in N(y))
        #   - Pair (v, y) for y != u: N(v)∩N(y) gains u (if u is in N(y))
        #   - Pair (u, v): N(u)∩N(v) = (old N(u) ∪ {v}) ∩ (old N(v) ∪ {u})
        #     = (old N(u) ∩ old N(v)) ∪ ({v} ∩ old N(v)) ∪ (old N(u) ∩ {u}) ∪ ({v}∩{u})
        #     Since u is not in N(u) and v is not in N(v) (simple graph), and u≠v:
        #     = old N(u) ∩ old N(v) ∪ ({v} if v in N(v) else empty) ∪ ({u} if u in N(u) else empty)
        #     = old N(u) ∩ old N(v)  (since v not in N(v), u not in N(u))
        #     Wait, that's not right either. Let me reconsider.
        
        # Simpler approach: just check all pairs after hypothetically adding the edge.
        # But that's O(n^2) per check. With n=20, that's 190 pairs, each requiring a bitmask AND and popcount.
        # That's fast enough.
        
        # Hypothetical adjacency
        new_adj_u = adj[u] | (1 << v)
        new_adj_v = adj[v] | (1 << u)
        
        # Check all pairs
        for a in range(n):
            if a == u:
                na = new_adj_u
            elif a == v:
                na = new_adj_v
            else:
                na = adj[a]
            for b in range(a + 1, n):
                if b == u:
                    nb = new_adj_u
                elif b == v:
                    nb = new_adj_v
                else:
                    nb = adj[b]
                if bin(na & nb).count('1') >= 3:
                    return False
        return True
    
    # Strategy: Start from complete graph, remove edges to fix violations
    # Initialize with complete graph
    for i in range(n):
        adj[i] = ((1 << n) - 1) ^ (1 << i)  # All bits set except self
    
    def has_violation():
        for a in range(n):
            for b in range(a + 1, n):
                if bin(adj[a] & adj[b]).count('1') >= 3:
                    return True
        return False
    
    def find_violation():
        for a in range(n):
            for b in range(a + 1, n):
                if bin(adj[a] & adj[b]).count('1') >= 3:
                    return (a, b)
        return None
    
    # Greedy removal: while there's a violation, remove an edge that's in the most violations
    # For speed, we'll just find any violation and remove one of the edges involved
    
    # Actually, let's be smarter: for a violation pair (a,b) with >= 3 common neighbors,
    # we need to remove edges so that |N(a)∩N(b)| <= 2.
    # We can remove edges from common neighbors to a or b.
    
    # Let's try: repeatedly find a violation, and remove an edge incident to one of the common neighbors
    # that participates in the most violations.
    
    def edge_violation_count(u, v):
        """Count how many pairs (a,b) have both u and v as common neighbors, 
        i.e., u and v are in N(a)∩N(b)."""
        # u and v are common neighbors of (a,b) iff a and b are both neighbors of u and v
        # So a must be in N(u)∩N(v) and b must be in N(u)∩N(v), a≠b
        # But also need a≠u, b≠v etc. Actually since it's a simple graph, u is not in N(u).
        common = adj[u] & adj[v]
        cnt = bin(common).count('1')
        return cnt * (cnt - 1) // 2  # number of pairs among common neighbors
    
    # Greedy edge removal
    removed = 0
    max_iterations = 5000
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        viol = find_violation()
        if viol is None:
            break
        a, b = viol
        common = adj[a] & adj[b]
        common_list = [i for i in range(n) if (common >> i) & 1]
        
        # We need to reduce |common| to at most 2.
        # Remove edges to achieve this. For each common neighbor c, 
        # we can remove edge (a,c) or (b,c).
        # Choose the removal that breaks the most other violations.
        
        best_removal = None
        best_score = -1
        for c in common_list:
            # Option 1: remove edge (a, c)
            # Option 2: remove edge (b, c)
            for (x, y) in [(a, c), (b, c)]:
                # Score: how many violations would be fixed?
                # Rough heuristic: count how many pairs have x and c as common neighbors, etc.
                # Simpler: just pick randomly or use a simple heuristic
                score = 0
                # Count pairs involving x that would be fixed
                for w in range(n):
                    if w == x or w == y:
                        continue
                    # After removing (x,y), check if pair (x,w) or (y,w) or others improve
                    pass
                # Let's just use a simple score: prefer removing edges that are in many violations
                score = edge_violation_count(x, y)
                if score > best_score:
                    best_score = score
                    best_removal = (x, y)
        
        if best_removal is None:
            # Fallback: remove any edge
            c = common_list[0]
            best_removal = (a, c)
        
        x, y = best_removal
        adj[x] &= ~(1 << y)
        adj[y] &= ~(1 << x)
        removed += 1
    
    # Now try to add back edges
    # Collect all non-edges
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not ((adj[i] >> j) & 1):
                non_edges.append((i, j))
    
    random.shuffle(non_edges)
    
    for (u, v) in non_edges:
        if time.time() > deadline - 5:
            break
        if is_valid_add(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
