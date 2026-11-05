import time
import random

def construct():
    N = 20
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin? No, 150s limit, use 140s.
    
    # Bitmask representation
    # mask[i] is a bitmask of neighbors of i
    masks = [0] * N
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_add(u, v):
        # Check if adding edge (u, v) creates a K3,3
        # This happens if there exist u2, u3 such that 
        # |N(u) ∩ N(u2) ∩ N(u3) ∩ {v}| >= 3? No.
        # The condition is: For all distinct a, b, c, |N(a) ∩ N(b) ∩ N(c)| <= 2.
        # Adding edge (u, v) changes N(u) to N(u) ∪ {v} and N(v) to N(v) ∪ {u}.
        # We only need to check triples involving u or v.
        # Triples not involving u or v are unchanged.
        # Triples involving u: (u, x, y). Common neighbors become (N(u) ∪ {v}) ∩ N(x) ∩ N(y).
        #   = (N(u) ∩ N(x) ∩ N(y)) ∪ ({v} ∩ N(x) ∩ N(y)).
        #   The size can increase by at most 1 (if v is in N(x) ∩ N(y) and not in N(u)).
        #   So we need to ensure that for all x, y, the new intersection size is <= 2.
        #   This is equivalent to: 
        #   If v is in N(x) ∩ N(y), then we need |N(u) ∩ N(x) ∩ N(y)| <= 1.
        #   If v is NOT in N(x) ∩ N(y), the size doesn't change.
        # Similarly for triples (x, y, v).
        #   Common neighbors become N(x) ∩ N(y) ∩ (N(v) ∪ {u}).
        #   If u is in N(x) ∩ N(y), we need |N(v) ∩ N(x) ∩ N(y)| <= 1.
        
        # Let's just check all triples involving u or v.
        # There are O(N^2) such triples. N=20, so 400 checks. Fast.
        
        # Triples (u, x, y) with x != y, x,y != u
        for x in range(N):
            if x == u: continue
            for y in range(N):
                if y == u or y == x: continue
                # Check triple (u, x, y)
                # New common neighbors: (masks[u] | (1 << v)) & masks[x] & masks[y]
                # Note: v is not in masks[u] yet.
                common = (masks[u] | (1 << v)) & masks[x] & masks[y]
                if popcount(common) >= 3:
                    return False
        
        # Triples (x, y, v) with x != y, x,y != v
        # Note: (u, x, y) covered triples where u is one of the vertices.
        # We also need triples where v is one of the vertices, but u is NOT.
        # i.e., (x, y, v) where x, y != u, v.
        for x in range(N):
            if x == v or x == u: continue
            for y in range(N):
                if y == v or y == u or y == x: continue
                # Check triple (x, y, v)
                # New common neighbors: masks[x] & masks[y] & (masks[v] | (1 << u))
                common = masks[x] & masks[y] & (masks[v] | (1 << u))
                if popcount(common) >= 3:
                    return False
        return True

    def try_add_edge(u, v):
        if u == v: return False
        if (masks[u] >> v) & 1: return False # already edge
        if is_valid_add(u, v):
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            return True
        return False

    # Greedy initialization
    random.seed(42)
    edges_to_try = [(u, v) for u in range(N) for v in range(u+1, N)]
    random.shuffle(edges_to_try)
    for u, v in edges_to_try:
        try_add_edge(u, v)
    
    current_edges = sum(popcount(m) for m in masks) // 2
    best_edges = current_edges
    best_masks = masks[:]
    
    # Local Search: Simulated Annealing or Random Restart Hill Climbing
    # Since we can only add valid edges, we might get stuck.
    # Strategy: 
    # 1. Try to add any possible edge.
    # 2. If stuck, remove a random edge and try again.
    
    while time.time() < deadline:
        # Try to add edges
        improved = False
        # Shuffle non-edges
        non_edges = [(u, v) for u in range(N) for v in range(u+1, N) if not ((masks[u] >> v) & 1)]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if try_add_edge(u, v):
                improved = True
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_masks = masks[:]
        
        if not improved:
            # Stuck. Remove a random edge to escape local optimum
            if current_edges == 0:
                break
            all_edges = [(u, v) for u in range(N) for v in range(u+1, N) if (masks[u] >> v) & 1]
            if not all_edges:
                break
            u, v = random.choice(all_edges)
            # Remove edge
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            current_edges -= 1
            
            # Try to add edges again
            non_edges = [(a, b) for a in range(N) for b in range(a+1, N) if not ((masks[a] >> b) & 1)]
            random.shuffle(non_edges)
            for a, b in non_edges:
                if try_add_edge(a, b):
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_masks = masks[:]

    # Reconstruct edge list from best_masks
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_masks[u] >> v) & 1:
                edges.append([u, v])
    return edges
