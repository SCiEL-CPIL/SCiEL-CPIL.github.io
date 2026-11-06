import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    n = 20
    
    def can_add(adj, u, v):
        """Check if adding edge (u,v) violates K3,3-free condition."""
        if adj[u] & (1 << v):
            return False
        
        # A K3,3 subgraph exists iff there is a set of 3 vertices whose common neighbor set has size >= 3.
        # Adding edge (u,v) can only create new common-neighbor relationships involving u or v as common neighbors.
        
        # Case 1: v becomes a new common neighbor of some triple {a,b,c} where a,b,c are all adjacent to v.
        # After adding (u,v), v is adjacent to u. So if u,a,b are all adjacent to v, then v is a common neighbor of {u,a,b}.
        # We need |N(u) ∩ N(a) ∩ N(b)| <= 1 (existing common neighbors, excluding v) to keep total <= 2.
        
        # Case 2: Similarly, u becomes new common neighbor of {v,a,b} where a,b adjacent to u.
        # Need |N(v) ∩ N(a) ∩ N(b)| <= 1.
        
        # Optimization: Pre-compute the set of vertices adjacent to v and u.
        adj_v = adj[v]
        adj_u = adj[u]
        
        # For case 1: find all a,b where both a and b are adjacent to v, and a,b != u,v
        # Then check common neighbors of {u,a,b}
        # Instead of O(n^2) pairs, we can iterate more efficiently.
        
        # For each a adjacent to v (a != u, a != v):
        #   For each b adjacent to v (b > a, b != u, b != v):
        #     Check if |N(u) ∩ N(a) ∩ N(b)| > 1
        
        # Let's use a more efficient approach:
        # For case 1, the constraint is: for all pairs (a,b) with a,b adjacent to v, a,b != u:
        #   |N(u) ∩ N(a) ∩ N(b)| <= 1
        
        # N(u) ∩ N(a) ∩ N(b) = (N(u) ∩ N(a)) ∩ N(b)
        # Let S_a = N(u) ∩ N(a) (vertices adjacent to both u and a)
        # Then we need |S_a ∩ N(b)| <= 1 for all b adjacent to v, b != u, b != a.
        
        # Alternative: iterate over all pairs (a,b) where both are adjacent to v.
        # This is at most C(19,2) = 171 pairs, which is fine.
        
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj_v & (1 << a)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj_v & (1 << b)):
                    continue
                common = adj_u & adj[a] & adj[b]
                if common.bit_count() > 1:
                    return False
        
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj_u & (1 << a)):
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj_u & (1 << b)):
                    continue
                common = adj_v & adj[a] & adj[b]
                if common.bit_count() > 1:
                    return False
                        
        return True

    def get_edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def greedy_fill(adj, time_limit=0.5):
        """Greedily add as many edges as possible."""
        current_edges = get_edge_count(adj)
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        random.shuffle(missing)
        
        for u, v in missing:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
        
        return adj, current_edges

    def local_search(adj, max_time=1.0):
        """Local search with kick mechanism."""
        ls_start = time.time()
        current_edges = get_edge_count(adj)
        best_adj = adj[:]
        best_edges = current_edges
        
        # First, greedy fill
        adj, current_edges = greedy_fill(adj)
        best_adj = adj[:]
        best_edges = current_edges
        
        no_improve_count = 0
        max_no_improve = 100
        
        while time.time() - ls_start < max_time and no_improve_count < max_no_improve:
            if time.time() > deadline - 3:
                break
            
            # Try random add/remove swaps
            # Strategy 1: Try to add a random missing edge
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
            
            if missing:
                u, v = random.choice(missing)
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1
                    no_improve_count = 0
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    no_improve_count += 1
            else:
                break
            
            # Strategy 2: Kick - remove 1-2 random edges, then greedy fill
            if no_improve_count > 20:
                kick_size = random.choice([1, 1, 2, 2, 3])
                existing = get_edge_list(adj)
                if existing:
                    removed = []
                    for _ in range(min(kick_size, len(existing))):
                        if not existing:
                            break
                        idx = random.randint(0, len(existing) - 1)
                        a, b = existing[idx]
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        current_edges -= 1
                        removed.append((a, b))
                        existing.pop(idx)
                    
                    # Greedy fill from current state
                    adj, current_edges = greedy_fill(adj)
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                        if no_improve_count >= max_no_improve:
                            adj = best_adj[:]
                            current_edges = best_edges
                            no_improve_count = 0
                            # Re-greedy fill
                            adj, current_edges = greedy_fill(adj)
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = adj[:]
                            break
        
        return best_adj, best_edges

    def spectral_init():
        """Initialize using spectral approach - create a dense bipartite-like structure."""
        adj = [0] * n
        # Try to create a dense graph by starting with a random dense structure
        # and then removing edges that violate K3,3-free
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        random.seed(num_restarts * 104729 + 7)
        
        # Different initialization strategies
        if num_restarts % 4 == 1:
            # Pure greedy
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        elif num_restarts % 4 == 2:
            # Start with random sparse, then greedy fill
            adj = [0] * n
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(pairs)
            for u, v in pairs[:random.randint(10, 40)]:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            adj, _ = greedy_fill(adj)
        elif num_restarts % 4 == 3:
            # Perturb best solution
            if best_edges > 0:
                adj = best_adj[:]
                existing = get_edge_list(adj)
                num_remove = random.randint(3, 10)
                random.shuffle(existing)
                for a, b in existing[:num_remove]:
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                adj, _ = greedy_fill(adj)
            else:
                adj = [0] * n
                pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
                random.shuffle(pairs)
                for u, v in pairs:
                    if can_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            # Spectral-style: start with complete bipartite structure
            adj = [0] * n
            # Split into two groups of 10
            group1 = list(range(10))
            group2 = list(range(10, 20))
            # Add all cross edges first (these are safe - no K3,3 in bipartite)
            for a in group1:
                for b in group2:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
            # Now try to add intra-group edges
            intra = [(i, j) for i in range(n) for j in range(i+1, n) 
                     if (i < 10 and j < 10) or (i >= 10 and j >= 10)]
            random.shuffle(intra)
            for u, v in intra:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            adj, _ = greedy_fill(adj)
        
        edges = get_edge_count(adj)
        
        # Local search with decreasing time allocation
        remaining = deadline - time.time()
        ls_time = min(2.0, max(0.3, remaining * 0.2))
        adj, edges = local_search(adj, max_time=ls_time)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
    
    # Final polish
    if time.time() < deadline - 2:
        best_adj, final_edges = greedy_fill(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges
    
    # Convert to list of lists
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
