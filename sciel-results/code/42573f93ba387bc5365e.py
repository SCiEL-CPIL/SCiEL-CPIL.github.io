import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_possible_edges = len(all_edges)
    
    def can_add(adj, u, v):
        """Check if adding edge (u,v) violates K3,3-free condition."""
        if adj[u] & (1 << v):
            return False
            
        # For K3,3-free: no three vertices can have 3+ common neighbors.
        # Adding (u,v) creates new triples where u and v are in different parts.
        # Key insight: check all pairs (a,b) - if v is common neighbor of a,b and u has 2+ common neighbors with a,b, 
        # OR u is common neighbor of a,b and v has 2+ common neighbors with a,b, violation occurs.
        
        # More precisely: after adding (u,v), for any pair (a,b):
        # - If both a,b are adjacent to v (in new graph), then common neighbors of {a,b} include v.
        #   We need |N(a) ∩ N(b) ∩ N(u)| <= 2 to avoid creating K3,3 with u,v and some third vertex.
        # Actually simpler: the constraint is that no 3 vertices share 3 common neighbors.
        # Adding (u,v): check all triples that now might have 3 common neighbors.
        
        # A triple {x,y,z} has a common neighbor w if w is adjacent to all three.
        # After adding (u,v), new common neighbor relationships only involve u or v as the new edge endpoint.
        
        # Case 1: v becomes a new common neighbor of some triple {a,b,c} where a,b,c are all adjacent to v.
        #   For v to be common neighbor of {a,b,c}, we need a,b,c all adjacent to v.
        #   The new edge (u,v) makes u adjacent to v, so if u,a,b are all adjacent to v, 
        #   then v is common neighbor of {u,a,b}. We need |N(u) ∩ N(a) ∩ N(b)| <= 2 (excluding v itself from count? No, v is the common neighbor).
        #   Actually: common neighbors of {u,a,b} are vertices adjacent to all of u,a,b.
        #   v is one such (after adding). We need total count <= 2, so existing common neighbors of {u,a,b} (excluding v) must be <= 1.
        
        # Case 2: Similarly, u becomes new common neighbor of {v,a,b} where a,b adjacent to u.
        #   Need |N(v) ∩ N(a) ∩ N(b)| <= 2 after adding, so existing (excluding u) must be <= 1.
        
        # Implementation:
        # For each pair (a,b) where a != u,v and b != u,v:
        #   If a and b are both adjacent to v: check common neighbors of {u,a,b} excluding v. If > 1, reject.
        #   If a and b are both adjacent to u: check common neighbors of {v,a,b} excluding u. If > 1, reject.
        
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                
                # Check if v is adjacent to both a and b
                if (adj[a] & (1 << v)) and (adj[b] & (1 << v)):
                    # Common neighbors of {u, a, b} in current graph (before adding u-v)
                    # These are vertices adjacent to u, a, and b
                    common = adj[u] & adj[a] & adj[b]
                    # v is not in common since we haven't added u-v yet, and v is adjacent to a,b but not necessarily u
                    # After adding u-v, v becomes a common neighbor of {u,a,b}
                    # So total common neighbors will be bit_count(common) + 1
                    # We need this <= 2, so bit_count(common) <= 1
                    if common.bit_count() > 1:
                        return False
                
                # Check if u is adjacent to both a and b
                if (adj[a] & (1 << u)) and (adj[b] & (1 << u)):
                    # Common neighbors of {v, a, b} in current graph
                    common = adj[v] & adj[a] & adj[b]
                    # After adding u-v, u becomes a common neighbor of {v,a,b}
                    if common.bit_count() > 1:
                        return False
                        
        return True

    def get_edge_list(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append((i, j))
        return edges

    def greedy_fill(adj):
        """Greedily add as many edges as possible. Returns updated adj and edge count."""
        current_edges = sum(x.bit_count() for x in adj) // 2
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        # Try multiple random orders for greedy fill
        best_adj = adj[:]
        best_edges = current_edges
        
        for attempt in range(5):
            temp_adj = adj[:]
            temp_missing = missing[:]
            random.shuffle(temp_missing)
            
            for u, v in temp_missing:
                if can_add(temp_adj, u, v):
                    temp_adj[u] |= (1 << v)
                    temp_adj[v] |= (1 << u)
            
            temp_edges = sum(x.bit_count() for x in temp_adj) // 2
            if temp_edges > best_edges:
                best_edges = temp_edges
                best_adj = temp_adj[:]
                
        return best_adj, best_edges

    def local_search(adj, max_time=1.0):
        """Local search with kick mechanism."""
        ls_start = time.time()
        current_edges = sum(x.bit_count() for x in adj) // 2
        best_adj = adj[:]
        best_edges = current_edges
        
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        # First, greedy fill
        adj, current_edges = greedy_fill(adj)
        best_adj = adj[:]
        best_edges = current_edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if not missing:
            return best_adj, best_edges
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() - ls_start < max_time and no_improve_count < max_no_improve:
            if time.time() > deadline - 3:
                break
                
            if not missing:
                break
                
            # Try to add a random missing edge
            idx = random.randint(0, len(missing) - 1)
            u, v = missing[idx]
            
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                missing.pop(idx)
                no_improve_count = 0
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Kick: remove 1-3 random edges, then greedy fill
                kick_size = random.choice([1, 1, 1, 2, 2, 3])
                existing = get_edge_list(adj)
                if not existing:
                    break
                    
                removed = []
                for _ in range(min(kick_size, len(existing))):
                    if not existing:
                        break
                    idx2 = random.randint(0, len(existing) - 1)
                    a, b = existing[idx2]
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    current_edges -= 1
                    removed.append((a, b))
                    existing.pop(idx2)
                
                # Greedy fill from current state
                adj, current_edges = greedy_fill(adj)
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # Revert to best if we're stuck
                    if no_improve_count >= max_no_improve:
                        adj = best_adj[:]
                        current_edges = best_edges
                        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                        no_improve_count = 0
                        if not missing:
                            break
                        continue
                        # Actually, let's just break and restart
                        break
                
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                
        return best_adj, best_edges

    best_adj = [0] * n
    best_edges = 0
    
    # Multiple restarts with varied strategies
    for i in range(10000):
        if time.time() > deadline - 8:
            break
            
        random.seed(i * 7 + 13)
        adj = [0] * n
        
        # Strategy 1: Pure greedy with random order
        if i % 3 == 0:
            pairs = all_edges[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Strategy 2: Greedy with bias toward high-degree vertices
        elif i % 3 == 1:
            # Start with some random edges, then fill greedily
            num_initial = random.randint(10, 30)
            pairs = all_edges[:]
            random.shuffle(pairs)
            for u, v in pairs[:num_initial]:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            # Now greedy fill
            adj, _ = greedy_fill(adj)
        
        # Strategy 3: Start from a known good structure if available
        else:
            if best_edges > 0 and i > 0:
                # Perturb best solution
                adj = best_adj[:]
                # Remove some random edges
                existing = get_edge_list(adj)
                num_remove = random.randint(3, 8)
                random.shuffle(existing)
                for a, b in existing[:num_remove]:
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                # Greedy fill
                adj, _ = greedy_fill(adj)
            else:
                pairs = all_edges[:]
                random.shuffle(pairs)
                for u, v in pairs:
                    if can_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        edges = sum(x.bit_count() for x in adj) // 2
        
        # Local search
        ls_time = min(2.0, max(0.5, (deadline - time.time()) * 0.3))
        adj, edges = local_search(adj, max_time=ls_time)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]
            
    # Final polish: try to add any remaining edges
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
