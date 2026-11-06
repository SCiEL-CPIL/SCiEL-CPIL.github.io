import time
import random

def construct():
    n = 20
    start_time = time.time()
    # Use a conservative deadline to ensure we return before the 120s kill
    deadline = start_time + 100.0 

    def is_add_valid(adj, u, v):
        """
        Fast check: does adding edge (u,v) create a K3,3?
        Only triples where the new edge adds a common neighbor matter:
        - Triple {u,a,b}: gains v as common neighbor if v~a and v~b (a,b != u)
          -> need existing common neighbors of {u,a,b} >= 2 (then +1 = 3)
        - Triple {v,a,b}: gains u as common neighbor if u~a and u~b (a,b != v, a,b != u)
          -> need existing common neighbors of {v,a,b} >= 2
        """
        # Get neighbor lists as bitmasks
        nu = adj[u]
        nv = adj[v]
        
        # Triples {u, a, b} where a,b are neighbors of v (a,b != u, a,b != v)
        # v's neighbors excluding u and v
        vn_mask = nv & ~(1 << u) & ~(1 << v)
        # For each pair (a,b) in vn_mask, check if common neighbors of {u,a,b} >= 2
        # (because adding v makes it 3)
        # common of {u,a,b} = nu & adj[a] & adj[b]
        # We need (nu & adj[a] & adj[b]).bit_count() >= 2
        # But note: nu already includes v (we added it), so we should exclude v
        # Actually, we haven't added it yet in this function - adj is pre-add state
        # Wait, let me reconsider. In the current code, adj is modified BEFORE calling is_valid_after_add.
        # Let me keep the same convention: adj already has the edge added.
        
        # With adj having edge (u,v):
        # For triple {u,a,b}: common neighbors = adj[u] & adj[a] & adj[b]
        # Before adding (u,v), common neighbors of {u,a,b} did NOT include v (if v~a and v~b)
        # After adding, if v~a and v~b, then v is now a common neighbor
        # So we need: old_common >= 2, where old_common = (adj[u] & adj[a] & adj[b]) & ~(1<<v)
        # i.e., (adj[u] & adj[a] & adj[b]).bit_count() - (1 if v in that set else 0) >= 2
        # Simpler: just check (adj[u] & adj[a] & adj[b]).bit_count() >= 3 directly,
        # but that's what the original does. The speedup comes from only checking relevant triples.
        
        # Let me just check bit_count >= 3 for the relevant triples
        vn_list = []
        m = vn_mask
        while m:
            lb = m & (-m)
            a = lb.bit_length() - 1
            vn_list.append(a)
            m ^= lb
        
        for i in range(len(vn_list)):
            a = vn_list[i]
            adj_a = adj[a]
            for j in range(i + 1, len(vn_list)):
                b = vn_list[j]
                common = nu & adj_a & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        # Triples {v, a, b} where a,b are neighbors of u (a,b != v, a,b != u)
        un_mask = nu & ~(1 << v) & ~(1 << u)
        un_list = []
        m = un_mask
        while m:
            lb = m & (-m)
            a = lb.bit_length() - 1
            un_list.append(a)
            m ^= lb
        
        for i in range(len(un_list)):
            a = un_list[i]
            adj_a = adj[a]
            for j in range(i + 1, len(un_list)):
                b = un_list[j]
                common = nv & adj_a & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def greedy_fill(adj):
        """
        Greedily adds edges in random order.
        Returns the number of edges in the final adj.
        """
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_add_valid(adj, u, v):
                pass
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return count_edges(adj)

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append((u, v))
        return edges

    def try_add_edge(adj, u, v):
        """Try to add edge (u,v), return True if successful."""
        if adj[u] & (1 << v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if is_add_valid(adj, u, v):
            return True
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            return False

    def local_search(adj, max_iter=200):
        """
        Local search with 1-edge swaps: remove one edge, add another.
        Returns improved adj and count.
        """
        current_count = count_edges(adj)
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            edges = get_edges(adj)
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) 
                        if not (adj[u] & (1 << v))]
            
            if not edges or not non_edges:
                break
            
            improved = False
            # Try random 1-edge swaps
            for _ in range(20):
                if time.time() > deadline:
                    break
                ru, rv = random.choice(edges)
                nu_, nv_ = random.choice(non_edges)
                
                # Remove (ru, rv), try add (nu_, nv_)
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                
                if try_add_edge(adj, nu_, nv_):
                    new_count = count_edges(adj)
                    if new_count > current_count:
                        current_count = new_count
                        improved = True
                        break
                    else:
                        # Rollback the add
                        adj[nu_] &= ~(1 << nv_)
                        adj[nv_] &= ~(1 << nu_)
                
                # Restore removed edge
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
            
            if improved:
                continue
            else:
                # No improvement, break
                break
        
        return current_count

    best_adj = [0] * n
    best_count = 0
    current_adj = [0] * n
    
    # Initial greedy fill from empty graph
    current_count = greedy_fill(current_adj)
    if current_count > best_count:
        best_count = current_count
        best_adj = current_adj[:]

    # Local search / Restart strategy
    iterations = 0
    no_improve_count = 0
    while time.time() < deadline:
        iterations += 1
        current_adj = best_adj[:]
        
        # Adaptive removal: start with fewer, increase if stuck
        if no_improve_count < 10:
            num_removals = random.randint(1, 3)
        elif no_improve_count < 30:
            num_removals = random.randint(2, 6)
        else:
            num_removals = random.randint(3, 10)
        
        edges_to_remove = get_edges(current_adj)
        
        if not edges_to_remove:
            continue
            
        random.shuffle(edges_to_remove)
        for u, v in edges_to_remove[:num_removals]:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
        
        current_count = greedy_fill(current_adj)
        
        # Also try local search on the result
        if current_count >= best_count:
            current_count = local_search(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # If stuck for too long, do a full restart
        if no_improve_count >= 50:
            current_adj = [0] * n
            current_count = greedy_fill(current_adj)
            current_count = local_search(current_adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = current_adj[:]
            no_improve_count = 0

    # Final polish: try adding any missing edge that doesn't create K3,3
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline:
                    break
                if best_adj[u] & (1 << v):
                    continue
                # Try adding
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                if is_add_valid(best_adj, u, v):
                    improved = True
                else:
                    best_adj[u] &= ~(1 << v)
                    best_adj[v] &= ~(1 << u)
            if time.time() > deadline:
                break

    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
