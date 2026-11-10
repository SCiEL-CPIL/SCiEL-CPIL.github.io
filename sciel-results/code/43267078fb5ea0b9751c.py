import time
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m_total = len(all_edges)
    
    def can_add(adj, u, v):
        """Fast check if adding edge (u,v) keeps graph K3,3-free."""
        bit_v = 1 << v
        bit_u = 1 << u
        
        for a in range(n):
            if a == u or a == v:
                continue
            ma = adj[a]
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                mb = adj[b]
                common_ab = ma & mb
                
                # Check triple {u, a, b}: does adding (u,v) make v a common neighbor?
                if (common_ab & bit_v) and (adj[u] & common_ab).bit_count() >= 2:
                    return False
                
                # Check triple {v, a, b}: does adding (u,v) make u a common neighbor?
                if (common_ab & bit_u) and (adj[v] & common_ab).bit_count() >= 2:
                    return False
        return True
    
    def is_valid(adj):
        """Verify graph is K3,3-free."""
        for u in range(n):
            mu = adj[u]
            for v in range(u + 1, n):
                muv = mu & adj[v]
                if muv.bit_count() < 3:
                    continue
                for w in range(v + 1, n):
                    if (muv & adj[w]).bit_count() >= 3:
                        return False
        return True
    
    def find_violation(adj):
        """Find a K3,3 violation. Returns (u,v,w, common_neighbors) or None."""
        for u in range(n):
            mu = adj[u]
            for v in range(u + 1, n):
                muv = mu & adj[v]
                if muv.bit_count() < 3:
                    continue
                for w in range(v + 1, n):
                    muvw = muv & adj[w]
                    if muvw.bit_count() >= 3:
                        neighbors = [i for i in range(n) if (muvw >> i) & 1]
                        return (u, v, w, neighbors)
        return None
    
    def repair(adj, edges, rng):
        """Remove edges to eliminate all K3,3 violations."""
        changed = True
        while changed:
            changed = False
            viol = find_violation(adj)
            if viol is None:
                break
            u, v, w, neighbors = viol
            # Remove a random edge from the K3,3
            x = neighbors[rng.integers(len(neighbors))]
            vertex = [u, v, w][rng.integers(3)]
            a, b = min(vertex, x), max(vertex, x)
            if (a, b) in edges:
                edges.remove((a, b))
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                changed = True
        return adj, edges
    
    def greedy_add(adj, edges, rng, deadline):
        """Greedily add all possible missing edges."""
        missing = [(i, j) for i, j in all_edges if (i, j) not in edges]
        rng.shuffle(missing)
        added = 0
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            if can_add(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
                added += 1
        return added
    
    def local_search(adj, edges, rng, deadline):
        """Local search with 1-for-2 and 2-for-3 swaps."""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_edges = list(edges)
            rng.shuffle(current_edges)
            
            for e in current_edges:
                if time.time() > deadline - 5:
                    break
                u, v = e
                # Remove (u, v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                edges.remove(e)
                
                # Try to add 2 edges
                added = 0
                candidates = [(i, j) for i, j in all_edges if (i, j) not in edges]
                rng.shuffle(candidates)
                
                for a, b in candidates:
                    if added >= 2:
                        break
                    if can_add(adj, a, b):
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        edges.add((a, b))
                        added += 1
                        
                if added >= 2:
                    improved = True
                elif added == 1:
                    pass  # Keep the swap, net zero but different configuration
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add(e)
                    
            # Try 2-for-3 swaps: remove 2 edges, add 3
            if time.time() < deadline - 5:
                current_edges = list(edges)
                rng.shuffle(current_edges)
                for i in range(min(len(current_edges) - 1, 100)):
                    if time.time() > deadline - 5:
                        break
                    e1 = current_edges[i]
                    e2 = current_edges[i + 1]
                    u1, v1 = e1
                    u2, v2 = e2
                    # Remove both
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    edges.remove(e1)
                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    edges.remove(e2)
                    
                    # Try to add 3 edges
                    added = 0
                    candidates = [(a, b) for a, b in all_edges if (a, b) not in edges]
                    rng.shuffle(candidates)
                    
                    for a, b in candidates:
                        if added >= 3:
                            break
                        if can_add(adj, a, b):
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
                            edges.add((a, b))
                            added += 1
                            
                    if added >= 3:
                        improved = True
                    elif added == 2:
                        pass
                    else:
                        # Revert
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                        edges.add(e1)
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        edges.add(e2)
                        
        return adj, edges
    
    best_edges = []
    best_count = 0
    
    # Multiple restarts with different strategies
    densities = [0.3, 0.4, 0.5, 0.6, 0.7]
    num_restarts = 80
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        rng = np.random.default_rng(42 + r * 7)
        p = densities[r % len(densities)] + 0.1 * rng.random()
        
        # Initialize
        adj = [0] * n
        edges = set()
        for i, j in all_edges:
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.add((i, j))
        
        # Repair
        adj, edges = repair(adj, edges, rng)
        
        if time.time() > deadline - 5:
            break
        
        # Greedy add
        greedy_add(adj, edges, rng, deadline)
        
        # Local search
        adj, edges = local_search(adj, edges, rng, deadline)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
            
        if best_count >= 56:
            break
    
    # Final verification
    if best_edges:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        if is_valid(adj_final):
            return best_edges
    
    # Fallback: deterministic construction
    # Use a known K3,3-free graph construction
    # The complement of a disjoint union of cliques is K3,3-free if cliques are large enough
    # Or use a random graph with p=0.5 and repair
    rng = np.random.default_rng(12345)
    adj = [0] * n
    edges = set()
    for i, j in all_edges:
        if rng.random() < 0.5:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            edges.add((i, j))
    adj, edges = repair(adj, edges, rng)
    greedy_add(adj, edges, rng, deadline)
    
    if edges and len(edges) > best_count:
        adj_final = [0] * n
        for u, v in edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        if is_valid(adj_final):
            return [list(e) for e in edges]
    
    return [list(e) for e in edges] if edges else []
