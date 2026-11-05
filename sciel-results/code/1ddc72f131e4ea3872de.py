"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a K3,3-free graph on 20 vertices maximizing edges.
    
    Strategy:
    1. Start with a greedy construction that builds a dense graph while maintaining K3,3-freeness.
    2. Use local search (edge swaps) to improve the number of edges.
    3. Use bitmask-based violation checks for efficiency.
    4. Multiple restarts with different random seeds.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    N = 20
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    pair_index = {p: idx for idx, p in enumerate(all_pairs)}
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_valid_k33(adj, n):
        """Check if graph with adjacency bitmasks is K3,3-free.
        A K3,3 exists if there are 3 vertices u,v,w with |N(u) ∩ N(v) ∩ N(w)| >= 3
        where the intersection excludes u,v,w themselves.
        """
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    # Common neighbors of u, v, w
                    common = adj[u] & adj[v] & adj[w]
                    # Remove u, v, w themselves from consideration
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    common &= ~(1 << w)
                    if popcount(common) >= 3:
                        return False
        return True
    
    def count_edges(adj, n):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2
    
    def can_add_edge(adj, u, v, n):
        """Check if adding edge (u,v) maintains K3,3-freeness.
        Only need to check triples that include u or v.
        """
        # After adding edge (u,v), new common neighbor sets change for triples containing u or v
        # Triple (u, v, w): common neighbors of u, v, w
        #   = (N(u) ∪ {v}) ∩ (N(v) ∪ {u}) ∩ N(w) minus {u,v,w}
        #   = (N(u) ∩ N(v) ∩ N(w)) ∪ (N(u) ∩ {u} ∩ N(w)) ∪ ... 
        #   Actually simpler: new N(u) = N(u) ∪ {v}, new N(v) = N(v) ∪ {u}
        #   For triple (u, v, w): common = (N(u)∪{v}) ∩ (N(v)∪{u}) ∩ N(w)
        #   = (N(u) ∩ N(v) ∩ N(w)) ∪ ({v} ∩ N(v) ∩ N(w)) ∪ (N(u) ∩ {u} ∩ N(w)) ∪ ({v} ∩ {u} ∩ N(w))
        #   v is in N(v)? No, v is not in N(v) (no self-loops). So {v} ∩ N(v) = empty.
        #   u is in N(u)? No. So N(u) ∩ {u} = empty.
        #   {v} ∩ {u} = empty.
        #   So common = N(u) ∩ N(v) ∩ N(w) (same as before, since u∉N(u), v∉N(v))
        #   Wait, that's not right. Let me reconsider.
        #   
        #   New adjacency: N'(u) = N(u) ∪ {v}, N'(v) = N(v) ∪ {u}, N'(w) = N(w)
        #   Common neighbors of u,v,w in new graph:
        #   N'(u) ∩ N'(v) ∩ N'(w) = (N(u) ∪ {v}) ∩ (N(v) ∪ {u}) ∩ N(w)
        #   = [N(u) ∩ (N(v) ∪ {u}) ∩ N(w)] ∪ [{v} ∩ (N(v) ∪ {u}) ∩ N(w)]
        #   = [N(u) ∩ N(v) ∩ N(w) ∪ N(u) ∩ {u} ∩ N(w)] ∪ [{v} ∩ N(v) ∩ N(w) ∪ {v} ∩ {u} ∩ N(w)]
        #   Since u ∉ N(u), N(u) ∩ {u} = ∅
        #   Since v ∉ N(v), {v} ∩ N(v) = ∅
        #   {v} ∩ {u} = ∅
        #   So = N(u) ∩ N(v) ∩ N(w)
        #   Same as before! So adding edge (u,v) doesn't change common neighbors for triple (u,v,w).
        #
        #   For triple (u, x, y) where x,y ≠ v:
        #   N'(u) ∩ N'(x) ∩ N'(y) = (N(u) ∪ {v}) ∩ N(x) ∩ N(y)
        #   = (N(u) ∩ N(x) ∩ N(y)) ∪ ({v} ∩ N(x) ∩ N(y))
        #   So we need to check: does v become a common neighbor of u, x, y?
        #   i.e., is v ∈ N(x) ∩ N(y) AND (after adding, v ∈ N'(u) which is true since we add (u,v))
        #   And we need to check if this creates 3 common neighbors.
        #
        #   More precisely: after adding (u,v), for any triple (u, x, y) with x,y ≠ v:
        #   New common = old_common ∪ ({v} if v ∈ N(x) ∩ N(y) else ∅)
        #   So violation occurs if old_common already had 2+ and v ∈ N(x)∩N(y), OR old_common had 3+.
        #   But old graph was valid, so old_common ≤ 2.
        #   So violation iff |old_common| = 2 AND v ∈ N(x) ∩ N(y)
        #
        #   Similarly for triple (v, x, y) with x,y ≠ u:
        #   New common = old_common ∪ ({u} if u ∈ N(x) ∩ N(y) else ∅)
        #   Violation iff |old_common| = 2 AND u ∈ N(x) ∩ N(y)
        #
        #   For triples not containing u or v: no change.
        
        # Check triples (u, x, y) for all x < y, x,y ≠ v
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                # Old common neighbors of u, x, y
                old_common = adj[u] & adj[x] & adj[y]
                old_common &= ~(1 << u) & ~(1 << x) & ~(1 << y)
                # New common adds v if v ∈ N(x) ∩ N(y)
                if (adj[x] & (1 << v)) and (adj[y] & (1 << v)):
                    # v becomes a common neighbor
                    if popcount(old_common) >= 2:
                        return False
        # Check triples (v, x, y) for all x < y, x,y ≠ u
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x+1, n):
                if y == u or y == v:
                    continue
                old_common = adj[v] & adj[x] & adj[y]
                old_common &= ~(1 << v) & ~(1 << x) & ~(1 << y)
                if (adj[x] & (1 << u)) and (adj[y] & (1 << u)):
                    if popcount(old_common) >= 2:
                        return False
        return True
    
    def greedy_construct(seed):
        """Build a dense K3,3-free graph using greedy edge addition."""
        rng = random.Random(seed)
        adj = [0] * N
        edge_count = 0
        
        # Order edges randomly
        edges = all_pairs[:]
        rng.shuffle(edges)
        
        for u, v in edges:
            if can_add_edge(adj, u, v, N):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
            if time.time() > deadline:
                break
        
        return adj, edge_count
    
    def local_search(adj, edge_count, seed, time_limit=10):
        """Improve graph via edge swaps (remove one edge, add another)."""
        rng = random.Random(seed)
        end_time = time.time() + time_limit
        best_adj = adj[:]
        best_count = edge_count
        
        while time.time() < end_time:
            # Try to add a random missing edge
            # Find missing edges
            missing = []
            for u in range(N):
                for v in range(u+1, N):
                    if not (adj[u] & (1 << v)):
                        missing.append((u, v))
            
            if not missing:
                break
            
            # Try random missing edges
            found_improvement = False
            for _ in range(50):  # Try up to 50 random additions
                u, v = rng.choice(missing)
                if can_add_edge(adj, u, v, N):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
                    found_improvement = True
                    break
            
            if not found_improvement:
                # Try edge swap: remove an edge, then try to add two edges
                # Or just try to find a better configuration
                # Try removing a random edge and see if we can add more
                present = []
                for u in range(N):
                    for v in range(u+1, N):
                        if adj[u] & (1 << v):
                            present.append((u, v))
                
                if present:
                    ru, rv = rng.choice(present)
                    # Remove edge
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    edge_count -= 1
                    
                    # Try to add edges
                    added = 0
                    for _ in range(30):
                        if not missing:
                            break
                        u, v = rng.choice(missing)
                        if can_add_edge(adj, u, v, N):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edge_count += 1
                            added += 1
                            # Remove from missing (approximate)
                            if (u,v) in missing:
                                missing.remove((u,v))
                    
                    if edge_count > best_count:
                        best_adj = adj[:]
                        best_count = edge_count
                    else:
                        # Revert if not improved
                        adj[ru] |= (1 << rv)
                        adj[rv] |= (1 << ru)
                        edge_count -= added
        
        return best_adj, best_count
    
    # Multiple restarts
    best_edges = None
    best_count = 0
    
    # First, try a few greedy constructions with different seeds
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        adj, ec = greedy_construct(seed)
        if ec > best_count:
            best_count = ec
            best_edges = adj[:]
        
        # Local search on the best result
        if time.time() > deadline - 30:
            break
        adj2, ec2 = local_search(adj, ec, seed, time_limit=min(8, deadline - time.time() - 5))
        if ec2 > best_count:
            best_count = ec2
            best_edges = adj2[:]
    
    # Final local search on the best result
    if best_edges is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            adj3, ec3 = local_search(best_edges, best_count, 12345, time_limit=remaining)
            if ec3 > best_count:
                best_count = ec3
                best_edges = adj3[:]
    
    if best_edges is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    # Convert adjacency to edge list
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if best_edges[u] & (1 << v):
                result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END
