"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 20
TOTAL_EDGES = N * (N - 1) // 2  # 190

def construct():
    deadline = time.time() + 135.0  # safety margin
    
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
    
    def is_k33_free(adj):
        # Check if any 3 vertices have 3+ common neighbors
        # For each triple (i,j,k), count common neighbors
        # A K3,3 exists iff there exist 3 vertices A,B,C and 3 vertices D,E,F
        # such that each of A,B,C is connected to each of D,E,F.
        # This means |N(A) ∩ N(B) ∩ N(C)| >= 3 where N(v) is the neighbor set.
        for i in range(N):
            for j in range(i+1, N):
                # common neighbors of i and j
                common_ij = adj[i] & adj[j]
                for k in range(j+1, N):
                    common_ijk = common_ij & adj[k]
                    if popcount(common_ijk) >= 3:
                        return False
        return True
    
    def count_k33_violations(adj):
        """Count the number of triples with >= 3 common neighbors."""
        count = 0
        for i in range(N):
            for j in range(i+1, N):
                common_ij = adj[i] & adj[j]
                for k in range(j+1, N):
                    common_ijk = common_ij & adj[k]
                    c = popcount(common_ijk)
                    if c >= 3:
                        count += c - 2  # penalty: how many over the threshold
        return count
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps the graph K3,3-free."""
        # Adding edge (u,v) means u becomes a neighbor of v and vice versa.
        # This can create new K3,3s only if there exist a triple containing u or v
        # that now has 3 common neighbors.
        # More precisely: after adding (u,v), for any triple {a,b,c}, the common
        # neighbor count can only increase for triples that include u or v.
        # For triple (u, x, y): common neighbors = N(x) ∩ N(y) ∩ N(u)
        #   Before: N(u) didn't include v, after it does.
        #   So new common neighbors for (u,x,y) = old + [v if v in N(x)∩N(y)]
        # For triple (x, y, z) not containing u or v: unchanged.
        # For triple (u, v, x): common neighbors = N(u) ∩ N(v) ∩ N(x)
        #   Before: u not in N(v) and v not in N(u), so this triple had
        #   common neighbors = (N(u)∪{v}) ∩ (N(v)∪{u}) ∩ N(x) minus the edge itself...
        #   Actually before adding, N(u) and N(v) don't include each other.
        #   After adding, N(u) includes v and N(v) includes u.
        #   Common neighbors of (u,v,x) after = (N(u) ∩ N(v) ∩ N(x))
        #   N(u) ∩ N(v) after = (old N(u) ∩ old N(v)) ∪ {v if v in old N(u)} ∪ {u if u in old N(v)}
        #   But v was not in old N(u) and u was not in old N(v), so
        #   N(u) ∩ N(v) after = old N(u) ∩ old N(v) (unchanged for the intersection)
        #   Wait, that's not right. N(u) after = old N(u) ∪ {v}, N(v) after = old N(v) ∪ {u}.
        #   N(u) ∩ N(v) after = (old N(u) ∪ {v}) ∩ (old N(v) ∪ {u})
        #   = (old N(u) ∩ old N(v)) ∪ (old N(u) ∩ {u}) ∪ ({v} ∩ old N(v)) ∪ ({v} ∩ {u})
        #   = (old N(u) ∩ old N(v)) ∪ {} ∪ {} ∪ {}  (since u not in old N(u), v not in old N(v))
        #   = old N(u) ∩ old N(v)
        #   So common neighbors of (u,v,x) after = (old N(u) ∩ old N(v) ∩ N(x))
        #   Before, (u,v,x) had common neighbors = (old N(u) ∩ old N(v) ∩ N(x)) as well
        #   (since u and v weren't neighbors of each other, but that doesn't affect
        #    common neighbors of the triple).
        #   
        # Let me reconsider. The common neighbors of a triple {a,b,c} are vertices
        # adjacent to all three. Adding edge (u,v):
        # - For triples not containing u or v: unchanged.
        # - For triple (u, x, y) where x,y ≠ v: 
        #   Common neighbors before = N(x) ∩ N(y) ∩ N(u)
        #   Common neighbors after = N(x) ∩ N(y) ∩ (N(u) ∪ {v})
        #   = (N(x) ∩ N(y) ∩ N(u)) ∪ (N(x) ∩ N(y) ∩ {v})
        #   New ones: v if v ∈ N(x) ∩ N(y)
        # - For triple (v, x, y) where x,y ≠ u:
        #   Similarly, new common neighbor: u if u ∈ N(x) ∩ N(y)
        # - For triple (u, v, x):
        #   Before: N(u) ∩ N(v) ∩ N(x) (u,v not neighbors of each other doesn't matter)
        #   After: (N(u)∪{v}) ∩ (N(v)∪{u}) ∩ N(x) = (N(u)∩N(v)) ∩ N(x) [as shown above]
        #   So unchanged.
        #
        # So the only potential new violations are:
        # 1. For some x,y ≠ v: if v ∈ N(x)∩N(y) and |N(x)∩N(y)∩N(u)| == 2, then after adding,
        #    |N(x)∩N(y)∩(N(u)∪{v})| = 3, creating a violation.
        # 2. For some x,y ≠ u: if u ∈ N(x)∩N(y) and |N(x)∩N(y)∩N(v)| == 2, then after adding,
        #    |N(x)∩N(y)∩(N(v)∪{u})| = 3, creating a violation.
        
        # Check condition 1: v is a common neighbor of some pair (x,y) where x,y are both
        # neighbors of u (or will be), and N(u)∩N(x)∩N(y) has exactly 2 elements.
        # Actually: v ∈ N(x) ∩ N(y) means x and y are both neighbors of v.
        # And we need |N(x) ∩ N(y) ∩ N(u)| == 2 before the addition.
        
        # Check condition 2: u is a common neighbor of some pair (x,y) where x,y are both
        # neighbors of v, and |N(x) ∩ N(y) ∩ N(v)| == 2 before the addition.
        
        # Condition 1:
        neighbors_of_v = [x for x in range(N) if x != u and x != v and (adj[v] >> x) & 1]
        for idx, x in enumerate(neighbors_of_v):
            for y in neighbors_of_v[idx+1:]:
                # Check if v ∈ N(x) ∩ N(y): yes by construction
                # Check |N(x) ∩ N(y) ∩ N(u)|
                common = adj[x] & adj[y] & adj[u]
                if popcount(common) == 2:
                    return False
        
        # Condition 2:
        neighbors_of_u = [x for x in range(N) if x != u and x != v and (adj[u] >> x) & 1]
        for idx, x in enumerate(neighbors_of_u):
            for y in neighbors_of_u[idx+1:]:
                # Check if u ∈ N(x) ∩ N(y): yes by construction
                # Check |N(x) ∩ N(y) ∩ N(v)|
                common = adj[x] & adj[y] & adj[v]
                if popcount(common) == 2:
                    return False
        
        return True
    
    def greedy_construct():
        """Build a dense K3,3-free graph using greedy edge addition."""
        adj = [0] * N
        edge_count = 0
        
        # Try all edges in random order
        edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(edges)
        
        for u, v in edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        return adj, edge_count
    
    def local_search(adj, edge_count, max_iters=50000):
        """Try to improve by removing edges that allow adding more edges."""
        adj = list(adj)
        edge_count = edge_count
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try to find an edge to remove that allows adding 2+ new edges
            # Or simpler: try random edge removals and see if we can add more back
            best_gain = 0
            best_remove = None
            best_adds = None
            
            # Sample a few edges to try removing
            current_edges = []
            for i in range(N):
                for j in range(i+1, N):
                    if (adj[i] >> j) & 1:
                        current_edges.append((i, j))
            
            if not current_edges:
                break
            
            sample_size = min(20, len(current_edges))
            random.shuffle(current_edges)
            
            for u, v in current_edges[:sample_size]:
                # Remove edge (u,v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                # Try to add edges back
                new_adj = list(adj)
                new_count = edge_count - 1
                added = []
                
                edges_to_try = [(i, j) for i in range(N) for j in range(i+1, N) 
                               if i != u or j != v]
                random.shuffle(edges_to_try)
                
                for a, b in edges_to_try:
                    if (new_adj[a] >> b) & 1:
                        continue
                    if can_add_edge(new_adj, a, b):
                        new_adj[a] |= (1 << b)
                        new_adj[b] |= (1 << a)
                        new_count += 1
                        added.append((a, b))
                    
                    if len(added) >= 3:  # enough to be worth it
                        break
                
                gain = new_count - edge_count
                if gain > best_gain:
                    best_gain = gain
                    best_remove = (u, v)
                    best_adds = list(added)
                    best_new_adj = list(new_adj)
                    best_new_count = new_count
                
                # Restore
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            if best_gain > 0 and best_remove is not None:
                adj = best_new_adj
                edge_count = best_new_count
            else:
                break
        
        return adj, edge_count
    
    def simulated_annealing(adj, edge_count, max_iters=100000):
        """SA that allows temporary violations to escape local optima."""
        adj = list(adj)
        edge_count = edge_count
        best_adj = list(adj)
        best_count = edge_count
        
        T = 10.0
        cooling = 0.9999
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Random move: add or remove a random edge
            i = random.randint(0, N-1)
            j = random.randint(0, N-1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            has_edge = (adj[i] >> j) & 1
            
            if has_edge:
                # Remove edge
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                new_count = edge_count - 1
                new_violations = count_k33_violations(adj)
                delta_e = -1
            else:
                # Add edge
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                new_count = edge_count + 1
                new_violations = count_k33_violations(adj)
                delta_e = 1
            
            # Objective: maximize edges, penalize violations
            # Energy = -edge_count + lambda * violations
            # We want to accept moves that increase edge_count or reduce violations
            
            # Simple acceptance: if new state is K3,3-free and has more edges, accept
            # If new state has violations, accept with probability exp(-violations/T)
            
            if new_violations == 0:
                # No violations: accept if edge count increased
                if new_count >= edge_count:
                    edge_count = new_count
                    if edge_count > best_count:
                        best_count = edge_count
                        best_adj = list(adj)
                else:
                    # Revert
                    if has_edge:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
            else:
                # Has violations: accept with low probability
                prob = math.exp(-new_violations / max(T, 0.01))
                if random.random() < prob:
                    edge_count = new_count
                    if new_violations == 0 and edge_count > best_count:
                        best_count = edge_count
                        best_adj = list(adj)
                else:
                    # Revert
                    if has_edge:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
            
            T *= cooling
        
        return best_adj, best_count
    
    # Run multiple random restarts
    best_adj = None
    best_count = 0
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Greedy construction
        adj, edge_count = greedy_construct()
        
        if edge_count > best_count:
            best_count = edge_count
            best_adj = list(adj)
        
        # Local search improvement
        adj, edge_count = local_search(adj, edge_count, max_iters=10000)
        
        if edge_count > best_count:
            best_count = edge_count
            best_adj = list(adj)
        
        # Simulated annealing from best
        if time.time() < deadline - 15:
            sa_adj, sa_count = simulated_annealing(adj, edge_count, max_iters=20000)
            if sa_count > best_count:
                best_count = sa_count
                best_adj = list(sa_adj)
        
        # Final local search from SA result
        if time.time() < deadline - 8:
            adj, edge_count = local_search(adj, edge_count, max_iters=5000)
            if edge_count > best_count:
                best_count = edge_count
                best_adj = list(adj)
    
    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback: K2,18
        return [[a, b] for a in range(2) for b in range(2, 20)]
    
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

# EVOLVE-BLOCK-END
