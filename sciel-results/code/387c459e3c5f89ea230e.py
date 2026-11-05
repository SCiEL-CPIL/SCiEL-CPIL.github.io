"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = 20
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Map edge index to (u, v)
    edge_to_uv = {i: all_edges[i] for i in range(num_edges)}
    uv_to_edge = {all_edges[i]: i for i in range(num_edges)}
    
    # Precompute the "conflict" graph.
    # Two edges e1=(u1, v1) and e2=(u2, v2) conflict if adding them to a set
    # creates a K2,3 subgraph.
    # A K2,3 subgraph consists of two vertices {a, b} and three common neighbors {c, d, e}.
    # The edges involved are (a,c), (a,d), (a,e), (b,c), (b,d), (b,e).
    # Actually, the constraint is: for ANY pair of vertices (a, b), they can share AT MOST 2 neighbors.
    # If we select a set of edges S, let N_S(v) be the neighbors of v in S.
    # The condition is |N_S(a) ∩ N_S(b)| <= 2 for all a != b.
    
    # This is a global constraint on the set of edges. It's not just pairwise edge conflicts.
    # However, we can view it as: if we already have edges such that a and b share 2 neighbors,
    # we cannot add any edge (a, k) or (b, k) where k is a common neighbor? No, that's not right.
    # We cannot add an edge (a, k) if b is already a neighbor of a and k is already a neighbor of b?
    # Let's re-evaluate.
    # If we add edge (a, k), it increases the common neighbor count for pairs (a, x) where x is a neighbor of k?
    # No. Adding (a, k) makes k a neighbor of a.
    # For any vertex b that is a neighbor of a, k becomes a common neighbor of a and b.
    # So, adding (a, k) is invalid if there exists a vertex b such that:
    # 1. (a, b) is in S (b is neighbor of a)
    # 2. (b, k) is in S (k is neighbor of b)
    # 3. The current number of common neighbors of a and b is already 2.
    # OR, if (a, b) is NOT in S, but we are building the set incrementally?
    
    # This dependency is complex for a simple pairwise conflict graph because the validity of adding
    # an edge depends on the current state of the graph (specifically, the common neighbor counts).
    # Therefore, a static MWIS formulation on pairwise conflicts is NOT sufficient unless we define
    # conflicts dynamically or use a different formulation.
    
    # Alternative: Use a greedy algorithm with sophisticated backtracking (DFS) and pruning.
    # Or, use the fact that K2,3-free graphs are related to polarity graphs or finite geometries?
    # For n=20, the maximum number of edges is known to be around 100-110.
    # The seed program achieves ~100 (score 1.8 implies value/seed = 1.8, if seed is ~55? No, seed is likely low).
    # Wait, score 1.8 means the value is 1.8 times the seed value.
    # The task says "Maximize the number of edges".
    # Let's check the seed value. If the seed program is random, it might get ~50-60 edges.
    # 1.8 * 55 = 99.
    # If the seed is the initial random graph, it might be lower.
    # Let's assume the target is to maximize edges.
    
    # Strategy: 
    # 1. Start with a strong heuristic initial solution (e.g., greedy with randomization + local search).
    # 2. Use a "Swap" based local search: try to replace one edge with two edges, or remove one to add two.
    # 3. To break stagnation, use a "kick" mechanism: randomly remove a few edges and re-add greedily.
    # 4. Use a fast validity checker using bitsets.
    
    # Bitset representation for neighbors
    # adj[u] is a Python int representing the set of neighbors of u.
    
    def count_bits(x):
        return bin(x).count('1')
    
    def is_valid_adj(adj):
        # Check all pairs (u, v)
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if count_bits(common) > 2:
                    return False
        return True
    
    def can_add_edge(adj, u, v):
        # Check if adding (u, v) keeps the graph valid.
        # Adding (u, v) means v becomes a neighbor of u and u becomes a neighbor of v.
        # This affects common neighbor counts for pairs (u, x) and (v, x).
        # Specifically, for any x such that (u, x) is an edge, v becomes a common neighbor of u and x.
        # So we must ensure that for all x in N(u), the number of common neighbors of u and x is < 2 
        # BEFORE adding v to N(u)? 
        # Wait. If we add (u, v), then v is in N(u).
        # For any x in N(u) (excluding v, since (u,v) is new), the set of common neighbors of u and x gains v.
        # So we need: for all x in N(u), |N(u) ∩ N(x)| < 2 is not quite right.
        # We need: | (N(u) ∪ {v}) ∩ N(x) | <= 2.
        # Since v is not in N(x) (assuming (v,x) is not an edge? If (v,x) IS an edge, then v is already in N(x)).
        # Case 1: (v, x) is not an edge. Then v is not in N(x). 
        #   New common neighbors = (N(u) ∩ N(x)) ∪ {v} (if v in N(u), which it is now).
        #   Wait, v is in N(u). Is v in N(x)? No.
        #   So new common neighbors = (N(u) ∩ N(x)) ∪ {v}? No.
        #   Common neighbors of u and x are vertices w such that (u,w) and (x,w) are edges.
        #   After adding (u,v), w can be v. So v is a common neighbor of u and x IF (x,v) is an edge.
        #   If (x,v) is NOT an edge, v is NOT a common neighbor of u and x.
        #   So adding (u,v) only increases common neighbor counts for pairs (u, x) where (x,v) is an edge?
        #   No. Let's trace carefully.
        #   Let S be the current set of edges.
        #   We add (u, v).
        #   New edge set S' = S ∪ {(u, v)}.
        #   Consider a pair (a, b).
        #   Common neighbors in S': CN'(a, b) = { w | (a,w) in S' and (b,w) in S' }.
        #   If w is not v, (a,w) in S' iff (a,w) in S. Same for (b,w).
        #   So for w != v, w is in CN'(a, b) iff w is in CN(a, b).
        #   If w is v:
        #     v is in CN'(a, b) iff (a,v) in S' and (b,v) in S'.
        #     (a,v) in S' is true if (a,v) in S OR a==u (since we added (u,v)).
        #     (b,v) in S' is true if (b,v) in S OR b==u.
        #     
        #   So v is a new common neighbor for pairs (a, b) if:
        #   1. a == u and (b, v) in S. (Pair (u, b) where b is a neighbor of v)
        #   2. b == u and (a, v) in S. (Pair (a, u) where a is a neighbor of v)
        #   3. (a, v) in S and (b, v) in S and a != u, b != u? 
        #      If a != u and b != u, then (a,v) in S' iff (a,v) in S.
        #      So if (a,v) and (b,v) were already in S, v was already a common neighbor.
        #      So no change for pairs not involving u.
        #      
        #   Conclusion: Adding (u, v) ONLY affects the common neighbor counts for pairs involving u.
        #   Specifically, for each neighbor b of v (i.e., (b, v) in S), the pair (u, b) gains a common neighbor v.
        #   Also, for each neighbor a of v, the pair (a, u) gains a common neighbor v. (Same set of pairs).
        #   So we need to check: for all b such that (b, v) in S, the current number of common neighbors of (u, b) must be <= 1.
        #   Because adding v makes it count + 1. If it was 2, it becomes 3 (invalid).
        #   Also, we must ensure that u and v do not already have 3 common neighbors? 
        #   If (u, v) is added, u and v become neighbors. 
        #   Do we care about common neighbors of neighbors? The constraint is for ANY two vertices.
        #   The pair (u, v) itself: their common neighbors are w such that (u,w) and (v,w) are edges.
        #   Adding (u,v) doesn't change the set of common neighbors of u and v, because w must be distinct from u and v.
        #   Wait, if w=u, (v,u) is the new edge, but (u,u) is not an edge. So u is not a common neighbor.
        #   So the common neighbor count for the pair (u, v) is unchanged by adding (u, v).
        #   
        #   Therefore, the condition to add (u, v) is:
        #   For all b in N(v), |CN(u, b)| <= 1.
        #   AND (u, v) is not already in S.
        
        if adj[u] & (1 << v):
            return False # Already exists
            
        # Check all neighbors b of v
        # N(v) is adj[v]
        # For each b in N(v), we need count_bits(adj[u] & adj[b]) <= 1
        # Note: adj[u] does NOT contain v yet.
        # The common neighbors of u and b are in adj[u] & adj[b].
        # If any of these counts is 2, adding v (which is a neighbor of b) will make v a common neighbor of u and b?
        # Wait. Is v a neighbor of u? Yes, we are adding it.
        # Is v a neighbor of b? Yes, b is in N(v).
        # So v is a common neighbor of u and b.
        # So the new count is |adj[u] & adj[b]| + 1.
        # We need |adj[u] & adj[b]| + 1 <= 2  => |adj[u] & adj[b]| <= 1.
        
        neighbors_v = adj[v]
        # Iterate through bits of neighbors_v
        # To be fast, we can check if there is any b in N(v) such that count_bits(adj[u] & adj[b]) >= 2.
        
        # Optimization: Precompute for each u, a bitmask of "forbidden" vertices?
        # No, let's just iterate. n=20 is small.
        # Max degree is ~10. Iteration is fast.
        
        temp_v = neighbors_v
        while temp_v:
            # Extract lowest set bit
            lsb = temp_v & -temp_v
            b = lsb.bit_length() - 1
            # Check count of common neighbors of u and b
            if count_bits(adj[u] & adj[b]) > 1:
                return False
            temp_v ^= lsb
            
        return True

    # Greedy construction with randomization
    def greedy_construction(seed_val):
        random.seed(seed_val)
        adj = [0] * n
        # Shuffle edges
        edges_list = list(range(num_edges))
        random.shuffle(edges_list)
        
        for e_idx in edges_list:
            u, v = edge_to_uv[e_idx]
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    # Local Search: Try to swap edges to increase count
    def local_search(adj, time_limit=10.0):
        start_ls = time.time()
        adj = adj.copy()
        # Calculate current score
        score = sum(count_bits(a) for a in adj) // 2
        
        # Precompute list of non-edges
        # This is expensive to update. Let's just iterate all pairs.
        
        improved = True
        while improved and (time.time() - start_ls) < time_limit and (time.time() - start_time) < deadline:
            improved = False
            # Try to add any edge
            # Random order to break symmetry
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if (time.time() - start_ls) > time_limit:
                    break
                if not (adj[u] & (1 << v)):
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        score += 1
                        improved = True
            
            if not improved:
                # Try swaps: remove one edge, add two?
                # Or remove one, add one (neutral) to move in search space?
                # Let's try a "kick": remove a random edge and re-add greedily?
                # Or try to find a 2-swap: remove (a,b), add (c,d) and (e,f)?
                # Simplest kick: remove a random edge, then run greedy add again.
                if score > 0:
                    # Pick a random edge to remove
                    # Get a random u
                    u = random.randint(0, n-1)
                    if adj[u]:
                        v_bit = (adj[u] & -adj[u])
                        v = v_bit.bit_length() - 1
                        # Remove (u, v)
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        score -= 1
                        improved = True # Force loop to continue and try to add back
                
        return adj, score

    # Main loop
    best_adj = [0] * n
    best_score = 0
    
    # Phase 1: Multiple Greedy Initializations
    num_inits = 20
    for i in range(num_inits):
        if time.time() > deadline - 30:
            break
        adj = greedy_construction(i)
        score = sum(count_bits(a) for a in adj) // 2
        if score > best_score:
            best_score = score
            best_adj = adj.copy()
        
        # Quick local search on this init
        adj_ls, score_ls = local_search(adj, time_limit=2.0)
        if score_ls > best_score:
            best_score = score_ls
            best_adj = adj_ls.copy()

    # Phase 2: Intensive Local Search on Best
    if time.time() < deadline - 10:
        # Run longer local search on the best solution
        adj_ls, score_ls = local_search(best_adj, time_limit=10.0)
        if score_ls > best_score:
            best_score = score_ls
            best_adj = adj_ls.copy()
            
        # Try "Reheating" or "Restart" from best with perturbation
        # Perturb best_adj: remove 5 random edges, then local search
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            perturbed_adj = best_adj.copy()
            # Remove 5 random edges
            for _ in range(5):
                u = random.randint(0, n-1)
                if perturbed_adj[u]:
                    v_bit = (perturbed_adj[u] & -perturbed_adj[u])
                    v = v_bit.bit_length() - 1
                    perturbed_adj[u] &= ~(1 << v)
                    perturbed_adj[v] &= ~(1 << u)
            
            adj_ls, score_ls = local_search(perturbed_adj, time_limit=5.0)
            if score_ls > best_score:
                best_score = score_ls
                best_adj = adj_ls.copy()

    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
# EVOLVE-BLOCK-END
