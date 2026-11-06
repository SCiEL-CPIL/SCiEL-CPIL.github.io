import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin under 120s limit

    n = 40
    
    # adj[i] is an integer bitmask where bit j is 1 if edge (i,j) exists
    adj = [0] * n
    
    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u,v) to the current graph creates a C6.
        Returns True if it is safe to add (no C6 created), False otherwise.
        Optimized for speed using bitmasks.
        """
        # Neighbors of u excluding v
        nu_mask = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv_mask = adj[v] & ~(1 << u)
        
        # We need to check if there is a path of length 3 between u and v
        # i.e., u-a-b-c-d-v
        # a in N(u), d in N(v), a != d
        # path a-b-c-d exists in G \ {u, v}
        
        a = nu_mask
        while a:
            lb_a = a & (-a)
            ai = lb_a.bit_length() - 1
            a ^= lb_a
            
            d = nv_mask & ~lb_a  # d != a
            while d:
                ld_d = d & (-d)
                di = ld_d.bit_length() - 1
                d ^= ld_d
                
                # We need a path a-b-c-d
                # b must be in N(a) \ {u, v, d, a}
                # c must be in N(d) \ {u, v, a, d}
                # and b ~ c
                
                # Mask for b candidates
                b_candidates = adj[ai] & ~(1 << u) & ~(1 << v) & ~(1 << di) & ~(1 << ai)
                
                # Mask for c candidates
                c_candidates = adj[di] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << di)
                
                # Check if any neighbor of any b in b_candidates is in c_candidates
                # Let U = union of neighbors of all b in b_candidates
                # If U & c_candidates is non-empty, then a C6 exists.
                
                union_neighbors = 0
                bm = b_candidates
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    union_neighbors |= adj[bi]
                    
                    # Early exit if we already found a connection
                    if union_neighbors & c_candidates:
                        return False
            # end d loop
        # end a loop
        
        return True

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def is_c6_free(adj):
        """Verify the whole graph is C6-free. Used for final validation."""
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
        
        # Re-check every edge to ensure no C6 was formed by previous operations
        # This is a safety net. can_add_edge is incremental, but swaps might be tricky.
        # Actually, for swaps, we can just check the new edge against the graph with the old edge removed.
        return True # Placeholder, we rely on incremental checks

    # --- Initialization Strategy: Strongly Regular / High Degree Base ---
    # Instead of 8 K5s (degree 4), let's try a structure with higher average degree.
    # The Petersen graph is C6-free? No, Petersen has girth 5. 
    # We want to start with a graph that is likely C6-free but dense.
    # Let's use a random regular graph of degree 5 or 6, filtered for C6-freeness.
    # Or better: Start with a known C6-free construction.
    # The incidence graph of a projective plane of order 3 (12 vertices) is bipartite, girth 6? No, girth 6 means C6 exists.
    # Wait, "Hexagon-free" means NO C6.
    # Bipartite graphs with girth > 6 (i.e. >= 8) are C6-free.
    # Let's try to construct a bipartite graph with girth 8 on 40 vertices.
    # Or simply start with the 8 K5s (degree 4) but use Simulated Annealing to push it to degree 5+.
    
    # Let's stick to the 8 K5s base as it is guaranteed C6-free (disconnected components).
    # K5 is C3, C4, C5, C6-free? K5 has triangles. C6 is a cycle of length 6.
    # A K5 contains no C6 because it only has 5 vertices. So 8 disjoint K5s is valid.
    # Base edges: 8 * 10 = 80.
    
    base_adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
    
    current_adj = base_adj[:]
    current_count = count_edges(current_adj)
    best_adj = current_adj[:]
    best_count = current_count

    # Precompute all pairs for random selection
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Helper to try adding an edge greedily to a copy
    def try_greedy_add(adj_copy, deadline):
        non_edges = [(u, v) for u, v in all_pairs if not ((adj_copy[u] >> v) & 1)]
        random.shuffle(non_edges)
        added = False
        for u, v in non_edges:
            if time.time() > deadline - 2:
                break
            if not ((adj_copy[u] >> v) & 1):
                if can_add_edge(adj_copy, u, v):
                    adj_copy[u] |= (1 << v)
                    adj_copy[v] |= (1 << u)
                    added = True
        return added

    # Initial Greedy Saturation
    try_greedy_add(current_adj, deadline)
    current_count = count_edges(current_adj)
    if current_count > best_count:
        best_count = current_count
        best_adj = current_adj[:]

    # Simulated Annealing Loop
    # Move: Pick a random existing edge (u,v) to remove, and a random non-edge (x,y) to add.
    # Check if the new graph is C6-free.
    # If yes, accept if better or with probability exp(-delta/T).
    # If no, reject.
    
    step = 0
    while time.time() < deadline - 5:
        step += 1
        
        # Get current edges and non-edges
        current_edges = []
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    current_edges.append((u, v))
                else:
                    non_edges.append((u, v))
        
        if not current_edges or not non_edges:
            break
            
        # Random Swap: Remove (u1, v1), Add (u2, v2)
        u1, v1 = random.choice(current_edges)
        u2, v2 = random.choice(non_edges)
        
        # Ensure we don't swap an edge with itself (impossible since one is in, one is out)
        # Create trial adjacency
        trial_adj = current_adj[:]
        
        # Remove u1-v1
        trial_adj[u1] &= ~(1 << v1)
        trial_adj[v1] &= ~(1 << u1)
        
        # Add u2-v2
        # Check if adding u2-v2 is safe in the graph WITHOUT u1-v1
        if can_add_edge(trial_adj, u2, v2):
            trial_adj[u2] |= (1 << v2)
            trial_adj[v2] |= (1 << u2)
            
            # Count edges is same (swap), so we need to evaluate "potential" or just accept?
            # Actually, a swap keeps edge count constant. 
            # The benefit is that it might allow *subsequent* additions.
            # So we should accept swaps that lead to a state where *more* edges can be added?
            # That's expensive to check every step.
            
            # Alternative Strategy: 
            # 1. Perform Swap.
            # 2. Try to Greedy Add.
            # 3. If new count > old count, accept.
            # 4. If new count == old count, accept with prob exp(-0/T) = 1? 
            #    No, if count is same, we might be wandering. 
            #    Let's use a "potential" metric: number of valid non-edges (edges that CAN be added).
            
            # Let's calculate "openness": how many non-edges can be added?
            # This is expensive.
            
            # Simpler SA: 
            # Accept if the swap itself is valid (C6-free).
            # Then periodically try greedy add.
            
            # Let's try: Accept swap if valid. 
            # Every 10 steps, do a greedy add pass.
            
            # Delta E = 0 for the swap itself.
            # But we want to maximize edges.
            # So, let's do:
            # 1. Swap.
            # 2. Greedy Add on trial.
            # 3. If trial_count > current_count: Accept.
            # 4. If trial_count == current_count: Accept with prob 0.5 (to explore).
            # 5. If trial_count < current_count: Reject.
            
            trial_count = count_edges(trial_adj)
            
            # Since we only added one edge in the swap step (net 0 change from swap, but we check validity),
            # trial_count is currently current_count.
            
            # Let's actually run the greedy add on the trial graph to see if it unlocks anything.
            # To save time, we only do this for every Nth step or if we are stuck.
            
            if step % 5 == 0:
                # Deep check: Greedy add
                trial_adj_greedy = trial_adj[:]
                try_greedy_add(trial_adj_greedy, deadline)
                trial_greedy_count = count_edges(trial_adj_greedy)
                
                if trial_greedy_count > current_count:
                    # Accept
                    current_adj = trial_adj_greedy
                    current_count = trial_greedy_count
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = current_adj[:]
                        if best_count >= 105:
                            break
                else:
                    # Reject or accept with low prob to escape?
                    # If we reject, we stay in local minimum.
                    # If we accept, we change structure without gaining edges.
                    # Let's accept with small probability to allow wandering.
                    if random.random() < 0.1:
                        current_adj = trial_adj
                        # current_count stays same
            else:
                # Shallow check: Just accept the valid swap to mix the graph
                # This changes the topology, potentially making other edges addable later.
                # We accept if valid.
                current_adj = trial_adj
                # current_count is unchanged
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final Greedy Pass on Best
    final_adj = best_adj[:]
    try_greedy_add(final_adj, deadline)
    final_count = count_edges(final_adj)
    
    if final_count > best_count:
        best_count = final_count
        best_adj = final_adj

    return get_edge_list(best_adj)

# Fallback if construct fails or times out unexpectedly? 
# The function handles its own deadline.

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Edges found: {len(edges)}")
