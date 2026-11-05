import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    n = 20
    vertices = list(range(n))
    
    # Precompute pairs for fast iteration
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Helper to check K3,3-free property
    # A graph contains K3,3 if and only if there exist two disjoint sets of 3 vertices 
    # such that every vertex in one set is connected to every vertex in the other.
    # However, the prompt defines it as: "no three vertices have three common neighbours".
    # This is equivalent to: For any triple {u, v, w}, |N(u) ∩ N(v) ∩ N(w)| < 3.
    # Note: This is a stronger condition than just being K3,3-free? 
    # Actually, K3,3 is a subgraph. If there is a K3,3, let the partitions be A={a1,a2,a3} and B={b1,b2,b3}.
    # Then a1, a2, a3 have common neighbors b1, b2, b3. So |N(a1) ∩ N(a2) ∩ N(a3)| >= 3.
    # Conversely, if some triple {u,v,w} has 3 common neighbors {x,y,z}, then {u,v,w} and {x,y,z} form a K3,3 subgraph.
    # So the condition is exactly: max over all triples T of |∩_{v in T} N(v)| <= 2.
    
    # Optimization: Instead of checking all triples (1140 of them) for every move,
    # we can update the "common neighbor counts" incrementally or just check the affected triples.
    # When adding edge (u, v), the common neighbor sets of triples containing u or v change.
    # Specifically, for any triple T containing u, the new common neighbors are old_common ∩ N(v) (if v not in T) 
    # or something similar.
    # Actually, it's easier to just re-evaluate the validity of the graph efficiently.
    # Let's maintain a list of "violating triples" or just a function that checks validity.
    # Given n=20, 1140 triples. Checking all is fast (~1140 * bit_count operations).
    # Let's stick to a robust check but optimize the local search.
    
    def is_valid(adj):
        """Check if the graph represented by adj bitmasks is K3,3-free."""
        # We need to ensure no 3 vertices have 3 common neighbors.
        # Iterate over all triples (i, j, k)
        # To speed up, we can precompute common neighbors for all pairs?
        # There are 190 pairs. common[i][j] = adj[i] & adj[j].
        # Then for triple (i,j,k), common_ijk = common[i][j] & adj[k].
        # If bit_count(common_ijk) >= 3, invalid.
        
        # Let's just do it directly. 1140 iterations is nothing.
        for i in range(n):
            ai = adj[i]
            for j in range(i+1, n):
                cij = ai & adj[j]
                if cij.bit_count() < 3: 
                    # If i and j have fewer than 3 common neighbors, no triple containing i,j can have 3 common neighbors
                    # because common_ijk is a subset of cij.
                    continue
                for k in range(j+1, n):
                    cijk = cij & adj[k]
                    if cijk.bit_count() >= 3:
                        return False
        return True

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    # Initial Strategy: Paley Graph on 19 vertices + 1 isolated vertex?
    # Paley(19) is a strongly regular graph (19, 9, 4, 4).
    # Let's check if Paley(19) is K3,3-free.
    # In SRG(v, k, lambda, mu), any two adjacent vertices have lambda common neighbors, non-adjacent have mu.
    # For Paley(19): k=9, lambda=4, mu=4.
    # Take a triple {u, v, w}.
    # Case 1: u, v, w form a triangle. 
    #   N(u) ∩ N(v) has size 4. One of these is w. The other 3 are common neighbors of u,v.
    #   Do they share common neighbors with w?
    #   This requires detailed structure.
    #   Generally, Paley graphs are not necessarily K3,3-free.
    #   However, it's a good dense starting point.
    
    # Let's try a different construction: 
    # The complement of the line graph of K6? No.
    # Let's just use a random restart with a strong local search.
    # The current program's weakness is likely the "try_add_edge" which is O(N^3) per attempt? 
    # No, is_k33_free is O(N^3). 
    # In the current code, try_add_edge calls is_k33_free.
    # Greedy phase: 190 edges * 1140 triples * bit_count. ~200k ops. Very fast.
    # Local search: Swaps. 
    # The current code does random swaps. This is inefficient.
    # Better: Systematic Best-Improvement Local Search.
    # 1. Start with a dense graph (e.g. Paley(19) + 1 node connected to all? or just random dense).
    # 2. If invalid, remove edges to make it valid.
    # 3. Try to add edges (best improvement).
    # 4. Try to swap edges (remove one, add one) to increase density while maintaining validity.
    
    def generate_paley_19():
        # Vertices 0..18
        adj = [0] * 19
        # Quadratic residues mod 19
        # x^2 mod 19 for x in 1..9
        residues = set((x*x) % 19 for x in range(1, 10))
        for u in range(19):
            for v in range(u+1, 19):
                if (v - u) % 19 in residues or (u - v) % 19 in residues:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Extend to 20 vertices: vertex 19 is isolated initially
        adj20 = adj[:] + [0]
        return adj20

    best_adj = [0] * n
    best_count = 0
    best_edges = []
    
    def update_best(adj):
        nonlocal best_adj, best_count, best_edges
        c = count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]
            best_edges = adj_to_edges(adj)

    # Strategy 1: Start with Paley(19) + isolated node, then optimize
    # Strategy 2: Start with random graph, optimize
    
    random.seed(42)
    
    # Run multiple optimizations with different seeds/starts
    # Time budget per start: ~10s. 10 starts = 100s.
    
    num_starts = 10
    for s in range(num_starts):
        if time.time() > deadline - 20:
            break
            
        # Initialize adj
        if s == 0:
            adj = generate_paley_19()
        else:
            # Random graph with edge probability 0.5
            adj = [0] * n
            for i, j in all_pairs:
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Phase 1: Make it valid by removing edges
        # Find violating triples and remove an edge from one of them
        valid = False
        attempts = 0
        while not valid and time.time() < deadline - 20:
            valid = True
            # Find first violation
            viol_u, viol_v, viol_w, viol_nbs = -1, -1, -1, 0
            for i in range(n):
                ai = adj[i]
                for j in range(i+1, n):
                    cij = ai & adj[j]
                    if cij.bit_count() < 3: continue
                    for k in range(j+1, n):
                        cijk = cij & adj[k]
                        if cijk.bit_count() >= 3:
                            valid = False
                            viol_u, viol_v, viol_w = i, j, k
                            viol_nbs = cijk
                            break
                    if not valid: break
                if not valid: break
            if not valid:
                # Remove an edge incident to one of the violating nodes or between common neighbors?
                # Removing an edge (u, x) where x is in N(u) will reduce common neighbors.
                # Let's remove a random edge incident to viol_u, viol_v, or viol_w
                # Or better, remove an edge between a common neighbor and one of the triple?
                # If we remove edge (viol_u, x) where x is in viol_nbs, then x is no longer common neighbor.
                # Let's pick a random x from viol_nbs and remove (viol_u, x)
                x = (viol_nbs & (-viol_nbs)).bit_length() - 1
                adj[viol_u] &= ~(1 << x)
                adj[x] &= ~(1 << viol_u)
                attempts += 1
                if attempts > 1000: break # Safety
        
        if not valid:
            continue
            
        # Phase 2: Local Search - Add edges
        improved = True
        while improved and time.time() < deadline - 20:
            improved = False
            # Try adding each non-edge
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                # Try adding
                old_u = adj[u]
                old_v = adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    improved = True
                    # Don't break, keep adding if possible? 
                    # Best improvement would pick the one that allows most future adds.
                    # Simple: add it and restart loop (since graph changed).
                    break
                else:
                    adj[u] = old_u
                    adj[v] = old_v
        
        # Phase 3: Local Search - Swaps
        # Try to remove an edge and add a different edge to increase total? 
        # Swapping doesn't change edge count. 
        # We want to *increase* edge count.
        # So we should look for "2-exchanges": Remove 1 edge, Add 2 edges? No, that changes count by +1.
        # Or Remove 2, Add 3?
        # Standard local search for max edges with constraint:
        # 1. Try adding 1 edge (done above).
        # 2. Try removing 1 edge and adding 1 edge (swap) - doesn't help count directly, but might unlock future adds.
        # 3. Try removing 1 edge and adding 2 edges (net +1).
        
        # Let's implement a "Remove 1, Add 2" search.
        # For each edge e to remove, and each pair of non-edges f, g to add.
        # This is expensive. 
        # Alternative: Just do more "Add 1" with random restarts of the local search state?
        # Or: "Add 1" is stuck. Try "Remove 1" (randomly) then "Add 1" multiple times?
        
        # Let's try a specific swap: Remove edge (u,v), Add edge (x,y).
        # This keeps count same. But it might change the structure to allow more adds later.
        # Let's do a few rounds of random swaps if stuck.
        
        stuck_rounds = 0
        while time.time() < deadline - 20 and stuck_rounds < 20:
            # Try random swaps
            edges_in_graph = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
            
            if not non_edges:
                break
                
            random.shuffle(edges_in_graph)
            random.shuffle(non_edges)
            
            found_swap = False
            for e_remove in edges_in_graph[:10]: # Limit checks
                for e_add in non_edges[:10]:
                    u1, v1 = e_remove
                    u2, v2 = e_add
                    if u1 == u2 and v1 == v2: continue
                    
                    old = (adj[u1], adj[v1], adj[u2], adj[v2])
                    # Remove
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    # Add
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    
                    if is_valid(adj):
                        found_swap = True
                        break
                    else:
                        adj[u1], adj[v1], adj[u2], adj[v2] = old
                if found_swap:
                    break
            
            if not found_swap:
                stuck_rounds += 1
            else:
                stuck_rounds = 0
                # After a swap, try to add more edges
                improved = True
                while improved and time.time() < deadline - 20:
                    improved = False
                    for u, v in all_pairs:
                        if adj[u] & (1 << v): continue
                        old_u, old_v = adj[u], adj[v]
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if is_valid(adj):
                            improved = True
                            break
                        else:
                            adj[u], adj[v] = old_u, old_v
        
        update_best(adj)
        
    # Fallback
    if not best_edges:
        # Return a simple valid graph, e.g. bipartite K10,10 is not K3,3 free?
        # K10,10 contains K3,3.
        # Use a sparse graph.
        return [[i, i+1] for i in range(19)]
        
    return best_edges

# EVOLVE-BLOCK-END
