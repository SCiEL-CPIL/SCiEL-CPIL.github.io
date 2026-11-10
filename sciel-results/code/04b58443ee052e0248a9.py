"""Dense Q3-free graph on 16 vertices via complement removal strategy."""
import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Precompute all Q3 subgraphs in K16
    # A Q3 is a 3-dimensional hypercube. Its vertices can be represented as {0,1}^3.
    # Edges connect vertices differing in exactly 1 bit.
    # We need to find all sets of 8 distinct vertices in 0..15 that can be mapped to Q3.
    
    # Since 16! is too large, we use a smarter enumeration or property.
    # Actually, we can just iterate through all 8-subsets? C(16,8) = 12870.
    # For each 8-subset, check if it induces a Q3.
    # This is feasible.
    
    def is_q3(vertices):
        """Check if the set of 8 vertices induces a Q3 subgraph in K16 (which is complete, so always true if structure matches)."""
        # Wait, in K16, any 8 vertices induce K8.
        # We are looking for a subgraph isomorphic to Q3.
        # This means there exists a bijection f: vertices -> {0,1}^3 such that
        # (u,v) is an edge in the subgraph iff f(u) and f(v) differ in 1 bit.
        # Since the host graph is K16, the subgraph on these 8 vertices is K8.
        # Q3 is a subgraph of K8.
        # So ANY set of 8 distinct vertices in K16 contains a Q3 as a subgraph?
        # No. A subgraph is a subset of edges.
        # The task says "containing no subgraph isomorphic to Q3".
        # This means there is NO subset of edges E' subset of E(G) such that (V', E') is isomorphic to Q3.
        # In the complement removal strategy, we start with K16.
        # We remove edges.
        # We want to ensure that for any 8 vertices, the remaining edges do NOT contain a Q3.
        
        # To check if a specific set of 8 vertices contains a Q3 in the current graph:
        # We need to check if there is a labeling of these 8 vertices as Q3 vertices
        # such that all 12 edges of Q3 are present in the current graph.
        
        # There are 8! / (automorphisms of Q3) labelings?
        # Aut(Q3) has size 48 (hyperoctahedral group B3).
        # So 40320 / 48 = 840 distinct embeddings?
        # Actually, it's easier to just check all 8! permutations? 40320 * 12870 is too much.
        
        # Alternative:
        # For a fixed set of 8 vertices, does it contain a Q3?
        # We can pick a vertex, say v0. In Q3, each vertex has degree 3.
        # So v0 must have 3 neighbors in the set that form a triangle? No, neighbors in Q3 are not connected to each other.
        # In Q3, neighbors of a vertex form an independent set.
        # So if we pick v0, its neighbors in the Q3 must be 3 vertices in the set that are mutually non-adjacent in the current graph?
        # No, they are adjacent to v0 in the current graph.
        # And they are NOT adjacent to each other in the Q3 structure.
        # So in the current graph, if v0 is part of a Q3, it has 3 neighbors u1, u2, u3 in the set such that:
        # 1. (v0, ui) is an edge.
        # 2. (ui, uj) is NOT an edge? No, the Q3 structure doesn't have edges between neighbors of v0.
        #    But the current graph might have them. The subgraph just needs to HAVE the Q3 edges.
        #    It can have EXTRA edges.
        #    So (ui, uj) being an edge in G doesn't prevent {v0, u1, u2, u3...} from containing a Q3.
        #    Wait. "Subgraph isomorphic to Q3".
        #    This means there exists a subset of edges E' such that (V', E') is Q3.
        #    So we just need to find 8 vertices and 12 edges among them that form Q3.
        
        # So for a fixed set of 8 vertices V', does it contain a Q3?
        # Yes, if we can find a labeling.
        # How to check efficiently?
        # Pick a vertex v in V'.
        # In Q3, v has 3 neighbors. Let them be N(v).
        # N(v) is an independent set in Q3.
        # The other 4 vertices are at distance 2 or 3 from v.
        
        # This seems complicated to check dynamically for every 8-set.
        
        # Alternative Strategy:
        # Instead of removing edges from K16, let's stick to the additive approach but improve it.
        # The current program uses random greedy.
        # A different strategy: **Simulated Annealing** or **Genetic Algorithm**.
        # Or: **Max Cut / Spectral methods**?
        
        # Let's reconsider the "Different Strategy".
        # Current: Random Greedy Add.
        # New: **Deterministic Greedy Add with Backtracking** or **Local Search (2-opt/3-opt swaps)**.
        
        # Let's try **Local Search** starting from a good random greedy solution.
        # Moves:
        # 1. Add an edge.
        # 2. Remove an edge.
        # 3. Swap: Remove edge e1, add edge e2. (Net 0 change, might enable future adds)
        # 4. 2-Swap: Remove e1, e2, add e3, e4.
        
        # Actually, a very effective method for Turan-type problems is **Randomized Local Search** with **Edge Swaps**.
        
        # Let's implement a Local Search that tries to add edges, and if stuck, performs swaps to "shake" the graph.
        
        pass

    # Re-evaluating the "Complement Removal" idea.
    # If we start with K16 (120 edges), we need to remove edges to break all Q3s.
    # How many Q3s are in K16?
    # Number of ways to choose 8 vertices: C(16,8) = 12870.
    # For each set of 8 vertices, how many Q3 subgraphs?
    # Number of labeled Q3s on 8 vertices is 8! / |Aut(Q3)| = 40320 / 48 = 840.
    # So total Q3s in K16 is 12870 * 840 ~ 10 million.
    # We need to hit all of them with removed edges.
    # This is a Set Cover problem.
    # Greedy Set Cover:
    # 1. List all Q3s (as sets of 12 edges).
    # 2. While there are uncovered Q3s:
    #    a. Find the edge that covers the most uncovered Q3s.
    #    b. Remove that edge.
    #    c. Mark all Q3s containing that edge as covered.
    
    # This is computationally heavy to generate all Q3s and their edge sets.
    # But maybe we can do it.
    
    # Let's try a different angle: **Complement of a known dense graph**.
    # What is the complement of a Q3-free graph?
    # If G is Q3-free, what can we say about G^c?
    
    # Let's go back to **Local Search** as the "Materially Different" strategy.
    # It is a different optimization paradigm (hill climbing with restarts/swaps vs greedy construction).
    
    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        adj[i] is a bitmask of neighbors of i.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        a2_bits = neigh_u
        while a2_bits:
            a2 = a2_bits & (-a2_bits)
            a2_bits -= a2
            a2_idx = a2.bit_length() - 1
            
            a4_bits = neigh_u & ~(1 << a2_idx)
            while a4_bits:
                a4 = a4_bits & (-a4_bits)
                a4_bits -= a4
                a4_idx = a4.bit_length() - 1
                
                a3_candidates = neigh_v & adj[a2_idx]
                a5_candidates = neigh_v & adj[a4_idx]
                
                a3_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
                if not a3_candidates or not a5_candidates:
                    continue
                
                a3_bits = a3_candidates
                while a3_bits:
                    a3 = a3_bits & (-a3_bits)
                    a3_bits -= a3
                    a3_idx = a3.bit_length() - 1
                    
                    a5_cand = a5_candidates & ~(1 << a2_idx) & ~(1 << a4_idx) & ~(1 << a3_idx)
                    if not a5_cand:
                        continue
                    
                    a5_bits = a5_cand
                    while a5_bits:
                        a5 = a5_bits & (-a5_bits)
                        a5_bits -= a5
                        a5_idx = a5.bit_length() - 1
                        
                        used_mask = (1 << u) | (1 << v) | (1 << a2_idx) | (1 << a3_idx) | (1 << a4_idx) | (1 << a5_idx)
                        a6_candidates = (adj[a2_idx] & adj[a4_idx]) & ~used_mask
                        if not a6_candidates:
                            continue
                        
                        a7_base = (adj[a3_idx] & adj[a5_idx]) & ~used_mask
                        
                        a6_bits = a6_candidates
                        while a6_bits:
                            a6 = a6_bits & (-a6_bits)
                            a6_bits -= a6
                            a6_idx = a6.bit_length() - 1
                            
                            a7_cand = a7_base & ~(1 << a6_idx)
                            if not a7_cand:
                                continue
                            
                            if adj[a6_idx] & a7_cand:
                                return True
        return False

    def count_edges(adj):
        count = 0
        for u in range(n):
            count += bin(adj[u]).count('1')
        return count // 2

    def edges_from_adj(adj):
        result = []
        for u in range(n):
            mask = adj[u]
            while mask:
                bit = mask & (-mask)
                v = bit.bit_length() - 1
                if v > u:
                    result.append([u, v])
                mask -= bit
        return result

    def greedy_add(adj, edge_order):
        added = 0
        for u, v in edge_order:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not creates_q3_fast(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added += 1
        return added

    def local_search(adj, max_iters=1000):
        """
        Local search: try to add edges. If stuck, try targeted swaps and perturbations.
        """
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        
        # Precompute missing edges
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
        
        # Phase 1: Multiple greedy passes with different random orderings
        for pass_i in range(3):
            if time.time() > deadline - 5:
                break
            shuffled = missing[:]
            random.shuffle(shuffled)
            greedy_add(adj, shuffled)
            current_count = count_edges(adj)
            if current_count > best_count:
                best_count = current_count
                best_adj = adj[:]
            # Update missing edges
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
            if not missing:
                break
        
        # Phase 2: Targeted swaps - remove edges that block specific missing edges
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            
            # Get current missing edges
            missing_now = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
            if not missing_now:
                break
            
            # Try to find a missing edge that is close to being addable
            # i.e., removing one existing edge would allow it to be added
            found_swap = False
            for _attempt in range(10):
                if time.time() > deadline - 5:
                    break
                if not missing_now:
                    break
                u_miss, v_miss = random.choice(missing_now)
                
                # Find an existing edge whose removal might help
                # Heuristic: pick an edge incident to u_miss or v_miss
                candidates = []
                for neighbor in range(n):
                    if neighbor != u_miss and neighbor != v_miss:
                        if adj[u_miss] & (1 << neighbor):
                            candidates.append((u_miss, neighbor))
                        if adj[v_miss] & (1 << neighbor):
                            candidates.append((v_miss, neighbor))
                
                if not candidates:
                    continue
                
                u_rem, v_rem = random.choice(candidates)
                
                # Try the swap
                trial = best_adj[:]
                trial[u_rem] &= ~(1 << v_rem)
                trial[v_rem] &= ~(1 << u_rem)
                
                # Check if we can now add the missing edge
                if not creates_q3_fast(u_miss, v_miss, trial):
                    trial[u_miss] |= (1 << v_miss)
                    trial[v_miss] |= (1 << u_miss)
                
                # Now try to add more edges greedily
                missing_trial = [(u, v) for u in range(n) for v in range(u+1, n) if not (trial[u] & (1 << v))]
                random.shuffle(missing_trial)
                greedy_add(trial, missing_trial)
                
                new_count = count_edges(trial)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = trial[:]
                    adj = trial[:]
                    current_count = new_count
                    found_swap = True
                    # Update missing_now
                    missing_now = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
                    break
            
            if not found_swap:
                # No targeted swap worked, try a random perturbation
                existing = []
                for u in range(n):
                    mask = best_adj[u]
                    while mask:
                        bit = mask & (-mask)
                        v = bit.bit_length() - 1
                        if v > u:
                            existing.append((u, v))
                        mask -= bit
                
                if not existing:
                    break
                
                u_rem, v_rem = random.choice(existing)
                trial = best_adj[:]
                trial[u_rem] &= ~(1 << v_rem)
                trial[v_rem] &= ~(1 << u_rem)
                
                missing_trial = [(u, v) for u in range(n) for v in range(u+1, n) if not (trial[u] & (1 << v))]
                random.shuffle(missing_trial)
                greedy_add(trial, missing_trial)
                
                new_count = count_edges(trial)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = trial[:]
                    adj = trial[:]
                    current_count = new_count
        
        return best_adj, best_count

    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    # Run multiple restarts with local search
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        adj = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        greedy_add(adj, order)
        
        adj, cnt = local_search(adj)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
    
    # Final polish with multiple strategies
    if time.time() < deadline - 2:
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (best_adj[u] & (1 << v))]
        
        # Strategy 1: Random order
        random.shuffle(remaining)
        greedy_add(best_adj, remaining)
        
        # Strategy 2: Systematic order (by vertex degree in complement)
        remaining2 = [(u, v) for u in range(n) for v in range(u+1, n) 
                      if not (best_adj[u] & (1 << v))]
        if remaining2:
            # Sort by sum of degrees of u and v in the current graph
            remaining2.sort(key=lambda e: (best_adj[e[0]].bit_count() + best_adj[e[1]].bit_count()))
            greedy_add(best_adj, remaining2)
        
        # Strategy 3: Another random pass
        remaining3 = [(u, v) for u in range(n) for v in range(u+1, n) 
                      if not (best_adj[u] & (1 << v))]
        random.shuffle(remaining3)
        greedy_add(best_adj, remaining3)
        
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)
