import time
import random
import math

def construct():
    deadline = time.time() + 115  # Safety margin for 120s limit
    n = 16
    
    # --- Helper Functions ---
    
    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        adj[i] is a bitmask of neighbors of i.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        # In Q3, u and v are connected.
        # Their neighbors in Q3 must be distinct.
        # Let N_u be neighbors of u in Q3 (excluding v).
        # Let N_v be neighbors of v in Q3 (excluding u).
        # |N_u| = 2, |N_v| = 2.
        # The 4 neighbors N_u ∪ N_v are all distinct.
        # Also, each node in N_u is connected to one node in N_v (the one at distance 2 from u).
        # Specifically, if a in N_u and b in N_v, they are connected in Q3 iff they are NOT opposite?
        # Let's trace the structure.
        # u = 000, v = 100.
        # N_u = {001, 010}. N_v = {101, 110}.
        # Edges in Q3 among these 6:
        # u-v
        # u-001, u-010
        # v-101, v-110
        # 001-101, 010-110
        # 001-011, 010-111, 101-111, 110-111
        # Wait, Q3 has 8 vertices. u, v, 001, 010, 101, 110, 011, 111.
        # Let's identify the remaining 2 vertices: 011 and 111.
        # 011 is connected to 001, 010, 111.
        # 111 is connected to 101, 110, 011.
        
        # So, given u, v:
        # 1. Pick a2, a4 in N_u (neighbors of u in G, not v).
        # 2. Pick a3, a5 in N_v (neighbors of v in G, not u).
        # 3. All 6 must be distinct.
        # 4. Check edges:
        #    (a2, a4) must NOT be edge? No, in Q3, 001 and 010 are NOT connected.
        #    Wait, subgraph isomorphic to Q3 means we need a SUBSET of edges.
        #    So extra edges in G don't matter. We just need the 12 Q3 edges to exist in G.
        #    So we need:
        #    (a2, a3) or (a2, a5)?
        #    In Q3: 001 (a2) connects to 101 (a3 or a5).
        #    010 (a4) connects to 110 (a5 or a3).
        #    001 and 010 are NOT connected.
        #    101 and 110 are NOT connected.
        #    011 (a6) connects to 001, 010, 111.
        #    111 (a7) connects to 101, 110, 011.
        
        # So, for a fixed choice of a2, a4 from N_u and a3, a5 from N_v:
        # We need to find a6, a7 such that:
        # a6 is neighbor of a2, a4, a7
        # a7 is neighbor of a3, a5, a6
        # And a6, a7 are distinct from u,v,a2,a3,a4,a5.
        
        # Also, we need to ensure that the specific pairing works.
        # There are two ways to pair N_u and N_v:
        # Case 1: a2 connects to a3, a4 connects to a5.
        # Case 2: a2 connects to a5, a4 connects to a3.
        
        # Let's check Case 1:
        # If (a2, a3) in E and (a4, a5) in E:
        # Then a6 must be neighbor of a2 and a4.
        # a7 must be neighbor of a3 and a5.
        # And (a6, a7) must be in E.
        
        # Let's check Case 2:
        # If (a2, a5) in E and (a4, a3) in E:
        # Then a6 must be neighbor of a2 and a4.
        # a7 must be neighbor of a3 and a5.
        # And (a6, a7) must be in E.
        
        # Notice that in both cases, a6 is in N(a2) ∩ N(a4) and a7 is in N(a3) ∩ N(a5).
        # The only difference is which edges we require between the N_u and N_v sets.
        # But wait, if we find a6 and a7, we must also verify the cross edges.
        
        # Efficient check:
        # Iterate a2 in N_u
        # Iterate a4 in N_u (a4 != a2)
        # Candidates for a3, a5 from N_v.
        # We need a3, a5 distinct, and distinct from a2, a4.
        
        # Let's iterate a2, a4 from N_u.
        # Let S_u = {a2, a4}.
        # Let S_v_candidates = N_v \ S_u.
        # We need to pick 2 from S_v_candidates, say a3, a5.
        # For a specific pairing (a2-a3, a4-a5):
        #   Check if (a2,a3) in E and (a4,a5) in E.
        #   Find a6 in N(a2) ∩ N(a4) \ {u,v,a2,a3,a4,a5}
        #   Find a7 in N(a3) ∩ N(a5) \ {u,v,a2,a3,a4,a5}
        #   Check if a6 != a7 and (a6,a7) in E.
        
        # There are C(|N_v|, 2) pairs. |N_v| can be up to 14. 14*13/2 = 91.
        # |N_u| can be up to 14. C(14,2) = 91.
        # 91 * 91 = 8281 checks.
        # Each check involves bit operations.
        
        # Let's optimize by precomputing intersections.
        
        # For each pair (a2, a4) in N_u:
        #   common_u = adj[a2] & adj[a4] & ~(1<<u) & ~(1<<v) & ~(1<<a2) & ~(1<<a4)
        #   For each pair (a3, a5) in N_v:
        #     if a3 in {a2,a4} or a5 in {a2,a4}: continue
        #     # Check cross edges
        #     if (adj[a2] & (1<<a3)) and (adj[a4] & (1<<a5)):
        #         common_v = adj[a3] & adj[a5] & ~(1<<u) & ~(1<<v) & ~(1<<a2) & ~(1<<a3) & ~(1<<a4) & ~(1<<a5)
        #         # a6 must be in common_u, a7 in common_v
        #         # a6 != a7, (a6,a7) in E
        #         # Also a6, a7 must not be u,v,a2,a3,a4,a5. (Already excluded in common sets)
        #         # We need to check if there exists a6 in common_u, a7 in common_v such that a6!=a7 and adj[a6] & (1<<a7)
        
        #     if (adj[a2] & (1<<a5)) and (adj[a4] & (1<<a3)):
        #         # Similar check
        
        # This is still a bit heavy. Let's try to simplify.
        # Instead of iterating all pairs, we can just check if the condition holds.
        
        # Let's stick to the previous implementation which was working, but maybe optimize the inner loop.
        # The previous implementation iterated a2, a4, then a3, a5, then a6, a7.
        
        # Let's reuse the logic from the current program but ensure it's correct.
        # The current program's `creates_q3_fast` seems to be the bottleneck or at least the core.
        # I will copy it and trust it, as it was part of the 1.51 score program.
        
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
        return sum(bin(a).count('1') for a in adj) // 2

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

    # --- Structured Initialization ---
    # Partition 16 vertices into 4 groups of 4: G0, G1, G2, G3.
    # Let's try a "Turan-like" structure.
    # Q3 is bipartite. If we make the graph 4-partite with no edges within parts,
    # is it Q3-free?
    # Q3 has vertices of degree 3. In a 4-partite graph, max degree is 12.
    # Q3 contains triangles? No, Q3 is bipartite, so no odd cycles.
    # So a bipartite graph is Q3-free?
    # Q3 is bipartite. So a bipartite graph CAN contain Q3.
    # K_{8,8} contains Q3.
    # So we need to remove edges from a dense bipartite graph to break Q3s.
    
    # Alternative: Start with a random graph with p=0.5 and run SA.
    # Or start with a specific structured graph.
    
    # Let's try: Random graph with p=0.5, then SA.
    # Or: Start with K_{8,8} (bipartite) and remove edges to break Q3s?
    # K_{8,8} has 64 edges.
    # ex(16; Q3) is likely higher than 64.
    # Random p=0.5 has 60 edges on average.
    
    # Let's use a "Greedy Add" to get a good starting point, then SA.
    # This is similar to the current program, but SA is the differentiator.
    
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

    # --- Simulated Annealing ---
    
    def simulated_annealing(adj, start_temp=10.0, end_temp=0.1, steps=5000):
        current_adj = adj[:]
        current_count = count_edges(current_adj)
        best_adj = current_adj[:]
        best_count = current_count
        
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # Precompute edge indices for fast access?
        # Or just use the list.
        
        for i in range(steps):
            if time.time() > deadline - 5:
                break
                
            temp = start_temp * (end_temp / start_temp) ** (i / steps)
            
            # Generate a move:
            # 1. Add a random missing edge
            # 2. Remove a random existing edge
            # 3. Swap: Remove existing, add missing
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Add edge
                missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (current_adj[u] & (1 << v))]
                if not missing:
                    continue
                u, v = random.choice(missing)
                if creates_q3_fast(u, v, current_adj):
                    continue
                
                trial = current_adj[:]
                trial[u] |= (1 << v)
                trial[v] |= (1 << u)
                new_count = current_count + 1
                
                # Accept if better or with probability
                if new_count > current_count:
                    current_adj = trial
                    current_count = new_count
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = trial[:]
                else:
                    # Should not happen for add, but just in case
                    pass
                    
            elif move_type < 0.7:
                # Remove edge
                existing = [(u, v) for u in range(n) for v in range(u+1, n) if (current_adj[u] & (1 << v))]
                if not existing:
                    continue
                u, v = random.choice(existing)
                
                trial = current_adj[:]
                trial[u] &= ~(1 << v)
                trial[v] &= ~(1 << u)
                new_count = current_count - 1
                
                # Always accept removal? No, only if it helps escape local minima.
                # Delta E = -1.
                # Accept with probability exp(-delta / T) = exp(1/T) > 1?
                # Standard SA: accept if E_new < E_old OR with prob exp((E_old - E_new)/T).
                # Here E is "badness" = -edges. So minimizing badness = maximizing edges.
                # Delta E = E_new - E_old = -(new_count) - (-current_count) = current_count - new_count = 1.
                # Prob = exp(-1/T).
                
                prob = math.exp(-1.0 / temp)
                if random.random() < prob:
                    current_adj = trial
                    current_count = new_count
                    
            else:
                # Swap
                existing = [(u, v) for u in range(n) for v in range(u+1, n) if (current_adj[u] & (1 << v))]
                missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (current_adj[u] & (1 << v))]
                if not existing or not missing:
                    continue
                
                u_rem, v_rem = random.choice(existing)
                u_add, v_add = random.choice(missing)
                
                # Check if swap is valid (no duplicate edges)
                if (u_add == u_rem or u_add == v_rem or v_add == u_rem or v_add == v_rem):
                    continue
                
                trial = current_adj[:]
                trial[u_rem] &= ~(1 << v_rem)
                trial[v_rem] &= ~(1 << u_rem)
                
                # Check if adding creates Q3
                if creates_q3_fast(u_add, v_add, trial):
                    continue
                
                trial[u_add] |= (1 << v_add)
                trial[v_add] |= (1 << u_add)
                
                new_count = current_count # Net change 0
                
                # Delta E = 0.
                # Accept with probability 1? Or exp(0)=1.
                # But we want to explore.
                # If we accept all swaps, we might drift.
                # Let's accept with prob 0.5 or based on some heuristic?
                # Actually, if count is same, we can accept to explore.
                if random.random() < 0.5:
                    current_adj = trial
                    current_count = new_count
                    
                    # Check if this leads to a better state in next steps?
                    # No, SA is memoryless.
                    
                    # Update best? No, count is same.
                    
        return best_adj, best_count

    # --- Main Loop ---
    
    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    # Phase 1: Greedy Initialization
    adj = [0] * n
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    greedy_add(adj, all_edges)
    
    # Phase 2: Simulated Annealing
    adj, cnt = simulated_annealing(adj)
    if cnt > best_count:
        best_count = cnt
        best_adj = adj[:]
        
    # Phase 3: Multiple Restarts
    for _ in range(10):
        if time.time() > deadline - 20:
            break
            
        adj = [0] * n
        random.shuffle(all_edges)
        greedy_add(adj, all_edges)
        
        adj, cnt = simulated_annealing(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
            
    # Final Polish: Greedy add
    if time.time() < deadline - 2:
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (best_adj[u] & (1 << v))]
        random.shuffle(remaining)
        greedy_add(best_adj, remaining)
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
