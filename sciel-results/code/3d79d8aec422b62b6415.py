"""Dense Q3-free graph on 16 vertices via complement removal with greedy set cover."""
import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    total_edges = len(all_edges)  # 120
    
    # Precompute edge indices
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
    
    # Generate all Q3 subgraphs in K16
    # A Q3 is isomorphic to the 3-dimensional hypercube
    # We need to find all sets of 8 vertices that can be labeled as {0,1}^3
    # with edges connecting vertices differing in exactly 1 bit
    
    # The 12 edges of Q3 in terms of bit positions:
    # Vertices are 3-bit strings. Edges connect strings differing in one bit.
    # For a specific labeling, the 12 edges are determined.
    
    # Instead of enumerating all 8-subsets, let's think differently.
    # We'll use the complement approach: start with K16, remove edges to break all Q3s.
    
    # Actually, let's try a direct approach with a better local search.
    # The key insight: Q3-free means no 8 vertices with exactly the Q3 edge pattern.
    
    # Let's precompute all possible Q3 edge sets.
    # A Q3 has 8 vertices and 12 edges.
    # For each set of 8 vertices, there are multiple ways to embed Q3.
    
    # Number of distinct Q3 embeddings on 8 labeled vertices:
    # |Aut(Q3)| = 48, so 8!/48 = 840 distinct edge sets per 8-vertex set.
    # Total: C(16,8) * 840 = 12870 * 840 ≈ 10.8 million Q3s.
    
    # This is too many to store explicitly. Let's use a different strategy.
    
    # Strategy: Start with a known dense Q3-free graph and improve via local search.
    # The current best known ex(16, Q3) is around 50-60 edges.
    
    # Let's use a bipartite graph approach: Q3 is bipartite, so we can look for
    # bipartite subgraphs.
    
    # Actually, let's try the complement removal with a smarter approach.
    # We'll use a greedy algorithm that removes edges to break as many Q3s as possible.
    
    # But first, let's try a different initialization: use a random 3-regular graph
    # and add edges greedily.
    
    def creates_q3_fast(u, v, adj):
        """
        Check if adding edge (u,v) creates a Q3.
        adj[i] is a bitmask of neighbors of i.
        """
        neigh_u = adj[u] & ~(1 << v)
        neigh_v = adj[v] & ~(1 << u)
        
        # In Q3, u and v are adjacent. Let's say u=000, v=001 (differ in bit 0).
        # Then neighbors of u: 001, 010, 100 -> but 001=v, so others are a2, a4
        # Neighbors of v: 000, 011, 101 -> but 000=u, so others are a3, a5
        # a2 and a4 are neighbors of u (not v)
        # a3 and a5 are neighbors of v (not u)
        # a2, a4, a3, a5 are all distinct
        # a6 is adjacent to a2 and a4 (and not u, v, a3, a5)
        # a7 is adjacent to a3 and a5 (and not u, v, a2, a4)
        # a6 and a7 are adjacent to each other? Let's check:
        # In Q3 with u=000, v=001:
        # a2=010, a4=100, a3=011, a5=101
        # a6 is adjacent to a2(010) and a4(100): a6=110
        # a7 is adjacent to a3(011) and a5(101): a7=111
        # a6(110) and a7(111) differ in bit 0, so YES they are adjacent.
        
        # So we need:
        # 1. a2, a4 in neigh_u (neighbors of u, not v)
        # 2. a3, a5 in neigh_v (neighbors of v, not u)
        # 3. a2, a4, a3, a5 all distinct and not u, v
        # 4. a6 = common neighbor of a2 and a4, not in {u,v,a2,a3,a4,a5}
        # 5. a7 = common neighbor of a3 and a5, not in {u,v,a2,a3,a4,a5,a6}
        # 6. a6 and a7 are adjacent
        
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
                a5_candidates &= ~(1 << a2_idx) & ~(1 << a4_idx)
                
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

    def greedy_add(adj, edge_order, deadline):
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

    def local_search(adj, deadline):
        """
        Local search with edge swaps to escape local optima.
        """
        current_count = count_edges(adj)
        best_adj = adj[:]
        best_count = current_count
        
        # Phase 1: Greedy add all missing edges in random order
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (adj[u] & (1 << v))]
        random.shuffle(missing)
        greedy_add(adj, missing, deadline)
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # Phase 2: Try swaps - remove one edge, add another
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            
            trial = best_adj[:]
            # Build list of existing edges
            existing = []
            for u in range(n):
                mask = trial[u]
                while mask:
                    bit = mask & (-mask)
                    v = bit.bit_length() - 1
                    if v > u:
                        existing.append((u, v))
                    mask -= bit
            
            if not existing:
                break
            
            # Try multiple random swaps
            for swap_attempt in range(20):
                if time.time() > deadline - 3:
                    break
                trial = best_adj[:]
                
                u_rem, v_rem = random.choice(existing)
                trial[u_rem] &= ~(1 << v_rem)
                trial[v_rem] &= ~(1 << u_rem)
                
                # Now try to add edges greedily
                missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (trial[u] & (1 << v))]
                random.shuffle(missing)
                greedy_add(trial, missing, deadline)
                
                new_count = count_edges(trial)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = trial[:]
        
        # Phase 3: 2-swaps - remove two edges, add two edges
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            
            trial = best_adj[:]
            existing = []
            for u in range(n):
                mask = trial[u]
                while mask:
                    bit = mask & (-mask)
                    v = bit.bit_length() - 1
                    if v > u:
                        existing.append((u, v))
                    mask -= bit
            
            if len(existing) < 2:
                break
            
            # Remove two random edges
            e1, e2 = random.sample(existing, 2)
            u1, v1 = e1
            u2, v2 = e2
            trial[u1] &= ~(1 << v1)
            trial[v1] &= ~(1 << u1)
            trial[u2] &= ~(1 << v2)
            trial[v2] &= ~(1 << u2)
            
            # Now try to add edges greedily
            missing = [(u, v) for u in range(n) for v in range(u+1, n) if not (trial[u] & (1 << v))]
            random.shuffle(missing)
            greedy_add(trial, missing, deadline)
            
            new_count = count_edges(trial)
            if new_count > best_count:
                best_count = new_count
                best_adj = trial[:]
        
        return best_adj, best_count

    random.seed(42)
    best_adj = [0] * n
    best_count = -1
    
    # Run multiple restarts with local search
    num_restarts = 40
    for i in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        adj = [0] * n
        order = all_edges[:]
        random.shuffle(order)
        greedy_add(adj, order, deadline)
        
        adj, cnt = local_search(adj, deadline)
        
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
        
        # Update best_adj reference
        best_adj = best_adj  # Already copied in local_search
    
    # Final polish
    if time.time() < deadline - 2:
        remaining = [(u, v) for u in range(n) for v in range(u+1, n) 
                     if not (best_adj[u] & (1 << v))]
        random.shuffle(remaining)
        greedy_add(best_adj, remaining, deadline)
        best_count = count_edges(best_adj)

    return edges_from_adj(best_adj)
