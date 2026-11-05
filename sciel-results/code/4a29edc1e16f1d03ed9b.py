import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin for 150s limit
    
    n = 16
    # Total possible edges in K16
    total_edges = n * (n - 1) // 2
    
    # Precompute all edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))
    
    # Precompute bitmasks for adjacency
    # adj_mask[v] is an integer where bit u is 1 if (u,v) is an edge
    adj_mask = [0] * n
    
    # Function to check if adding edge (u, v) creates a Q3
    # Q3 is a 3-cube. It is a bipartite graph with parts of size 4, 
    # 3-regular.
    # If we add (u, v), any Q3 containing this edge must have u and v in opposite parts.
    # Let Part A contain u and Part B contain v.
    # |Part A| = 4, |Part B| = 4.
    # u is in A, v is in B.
    # The other 3 vertices in A must be neighbors of exactly 3 vertices in B.
    # The other 3 vertices in B must be neighbors of exactly 3 vertices in A.
    # Also, u must be connected to exactly 3 vertices in B.
    # And v must be connected to exactly 3 vertices in A.
    
    def creates_q3(u, v):
        # u and v are the new edge endpoints.
        # We need to choose 3 more vertices for A (from N(u) \ {v} and potentially others? No, they must be connected to B).
        # Actually, a faster check:
        # A Q3 containing (u,v) is determined by the sets A and B.
        # u in A, v in B.
        # N(u) intersect B must have size 3.
        # N(v) intersect A must have size 3.
        # For any w in A\{u}, N(w) intersect B must have size 3.
        # For any z in B\{v}, N(z) intersect A must have size 3.
        
        # Candidates for A\{u}: Vertices connected to at least 3 vertices in B?
        # This is getting complex to optimize dynamically without precomputing all subsets.
        # However, n=16 is small.
        
        # Let's iterate over possible sets A of size 4 containing u.
        # And sets B of size 4 containing v.
        # Disjoint A and B? Yes, bipartite.
        # Edges only between A and B in the subgraph? Yes.
        # All vertices in A have degree 3 in B.
        # All vertices in B have degree 3 in A.
        
        # Optimization:
        # 1. Identify candidates for A\{u}. They must be neighbors of v? No.
        #    They must have 3 neighbors in B.
        # 2. Identify candidates for B\{v}. They must be neighbors of u? No.
        #    They must have 3 neighbors in A.
        
        # Let's just iterate over combinations of 3 vertices from the remaining 14 for A,
        # and 3 vertices from the remaining 14 for B.
        # 14 choose 3 is 364. 364 * 364 is ~130k checks. Too slow per edge if done naively.
        
        # Better: 
        # u must have degree 3 in B. So B must contain exactly 3 neighbors of u (excluding v? No, v is in B).
        # Wait, u is in A. v is in B.
        # u is connected to v.
        # u must be connected to 3 vertices in B. One is v. So u must be connected to 2 other vertices in B\{v}.
        # Let N_u_in_B = {w in B | w != v and adj[u][w]}. Size must be 2.
        # Let N_v_in_A = {w in A | w != u and adj[v][w]}. Size must be 2.
        
        # So B is determined by choosing 2 neighbors of u (other than v) to join v.
        # Let candidates_B = [w for w in range(n) if w != u and w != v and adj_mask[u] & (1 << w)]
        # We need to choose 2 from candidates_B.
        # Let candidates_A = [w for w in range(n) if w != u and w != v and adj_mask[v] & (1 << w)]
        # We need to choose 2 from candidates_A.
        
        # If len(candidates_B) < 2 or len(candidates_A) < 2, return False.
        
        cand_B = [w for w in range(n) if w != u and w != v and (adj_mask[u] >> w) & 1]
        cand_A = [w for w in range(n) if w != u and w != v and (adj_mask[v] >> w) & 1]
        
        if len(cand_B) < 2 or len(cand_A) < 2:
            return False
            
        # Iterate over pairs in cand_B to form B
        # B = {v, b1, b2}
        for b1, b2 in itertools.combinations(cand_B, 2):
            B = {v, b1, b2}
            B_mask = (1 << v) | (1 << b1) | (1 << b2)
            
            # Now we need to find A = {u, a1, a2} such that:
            # 1. a1, a2 are in cand_A (neighbors of v).
            # 2. u, a1, a2 are NOT in B. (Since A and B are disjoint in bipartite Q3)
            #    Note: a1, a2 are neighbors of v, so they are not v.
            #    Are they b1 or b2? If a1 == b1, then a1 is in B. But a1 is in A. Contradiction.
            #    So we must ensure a1, a2 not in {b1, b2}.
            
            valid_cand_A = [a for a in cand_A if a not in (b1, b2)]
            if len(valid_cand_A) < 2:
                continue
                
            for a1, a2 in itertools.combinations(valid_cand_A, 2):
                A = {u, a1, a2}
                A_mask = (1 << u) | (1 << a1) | (1 << a2)
                
                # Check if A and B are disjoint
                if A_mask & B_mask:
                    continue
                    
                # Check degrees in the subgraph induced by A and B
                # For each a in A, count neighbors in B. Must be 3.
                ok = True
                for a in A:
                    deg = bin(adj_mask[a] & B_mask).count('1')
                    if deg != 3:
                        ok = False
                        break
                if not ok:
                    continue
                    
                for b in B:
                    deg = bin(adj_mask[b] & A_mask).count('1')
                    if deg != 3:
                        ok = False
                        break
                if not ok:
                    continue
                    
                return True
                
        return False

    # Randomized Greedy Construction
    # We try multiple random seeds/orders to find a better solution
    best_edge_count = 0
    best_adj_mask = [0] * n
    
    # Time budget for search
    # We will run the greedy algorithm with different random seeds
    # until we run out of time.
    
    # To maximize performance, we should probably just run one very good seed
    # or a few. The check creates_q3 is fast enough (~microseconds).
    # 120 edges * 364 * 364 is worst case, but early pruning helps.
    # Let's estimate: 120 edges added. Each check ~100us? 12ms total.
    # So we can run many iterations.
    
    # Strategy: Run greedy with random edge order.
    # Keep the best result.
    
    for seed in range(1000): # Limit iterations, but break on time
        if time.time() > DEADLINE:
            break
            
        random.seed(seed)
        random.shuffle(edges)
        
        current_adj_mask = [0] * n
        current_edge_count = 0
        
        for u, v in edges:
            if current_adj_mask[u] & (1 << v):
                continue # Already added
            
            # Check if adding (u, v) creates Q3
            if not creates_q3(u, v):
                current_adj_mask[u] |= (1 << v)
                current_adj_mask[v] |= (1 << u)
                current_edge_count += 1
        
        if current_edge_count > best_edge_count:
            best_edge_count = current_edge_count
            best_adj_mask = current_adj_mask[:]
            
            # If we reach a very high number, we might stop early?
            # Max edges in Q3-free graph on 16 vertices?
            # ex(16, Q3). 
            # If we hit 60+, it's likely optimal or near-optimal.
            # But let's keep going to be sure.
            
    # Construct the final edge list from best_adj_mask
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj_mask[u] & (1 << v):
                final_edges.append([u, v])
                
    return final_edges
