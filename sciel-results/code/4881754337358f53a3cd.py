"""Spectral-aware C4 minimization for 31 vertices, 91 edges."""
import numpy as np
from collections import defaultdict
import random
import time

def construct():
    n = 31
    m = 91
    
    start_time = time.time()
    timeout = 110  # leave 10s margin
    
    best_edges = None
    best_c4 = float('inf')
    
    def count_c4s(A_bits):
        """Count C4s using bitsets.
        C4s are pairs of vertices (u, v) with >= 2 common neighbors.
        Count = sum_{u < v} C(common(u, v), 2)
        """
        c4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = A_bits[u] & A_bits[v]
                c = bin(common).count('1')
                if c >= 2:
                    c4_count += c * (c - 1) // 2
        return c4_count

    def edges_between(A_bits, u, v):
        """Count edges between N(u) and N(v).
        This is the number of new C4s if we add edge (u, v).
        Also the number of C4s destroyed if we remove edge (u, v) 
        (assuming u, v are adjacent, we look at N'(u) and N'(v)).
        """
        # We need sum_{w in N(u)} |N(w) & N(v)|
        # Note: if u and v are adjacent, N(u) contains v and N(v) contains u.
        # For adding (u, v) where they are NOT adjacent:
        # N(u) and N(v) are disjoint from each other? No, they can overlap.
        # The C4 is u-w-x-v-u.
        # w in N(u), x in N(v), (w, x) in E.
        # So we count edges between N(u) and N(v).
        
        count = 0
        # Iterate over neighbors of u
        neighbors_u = [i for i in range(n) if (A_bits[u] >> i) & 1]
        for w in neighbors_u:
            # Count common neighbors of w and v
            common_wv = A_bits[w] & A_bits[v]
            count += bin(common_wv).count('1')
        return count

    # Greedy Initialization
    # We will run multiple restarts with different random tie-breaking
    for restart in range(10):
        random.seed(restart)
        
        A_bits = [0] * n
        degrees = [0] * n
        edges = set()
        
        # Precompute all potential edges
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        current_c4 = 0
        
        for _ in range(m):
            min_cost = float('inf')
            candidates = []
            
            for (u, v) in all_edges:
                if (u, v) in edges:
                    continue
                
                # Cost is number of edges between N(u) and N(v)
                # If u, v are not connected, this is the number of new C4s.
                cost = edges_between(A_bits, u, v)
                
                # Tie-breaker: prefer lower degree vertices? 
                # Or just random.
                # Let's add a small random noise to break ties
                # Or just collect all min costs and pick one randomly.
                if cost < min_cost:
                    min_cost = cost
                    candidates = [(u, v)]
                elif cost == min_cost:
                    candidates.append((u, v))
            
            if not candidates:
                break
                
            # Pick randomly from candidates
            u, v = random.choice(candidates)
            
            # Add edge
            edges.add((u, v))
            A_bits[u] |= (1 << v)
            A_bits[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
            
            # Update current_c4? 
            # It's easier to just recalculate at the end or update incrementally.
            # Incremental update: new C4s = cost.
            current_c4 += cost

        # Local Search
        # We will try to improve by swapping edges.
        # A swap (remove (a, b), add (c, d)) changes C4 count by:
        # - (C4s destroyed by removing (a, b)) + (C4s created by adding (c, d))
        
        # Destroyed C4s for (a, b):
        # These are C4s using (a, b). 
        # They correspond to paths a-w-x-b.
        # w in N(a)\{b}, x in N(b)\{a}, (w, x) in E.
        # This is the number of edges between N'(a) and N'(b).
        
        def edges_between_excl(A_bits, u, v):
            """Count edges between N(u)\{v} and N(v)\{u}."""
            count = 0
            # Neighbors of u excluding v
            neighbors_u = [i for i in range(n) if i != v and ((A_bits[u] >> i) & 1)]
            for w in neighbors_u:
                # Common neighbors of w and v, excluding u?
                # If w is neighbor of u, and we look at common neighbors of w and v:
                # One common neighbor might be u? 
                # u is neighbor of w (since w in N(u)).
                # u is neighbor of v (since (u, v) in E).
                # So u is in N(w) & N(v).
                # We must exclude u from the count of common neighbors if we want to count edges (w, x) where x != u.
                # Wait, the edge is (w, x). x must be in N(v)\{u}.
                # So we count x in N(v) such that (w, x) in E and x != u.
                # x cannot be u because u is not in N(v)\{u}.
                # So we just count common neighbors of w and v, but ensure x is not u?
                # Actually, if x=u, then edge is (w, u). But u is in N(w).
                # Is u in N(v)? Yes.
                # So (w, u) is an edge between N'(u) and N(v)?
                # No, w in N'(u) means w != v.
                # x in N'(v) means x != u.
                # So we must exclude x=u.
                
                common_wv = A_bits[w] & A_bits[v]
                # Remove u from common_wv if it's there
                if (common_wv >> u) & 1:
                    common_wv &= ~(1 << u)
                count += bin(common_wv).count('1')
            return count

        improved = True
        while improved and (time.time() - start_time) < timeout:
            improved = False
            current_c4 = count_c4s(A_bits)
            
            # Generate list of edges and non-edges
            edge_list = list(edges)
            random.shuffle(edge_list)
            
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) not in edges:
                        non_edges.append((i, j))
            
            # Try swaps
            # To save time, we can limit the number of non-edges tried per edge
            # Or just iterate through all if fast enough.
            # n=31, m=91. ~370 non-edges.
            # 91 * 370 = 33k swaps.
            # Each swap check: edges_between_excl (O(deg)) + edges_between (O(deg)).
            # ~30 ops per swap. 1M ops per pass. Very fast.
            
            for (a, b) in edge_list:
                # Calculate cost to remove (a, b)
                # This is the number of C4s destroyed.
                rem_cost = edges_between_excl(A_bits, a, b)
                
                # Try adding random non-edges
                # We can shuffle non_edges once per pass or per edge?
                # Per pass is better for diversity.
                # But we need to ensure (c, d) is not (a, b).
                
                # Optimization: Only try a subset of non-edges if time is tight
                # But 33k is small. Let's try all.
                
                for (c, d) in non_edges:
                    if (c, d) == (a, b):
                        continue
                    
                    # Calculate cost to add (c, d)
                    # This is the number of new C4s created.
                    add_cost = edges_between(A_bits, c, d)
                    
                    delta = add_cost - rem_cost
                    
                    if delta < 0:
                        # Perform swap
                        edges.remove((a, b))
                        edges.add((c, d))
                        
                        A_bits[a] &= ~(1 << b)
                        A_bits[b] &= ~(1 << a)
                        A_bits[c] |= (1 << d)
                        A_bits[d] |= (1 << c)
                        
                        current_c4 += delta
                        improved = True
                        break # Restart scanning for this edge? Or just continue?
                        # Breaking inner loop and continuing outer is standard hill climbing.
                        # But we changed A_bits, so rem_cost for other edges is stale.
                        # However, we are just doing one swap per "attempt" to find improvement.
                        # Actually, if we break, we skip other non-edges for this (a, b).
                        # We also skip other edges (a, b') because we break out of the inner loop,
                        # but the outer loop continues to the next (a, b).
                        # This is fine.
                        # But we should update non_edges list?
                        # (a, b) is now a non-edge, (c, d) is now an edge.
                        # We can just rebuild non_edges at the start of the next `while improved` iteration.
                        # Or update it here.
                        # Rebuilding is O(n^2) = 1000 ops. Negligible.
                        break
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = set(edges)

    # Convert best_edges to list of lists
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
