import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Helper to run a single greedy pass
    def run_greedy(initial_edges, seed):
        random.seed(seed)
        edges = set(initial_edges)
        
        # Precompute bitmask for each vertex
        adj_mask = [0] * 16
        for (u, v) in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
        # Precompute perms24 (might not be needed if we use the specific structure check)
        # But we might need it for general Q3 check if we fallback.
        # For the optimized check, we don't need perms24.
        
        def contains_q3_with_edge(u, v):
            """
            Checks if adding edge (u, v) creates a Q3 subgraph containing (u, v).
            Assumes (u, v) is NOT yet in the graph.
            """
            # Neighbors of u (excluding v, since v is not connected yet in the 'check' context, 
            # but wait, we are checking if ADDING it creates a Q3.
            # So we consider the graph WITH (u,v).
            # So N(u) in the new graph includes v.
            # N(u) in old graph does not include v.
            # In Q3, u has degree 3.
            # So u must be connected to v and 2 other vertices in B.
            # So we need 2 neighbors of u in the OLD graph.
            
            n_u = [x for x in range(16) if x != u and x != v and ((adj_mask[u] >> x) & 1)]
            n_v = [x for x in range(16) if x != u and x != v and ((adj_mask[v] >> x) & 1)]
            
            if len(n_u) < 2 or len(n_v) < 2:
                return False
            
            # Candidates for B' (2 neighbors of u in B, excluding v)
            # Candidates for A' (2 neighbors of v in A, excluding u)
            
            # Iterate over pairs in n_u and n_v
            for b1, b2 in itertools.combinations(n_u, 2):
                for a1, a2 in itertools.combinations(n_v, 2):
                    # We have u, v, a1, a2, b1, b2.
                    # We need a3 and b3.
                    # a3 in A, b3 in B.
                    # a3 not connected to v (in new graph? v is connected to a1, a2, u. a3 is NOT connected to v).
                    # b3 not connected to u (u is connected to v, b1, b2. b3 is NOT connected to u).
                    
                    # Also a3, b3 must be distinct from {u, v, a1, a2, b1, b2}
                    used = {u, v, a1, a2, b1, b2}
                    
                    # Candidates for a3:
                    # x not in used
                    # x not connected to v (adj_mask[v] & (1<<x) == 0)
                    # Note: x could be connected to u? No, a3 in A, u in A. No edge.
                    # So x not connected to u as well?
                    # If x is not connected to u and not connected to v, it's a candidate for A or B.
                    # But we need a3 in A.
                    # In Q3, a3 is connected to 3 vertices in B.
                    
                    # Candidates for b3:
                    # y not in used
                    # y not connected to u
                    # y not connected to v? No, b3 in B, v in B. No edge.
                    
                    # So:
                    # a3 candidates: x not in used, not adj to u, not adj to v.
                    # b3 candidates: y not in used, not adj to u, not adj to v.
                    
                    # Wait, if x is not adj to u and not adj to v, it can be in A or B.
                    # We assign x to A (as a3) and y to B (as b3).
                    
                    a3_candidates = []
                    b3_candidates = []
                    for x in range(16):
                        if x in used: continue
                        if (adj_mask[u] >> x) & 1: continue # Must not be adj to u
                        if (adj_mask[v] >> x) & 1: continue # Must not be adj to v
                        a3_candidates.append(x)
                        b3_candidates.append(x)
                    
                    if not a3_candidates or not b3_candidates:
                        continue
                    
                    for a3 in a3_candidates:
                        for b3 in b3_candidates:
                            if a3 == b3: continue
                            
                            # Check if {u, a1, a2, a3} and {v, b1, b2, b3} form Q3
                            A = [u, a1, a2, a3]
                            B = [v, b1, b2, b3]
                            
                            # Check edges
                            # 1. u connected to v, b1, b2. NOT b3.
                            if not ((adj_mask[u] >> v) & 1): return True # Edge (u,v) is being added, so assume present for check? 
                            # Actually, we are checking if the NEW graph contains Q3.
                            # So (u,v) IS present.
                            # But adj_mask does NOT have (u,v) yet.
                            # So we must treat (u,v) as present.
                            # For other edges, we use adj_mask.
                            
                            # Let's define a helper check_edge(x, y)
                            def is_edge(x, y):
                                if (x == u and y == v) or (x == v and y == u):
                                    return True
                                return (adj_mask[x] >> y) & 1
                            
                            # u (A) connects to: v, b1, b2. Not b3.
                            if not is_edge(u, v): return False # Should be true
                            if not is_edge(u, b1): continue
                            if not is_edge(u, b2): continue
                            if is_edge(u, b3): continue # Must NOT be edge
                            
                            # v (B) connects to: u, a1, a2. Not a3.
                            if not is_edge(v, u): return False
                            if not is_edge(v, a1): continue
                            if not is_edge(v, a2): continue
                            if is_edge(v, a3): continue
                            
                            # a1 (A) connects to: v, and 2 of {b1, b2, b3}. Not u, a2, a3.
                            # In Q3, each a in A connects to 3 in B.
                            # a1 connects to v.
                            # a1 connects to exactly 2 of b1, b2, b3.
                            # a1 NOT connected to u, a2, a3.
                            if is_edge(a1, u) or is_edge(a1, a2) or is_edge(a1, a3): continue
                            if not is_edge(a1, v): continue # Already checked v-a1
                            
                            count_a1_b = 0
                            if is_edge(a1, b1): count_a1_b += 1
                            if is_edge(a1, b2): count_a1_b += 1
                            if is_edge(a1, b3): count_a1_b += 1
                            if count_a1_b != 2: continue
                            
                            # a2 (A) connects to: v, and 2 of {b1, b2, b3}. Not u, a1, a3.
                            if is_edge(a2, u) or is_edge(a2, a1) or is_edge(a2, a3): continue
                            if not is_edge(a2, v): continue
                            
                            count_a2_b = 0
                            if is_edge(a2, b1): count_a2_b += 1
                            if is_edge(a2, b2): count_a2_b += 1
                            if is_edge(a2, b3): count_a2_b += 1
                            if count_a2_b != 2: continue
                            
                            # a3 (A) connects to: NOT v. And 3 of {b1, b2, b3}. Not u, a1, a2.
                            if is_edge(a3, u) or is_edge(a3, a1) or is_edge(a3, a2): continue
                            if is_edge(a3, v): continue # Already checked
                            
                            count_a3_b = 0
                            if is_edge(a3, b1): count_a3_b += 1
                            if is_edge(a3, b2): count_a3_b += 1
                            if is_edge(a3, b3): count_a3_b += 1
                            if count_a3_b != 3: continue
                            
                            # Now check B vertices consistency?
                            # If A-B edges are correct, and degrees match, is it Q3?
                            # Q3 is unique bipartite graph with 4+4 vertices, degree 3.
                            # Yes, if the bipartite graph is 3-regular, it is Q3 (or K4,4 minus perfect matching? No, Q3 is the 3-cube).
                            # Actually, there are multiple 3-regular bipartite graphs on 8 vertices?
                            # Q3 is the only one that is a cube.
                            # But we checked specific connections.
                            # Let's verify b vertices degrees.
                            
                            # b1 (B) connects to: u, and 2 of {a1, a2, a3}. Not v, b2, b3.
                            if is_edge(b1, v) or is_edge(b1, b2) or is_edge(b1, b3): continue
                            if not is_edge(b1, u): continue
                            
                            count_b1_a = 0
                            if is_edge(b1, a1): count_b1_a += 1
                            if is_edge(b1, a2): count_b1_a += 1
                            if is_edge(b1, a3): count_b1_a += 1
                            if count_b1_a != 2: continue
                            
                            # b2 (B) connects to: u, and 2 of {a1, a2, a3}. Not v, b1, b3.
                            if is_edge(b2, v) or is_edge(b2, b1) or is_edge(b2, b3): continue
                            if not is_edge(b2, u): continue
                            
                            count_b2_a = 0
                            if is_edge(b2, a1): count_b2_a += 1
                            if is_edge(b2, a2): count_b2_a += 1
                            if is_edge(b2, a3): count_b2_a += 1
                            if count_b2_a != 2: continue
                            
                            # b3 (B) connects to: NOT u. And 3 of {a1, a2, a3}. Not v, b1, b2.
                            if is_edge(b3, v) or is_edge(b3, b1) or is_edge(b3, b2): continue
                            if is_edge(b3, u): continue
                            
                            count_b3_a = 0
                            if is_edge(b3, a1): count_b3_a += 1
                            if is_edge(b3, a2): count_b3_a += 1
                            if is_edge(b3, a3): count_b3_a += 1
                            if count_b3_a != 3: continue
                            
                            return True
            return False

        all_pairs = set(itertools.combinations(range(16), 2))
        non_edges = list(all_pairs - edges)
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            random.shuffle(non_edges)
            
            current_non_edges = [e for e in non_edges if e not in edges]
            
            def degree(v):
                return bin(adj_mask[v]).count('1')
            
            current_non_edges.sort(key=lambda e: degree(e[0]) + degree(e[1]) + random.uniform(0, 1), reverse=True)
            
            for e in current_non_edges:
                if time.time() > deadline:
                    break
                u, v = e
                
                # Check safety
                if not contains_q3_with_edge(u, v):
                    edges.add((u, v))
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    changed = True
                # If unsafe, do nothing (edge remains non-edge)
                    
        return edges

    # Initial seed: K_{3,13}
    initial_edges = set()
    for u in range(3):
        for v in range(3, 16):
            initial_edges.add((u, v))
            
    best_edges = set()
    best_count = -1
    
    seed = 42
    while time.time() < deadline:
        current_edges = run_greedy(initial_edges, seed)
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges
        seed += 1
        
    return [list(e) for e in best_edges]
