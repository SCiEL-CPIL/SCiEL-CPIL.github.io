import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # We want to maximize n.
    # Try n from 47 down to 40.
    for n in range(47, 39, -1):
        if time.time() > deadline - 10:
            break
            
        min_deg_req = n - 39
        
        # Multiple attempts for each n
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            # Initialize adjacency matrix as bitsets
            adj = [0] * n
            degrees = [0] * n
            
            # 1. Generate a random graph with target average degree
            # Target degree is min_deg_req. Let's aim for min_deg_req + 1 to have slack.
            target_deg = min_deg_req + 1
            if target_deg > n - 1:
                target_deg = n - 1
            
            # Random initialization: Add random edges
            # We want to ensure we don't start with too many C4s.
            # A sparse random graph is less likely to have C4s.
            # But we need high degree.
            # Let's start with a random regular-like graph.
            
            # Simple approach: Shuffle all possible edges and add until degrees are close to target.
            edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges)
            
            for u, v in edges:
                if time.time() > deadline - 2:
                    break
                # Add edge if it doesn't immediately violate degree constraints too much
                # and ideally doesn't create a C4.
                # But for initialization, we can just add random edges.
                # We will fix C4s in the next step.
                # To be safe, let's check C4 condition during init too.
                c4_found = False
                # Check if adding (u,v) creates a C4
                # Condition: |N(u) \cap N(v)| >= 1 ? No.
                # If we add (u,v), a C4 is formed if there exists a path u-x-y-v of length 3.
                # This is equivalent to N(u) \cap N(N(v)) != empty?
                # Or simply: if there is a common neighbor? No.
                # If u and v have a common neighbor w, then u-w-v is a path of length 2.
                # Adding (u,v) creates a triangle, not a C4.
                # A C4 requires path of length 3.
                # So we need to check if N(u) and N(v) have an edge between them?
                # Let x in N(u), y in N(v). If (x,y) is an edge, then u-x-y-v-u is a C4.
                # So the check is: exists x in N(u), y in N(v) such that (x,y) is in E.
                
                # Efficient check:
                # Iterate x in N(u). Check if N(x) intersects N(v).
                temp = adj[u]
                while temp:
                    lsb = temp & (-temp)
                    x = lsb.bit_length() - 1
                    if adj[x] & adj[v]:
                        c4_found = True
                        break
                    temp ^= lsb
                
                if not c4_found:
                    # Check degrees
                    if degrees[u] < target_deg + 2 and degrees[v] < target_deg + 2:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        
                        # Stop if we have enough edges?
                        # Sum of degrees / 2 = m.
                        # We want m approx n * target_deg / 2.
                        if sum(degrees) >= n * target_deg:
                            break

            # 2. Local Search to fix C4s and balance degrees
            # Badness function: Number of pairs (u,v) with |N(u) \cap N(v)| >= 2
            # Actually, ANY common neighbor pair doesn't form a C4 by itself.
            # Only if there is an edge between the common neighbors? No.
            # Let's stick to the direct C4 check: "Is there a cycle of length 4?"
            # It is faster to count C4s directly?
            # Number of C4s = sum_{u < v} (|N(u) \cap N(v)| choose 2) - (edges in common neighborhood?)
            # This formula is for counting C4s in a graph.
            # If we just want to know if any C4 exists, we can check:
            # For each pair (u,v), if |N(u) \cap N(v)| >= 2, it DOESN'T mean a C4 exists.
            # It means there are two paths of length 2.
            # A C4 is u-x-v-y-u.
            # This requires x in N(u) \cap N(v) and y in N(u) \cap N(v) with x != y AND (x,y) is NOT an edge?
            # No. If (x,y) is an edge, then u-x-y-v-u is a cycle of length 4?
            # Edges: (u,x), (x,y), (y,v), (v,u).
            # Yes.
            # So if u,v have common neighbors x,y and (x,y) is an edge, we have a C4.
            # What if (x,y) is NOT an edge?
            # Then u-x-v-y-u is not a cycle because (x,y) is missing.
            # So the condition for C4 is:
            # Exists u,v,x,y distinct such that (u,x), (x,v), (v,y), (y,u) are edges.
            
            # Let's use a different badness metric:
            # Count the number of C4s.
            # We can estimate or count exactly.
            # Exact count:
            # For each pair of edges (u,v) and (x,y) with u!=x, v!=y, etc?
            # Standard algorithm:
            # c4_count = 0
            # for u in range(n):
            #   for v in range(u+1, n):
            #     common = N(u) & N(v)
            #     if len(common) >= 2:
            #        # Check if any pair in common has an edge?
            #        # No, the C4 is u-a-v-b-u.
            #        # a,b are in common.
            #        # The cycle is u-a-v-b-u.
            #        # Edges: (u,a), (a,v), (v,b), (b,u).
            #        # This is always a C4 if a!=b.
            #        # Wait, is it?
            #        # u is connected to a. a is connected to v. v is connected to b. b is connected to u.
            #        # Yes.
            #        # So IF |N(u) \cap N(v)| >= 2, THEN there is a C4.
            #        # Let's verify.
            #        # Graph: u-a, a-v, v-b, b-u.
            #        # a is in N(u) and N(v).
            #        # b is in N(u) and N(v).
            #        # So a,b in N(u) \cap N(v).
            #        # The cycle u-a-v-b-u uses edges (u,a), (a,v), (v,b), (b,u).
            #        # Are these edges present?
            #        # (u,a) yes. (a,v) yes (since a in N(v)).
            #        # (v,b) yes. (b,u) yes (since b in N(u)).
            #        # So YES. Any pair of vertices with >= 2 common neighbors forms a C4.
            
            # So the condition is simply: **Max_{u,v} |N(u) \cap N(v)| <= 1**.
            
            # This simplifies the check immensely.
            
            improved = True
            polish_deadline = deadline - 2
            while improved and time.time() < polish_deadline:
                improved = False
                min_deg = min(degrees)
                
                # Check if we have a C4
                has_c4 = False
                bad_pair = None
                for u in range(n):
                    if time.time() > polish_deadline:
                        break
                    for v in range(u+1, n):
                        # Check intersection
                        inter = adj[u] & adj[v]
                        # Count bits
                        # If popcount >= 2, C4 exists
                        if inter & (inter - 1): # At least 2 bits set
                            has_c4 = True
                            bad_pair = (u, v)
                            break
                    if has_c4:
                        break
                
                if not has_c4:
                    # Check min degree
                    if min_deg >= min_deg_req:
                        # Valid graph found
                        edges_list = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if (adj[i] >> j) & 1:
                                    edges_list.append([i, j])
                        return {'n': n, 'edges': edges_list}
                    
                    # If no C4 but min degree too low, try to add edges
                    # Find vertex with min degree
                    u = degrees.index(min_deg)
                    # Try to add edge (u, v) where v is non-neighbor
                    # Condition: adding (u,v) must not create a C4.
                    # New condition: |N(u) \cap N(v)| <= 1 (already true for all pairs)
                    # Adding (u,v) makes u and v neighbors.
                    # Does it create a new C4?
                    # A new C4 would involve (u,v).
                    # Cycle u-v-x-y-u?
                    # Edges (u,v), (v,x), (x,y), (y,u).
                    # x in N(v), y in N(u).
                    # We need (x,y) to be an edge.
                    # So we need to check if there exists x in N(v), y in N(u) such that (x,y) is an edge.
                    # This is the same check as before: N(N(u)) intersects N(N(v))?
                    # Specifically: exists x in N(v) such that N(x) intersects N(u).
                    
                    non_neighbors = [v for v in range(n) if v != u and not ((adj[u] >> v) & 1)]
                    random.shuffle(non_neighbors)
                    added = False
                    for v in non_neighbors:
                        # Check if adding (u,v) creates C4
                        c4_found = False
                        temp = adj[v]
                        while temp:
                            lsb = temp & (-temp)
                            x = lsb.bit_length() - 1
                            if adj[x] & adj[u]:
                                c4_found = True
                                break
                            temp ^= lsb
                        
                        if not c4_found:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            degrees[u] += 1
                            degrees[v] += 1
                            improved = True
                            added = True
                            break
                    
                    if added:
                        continue # Restart loop to re-check C4s and min degree
                
                # If has_c4, try to fix
                if has_c4:
                    u, v = bad_pair
                    # Find common neighbors
                    inter = adj[u] & adj[v]
                    # Get list of common neighbors
                    common = []
                    temp = inter
                    while temp:
                        lsb = temp & (-temp)
                        a = lsb.bit_length() - 1
                        common.append(a)
                        temp ^= lsb
                    
                    # We have a C4 u-a-v-b-u.
                    # We want to break it.
                    # Strategy: Swap an edge.
                    # Try to remove (u,a) and add (u,c) where c is not in N(u) and doesn't create C4.
                    # This changes degree of u, a.
                    # To keep it simple, let's try a 2-opt swap:
                    # Remove (u,a) and (v,b). Add (u,b) and (v,a).
                    # This preserves degrees.
                    # Check if new edges create C4s.
                    
                    a = common[0]
                    b = common[1]
                    
                    # Check if (u,b) and (v,a) are non-edges
                    if not ((adj[u] >> b) & 1) and not ((adj[v] >> a) & 1):
                        # Remove (u,a) and (v,b)
                        adj[u] &= ~(1 << a)
                        adj[a] &= ~(1 << u)
                        adj[v] &= ~(1 << b)
                        adj[b] &= ~(1 << v)
                        degrees[u] -= 1
                        degrees[a] -= 1
                        degrees[v] -= 1
                        degrees[b] -= 1
                        
                        # Add (u,b) and (v,a)
                        # Check C4 for (u,b)
                        c41 = False
                        temp = adj[u]
                        while temp:
                            lsb = temp & (-temp)
                            x = lsb.bit_length() - 1
                            if adj[x] & adj[b]:
                                c41 = True
                                break
                            temp ^= lsb
                            
                        c42 = False
                        if not c41:
                            temp = adj[v]
                            while temp:
                                lsb = temp & (-temp)
                                x = lsb.bit_length() - 1
                                if adj[x] & adj[a]:
                                    c42 = True
                                    break
                                temp ^= lsb
                        
                        if not c41 and not c42:
                            # Accept swap
                            adj[u] |= (1 << b)
                            adj[b] |= (1 << u)
                            adj[v] |= (1 << a)
                            adj[a] |= (1 << v)
                            degrees[u] += 1
                            degrees[b] += 1
                            degrees[v] += 1
                            degrees[a] += 1
                            improved = True
                        else:
                            # Revert
                            adj[u] |= (1 << a)
                            adj[a] |= (1 << u)
                            adj[v] |= (1 << b)
                            adj[b] |= (1 << v)
                            degrees[u] += 1
                            degrees[a] += 1
                            degrees[v] += 1
                            degrees[b] += 1

    # Fallback
    return {'n': 39, 'edges': []}
