import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    best_n = 0
    best_edges = []

    # Try n=47 first, then 46, then 45...
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        if min_deg_needed < 0:
            min_deg_needed = 0
            
        found = False
        
        # Number of restarts depends on time available
        # With bitsets, we can do many more restarts
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            random.seed(restart * 100 + target_n)
            
            # N[u] is an integer bitmask of neighbors of u
            N = [0] * target_n
            deg = [0] * target_n
            
            def can_add(u, v):
                # 1. u and v must not be connected
                if (N[u] >> v) & 1:
                    return False
                
                # 2. |N(u) cap N(v)| <= 1
                common = N[u] & N[v]
                if common != 0:
                    # Check if common has exactly 1 bit
                    if common & (common - 1) != 0:
                        return False
                
                # 3. For all w in N(v), N(u) cap N(w) == 0
                # This is equivalent to N(u) & (Union of N(w) for w in N(v)) == 0
                # We can just iterate through set bits of N[v]
                # But note: w cannot be u (since u not in N(v) initially)
                # And w cannot be v (no self loop)
                # N[v] contains neighbors of v.
                # If we add (u,v), u becomes neighbor of v.
                # The condition is derived from the fact that u becomes a common neighbor of v and any w in N(v).
                # So we need u NOT to be in N(w) for any w in N(v).
                # i.e., N[u] & N[w] == 0? No.
                # u is a vertex. We check if u is in N(w).
                # That is (N[w] >> u) & 1.
                # So we need for all w in N(v), (N[w] >> u) & 1 == 0.
                # This means u is not a neighbor of any neighbor of v.
                # i.e., u is not at distance 2 from v via a path v-w-u?
                # Wait. If u is neighbor of w, and w is neighbor of v, then u-w-v is a path.
                # Adding u-v creates cycle u-v-w-u? That's a triangle (C3).
                # The task says C4-free. Triangles are allowed?
                # "containing no C4". It does not say triangle-free.
                # However, the condition derived earlier:
                # "For all w in N(v), common[u][w] must be 0"
                # common[u][w] is |N(u) cap N(w)|.
                # If we add (u,v), v becomes a common neighbor of u and w (if w in N(v)).
                # So common[u][w] increases by 1.
                # We need new common[u][w] <= 1.
                # So old common[u][w] must be 0.
                # Old common[u][w] = |N(u) cap N(w)|.
                # So we need N[u] & N[w] == 0 for all w in N(v).
                
                # Let's verify this logic.
                # If N[u] & N[w] != 0, there is x in N(u) cap N(w).
                # Then x-u-w-x is a triangle? No, x-u, u-w, w-x.
                # If we add u-v, and w is in N(v), then v-u-w-v is a C4?
                # v-u (new), u-w (exists), w-v (exists).
                # This is a triangle v-u-w-v.
                # A triangle is not a C4.
                # So why did the original code check common[u][w] > 0?
                # Let's re-read the original code logic.
                # "For all w in N(v) (w != u), common[u][w] must be 0"
                # "Because if w is neighbor of v, and we add u-v, v becomes a common neighbor of u and w."
                # "So common[u][w] increases by 1."
                # "We need new common[u][w] <= 1 => old common[u][w] <= 0."
                # This logic assumes that we need to maintain |N(a) cap N(b)| <= 1 for ALL pairs a,b.
                # Is that equivalent to C4-free?
                # Yes, as proven before: Graph is C4-free iff |N(a) cap N(b)| <= 1 for all a != b.
                # So we MUST maintain this invariant.
                # So yes, we need N[u] & N[w] == 0 for all w in N(v).
                
                # So the check is:
                # For all w such that (N[v] >> w) & 1:
                #    if (N[u] & N[w]) != 0: return False
                
                # Similarly for u's neighbors:
                # For all w such that (N[u] >> w) & 1:
                #    if (N[v] & N[w]) != 0: return False

                # Optimization:
                # We can check if (N[u] & N[w]) is 0 for all w in N[v].
                # This is equivalent to: N[u] is disjoint from the union of N[w] for w in N[v].
                # Let U_v = 0
                # for w in range(target_n):
                #   if (N[v] >> w) & 1:
                #     U_v |= N[w]
                # if N[u] & U_v: return False
                
                # And symmetric for N[v] and U_u.
                
                # Since n is small (47), we can just iterate.
                # But we can make it faster by precomputing? No, N changes.
                # We'll just iterate over set bits.
                
                # Iterate w in N[v]
                mask = N[v]
                while mask:
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    if N[u] & N[w]:
                        return False
                    mask ^= lsb
                    
                # Iterate w in N[u]
                mask = N[u]
                while mask:
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    if N[v] & N[w]:
                        return False
                    mask ^= lsb

                return True

            def add_edge(u, v):
                N[u] |= (1 << v)
                N[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1

            # Greedy loop
            success = True
            # Max steps: total edges is n(n-1)/2.
            # But we stop when min_deg is met.
            # min_deg_needed is small (n-39).
            # For n=47, min_deg >= 8.
            # Total edges at least n*8/2 = 188.
            # Max edges ~ 1000.
            
            for _ in range(500): # Max steps
                if time.time() > deadline:
                    success = False
                    break
                    
                # Find vertex with min degree
                min_deg = min(deg)
                if min_deg >= min_deg_needed:
                    success = True
                    break
                    
                low_vertices = [i for i in range(target_n) if deg[i] < min_deg_needed]
                if not low_vertices:
                    success = True
                    break
                    
                u = random.choice(low_vertices)
                
                # Find a valid v to connect to u
                candidates = []
                for v in range(target_n):
                    if v == u:
                        continue
                    if can_add(u, v):
                        candidates.append(v)
                        
                if not candidates:
                    success = False
                    break
                
                v = random.choice(candidates)
                add_edge(u, v)
                
            if success:
                # Final verification
                # Check C4-free: |N[u] & N[v]| <= 1 for all u != v
                c4_free = True
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        common = N[i] & N[j]
                        if common != 0 and (common & (common - 1)) != 0:
                            c4_free = False
                            break
                    if not c4_free:
                        break
                
                if c4_free and min(deg) >= min_deg_needed:
                    final_edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if (N[i] >> j) & 1:
                                final_edges.append([i, j])
                    
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = final_edges
                    found = True
                    break # Break restart loop
        
        if best_n == 47:
            break
        if found:
            break

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback: n=39, empty graph is valid (min deg 0 >= 0)
        return {'n': 39, 'edges': []}
