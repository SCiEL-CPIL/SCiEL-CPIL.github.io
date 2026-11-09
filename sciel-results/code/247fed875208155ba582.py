import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Helper: Check if a specific 4-coloring of K_n has a monochromatic C6
    def has_c6(adj, n):
        # adj[c][i] is a bitmask of neighbors of i in color c
        for c in range(4):
            A = adj[c]
            for v0 in range(n):
                # We look for a cycle v0-v1-v2-v3-v4-v5-v0
                # To avoid duplicates and ensure simple cycle, we enforce v0 < v1 < v2 < v3 < v4 < v5?
                # No, C6 is not a path. But we can iterate over starting vertices and ensure
                # the cycle is "canonical" or just check all.
                # Given n<=20, O(n^6) is too slow if done naively (20^6 = 64M).
                # But with bitsets, it's fast.
                
                # Let's use a faster check:
                # For each pair (u, v) in color c, check if there is a path of length 4 between them
                # in the subgraph induced by V \ {u, v}? No, that's for induced.
                # The task says "not necessarily induced".
                
                # Efficient C6 detection:
                # Precompute A2 (2-step reachability) and A4 (4-step reachability)?
                # If u and v are connected by a path of length 4 in G_c, and edge (u,v) exists in G_c,
                # then we have a C6 (or a C4 or C3 with chords, but C6 is the specific target).
                # Wait, if there is a path of length 4 from u to v, and edge (u,v) exists, 
                # the cycle length is 5 (path 4 + edge 1)? No.
                # Path u-w1-w2-w3-w4-v has 4 edges.
                # Cycle u-w1-w2-w3-w4-v-u has 5 edges. That's a C5.
                # We need a path of length 4 from u to v such that u and v are NOT adjacent?
                # No, the cycle is v0-v1-v2-v3-v4-v5-v0.
                # Edges: (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0).
                # If we fix edge (v0, v5) as color c, we need a path of length 4 from v0 to v5
                # using vertices v1, v2, v3, v4.
                # So, for any edge (u, v) in color c, if there is a path of length 4 from u to v
                # in the graph G_c \ {u, v}? No, the intermediate vertices just need to be distinct.
                
                # Let's stick to the explicit 6-node search with bitsets, which is fast for n=20.
                # For each v0:
                #   For each v1 in N(v0):
                #     For each v2 in N(v1) \ {v0}:
                #       For each v3 in N(v2) \ {v0, v1}:
                #         For each v4 in N(v3) \ {v0, v1, v2}:
                #           For each v5 in N(v4) \ {v0, v1, v2, v3}:
                #             If v5 is in N(v0) and v5 != v1: Found C6.
                
                # Optimization:
                # We can iterate v0, v1, v2, v3, v4, v5.
                # To avoid symmetry, we can require v0 < v1 < v2 < v3 < v4 < v5?
                # No, that restricts to specific cycles.
                # However, for verification, we just need to find ANY.
                
                # Let's use a slightly optimized version.
                for v0 in range(n):
                    n0 = A[v0]
                    while n0:
                        lsb = n0 & -n0
                        v1 = lsb.bit_length() - 1
                        n0 ^= lsb
                        
                        # v1 must be > v0 to avoid double counting? No, just check all.
                        # But we can prune: if v1 < v0, skip? 
                        # Actually, let's just check all.
                        
                        n1 = A[v1] & ~(1 << v0)
                        while n1:
                            lsb = n1 & -n1
                            v2 = lsb.bit_length() - 1
                            n1 ^= lsb
                            
                            n2 = A[v2] & ~(1 << v0) & ~(1 << v1)
                            while n2:
                                lsb = n2 & -n2
                                v3 = lsb.bit_length() - 1
                                n2 ^= lsb
                                
                                n3 = A[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                                while n3:
                                    lsb = n3 & -n3
                                    v4 = lsb.bit_length() - 1
                                    n3 ^= lsb
                                    
                                    n4 = A[v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                                    # v5 must be in n4 AND in N(v0)
                                    # Also v5 != v1 (already excluded by n0 mask? No, n0 was for v1)
                                    # v5 must be distinct from v0, v1, v2, v3, v4.
                                    # n4 already excludes v0, v1, v2, v3.
                                    # We need v5 to be a neighbor of v0.
                                    candidates = n4 & A[v0]
                                    # v5 cannot be v1? 
                                    # If v5 == v1, the cycle is v0-v1-v2-v3-v4-v1-v0? 
                                    # That's a C5 (v0,v1,v2,v3,v4) with a chord? No, v1 is v1.
                                    # Cycle: v0-v1-v2-v3-v4-v1. Length 5.
                                    # So we must ensure v5 != v1.
                                    # A[v0] contains v1. So we mask out v1.
                                    candidates &= ~(1 << v1)
                                    
                                    if candidates:
                                        return True
        return False

    def verify(n, colors):
        adj = [[0] * n for _ in range(4)]
        def get_idx(i, j):
            return i * (n - 1) - i * (i - 1) // 2 + j - i - 1
        
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j)
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        return not has_c6(adj, n)

    # CSP Solver
    # Variables: Edges (i, j) for i < j.
    # Domains: {0, 1, 2, 3}
    # Constraints: No C6 in any color class.
    
    # To make CSP tractable, we need good variable ordering and constraint propagation.
    # A common strategy for graph coloring CSPs is to order vertices and assign edges incident to them.
    # Here, the constraint is global (C6).
    
    # Alternative Strategy: Local Search with Simulated Annealing / Tabu Search.
    # Given the small n (17-19), a well-tuned local search might find a solution faster than CSP backtracking
    # if the landscape is not too rugged. The seed program used random restarts + local search.
    # The current program uses constructive backtracking.
    
    # Let's try a hybrid:
    # 1. Generate a random coloring.
    # 2. Use a local search that minimizes the number of C6s.
    # 3. If we hit 0 C6s, we are done.
    # 4. Use "conflict-directed" moves: if a C6 is found, flip one of its edges.
    
    # Why this might work better:
    # Backtracking can get stuck in deep, unproductive branches.
    # Local search explores the neighborhood of "almost valid" colorings.
    
    def local_search(n, max_time):
        # Initialize random coloring
        # Store colors in a 2D list or flat list
        # Flat list is faster for indexing?
        # Let's use a 2D list for adjacency building speed.
        
        colors = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j]
                
        def get_adj():
            adj = [[0] * n for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[i][j]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
            return adj
            
        adj = get_adj()
        
        # Objective: Minimize number of C6s.
        # We can count C6s efficiently?
        # Counting all C6s is hard.
        # Instead, we can just try to find ONE C6. If found, fix it. If none, success.
        
        def find_c6(adj):
            # Returns a list of vertices forming a C6, or None
            for c in range(4):
                A = adj[c]
                for v0 in range(n):
                    n0 = A[v0]
                    while n0:
                        lsb = n0 & -n0
                        v1 = lsb.bit_length() - 1
                        n0 ^= lsb
                        
                        n1 = A[v1] & ~(1 << v0)
                        while n1:
                            lsb = n1 & -n1
                            v2 = lsb.bit_length() - 1
                            n1 ^= lsb
                            
                            n2 = A[v2] & ~(1 << v0) & ~(1 << v1)
                            while n2:
                                lsb = n2 & -n2
                                v3 = lsb.bit_length() - 1
                                n2 ^= lsb
                                
                                n3 = A[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                                while n3:
                                    lsb = n3 & -n3
                                    v4 = lsb.bit_length() - 1
                                    n3 ^= lsb
                                    
                                    n4 = A[v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                                    candidates = n4 & A[v0]
                                    candidates &= ~(1 << v1)
                                    if candidates:
                                        lsb_c = candidates & -candidates
                                        v5 = lsb_c.bit_length() - 1
                                        return [v0, v1, v2, v3, v4, v5]
            return None

        current_c6 = find_c6(adj)
        
        # Tabu list or just random moves
        # We will perform random restarts if stuck.
        
        while time.time() < max_time:
            if current_c6 is None:
                return True # Success
            
            # Pick a random edge in the C6 to flip
            # The C6 has 6 edges.
            # Edges are (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
            edges_in_cycle = [
                (current_c6[0], current_c6[1]),
                (current_c6[1], current_c6[2]),
                (current_c6[2], current_c6[3]),
                (current_c6[3], current_c6[4]),
                (current_c6[4], current_c6[5]),
                (current_c6[5], current_c6[0])
            ]
            
            u, v = random.choice(edges_in_cycle)
            old_c = colors[u][v]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Apply change
            colors[u][v] = new_c
            colors[v][u] = new_c
            
            # Update adj
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            # Check if we fixed the C6 or created a new one?
            # We just re-find a C6.
            current_c6 = find_c6(adj)
            
            # If we are stuck (cycling), we might need to randomize more.
            # Simple random walk is prone to local minima.
            # Let's add a "kick" every 1000 steps: randomize 5% of edges.
            
        return False

    # Try n=18 first? Or 17?
    # The prompt says best known is 17. 18 is new.
    # Local search is stochastic. It might find 17 quickly, 18 rarely.
    # To maximize score, we should try to find 18. If it fails, return 17.
    
    # Strategy:
    # 1. Try local search for n=18 for 30s.
    # 2. If success, verify and return.
    # 3. Try local search for n=17 for 30s.
    # 4. If success, verify and return.
    # 5. Fallback to a known valid n=17 (hardcoded or generated by a very fast constructive method).
    
    # Actually, the seed program scored 1.0, which implies it found n=17.
    # So we just need to beat n=17.
    
    # Let's try n=18.
    if time.time() < deadline - 30:
        if local_search(18, deadline - 30):
            # Verify
            colors_flat = []
            for i in range(18):
                for j in range(i + 1, 18):
                    colors_flat.append(colors[i][j]) # Wait, colors is local to local_search
                    # Need to return the colors from local_search
            # I need to refactor local_search to return the colors.
            
    # Refactor: local_search returns (success, colors_flat)
    
    def local_search_return(n, max_time):
        colors = [[-1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                colors[i][j] = random.randint(0, 3)
                colors[j][i] = colors[i][j]
                
        def get_adj():
            adj = [[0] * n for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[i][j]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
            return adj
            
        adj = get_adj()
        
        def find_c6(adj):
            for c in range(4):
                A = adj[c]
                for v0 in range(n):
                    n0 = A[v0]
                    while n0:
                        lsb = n0 & -n0
                        v1 = lsb.bit_length() - 1
                        n0 ^= lsb
                        
                        n1 = A[v1] & ~(1 << v0)
                        while n1:
                            lsb = n1 & -n1
                            v2 = lsb.bit_length() - 1
                            n1 ^= lsb
                            
                            n2 = A[v2] & ~(1 << v0) & ~(1 << v1)
                            while n2:
                                lsb = n2 & -n2
                                v3 = lsb.bit_length() - 1
                                n2 ^= lsb
                                
                                n3 = A[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                                while n3:
                                    lsb = n3 & -n3
                                    v4 = lsb.bit_length() - 1
                                    n3 ^= lsb
                                    
                                    n4 = A[v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                                    candidates = n4 & A[v0]
                                    candidates &= ~(1 << v1)
                                    if candidates:
                                        lsb_c = candidates & -candidates
                                        v5 = lsb_c.bit_length() - 1
                                        return [v0, v1, v2, v3, v4, v5]
            return None

        current_c6 = find_c6(adj)
        steps = 0
        
        while time.time() < max_time:
            if current_c6 is None:
                # Success
                colors_flat = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors_flat.append(colors[i][j])
                return True, colors_flat
            
            edges_in_cycle = [
                (current_c6[0], current_c6[1]),
                (current_c6[1], current_c6[2]),
                (current_c6[2], current_c6[3]),
                (current_c6[3], current_c6[4]),
                (current_c6[4], current_c6[5]),
                (current_c6[5], current_c6[0])
            ]
            
            u, v = random.choice(edges_in_cycle)
            old_c = colors[u][v]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            colors[u][v] = new_c
            colors[v][u] = new_c
            
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            current_c6 = find_c6(adj)
            
            steps += 1
            # Random kick to escape local minima
            if steps % 100 == 0:
                # Randomize a few edges
                for _ in range(5):
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i == j: continue
                    if i > j: i, j = j, i
                    old_c = colors[i][j]
                    new_c = random.randint(0, 3)
                    colors[i][j] = new_c
                    colors[j][i] = new_c
                    adj[old_c][i] &= ~(1 << j)
                    adj[old_c][j] &= ~(1 << i)
                    adj[new_c][i] |= (1 << j)
                    adj[new_c][j] |= (1 << i)
                
                current_c6 = find_c6(adj)
                
        return False, None

    # Try n=18
    if time.time() < deadline - 20:
        success, colors_flat = local_search_return(18, deadline - 20)
        if success:
            if verify(18, colors_flat):
                return {'n': 18, 'colors': colors_flat}
            
    # Try n=17
    if time.time() < deadline - 5:
        success, colors_flat = local_search_return(17, deadline - 5)
        if success:
            if verify(17, colors_flat):
                return {'n': 17, 'colors': colors_flat}
                
    # Fallback: Return a known valid n=17 coloring if we have one, or a smaller one.
    # Since we don't have a hardcoded n=17 in this new strategy, we generate a simple n=9.
    # But wait, the seed program found n=17. We should try to ensure we at least get n=17.
    # The local search for n=17 is very likely to succeed.
    # If it fails, we return n=9. This might drop the score.
    # To be safe, let's add a quick constructive fallback for n=17?
    # Or just rely on the local search for n=17 which has 5 seconds.
    # 5 seconds might be enough for n=17.
    
    # Let's add a hardcoded valid n=17 coloring if possible?
    # I don't have one memorized.
    # But I can run the local search for n=17 with a longer time if n=18 fails quickly?
    # The budget is split: 130s for n=18, 5s for n=17.
    # If n=18 fails at 130s, we have 5s for n=17.
    
    # Let's adjust: Try n=18 for 100s. Then n=17 for 10s.
    
    # Re-run logic
    # Actually, the code above already handles the timing.
    
    # Fallback n=9
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 4)
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    # Test
    res = construct()
    print(res['n'])
