import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def to_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def count_k4(adj, n):
        """Count all K4 subgraphs exactly."""
        count = 0
        for i in range(n - 3):
            for j in range(i + 1, n - 2):
                if not ((adj[i] >> j) & 1):
                    continue
                for k in range(j + 1, n - 1):
                    if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1):
                        continue
                    for l in range(k + 1, n):
                        if ((adj[i] >> l) & 1) and ((adj[j] >> l) & 1) and ((adj[k] >> l) & 1):
                            count += 1
        return count

    def has_i6(adj, n):
        """Check if there's an independent set of size 6 via backtracking."""
        # In complement, we need a K6.
        # Backtracking: pick vertices one by one, each must be non-adjacent to all previously picked.
        def bt(candidates, depth):
            if depth == 6:
                return True
            if len(candidates) < depth:
                return False
            for idx in range(len(candidates)):
                v = candidates[idx]
                new_cands = []
                for u in candidates[idx + 1:]:
                    if not ((adj[v] >> u) & 1):
                        new_cands.append(u)
                if bt(new_cands, depth + 1):
                    return True
            return False
        return bt(list(range(n)), 0)

    def is_valid(adj, n):
        if count_k4(adj, n) > 0:
            return False
        if has_i6(adj, n):
            return False
        return True

    # Known valid constructions as fallbacks
    def paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def clebsch_16():
        # Clebsch graph: 16 vertices, complement of Shrikhande
        # Actually let's use a known K4-free graph
        # The Clebsch graph is K4-free? Let me verify.
        # Vertices are binary strings of length 4. Edge if Hamming distance 1 or 6? No.
        # Clebsch: vertices {0,1}^4, edge if dist=1 or dist=... 
        # Actually the Clebsch graph has 16 vertices, 120 edges, is strongly regular (16,5,0,2)
        # It is K4-free (since lambda=0 means no triangles among neighbors... wait)
        # lambda=0: adjacent vertices have 0 common neighbors -> no triangles at all!
        # So definitely K4-free.
        # Independence number of Clebsch is 5? Let me check.
        # Actually alpha(Clebsch) = 5. So no I6. 
        # Wait, if alpha=5, then it's valid!
        # Let me construct it.
        # Clebsch: vertices are all 4-bit strings. Two vertices adjacent if they differ in exactly 1 or... 
        # No. The Clebsch graph is the folded 5-cube? No.
        # The Clebsch graph: V = {0,1}^4. Adjacent if Hamming distance is 1 or... 
        # Actually it's the graph where two 4-bit strings are adjacent if their dot product mod 2 is 0 and they're different? No.
        # Let me just use a different known graph.
        # The 16-vertex Clebsch: strongly regular (16,5,0,2). 
        # Construction: vertices are subsets of {1,2,3,4} of even size? No, that's 8 vertices.
        # Actually: take all 16 binary strings of length 4. Two are adjacent if they differ in exactly 1 position AND... no.
        # Let me look this up mentally: The Clebsch graph is the complement of the Shrikhande graph? No.
        # The Clebsch graph: V = {0,1}^4, edge between x and y if d(x,y) = 1 or d(x,y) = ... 
        # I recall the Clebsch graph has no triangles (lambda=0). Let me just construct it as:
        # Two vertices are adjacent if their bitwise XOR has popcount 1 or popcount 6? No, max popcount is 4.
        # OK let me just use a different approach. I'll use the 17-vertex Paley as fallback.
        return None

    # Try incremental backtracking for larger n
    # We build the graph one vertex at a time.
    # For each new vertex, we decide its edges to previous vertices.
    # We maintain: for each triple of previous vertices that form a path, track if adding edges would create K4.
    # This is complex. Let me try a different structural approach:
    
    # Random restart local search with smarter initialization and repair
    def count_k3_fast(adj, n):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    mask = ((1 << n) - 1) ^ ((1 << (j + 1)) - 1)
                    count += bin(adj[i] & adj[j] & mask).count('1')
        return count

    def count_i3_fast(adj, n):
        count = 0
        mask = (1 << n) - 1
        for i in range(n):
            for j in range(i + 1, n):
                if not ((adj[i] >> j) & 1):
                    non_neigh_i = ~adj[i] & mask
                    non_neigh_j = ~adj[j] & mask
                    mask_k = mask ^ ((1 << (j + 1)) - 1)
                    count += bin(non_neigh_i & non_neigh_j & mask_k).count('1')
        return count

    def repair_adj(adj, n, max_iters=2000):
        """Iteratively repair K4 and I6 violations by flipping edges."""
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Check K4
            k4_found = False
            for i in range(n - 3):
                if k4_found:
                    break
                for j in range(i + 1, n - 2):
                    if k4_found:
                        break
                    if not ((adj[i] >> j) & 1):
                        continue
                    for k in range(j + 1, n - 1):
                        if k4_found:
                            break
                        if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1):
                            continue
                        for l in range(k + 1, n):
                            if ((adj[i] >> l) & 1) and ((adj[j] >> l) & 1) and ((adj[k] >> l) & 1):
                                # Found K4: {i,j,k,l}. Remove a random edge.
                                edge_pairs = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
                                a, b = random.choice(edge_pairs)
                                adj[a] &= ~(1 << b)
                                adj[b] &= ~(1 << a)
                                k4_found = True
                                break
            
            if k4_found:
                continue
            
            # Check I6
            i6_found = False
            # Use backtracking to find I6
            def find_i6(adj, n):
                def bt(candidates, depth, chosen):
                    if depth == 6:
                        return chosen[:]
                    if len(candidates) < depth:
                        return None
                    for idx in range(len(candidates)):
                        v = candidates[idx]
                        new_cands = []
                        for u in candidates[idx + 1:]:
                            if not ((adj[v] >> u) & 1):
                                new_cands.append(u)
                        result = bt(new_cands, depth + 1, chosen + [v])
                        if result is not None:
                            return result
                    return None
                return bt(list(range(n)), 0, [])
            
            i6 = find_i6(adj, n)
            if i6 is not None:
                # Add a random edge within the independent set
                a, b = random.sample(i6, 2)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                i6_found = True
            
            if not k4_found and not i6_found:
                return True  # Valid!
        
        return False

    # Main search: try multiple n values with random init + repair
    best_result = {'n': 17, 'edges': to_edges(paley_17(), 17)}
    
    for target_n in [35, 34, 33, 32, 30, 28, 25, 22, 20]:
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts
        max_restarts = 5
        if time.time() > deadline - 60:
            max_restarts = 2
        
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break
            
            # Random initialization with edge probability tuned to avoid K4
            # For K4-free, max edges ~ n^2/4 (Mantel's theorem for triangle-free, but K4-free allows more)
            # Actually K4-free graphs can have up to ~n^2/3 edges (by Turan's theorem, T_3(n))
            # But we also need no I6, so density matters.
            # Let's try p around 0.3-0.5
            p = random.uniform(0.25, 0.5)
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Repair
            if repair_adj(adj, target_n, max_iters=3000):
                if target_n > best_result['n']:
                    best_result = {'n': target_n, 'edges': to_edges(adj, target_n)}
                # Found valid, try larger n
                break
    
    # Final check: if we found n > 17, return it; else return Paley 17
    if best_result['n'] > 17:
        return best_result
    
    # Fallback: try a few more quick repairs on n=20
    if time.time() < deadline - 20:
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            adj = [0] * 20
            for i in range(20):
                for j in range(i + 1, 20):
                    if random.random() < 0.4:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if repair_adj(adj, 20, max_iters=5000):
                return {'n': 20, 'edges': to_edges(adj, 20)}
    
    return {'n': 17, 'edges': to_edges(paley_17(), 17)}
