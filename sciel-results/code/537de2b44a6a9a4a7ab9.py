import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Fallback solution: Andrasfai graph And(9) with n=26
    def get_andrasfai_9():
        n = 26
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    best_result = get_andrasfai_9()
    best_n = 26

    # Helper to check if a graph (defined by adjacency matrix or set) is valid
    def is_valid_graph(n, adj_matrix):
        # Check triangle-free
        # adj_matrix[i][j] is 1 if edge exists
        # Triangle i,j,k exists if adj[i][j] and adj[j][k] and adj[k][i]
        # We can do this efficiently with bitsets or loops
        # For n<=40, O(n^3) is 64000 ops, very fast.
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i][j]:
                    for k in range(j + 1, n):
                        if adj_matrix[i][k] and adj_matrix[j][k]:
                            return False
        # Check alpha < 10
        # Use a simple branch and bound for Max Independent Set
        # Since n is small, we can use a bitset-based B&B
        
        # Precompute neighbors as bitmasks for speed
        # Actually, for n<=40, we can use Python integers as bitsets.
        # adj_mask[i] has bit j set if i~j
        
        adj_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and adj_matrix[i][j]:
                    mask |= (1 << j)
            adj_masks[i] = mask
        
        # Max Independent Set via B&B
        # State: available vertices (bitmask), current independent set size
        # We want to know if there is an independent set of size 10.
        # If yes, return False (invalid). If max size < 10, return True.
        
        # To speed up, we can use a greedy upper bound.
        # But B&B with bitsets is fast for n=40.
        
        def has_independent_set_of_size_k(k, masks, n):
            # Returns True if there is an independent set of size k
            # Use DFS
            # Optimization: if popcount(avail) < k, return False
            # Order vertices by degree (heuristic)
            
            # We'll implement a simple recursive search with memoization or pruning
            # Given time constraints, a simple DFS might be enough if k=10 is small relative to n=40?
            # Actually, finding an independent set of size 10 in a random graph is easy.
            # Finding that NO such set exists (i.e., proving alpha < 10) is the hard part.
            # So we need to prove alpha <= 9.
            
            # This is equivalent to checking if the independence number is <= 9.
            # We can use a standard B&B algorithm for Maximum Independent Set.
            
            # Let's use a simple recursive function
            # avail: bitmask of vertices that can still be added to the independent set
            # current_size: size of current independent set
            # best_size: global best found so far (to prune)
            
            # Actually, we just need to check if there EXISTS an independent set of size 10.
            # If we find one, we stop (invalid).
            # If we exhaust all possibilities and don't find one, it's valid.
            
            # To make it fast, we should order vertices.
            # Let's just do a DFS that tries to build an independent set of size 10.
            
            def dfs(avail, remaining):
                if remaining == 0:
                    return True
                if avail == 0:
                    return False
                # If popcount(avail) < remaining, return False
                if bin(avail).count('1') < remaining:
                    return False
                
                # Pick a vertex. Heuristic: pick vertex with lowest degree in avail?
                # Or just lowest index?
                # To find a solution quickly (to invalidate), picking any is fine.
                # To prove non-existence, we need good pruning.
                
                # Get lowest set bit
                v = (avail & -avail).bit_length() - 1
                
                # Option 1: Include v
                # If we include v, we remove v and its neighbors from avail
                new_avail = avail & ~(1 << v) & ~masks[v]
                if dfs(new_avail, remaining - 1):
                    return True
                
                # Option 2: Exclude v
                # Just remove v
                new_avail = avail & ~(1 << v)
                if dfs(new_avail, remaining):
                    return True
                
                return False
            
            initial_avail = (1 << n) - 1
            return dfs(initial_avail, k)

        if has_independent_set_of_size_k(10, adj_masks, n):
            return False
        return True

    # Search for Circulant Graphs
    # A circulant graph is defined by a set S of distances.
    # adj[i][j] is 1 if (j-i) % n is in S or (i-j) % n is in S.
    # Since S is symmetric, we just check (j-i) % n in S.
    
    # We will iterate n from 40 down to best_n + 1
    for n in range(40, best_n, -1):
        if time.time() > deadline:
            break
            
        # Try to find a valid S for this n
        # We can use a random local search
        # Start with a random S
        # Or start with a known good structure?
        
        # Let's try a few random restarts
        found = False
        for restart in range(50): # Max 50 restarts per n
            if time.time() > deadline:
                break
                
            # Initialize S
            # S must be symmetric: if d in S, n-d in S.
            # So we pick from 1 to n//2
            possible_distances = list(range(1, n // 2 + 1))
            random.shuffle(possible_distances)
            
            # Start with a small set, maybe 2 or 3 distances
            # Density needs to be high enough to keep alpha low, but low enough to be triangle-free.
            # Triangle-free condition for circulant:
            # If a, b in S, then a-b not in S and a+b not in S (mod n, careful with symmetry)
            # Actually, triangle 0, a, b exists if a in S, b in S, b-a in S.
            
            # Let's just generate a random S and check/repair
            # Better: Start with empty S, add edges randomly, if triangle forms, reject.
            # This generates a random triangle-free graph, but not necessarily circulant.
            # If we restrict to circulant, we can't just "reject edge" easily without breaking symmetry.
            
            # Alternative: Random S, then check validity.
            # Probability of random S being valid is low.
            
            # Let's use a heuristic:
            # Try to construct S such that it is a "sum-free" set in Z_n (for triangle-free).
            # A set S is sum-free if x+y in S implies x,y not in S? No.
            # Triangle 0, x, y: x in S, y in S, y-x in S.
            # So S must not contain {x, y, y-x}.
            
            # We can use a greedy construction of S.
            # Start with S = {}
            # Try to add distances d.
            # Check if adding d creates a triangle.
            # A triangle is formed if there exist a, b in S such that:
            # 1. d = a - b (mod n) -> Triangle 0, b, a (edges b, a, a-b=d). Wait, if d is added, we check against existing S.
            # If we add d, we check if d is of the form a-b for a,b in S?
            # If a, b in S, and d = a-b, then 0, b, a is a triangle (edges b, a, d).
            # Also d = a+b? Triangle 0, a, -b?
            
            # This is getting complicated. Let's just use the general random triangle-free graph generator.
            # Generate random edges, remove triangles.
            # Then check alpha.
            
            # But we want to maximize n.
            # Let's just try n=40, 39, etc. with a simple random triangle-free graph.
            
            # Generate random triangle-free graph on n vertices
            adj_matrix = [[0]*n for _ in range(n)]
            # Try to add edges
            # Random order of pairs
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(pairs)
            
            for i, j in pairs:
                # Check if adding (i,j) creates a triangle
                # Triangle with k exists if adj[i][k] and adj[j][k]
                # We can maintain a set of common neighbors
                # For simplicity, just check all k
                creates_triangle = False
                for k in range(n):
                    if k != i and k != j:
                        if adj_matrix[i][k] and adj_matrix[j][k]:
                            creates_triangle = True
                            break
                if not creates_triangle:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
                    
            # Check alpha
            if is_valid_graph(n, adj_matrix):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj_matrix[i][j]:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
                
    return best_result
