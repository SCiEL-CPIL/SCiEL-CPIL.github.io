import time
import random
import itertools

def is_valid_coloring(colors, n):
    """
    Check if the coloring of K_n is valid (no monochromatic K2,2,2).
    colors: list of 0/1 for each pair (i,j) i<j in lexicographic order.
    """
    # Build adjacency matrices for color 0 and color 1
    adj0 = [[0]*n for _ in range(n)]
    adj1 = [[0]*n for _ in range(n)]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            if c == 0:
                adj0[i][j] = 1
                adj0[j][i] = 1
            else:
                adj1[i][j] = 1
                adj1[j][i] = 1
            idx += 1
            
    # Function to check if a graph (adj matrix) contains K2,2,2
    def contains_k222(adj, n):
        # Find all non-edges
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == 0:
                    non_edges.append((i, j))
        
        # If there are fewer than 3 non-edges, no K2,2,2 possible
        if len(non_edges) < 3:
            return False, None

        # Precompute neighbors for speed? 
        # For each pair of disjoint non-edges, check if they form a K2,2
        # and if there is a third non-edge connecting to all 4.
        
        # Optimization: Group non-edges by their "connection profile" or just brute force with pruning.
        # n is small (<=31), so brute force on pairs of non-edges is okay.
        
        L = len(non_edges)
        for i in range(L):
            a, b = non_edges[i]
            for j in range(i+1, L):
                c, d = non_edges[j]
                # Check disjointness
                if a == c or a == d or b == c or b == d:
                    continue
                
                # Check if {a,b} and {c,d} form a K2,2 in adj
                # Need adj[a][c], adj[a][d], adj[b][c], adj[b][d] to be 1
                if adj[a][c] and adj[a][d] and adj[b][c] and adj[b][d]:
                    # We have a K2,2 on {a,b,c,d}.
                    # Now check if there exists a non-edge {e,f} disjoint from {a,b,c,d}
                    # such that e and f are connected to all a,b,c,d.
                    
                    # Candidates for e,f must be in N(a) ∩ N(b) ∩ N(c) ∩ N(d)
                    # And {e,f} must be a non-edge.
                    
                    # Find intersection of neighbors
                    # n is small, so we can just iterate
                    valid_third_pair = False
                    for k in range(j+1, L): # We can start from j+1 to avoid rechecking, but need to ensure disjointness with a,b,c,d
                        e, f = non_edges[k]
                        if e in (a,b,c,d) or f in (a,b,c,d):
                            continue
                        # Check if e and f are connected to all a,b,c,d
                        if (adj[e][a] and adj[e][b] and adj[e][c] and adj[e][d] and
                            adj[f][a] and adj[f][b] and adj[f][c] and adj[f][d]):
                            return True, [(a,b), (c,d), (e,f)]
        return False, None

    # Check color 0
    valid0, witness0 = contains_k222(adj0, n)
    if valid0:
        return False, 0, witness0
        
    # Check color 1
    valid1, witness1 = contains_k222(adj1, n)
    if valid1:
        return False, 1, witness1
        
    return True, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # 1. Try Paley(29) first as a strong baseline
    # Paley graph of order p (prime, p=1 mod 4)
    # Color 0 if (j-i) is a quadratic residue mod p
    def paley_coloring(n):
        if n == 29:
            p = 29
            squares = {x*x % p for x in range(1, p)}
            colors = []
            for i in range(p):
                for j in range(i+1, p):
                    diff = (j - i) % p
                    colors.append(0 if diff in squares else 1)
            return colors
        
    best_colors = None
    best_n = 0
    
    # Check Paley 29
    colors29 = paley_coloring(29)
    is_valid, _, _ = is_valid_coloring(colors29, 29)
    if is_valid:
        best_colors = colors29
        best_n = 29
    
    # 2. Try to find a valid coloring for n=30 using randomized local search
    # We will run simulated annealing for a few seconds.
    if time.time() < deadline:
        n = 30
        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
        
        # Helper to get/set edge
        def get_edge(i, j):
            if i > j: i, j = j, i
            return i*(n-1) - i*(i+1)//2 + (j - i - 1)
            # Formula for index of (i,j) in lexicographic list of pairs (0,1)..(0,n-1)..(n-2,n-1)
            # Count edges before i: 0..i-1. 
            # Edges starting with 0: n-1
            # Edges starting with 1: n-2
            # ...
            # Edges starting with i-1: n-i
            # Sum_{k=0}^{i-1} (n-k) = i*n - i(i+1)/2
            # Then offset by j-i-1? 
            # Pairs starting with i: (i, i+1), (i, i+2), ...
            # Index of (i,j) is sum_{k=0}^{i-1} (n-k) + (j - i - 1)
            # Let's re-verify.
            # n=3. Pairs: (0,1), (0,2), (1,2).
            # i=0, j=1: sum=0, offset=0. idx=0. Correct.
            # i=0, j=2: sum=0, offset=1. idx=1. Correct.
            # i=1, j=2: sum_{k=0}^{0} (3-k) = 3. Offset=0. idx=3. 
            # Wait, list is [ (0,1), (0,2), (1,2) ]. Indices 0, 1, 2.
            # My formula gives 3 for (1,2). Incorrect.
            # Sum of edges for vertices < i:
            # Vertex 0 has n-1 edges.
            # Vertex 1 has n-2 edges.
            # ...
            # Vertex i-1 has n-(i-1) edges.
            # Total = sum_{k=0}^{i-1} (n-1-k) = i(n-1) - i(i-1)/2.
            # For i=1: 1*(2) - 0 = 2. Correct (edges (0,1), (0,2)).
            # For i=0: 0. Correct.
            # So index = i(n-1) - i(i-1)/2 + (j - i - 1).
            
        def get_edge_idx(i, j):
            if i > j: i, j = j, i
            return i*(n-1) - i*(i-1)//2 + (j - i - 1)

        # Check validity
        valid, color_c, witness = is_valid_coloring(colors, n)
        
        # Simulated Annealing
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        
        while time.time() < deadline and T > min_T:
            if valid:
                # We found a solution!
                best_colors = colors
                best_n = n
                break
                
            # Pick a random edge to flip
            # Prefer edges in the witness if possible to break the violation
            if witness:
                # Witness is a list of 3 pairs of vertices
                # We can flip an edge between the parts or within? 
                # In K2,2,2, edges are between parts. 
                # To break it, we can flip one of the cross edges.
                # Or flip an edge within a part? No, parts are non-edges in the color graph.
                # If we flip an edge within a part to be present, it breaks the "non-edge" requirement for that pair?
                # Wait, the definition of K2,2,2 subgraph relies on the 3 pairs being independent sets in the color graph.
                # If we add an edge within a pair, that pair is no longer an independent set, so that specific K2,2,2 is broken.
                # So flipping an edge within a witness pair is a good move.
                # Also flipping a cross edge removes the connection.
                
                # Let's just pick a random edge from the witness cross-edges or internal edges
                # Witness: [(a,b), (c,d), (e,f)]
                # Internal edges: (a,b), (c,d), (e,f)
                # Cross edges: all pairs between these sets.
                # We pick one randomly.
                
                # For simplicity, just pick a random edge in the graph.
                # Or pick an edge incident to a witness vertex.
                u, v = random.choice(witness[0])
                # Flip a random edge connected to u?
                # Let's just pick a random edge index.
                idx = random.randint(0, n*(n-1)//2 - 1)
            else:
                idx = random.randint(0, n*(n-1)//2 - 1)
                
            # Flip
            colors[idx] = 1 - colors[idx]
            
            # Check validity
            valid, color_c, witness = is_valid_coloring(colors, n)
            
            # Update T
            T *= cooling_rate
            
        # If we didn't find a valid one, we might have improved the "validity score" but not reached 0 violations.
        # We return the best valid coloring found so far (best_n, best_colors).
        
    if best_colors is None:
        # Fallback to a small valid coloring if 29 failed (it shouldn't)
        # Paley 17
        p = 17
        squares = {x*x % p for x in range(1, p)}
        colors = []
        for i in range(p):
            for j in range(i+1, p):
                diff = (j - i) % p
                colors.append(0 if diff in squares else 1)
        return {'n': 17, 'colors': colors}
        
    return {'n': best_n, 'colors': best_colors}
