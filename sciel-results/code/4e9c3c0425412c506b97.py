import time
from math import gcd

def construct():
    N = 128
    deadline = time.time() + 50
    
    # 1. Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # 2. Build Conflict Graph G
    # Edge (u, v) in G if there exists w != u, v such that gcd(u,v) == gcd(u,w) == gcd(v,w)
    # We represent G and its complement H using bitsets (integers)
    # adj_G[i] is a bitmask of neighbors of i in G
    adj_G = [0] * (N + 1)
    
    # Iterate over all triples to find conflicts
    # If (u, v, w) is a bad triple, add edges (u,v), (v,w), (u,w) to G
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            g_uv = gcd_table[u][v]
            for w in range(v + 1, N + 1):
                if gcd_table[v][w] == g_uv and gcd_table[w][u] == g_uv:
                    # Conflict found
                    adj_G[u] |= (1 << v)
                    adj_G[v] |= (1 << u)
                    adj_G[v] |= (1 << w)
                    adj_G[w] |= (1 << v)
                    adj_G[u] |= (1 << w)
                    adj_G[w] |= (1 << u)

    # 3. Build Complement Graph H
    # Edge (u, v) in H if NO conflict exists (i.e., no edge in G)
    # Note: We only consider u < v for the graph structure, but bitsets are symmetric
    adj_H = [0] * (N + 1)
    for i in range(1, N + 1):
        # All other nodes
        all_nodes = ((1 << (N + 1)) - 1) ^ (1 << i) # Mask with i removed
        # H neighbors are those NOT in G neighbors
        adj_H[i] = all_nodes & ~adj_G[i]

    # 4. Tomita's Algorithm for Maximum Clique in H
    # Vertices are 1..N
    
    # Pre-sort vertices by degree in H (descending) to improve heuristic
    # Degree in H is popcount(adj_H[i])
    degrees = [bin(adj_H[i]).count('1') for i in range(N + 1)]
    # Sort indices 1..N by degree descending
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda x: degrees[x], reverse=True)
    
    # Map original vertex to new index? 
    # Tomita's algorithm usually works on the set of vertices directly.
    # We can just use the bitset operations.
    
    best_clique = 0
    best_set = 0
    
    def max_clique(P, X):
        nonlocal best_clique, best_set
        if time.time() > deadline:
            return
            
        if P == 0:
            if len(bin(X).count('1')) > best_clique:
                best_clique = len(bin(X).count('1'))
                best_set = X
            return
        
        # Greedy Coloring of P
        # We want to color P such that adjacent vertices in H have different colors.
        # The max color is an upper bound on the clique size extendable from P.
        U = P
        colors = {} # vertex -> color
        max_color = 0
        color_order = [] # List of (vertex, color) to iterate in reverse color order
        
        while U != 0:
            max_color += 1
            W = U
            while W != 0:
                # Pick a vertex from W
                # To make it deterministic and fast, pick the lowest set bit
                v = (W & -W).bit_length() - 1
                # Remove v from W and U
                W ^= (1 << v)
                U ^= (1 << v)
                colors[v] = max_color
                color_order.append((v, max_color))
                # Remove neighbors of v in H from W
                # Neighbors are in adj_H[v]
                W &= ~adj_H[v]
                
        # Pruning
        if len(bin(X).count('1')) + max_color <= best_clique:
            return
            
        # Iterate vertices in reverse order of color (highest color first)
        # This is the key to Tomita's algorithm
        color_order.sort(key=lambda x: x[1], reverse=True)
        
        for v, c in color_order:
            # Check if v is still in P (it should be, as we iterated over a copy)
            if not (P & (1 << v)):
                continue
                
            # Prune based on color
            if len(bin(X).count('1')) + c <= best_clique:
                break
                
            # Add v to clique
            new_X = X | (1 << v)
            # New P is P intersected with neighbors of v in H
            new_P = P & adj_H[v]
            
            max_clique(new_P, new_X)
            
            # Remove v from P for the next iteration
            P &= ~(1 << v)
            
            # Update bound
            # The color of remaining vertices might effectively decrease, 
            # but Tomita's original implementation relies on the initial coloring 
            # and the fact that we remove vertices. 
            # Actually, the standard implementation just iterates.

    # Initial call
    # P is all vertices 1..N
    initial_P = 0
    for i in range(1, N + 1):
        initial_P |= (1 << i)
        
    max_clique(initial_P, 0)
    
    # Extract result
    result = []
    for i in range(1, N + 1):
        if best_set & (1 << i):
            result.append(i)
            
    if not result:
        return [1]
        
    return result
