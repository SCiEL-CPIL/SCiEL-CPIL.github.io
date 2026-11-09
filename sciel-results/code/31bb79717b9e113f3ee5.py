import time
import math
from math import gcd

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    N = 128
    full_mask = (1 << N) - 1  # bits 0..N-1 set, representing numbers 1..N
    
    # Precompute LCM table
    # Map number i to index i-1
    lcm_table = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i, N):
            a = i + 1
            b = j + 1
            lcm_table[i][j] = a * b // gcd(a, b)
            lcm_table[j][i] = lcm_table[i][j]
            
    # Identify bad triples and build conflict graph
    # conflict[i] is a bitmask of j such that (i, j) is in a bad triple
    conflict = [0] * N
    
    # For each LCM value, collect pairs that have that LCM
    # If a value L has 3 or more pairs sharing it, we need to check combinations
    # Actually, a triple (a,b,c) is bad if lcm(a,b)=lcm(a,c)=lcm(b,c)=L.
    # This implies that for a specific L, the set of numbers that form pairs with LCM L
    # must contain a triangle.
    
    # More efficient:
    # Iterate all pairs (a,b). Let L = lcm(a,b).
    # Find all c such that lcm(a,c)=L and lcm(b,c)=L.
    # If such c exists, then (a,b) is part of a bad triple.
    
    # Optimization: Group numbers by their prime factorization properties?
    # Simpler: Just iterate. N=128 is small.
    # Pairs: ~8000. For each pair, check against all other ~128 numbers. 8000*128 = 1M ops. Fast.
    
    for a in range(N):
        for b in range(a + 1, N):
            lab = lcm_table[a][b]
            for c in range(b + 1, N):
                if lab == lcm_table[a][c] == lcm_table[b][c]:
                    # (a, b, c) is a bad triple
                    conflict[a] |= (1 << b)
                    conflict[a] |= (1 << c)
                    conflict[b] |= (1 << a)
                    conflict[b] |= (1 << c)
                    conflict[c] |= (1 << a)
                    conflict[c] |= (1 << b)
    
    # We want Max Independent Set in the conflict graph.
    # This is equivalent to Max Clique in the complement graph.
    # Let's build the complement adjacency lists (bitmasks).
    # comp_adj[i] contains bits j where NOT (i==j or conflict[i] has bit j)
    
    comp_adj = [0] * N
    for i in range(N):
        # Start with all bits set
        mask = full_mask
        # Remove self
        mask &= ~(1 << i)
        # Remove conflict neighbors
        mask &= ~conflict[i]
        comp_adj[i] = mask

    best_size = 0
    best_set = []
    
    # Bron-Kerbosch with Pivoting for Max Clique in comp_graph
    # R: current clique
    # P: candidates that can extend the clique
    # X: candidates that have already been processed (to avoid duplicates)
    
    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        # Pruning: even if we add all P, we can't beat best_size
        if len(R) + P.bit_count() <= best_size:
            return
            
        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_set = R[:]
            return
        
        # Choose pivot u from P | X with maximum degree in P
        # To speed up, we can just pick the first one or one with high degree.
        # Let's find a good pivot.
        union = P | X
        if union == 0:
            return
            
        # Heuristic: pick u in P|X that maximizes popcount(P & comp_adj[u])
        # This minimizes the number of branches.
        # Since N is small, we can iterate.
        pivot = -1
        max_deg = -1
        temp = union
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            deg = (P & comp_adj[u]).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = u
            temp ^= lsb
            
        # If no pivot found (should not happen if union!=0), pick any
        if pivot == -1:
            lsb = union & -union
            pivot = lsb.bit_length() - 1
            
        # Iterate over candidates v in P \ N(pivot)
        # N(pivot) in the context of the subgraph induced by P
        # The neighbors of pivot in the complement graph are comp_adj[pivot].
        # We want v such that v is in P and v is NOT a neighbor of pivot in the comp graph?
        # Wait. Bron-Kerbosch on Graph G.
        # P is the set of vertices adjacent to all vertices in R.
        # We want to extend R by v. v must be in P.
        # Pivoting: Let u be a vertex in P ∪ X.
        # We only need to consider v in P \ N(u).
        # Why? Because any clique containing v and u will be found when u is processed (or was processed).
        # Specifically, if v is adjacent to u, the clique R ∪ {v, ...} that includes u will be generated in the branch for u.
        # So we skip v if v is adjacent to u.
        
        candidates = P & ~comp_adj[pivot]
        
        while candidates:
            if time.time() > deadline:
                return
            
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # Recurse
            new_R = R + [v]
            new_P = P & comp_adj[v]
            new_X = X & comp_adj[v]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            # Update P and X
            P ^= lsb
            X |= lsb
            candidates ^= lsb

    # Initial call
    # R is empty, P is all vertices, X is empty
    bron_kerbosch([], full_mask, 0)
    
    # If best_set is empty (should not happen, at least 1 element is valid), return [1]
    if not best_set:
        return [1]
        
    # Convert indices back to values (1-based)
    return [x + 1 for x in best_set]
