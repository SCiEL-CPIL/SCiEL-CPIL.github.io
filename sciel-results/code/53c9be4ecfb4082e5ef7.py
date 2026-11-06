import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 512
    # 1. Precompute LCM compatibility
    # neighbors[i] is a bitmask where bit j is set if lcm(i, j) <= 512
    # We use 0-indexed for the bitmask, so vertex v corresponds to bit (v-1)
    neighbors = [0] * (N + 1)
    
    # Optimization: Only compute for i <= j, then symmetrize
    # But bitmask construction is easier if we just loop all pairs
    # 512*512 = 262144 operations. Very fast.
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            # Calculate lcm
            # lcm(i, j) = i * j // gcd(i, j)
            # To avoid overflow (not an issue in Python) and speed up:
            # if i // gcd(i, j) * j <= 512
            g = gcd(i, j)
            if (i // g) * j <= 512:
                mask |= (1 << (j - 1))
        neighbors[i] = mask
        
    # 2. Find Maximum Clique using Bron-Kerbosch with pivot
    # We represent sets of vertices as Python integers (bitsets)
    
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        # Check time limit
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            # Found a maximal clique
            # R is a list of vertices
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
        
        # Choose pivot u from P union X
        # We want to minimize |P & neighbors[u]|
        # Iterate through candidates in P | X
        # To speed up, we can pick a random one or the one with highest degree in P?
        # Standard: pick any u in P|X.
        # Better: pick u that maximizes |P & neighbors[u]| to minimize the loop?
        # No, Bron-Kerbosch says: pick u in P|X such that |P & neighbors[u]| is MAXIMIZED?
        # Actually, the pivot rule is: choose u in P ∪ X.
        # Then iterate over v in P \ N(u).
        # This reduces the number of recursive calls.
        # So we want N(u) to cover as much of P as possible.
        # So we want to MAXIMIZE |P & neighbors[u]|.
        
        PX = P | X
        if PX == 0:
            return
            
        # Find pivot u
        # Iterate through bits in PX
        # For efficiency, we can just pick the first one, but a good pivot is better.
        # Let's pick u with max |P & neighbors[u]|
        # This requires iterating through all bits in PX.
        # PX has at most 512 bits.
        max_overlap = -1
        pivot = -1
        temp = PX
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            u = lsb.bit_length() - 1 + 1 # index 1..512
            # Calculate overlap
            overlap = P & neighbors[u]
            # Count bits
            # Python int.bit_count() is fast
            count = overlap.bit_count()
            if count > max_overlap:
                max_overlap = count
                pivot = u
            temp ^= lsb
            
        if pivot == -1:
            return

        # Candidates to recurse on: P \ N(pivot)
        # N(pivot) is neighbors[pivot]
        candidates = P & ~neighbors[pivot]
        
        # Iterate through candidates
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1 + 1
            temp ^= lsb
            
            # Recursive call
            # R' = R + {v}
            # P' = P & N(v)
            # X' = X & N(v)
            bron_kerbosch(R + [v], P & neighbors[v], X & neighbors[v])
            
            # Update P and X
            P &= ~lsb
            X |= lsb
            
            if time.time() > deadline:
                return

    # Initial call
    # P = all vertices (1..512)
    # X = 0
    initial_P = 0
    for i in range(1, N + 1):
        initial_P |= (1 << (i - 1))
        
    # To help BK, we can order vertices by degree?
    # BK doesn't strictly require ordering, but it helps if we process high degree first?
    # The pivot selection handles some of this.
    # However, if we just run it, it might be slow if the first maximal clique is small.
    # But we track best_clique.
    
    # One optimization: Greedy initialization of best_clique
    # Let's find a greedy clique first to have a good lower bound.
    # Greedy: pick vertex with highest degree, add to clique, remove non-neighbors.
    # Repeat.
    
    def greedy_clique():
        P = initial_P
        R = []
        while P:
            # Pick vertex with max degree in P
            best_v = -1
            max_deg = -1
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1 + 1
                deg = (P & neighbors[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
                
            R.append(best_v)
            # Remove best_v from P
            P &= ~(1 << (best_v - 1))
            # Remove non-neighbors of best_v from P
            P &= neighbors[best_v]
            
        return R

    # Initialize best_clique with greedy result
    best_clique = greedy_clique()
    
    # Run BK
    # To avoid deep recursion limits or time issues, we rely on the deadline check.
    # Also, we can limit the depth or use a heuristic to prune if current R size + P size <= best_clique size?
    # Pruning: if len(R) + P.bit_count() <= len(best_clique), return.
    
    def bron_kerbosch_optimized(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        # Pruning
        if len(R) + P.bit_count() <= len(best_clique):
            return
            
        PX = P | X
        if PX == 0:
            return
            
        # Find pivot u maximizing |P & N(u)|
        max_overlap = -1
        pivot = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length() - 1 + 1
            overlap = P & neighbors[u]
            count = overlap.bit_count()
            if count > max_overlap:
                max_overlap = count
                pivot = u
            temp ^= lsb
            
        if pivot == -1:
            return

        candidates = P & ~neighbors[pivot]
        
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1 + 1
            temp ^= lsb
            
            bron_kerbosch_optimized(R + [v], P & neighbors[v], X & neighbors[v])
            
            P &= ~lsb
            X |= lsb
            
            if time.time() > deadline:
                return

    bron_kerbosch_optimized([], initial_P, 0)
    
    return best_clique
