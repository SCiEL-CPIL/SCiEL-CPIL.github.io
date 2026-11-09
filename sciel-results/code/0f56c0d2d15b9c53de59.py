import time
import math

def construct():
    start_time = time.perf_counter()
    LIMIT = 512
    N = LIMIT
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # Precompute adjacency bitmasks
    # adj[i] is a bitmask where bit j is 1 if lcm(i, j+1) <= LIMIT
    # Note: We use 0-indexed bits for 1..N
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                # j is compatible with i
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
    
    # Self-compatibility is not needed for clique logic usually, but for completeness
    # A node is always compatible with itself in the context of "can I add it?"
    # But in Bron-Kerbosch, P contains candidates not yet in R.
    # We don't need self-loops in adj for the algorithm logic if we handle P correctly.

    best_clique_size = 0
    best_clique_mask = 0

    # Greedy Seed
    def greedy_seed():
        # Order by degree descending
        degrees = [(adj[i].bit_count(), i) for i in range(1, N + 1)]
        degrees.sort(reverse=True)
        
        R_mask = 0
        P_mask = (1 << N) - 1
        
        while P_mask:
            best_v = -1
            best_deg = -1
            temp = P_mask
            while temp:
                lsb = temp & -temp
                v_idx = (lsb.bit_length() - 1)
                v = v_idx + 1
                # Degree in P
                d = (adj[v] & P_mask).bit_count()
                if d > best_deg:
                    best_deg = d
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
            
            R_mask |= (1 << (best_v - 1))
            # Update P: must be compatible with best_v
            P_mask = P_mask & adj[best_v]
            P_mask &= ~(1 << (best_v - 1))
            
        return R_mask

    seed_mask = greedy_seed()
    best_clique_size = seed_mask.bit_count()
    best_clique_mask = seed_mask

    # Coloring Bound for Pruning
    # Returns an upper bound on the clique size in the subgraph induced by P
    def coloring_bound(P):
        U = P
        count = 0
        while U:
            count += 1
            I = 0
            # Find a maximal independent set in U
            V_avail = U
            while V_avail:
                # Pick a vertex, e.g., the one with lowest index for determinism/speed
                lsb = V_avail & -V_avail
                v_idx = lsb.bit_length() - 1
                v = v_idx + 1
                I |= lsb
                # Remove v and its neighbors from V_avail
                # Neighbors of v in the graph are adj[v]
                # We want to remove v and any node connected to v
                V_avail &= ~(adj[v] | lsb)
            # Remove the independent set I from U
            U &= ~I
        return count

    def bron_kerbosch_opt(R, P, X):
        nonlocal best_clique_size, best_clique_mask, deadline
        
        if time.perf_counter() > deadline:
            return False
        
        # Pruning
        r_size = R.bit_count()
        p_size = P.bit_count()
        
        # Simple bound
        if r_size + p_size <= best_clique_size:
            return True
            
        if P == 0:
            if r_size > best_clique_size:
                best_clique_size = r_size
                best_clique_mask = R
            return True
        
        PX = P | X
        if PX == 0:
            return True

        # Coloring Bound Pruning
        # If the max clique in P is at most K, then total clique is at most r_size + K
        # If r_size + K <= best_clique_size, prune
        # Computing coloring bound takes time, so only do it if p_size is small enough or if simple bound didn't prune
        # Heuristic: use coloring bound if p_size < 200 or so, or always?
        # Let's try always, as it's O(N^2) worst case but usually fast.
        # Optimization: Only compute if r_size + p_size > best_clique_size (which is true since we passed the first check)
        
        # A cheaper bound: Max degree in P + 1? No.
        # Let's use the coloring bound.
        
        # To speed up coloring bound, we can skip it if p_size is very large?
        # Actually, for N=512, bitset operations are fast.
        
        # We can also use a simpler bound: 
        # The number of vertices in P is an upper bound.
        # A better upper bound is the size of the maximum independent set? No, max clique <= min vertex cover + ... no.
        # Max clique <= number of colors in a greedy coloring of P.
        
        # Let's implement a fast check:
        # If r_size + coloring_bound(P) <= best_clique_size: return True
        
        # To avoid overhead of coloring bound on large P, we can use a heuristic.
        # But let's try it.
        
        # Select pivot u from P U X to minimize candidates
        # Heuristic: max degree in P
        max_pop = -1
        pivot = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u_idx = (lsb.bit_length() - 1)
            u = u_idx + 1
            pop = (adj[u] & P).bit_count()
            if pop > max_pop:
                max_pop = pop
                pivot = u
            temp ^= lsb
        
        if pivot == -1:
            return True

        candidates = P & ~adj[pivot]
        
        # Optional: Apply coloring bound here if candidates are small?
        # Or apply to P before branching?
        # Standard Tomita algorithm applies coloring to P to get an order and bound.
        
        # For now, let's stick to the pivot selection and simple bound.
        # If performance is an issue, we can add the coloring bound check here.
        
        # Let's add the coloring bound check for P
        # It might be expensive if P is large. 
        # Let's only do it if p_size < 100? Or just try it.
        
        # Actually, the simple bound `r_size + p_size` is weak.
        # Let's try to compute a tighter bound using a greedy coloring on P.
        
        # To save time, we can approximate or limit the coloring depth.
        
        # Let's try the coloring bound.
        # if r_size + coloring_bound(P) <= best_clique_size:
        #     return True
        
        # This line is commented out for now to see if it helps or hurts speed.
        # Given N=512, the search space might be small enough.
        
        while candidates:
            if time.perf_counter() > deadline:
                return False
            
            lsb = candidates & -candidates
            v_idx = (lsb.bit_length() - 1)
            v = v_idx + 1
            
            new_R = R | lsb
            new_P = P & ~adj[v]
            new_X = X & ~adj[v]
            
            if not bron_kerbosch_opt(new_R, new_P, new_X):
                return False
            
            P = P & ~lsb
            X = X | lsb
            candidates = candidates & ~lsb
            
        return True

    success = bron_kerbosch_opt(0, (1 << N) - 1, 0)
    
    result = [i + 1 for i in range(N) if (best_clique_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result
