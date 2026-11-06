import math
import time

def construct():
    N = 512
    LIMIT = 512
    
    # Precompute LCM compatibility
    # adj[i] is a bitmask where bit j is set if lcm(i+1, j+1) <= LIMIT
    # Note: i, j are 0-indexed, values are 1..N
    adj = [0] * N
    
    for i in range(N):
        u = i + 1
        mask = 0
        for j in range(N):
            v = j + 1
            # Optimization: if u * v > LIMIT * gcd(u, v), skip
            # lcm(u, v) = u * v // gcd(u, v)
            # We can just compute it
            g = math.gcd(u, v)
            if (u * v) // g <= LIMIT:
                mask |= (1 << j)
        adj[i] = mask

    best_clique = []
    best_size = 0
    
    # Greedy initialization
    # Try to build a large clique greedily
    # Start with 1, then add smallest possible, etc?
    # Or just run BK.
    
    start_time = time.time()
    TIME_LIMIT = 50.0  # Safety margin for 60s limit
    
    # Bron-Kerbosch with Pivoting
    # P: candidates, X: excluded, R: current clique (list of indices)
    
    def bron_kerbosch(P, X, R):
        nonlocal best_clique, best_size
        
        # Check time
        if time.time() - start_time > TIME_LIMIT:
            return False # Signal timeout
        
        # If P and X are empty, we found a maximal clique
        if P == 0 and X == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R[:]
            return True
        
        # If P is empty, no more cliques in this branch
        if P == 0:
            return True
            
        # Pruning: if |R| + popcount(P) <= best_size, prune
        if len(R) + P.bit_count() <= best_size:
            return True

        # Pivot selection
        # Choose u in P U X that maximizes |N(u) ∩ P|
        PX = P | X
        pivot = -1
        max_deg = -1
        
        # Iterate through bits in PX
        # This can be slow if PX is dense.
        # Optimization: We can just pick any u, or iterate.
        # For 512 bits, iterating all is 512 ops.
        # Let's iterate.
        temp_px = PX
        while temp_px:
            # Get lowest set bit
            lsb = temp_px & -temp_px
            u_idx = lsb.bit_length() - 1
            temp_px ^= lsb
            
            deg = (P & adj[u_idx]).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = u_idx
                
        # If P is subset of N(pivot), then no need to recurse?
        # Standard BK: iterate over P \ N(pivot)
        # If P \ N(pivot) is empty, we are done with this node?
        # Actually, if P is empty after removing N(pivot), we stop.
        
        candidates = P & ~adj[pivot]
        
        # Iterate over candidates
        while candidates:
            if time.time() - start_time > TIME_LIMIT:
                return False
                
            lsb = candidates & -candidates
            v_idx = lsb.bit_length() - 1
            candidates ^= lsb
            
            # Recurse
            # New P = P & N(v)
            # New X = X & N(v)
            # New R = R + [v]
            
            # Check if adding v improves potential
            if len(R) + 1 + (P & adj[v_idx]).bit_count() <= best_size:
                continue
                
            if not bron_kerbosch(P & adj[v_idx], X & adj[v_idx], R + [v_idx]):
                return False
            
            # Move v from P to X
            P &= ~lsb
            X |= lsb
            
    # Initial call
    # P = all nodes, X = empty
    initial_P = (1 << N) - 1
    initial_X = 0
    initial_R = []
    
    bron_kerbosch(initial_P, initial_X, initial_R)
    
    # Convert indices back to values
    if best_clique:
        return [i + 1 for i in best_clique]
    else:
        # Fallback to seed if something went wrong or timeout immediately
        return list(range(1, int(math.isqrt(LIMIT)) + 1))
