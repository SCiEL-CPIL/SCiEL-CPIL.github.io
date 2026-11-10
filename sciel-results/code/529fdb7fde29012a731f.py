"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 512
    n = N
    
    # Build adjacency
    adj = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            g = gcd(a, b)
            if a * b // g <= 512:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Greedy initial solution
    def greedy_clique():
        remaining = (1 << n) - 1
        clique = []
        while remaining:
            best_node = -1
            best_deg = -1
            r = remaining
            while r:
                lsb = r & (-r)
                node = lsb.bit_length() - 1
                deg = bin(remaining & adj[node]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_node = node
                r ^= lsb
            if best_node < 0:
                break
            clique.append(best_node)
            remaining &= adj[best_node]
            remaining &= ~(1 << best_node)
        return clique
    
    best_clique = greedy_clique()
    best_size = len(best_clique)
    
    # Branch and bound
    # Order by degree
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -degrees[x])
    
    # Search using lists for P
    # To make it faster, precompute neighbor masks
    
    def search(P, C_size):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P:
            if C_size > best_size:
                best_size = C_size
                # We'd need to track the actual clique, but for now just size
            return
        
        if C_size + len(P) <= best_size:
            return
        
        # Process in reverse
        for i in range(len(P) - 1, -1, -1):
            if time.time() > deadline:
                return
            
            v = P[i]
            
            if C_size + 1 + i <= best_size:
                break
            
            # New P: neighbors of v in P[0:i]
            v_mask = adj[v]
            new_P = []
            for j in range(i):
                u = P[j]
                if (v_mask >> u) & 1:
                    new_P.append(u)
            
            search(new_P, C_size + 1)
    
    # We need to track the actual best clique, not just size
    # Let me redo this properly
    
    best_clique = greedy_clique()
    best_size = len(best_clique)
    
    def search2(P, C):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P:
            if len(C) > best_size:
                best_size = len(C)
                best_clique = C[:]
            return
        
        if len(C) + len(P) <= best_size:
            return
        
        for i in range(len(P) - 1, -1, -1):
            if time.time() > deadline:
                return
            
            v = P[i]
            
            if len(C) + 1 + i <= best_size:
                break
            
            v_mask = adj[v]
            new_P = [P[j] for j in range(i) if (v_mask >> P[j]) & 1]
            
            search2(new_P, C + [v])
    
    search2(order[:], [])
    
    if not best_clique:
        return [1]
    
    return sorted(best_clique)

# EVOLVE-BLOCK-END
