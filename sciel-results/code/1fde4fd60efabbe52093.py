"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    n = 40
    best_edges = []
    best_count = 0
    
    # Run multiple restarts with different random orderings
    # We use bitsets for adjacency to speed up C6 checks
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with 8 K5s
        adj_bits = [0] * n
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    edges.append([u, v])
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        
        # List of candidate edges
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_bits[u] >> v) & 1:
                    candidates.append((u, v))
        
        random.shuffle(candidates)
        
        # Greedy addition
        current_edges = list(edges)
        current_count = len(edges)
        
        for u, v in candidates:
            if time.time() > deadline - 5.0:
                break
            
            # Check if adding (u,v) creates C6
            # Path u-x1-x2-x3-x4-v
            # x1 in N(u), x4 in N(v)
            # Need path x1-x2-x3-x4 of length 3
            
            nu = adj_bits[u] & ~(1 << v)
            nv = adj_bits[v] & ~(1 << u)
            
            c6_found = False
            
            # Get lists of neighbors for faster iteration if needed, 
            # but bit iteration is fine for n=40
            x1s = []
            temp = nu
            while temp:
                lsb = temp & -temp
                x1 = lsb.bit_length() - 1
                x1s.append(x1)
                temp ^= lsb
            
            x4s = []
            temp = nv
            while temp:
                lsb = temp & -temp
                x4 = lsb.bit_length() - 1
                x4s.append(x4)
                temp ^= lsb
            
            for x1 in x1s:
                if c6_found:
                    break
                for x4 in x4s:
                    if x1 == x4:
                        continue
                    
                    # Check if there is a path x1-x2-x3-x4
                    # A = N(x1) \ {u, v, x4}
                    A = adj_bits[x1] & ~(1 << u) & ~(1 << v) & ~(1 << x4)
                    # B = N(x4) \ {u, v, x1}
                    B = adj_bits[x4] & ~(1 << u) & ~(1 << v) & ~(1 << x1)
                    
                    # We need x2 in A such that (N(x2) \ {u, v, x1} intersects B)
                    # Actually, x3 must be in N(x2) and N(x4).
                    # So we need x2 in A such that (adj_bits[x2] & B) != 0.
                    # Note: adj_bits[x2] includes x4 if x2~x4.
                    # B includes x4? No, B = N(x4) \ ... so x4 is not in B.
                    # But adj_bits[x2] might contain x4.
                    # If adj_bits[x2] & B is non-zero, let x3 be that bit.
                    # x3 is in B, so x3 is in N(x4).
                    # x3 is in adj_bits[x2], so x3 is in N(x2).
                    # So x2-x3-x4 is a path.
                    # We also need x1-x2. x2 is in A, so x2 in N(x1).
                    # Distinctness:
                    # x2 in A => x2 != u, v, x4.
                    # x3 in B => x3 != u, v, x1.
                    # x3 in N(x2) => x3 != x2.
                    # x2 in N(x1) => x2 != x1.
                    # x1 != x4 checked.
                    # So all distinct.
                    
                    temp_a = A
                    while temp_a:
                        lsb = temp_a & -temp_a
                        x2 = lsb.bit_length() - 1
                        if (adj_bits[x2] & B) != 0:
                            c6_found = True
                            break
                        temp_a ^= lsb
                    
                    if c6_found:
                        break
            
            if not c6_found:
                # Add edge
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                current_edges.append([u, v])
                current_count += 1
        
        if current_count > best_count:
            best_count = current_count
            best_edges = current_edges
    
    return best_edges
# EVOLVE-BLOCK-END
