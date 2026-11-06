import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Generate candidates: numbers in [1, 255] divisible by 3, 5, or 17
    # (must have gcd > 1 with 255 = 3*5*17)
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    idx = {v: i for i, v in enumerate(candidates)}
    
    # Build adjacency: adj[i] is a set of indices j where gcd(candidates[i], candidates[j]) > 1
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # Convert to bitsets for speed
    adj_bits = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_bits[i] |= (1 << j)
    
    # Bron-Kerbosch with pivot, using bitsets
    best_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Choose pivot u from P ∪ X to minimize |P \ N(u)|
        # We want to pick u that maximizes |P ∩ N(u)| to minimize the branching
        PX = P | X
        if PX:
            # Pick pivot with maximum |P & adj_bits[u]|
            best_u = -1
            best_count = -1
            temp = PX
            while temp:
                u = temp & (-temp)
                u_idx = u.bit_length() - 1
                temp ^= u
                count = bin(P & adj_bits[u_idx]).count('1')
                if count > best_count:
                    best_count = count
                    best_u = u_idx
            if best_u >= 0:
                candidates_to_branch = P & ~adj_bits[best_u]
            else:
                candidates_to_branch = P
        else:
            candidates_to_branch = P
        
        temp = candidates_to_branch
        while temp:
            v = temp & (-temp)
            v_idx = v.bit_length() - 1
            temp ^= v
            
            new_R = R | v
            new_P = P & adj_bits[v_idx]
            new_X = X & adj_bits[v_idx]
            
            bron_kerbosch(new_R, new_P, new_X)
            
            P ^= v
            X |= v
    
    # Initial call: R = empty, P = all nodes, X = empty
    all_nodes = (1 << n) - 1
    bron_kerbosch(0, all_nodes, 0)
    
    if not best_clique:
        # Fallback: just return multiples of 5 (which includes 255)
        return list(range(5, 256, 5))
    
    return sorted([candidates[i] for i in best_clique])

