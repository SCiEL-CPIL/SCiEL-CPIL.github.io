import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 512
    
    # 1. Precompute LCM compatibility
    neighbors = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i // g) * j <= 512:
                mask |= (1 << (j - 1))
        neighbors[i] = mask
        
    # Precompute degrees
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = neighbors[i].bit_count()
        
    # Initial P: all vertices
    initial_P = 0
    for i in range(1, N + 1):
        initial_P |= (1 << (i - 1))
        
    best_clique = []
    
    # Greedy initial clique
    def greedy_clique():
        P = initial_P
        R = []
        while P:
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
            P &= ~(1 << (best_v - 1))
            P &= neighbors[best_v]
            
        return R
        
    best_clique = greedy_clique()
    
    def get_color_bound(P):
        if P == 0:
            return 0, []
            
        verts = []
        temp = P
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1 + 1
            verts.append(v)
            temp ^= lsb
            
        # Sort by degree descending
        verts.sort(key=lambda v: -degrees[v])
        
        color_of = [0] * (N + 1)
        max_c = 0
        
        for v in verts:
            used = set()
            nbrs = neighbors[v] & P
            temp_n = nbrs
            while temp_n:
                lsb_n = temp_n & -temp_n
                u = lsb_n.bit_length() - 1 + 1
                if color_of[u] > 0:
                    used.add(color_of[u])
                temp_n ^= lsb_n
                
            c = 1
            while c in used:
                c += 1
            color_of[v] = c
            if c > max_c:
                max_c = c
                
        verts.sort(key=lambda v: color_of[v])
        
        return max_c, verts

    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        max_c, ordered_P = get_color_bound(P)
        
        if len(R) + max_c <= len(best_clique):
            return
            
        for v in reversed(ordered_P):
            if time.time() > deadline:
                return
            
            nbr_v = neighbors[v]
            R.append(v)
            bron_kerbosch(R, P & nbr_v, X & nbr_v)
            R.pop()
            
            P &= ~(1 << (v - 1))
            X |= (1 << (v - 1))
            
    bron_kerbosch([], initial_P, 0)
    
    return best_clique
