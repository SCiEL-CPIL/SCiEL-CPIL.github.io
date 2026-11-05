import time
import sys

def construct(k, g):
    # Use the lift of K4 over F_p with constant voltage 1.
    # Girth will be at least p.
    # We need girth >= g. So pick p >= g.
    # To minimize n, pick smallest prime p >= g.
    # n = 4 * p.
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    p = g
    while not is_prime(p):
        p += 1
        
    # K4 vertices: 0, 1, 2, 3
    # K4 edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Lift: vertices (u, a) for u in 0..3, a in 0..p-1
    # Edge (u, a) -- (v, a+1) if (u,v) in K4
    
    n = 4 * p
    edges = []
    
    # Map (u, a) to index
    # index = u * p + a
    
    for u in range(4):
        for v in range(4):
            if u < v:
                for a in range(p):
                    i = u * p + a
                    j = v * p + (a + 1) % p
                    edges.append((i, j))
                    
    return n, edges
