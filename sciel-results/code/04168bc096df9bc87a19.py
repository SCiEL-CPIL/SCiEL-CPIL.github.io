import time

def construct():
    # The known optimal construction for R(C5, K9) lower bound is 8 disjoint K4s.
    # n = 32
    # alpha = 8 (one vertex from each K4)
    # C5-free (max clique size 4)
    
    n = 32
    edges = []
    
    for b in range(8):
        base = 4 * b
        # Add edges for K4 on vertices base, base+1, base+2, base+3
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
