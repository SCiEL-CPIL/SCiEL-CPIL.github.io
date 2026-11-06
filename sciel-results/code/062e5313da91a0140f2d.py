import random
import time

def construct():
    # The maximum n for a C5-free graph with alpha < 9 is likely 32.
    # A valid construction is 8 disjoint K4s.
    # n = 32
    # alpha = 8 (one from each K4) < 9
    # C5-free: K4 has no C5, disjoint union has no C5.
    
    n = 32
    edges = []
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    return {'n': n, 'edges': edges}
