import time
import random
from collections import deque

# EVOLVE-BLOCK-START
import itertools

def construct(k, g):
    deadline = time.time() + 110.0
    
    # Strategy: Lift the (3,8)-cage (Tutte-Coxeter graph, 30 vertices) to a (3,16)-cage.
    # We search for a voltage assignment over Z_M that ensures girth >= 16.
    # If successful, n = 30 * M.
    # We try M=32 (n=960), M=31 (n=930), etc.
    # The (3,8) cage has no cycles < 8.
    # We must ensure no cycle of length 8, 10, 12, 14 in the base lifts to a 0-sum cycle.
    
    # Tutte-Coxeter Graph Edges (30 vertices, 45 edges)
    # This is the (3,8)-cage.
    # Vertices 0..29.
    tc_edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 7), (2, 8), (2, 9),
        (3, 10), (3, 11), (3, 12),
        (4, 13), (4, 14), (4, 15),
        (5, 16), (5, 17), (5, 18),
        (6, 19), (6, 20), (6, 21),
        (7, 22), (7, 23), (7, 24),
        (8, 25), (8, 26), (8, 27),
        (9, 28), (9, 29), (9, 0), # Wait, 0 is already used. Let's use a known list.
        # I will use a standard representation.
        # Actually, I'll generate the Tutte-Coxeter graph programmatically if possible, 
        # or use a known edge list.
        # The Tutte-Coxeter graph is a Cayley graph of the group of order 30? No.
        # It is bipartite.
    ]
    
    # Since I don't have the exact edges hardcoded reliably, I will fallback to LUW 
    # but with an optimization: if k=3 and g=16, I will try to find a 
    # voltage assignment on a *random* 3-regular graph of small size?
    # No, that's hard.
    
    # Let's stick to the LUW but optimize the integer encoding.
    # This ensures correctness and speed, even if n is large.
    # However, the score is n.
    
    # Let's try one more thing: The (3,16) cage is 936.
    # If I can't get 936, I can't beat the seed if the seed is LUW.
    # But the seed IS LUW.
    # So I must get 936 to improve.
    
    # I will hardcode the Foster graph edges now.
    # I will use a list of 234 edges.
    # I will generate them from a known source in my memory.
    # The Foster graph is the (3,12) cage.
    # I will use the following edges (from a known database):
    
    foster_edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 7), (2, 8), (2, 9),
        (3, 10), (3, 11), (3, 12),
        (4, 13), (4, 14), (4, 15),
        (5, 16), (5, 17), (5, 18),
        (6, 19), (6, 20), (6, 21),
        (7, 22), (7, 23), (7, 24),
        (8, 25), (8, 26), (8, 27),
        (9, 28), (9, 29), (9, 30),
        (10, 31), (10, 32), (10, 33),
        (11, 34), (11, 35), (11, 36),
        (12, 37), (12, 38), (12, 39),
        (13, 40), (13, 41), (13, 42),
        (14, 43), (14, 44), (14, 45),
        (15, 46), (15, 47), (15, 48),
        (16, 49), (16, 50), (16, 51),
        (17, 52), (17, 53), (17, 54),
        (18, 55), (18, 56), (18, 57),
        (19, 58), (19, 59), (19, 60),
        (20, 61), (20, 62), (20, 63),
        (21, 64), (21, 65), (21, 66),
        (22, 67), (22, 68), (22, 69),
        (23, 70), (23, 71), (23, 72),
        (24, 73), (24, 74), (24, 75),
        (25, 76), (25, 77), (25, 78),
        (26, 79), (26, 80), (26, 81),
        (27, 82), (27, 83), (27, 84),
        (28, 85), (28, 86), (28, 87),
        (29, 88), (29, 89), (29, 90),
        (30, 91), (30, 92), (30, 93),
        (31, 94), (31, 95), (31, 96),
        (32, 97), (32, 98), (32, 99),
        (33, 100), (33, 101), (33, 102),
        (34, 103), (34, 104), (34, 105),
        (35, 106), (35, 107), (35, 108),
        (36, 109), (36, 110), (36, 111),
        (37, 112), (37, 113), (37, 114),
        (38, 115), (38, 116), (38, 117),
        (39, 118), (39, 119), (39, 120),
        (40, 121), (40, 122), (40, 123),
        (41, 124), (41, 125), (41, 126),
        (42, 127), (42, 128), (42, 129),
        (43, 130), (43, 131), (43, 132),
        (44, 133), (44, 134), (44, 135),
        (45, 136), (45, 137), (45, 138),
        (46, 139), (46, 140), (46, 141),
        (47, 142), (47, 143), (47, 144),
        (48, 145), (48, 146), (48, 147),
        (49, 148), (49, 149), (49, 150),
        (50, 151), (50, 152), (50, 153),
        (51, 154), (51, 155), (51, 0), # This is a guess, likely wrong.
    ]
    
    # This hardcoded list is risky. I will use the LUW construction but with integer optimization.
    # This guarantees a valid graph, even if n is large.
    
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]
    
    rule_list = [None] * (d + 1)
    for c in range(2, d + 1):
        rule_list[c] = rule[c]
    
    # Integer encoding for LUW
    q_d = q ** d
    q_d_plus_1 = q_d * q
    
    # Map from (side, x_val) to index
    # We will generate vertices in BFS order
    # State: (side, x_val)
    # x_val is an integer 0..q_d-1 representing the tuple
    
    index = {}
    order = []
    edges = []
    
    # Start vertex
    start_side = 0
    start_val = 0
    start_state = (start_side, start_val)
    index[start_state] = 0
    order.append(start_state)
    
    for i, state in enumerate(order):
        if time.time() > deadline:
            break
        side, x_val = state
        
        # Decode x_val to get coordinates
        # x_val = x_1 * q^(d-1) + ... + x_d
        coords = [0] * (d + 1)
        temp = x_val
        for c in range(d, 0, -1):
            coords[c] = temp % q
            temp //= q
        
        for first in range(q):
            y_coords = [0] * (d + 1)
            y_coords[1] = first
            for c in range(2, d + 1):
                a, b = rule_list[c]
                if side == 0:
                    y_coords[c] = add(coords[c], mul(y_coords[a], coords[b]))
                else:
                    y_coords[c] = sub(coords[c], mul(coords[a], y_coords[b]))
            
            # Encode y_coords
            y_val = 0
            for c in range(1, d + 1):
                y_val = y_val * q + y_coords[c]
            
            other_state = (1 - side, y_val)
            if other_state not in index:
                index[other_state] = len(order)
                order.append(other_state)
            
            if side == 0:
                edges.append((index[state], index[other_state]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
