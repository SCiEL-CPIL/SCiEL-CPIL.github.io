"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((e[0], e[1]))
        base_edges.add((e[1], e[0]))
        
    # Strategy: Use hubs 10 and 12 (even vertices) to connect all odd vertices.
    # And use hub 11 (odd vertex) to connect all even vertices?
    # No, the 19-edge solution was:
    # Hubs 10, 12.
    # Connect all Odds to 10 or 12.
    # But Evens?
    # In the 19-edge calculation:
    # H={10, 12}. F={9, 11, 13}.
    # Vertices to connect: V \ {10, 12, 9, 11, 13}.
    # This implies ALL other vertices are connected to 10 or 12.
    # But 10, 12 are Even. They connect to Odds.
    # So Evens (other than 10, 12) cannot be connected to 10 or 12 directly.
    # So the 19-edge solution was flawed in my derivation because I ignored parity.
    
    # Correct derivation:
    # We need to cover A (Evens) and B (Odds).
    # Hubs in A connect to B. Hubs in B connect to A.
    # To minimize edges, we want to maximize free connections.
    
    # Let's try H_A = {10, 12}, H_B = {11}.
    # Edges for B: Connect B \ {11} to 10 or 12.
    # Free in B: N(10)={9,11}, N(12)={11,13}.
    # 11 is hub. 9, 13 are free.
    # B size 12. H_B size 1. Free 2. To connect: 12 - 1 - 2 = 9.
    # Edges for A: Connect A \ {10, 12} to 11.
    # Free in A: N(11)={10,12}.
    # 10, 12 are hubs. Free 0.
    # A size 12. H_A size 2. Free 0. To connect: 12 - 2 - 0 = 10.
    # Total: 9 + 10 = 19.
    
    added_edges = []
    
    # Connect B to H_A
    H_A = [10, 12]
    H_B = [11]
    
    # B vertices
    B_verts = [i for i in range(n) if i % 2 != 0]
    for v in B_verts:
        if v in H_B:
            continue
        # Check if free (neighbor of 10 or 12)
        if (v, 10) in base_edges or (10, v) in base_edges or (v, 12) in base_edges or (12, v) in base_edges:
            continue
        # Assign to 10 or 12 arbitrarily (or closest)
        added_edges.append([10, v])
        
    # A vertices
    A_verts = [i for i in range(n) if i % 2 == 0]
    for u in A_verts:
        if u in H_A:
            continue
        # Check if free (neighbor of 11)
        if (u, 11) in base_edges or (11, u) in base_edges:
            continue
        # Assign to 11
        added_edges.append([11, u])
        
    # Verify triangle-free and diameter
    # For now, return this.
    
    # Convert to list of lists of ints
    result = [[int(u), int(v)] for u, v in added_edges]
    
    return result

# EVOLVE-BLOCK-END
