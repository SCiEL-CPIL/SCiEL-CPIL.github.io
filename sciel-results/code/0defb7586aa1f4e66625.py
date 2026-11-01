import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # For k=4, g=10, the Moore bound is 320.
    # The (4,10)-cage is a 4-regular graph of girth 10 with 320 vertices.
    # It can be constructed as a voltage graph over the Petersen graph.
    
    if k == 4 and g == 10:
        # Petersen graph edges
        # Vertices 0..9
        # Petersen graph is the Kneser graph K(5,2)
        # Vertices are 2-subsets of {0,1,2,3,4}
        # We map them to 0..9
        # 0: {0,1}, 1: {0,2}, 2: {0,3}, 3: {0,4}, 4: {1,2}
        # 5: {1,3}, 6: {1,4}, 7: {2,3}, 8: {2,4}, 9: {3,4}
        
        # Edges in Petersen graph (disjoint sets)
        # 0({0,1}) connected to 7({2,3}), 8({2,4}), 9({3,4})
        # 1({0,2}) connected to 6({1,4}), 8({2,4})? No, 8 is {2,4} shares 2.
        # Let's list edges explicitly.
        # 0: {0,1} -> disjoint with {2,3}(7), {2,4}(8), {3,4}(9)
        # 1: {0,2} -> disjoint with {1,3}(5), {1,4}(6), {3,4}(9)
        # 2: {0,3} -> disjoint with {1,2}(4), {1,4}(6), {2,4}(8)
        # 3: {0,4} -> disjoint with {1,2}(4), {1,3}(5), {2,3}(7)
        # 4: {1,2} -> disjoint with {0,3}(2), {0,4}(3), {3,4}(9)
        # 5: {1,3} -> disjoint with {0,2}(1), {0,4}(3), {2,4}(8)
        # 6: {1,4} -> disjoint with {0,2}(1), {0,3}(2), {2,3}(7)
        # 7: {2,3} -> disjoint with {0,1}(0), {0,4}(3), {1,4}(6)
        # 8: {2,4} -> disjoint with {0,1}(0), {0,3}(2), {1,3}(5)
        # 9: {3,4} -> disjoint with {0,1}(0), {0,2}(1), {1,2}(4)
        
        petersen_edges = [
            (0, 7), (0, 8), (0, 9),
            (1, 5), (1, 6), (1, 9),
            (2, 4), (2, 6), (2, 8),
            (3, 4), (3, 5), (3, 7),
            (4, 9),
            (5, 8),
            (6, 7)
        ]
        
        # Voltage group: Z_2 x Z_2 x Z_2 x Z_2 x Z_2 (size 32)
        # We need to assign voltages to edges such that the lift has girth 10.
        # This is a known construction.
        # Let's use a specific voltage assignment.
        # One known assignment for the (4,10)-cage:
        # We can use the fact that the Petersen graph is a (3,5)-cage.
        # The lift by Z_2^5 will be 4-regular (since Petersen is 3-regular, wait).
        # Petersen graph is 3-regular. The lift will be 3-regular.
        # But we need 4-regular.
        # So the voltage graph approach over Petersen (3-regular) gives a 3-regular graph.
        # We need a 4-regular base graph.
        
        # The base graph for the (4,10)-cage is not the Petersen graph directly in a simple lift.
        # The (4,10)-cage is not a lift of a smaller graph in a simple way.
        
        # Alternative: Use the LUW construction with better parameters?
        # Or use a different algebraic construction.
        
        # Since finding the exact (4,10)-cage is complex, and the time limit is tight,
        # I will use a heuristic search to find a 4-regular graph of girth 10 with minimal vertices.
        # However, a general search is too slow.
        
        # Let's try the LUW construction again but check if we can reduce n.
        # The LUW construction for k=4, g=10 gives n=2048.
        # The Moore bound is 320.
        
        # There is a known construction for the (4,10)-cage using the dodecahedron?
        # The dodecahedron is (3,10)-cage with 20 vertices.
        
        # I will implement a simple backtracking search for small n, but it might not finish.
        # Given the constraints, I will return the LUW graph as a fallback, but try to optimize it.
        
        # Actually, the (4,10)-cage is the graph with 320 vertices.
        # I can hardcode the edge list if I can find it.
        # But I don't have it.
        
        # Let's use the LUW construction as is, but maybe with q=4, d=5.
        # It is valid, but n=2048.
        
        # Is there a better construction?
        # The (4,10)-cage is the **Birkhoff graph** B(4)?
        # Birkhoff graph B(k) is a k-regular graph of girth 2 log_2(k+1) + 2?
        # For k=4, girth = 2 log_2(5) + 2 ~ 4.6 + 2 = 6.6. Not 10.
        
        # I will use the LUW construction. It is safe and valid.
        pass
    
    # Fallback: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
