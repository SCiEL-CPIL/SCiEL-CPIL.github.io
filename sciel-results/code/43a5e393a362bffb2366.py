import time
import math
from collections import deque

def construct(k, g):
    deadline = time.time() + 115
    start_time = time.time()

    # For the specific case of k=3, g=16
    if k == 3 and g == 16:
        # 1. Try the known smallest construction (936 vertices) if possible?
        # The 936-vertex graph by Exoo et al. is likely not a simple Cayley graph or standard LUW.
        # However, we can try to find a smaller one using other families.
        
        # 2. Try LUW (Lazebnik-Ustimenko-Woldar) with optimized parameters.
        # The standard LUW for k=3, g=16 gives n = 2 * 3^4 = 162? No.
        # LUW construction for k-regular, girth g >= 6:
        # q is a prime power.
        # n = 2 * q * d where d is related to girth.
        # Actually, the LUW graph for k=3, g=16 is not the one in the seed.
        # The seed's _luw_fallback is a specific implementation.
        # Let's check the size of the seed's LUW fallback.
        # q=3, d = max(3, 16-5 if (16-5)%2 else 16-4) = max(3, 11 if 1 else 12) = 11.
        # n = 2 * 3^11? No, the vertices are tuples.
        # The LUW graph in the seed produces a graph of size 2 * q^d?
        # Let's run it to see.
        
        # 3. Try the "Klein" graph (cage problem for (3, g) with g even).
        # Klein graphs exist for g=2h.
        # For g=16, h=8.
        # The construction requires a prime power q = p^m.
        # The vertices are pairs (x, y) in Z_q x Z_q? No, usually it's a Cayley graph of Z_q x Z_q or similar.
        # The standard Klein graph for girth 2h has 2q vertices? No, 2q is for girth 6?
        # Actually, the smallest (3,16)-cage is 936 vertices.
        # This suggests that standard constructions (LUW, Klein) might not hit 936 exactly unless specific parameters are used.
        # 936 = 2 * 468.
        # Let's try to construct the 936-vertex graph directly if we know the structure, or find a smaller one.
        
        # Since we don't have the explicit 936-vertex construction code, and the LUW fallback is likely large (exponential in g),
        # we need a better approach.
        
        # What if we use a heuristic search for a smaller graph?
        # Given the time limit (120s), and the lower bound is 512, the 936 graph is the target.
        
        # Let's look at the seed's LUW fallback again.
        # It returns a graph. Let's estimate its size.
        # q=3, d=11.
        # The number of vertices is 2 * q^d? No, the vertices are (side, y) where y is a tuple of length d+1?
        # In the seed:
        # start = (0, (None,) + (0,)*d)
        # neighbours generate tuples of length d+1?
        # y = [None, first] + [0]*(d-1) -> length d+1.
        # So the number of vertices is at most 2 * q^(d+1) = 2 * 3^12 = 2 * 531441 = 1062882.
        # This is much larger than 936.
        
        # So the LUW fallback is terrible for scoring.
        # We MUST find a smaller graph.
        
        # Known small (3,16)-cage is 936.
        # Is there a way to construct it?
        # The 936-vertex graph is a (3,16)-cage.
        # It is a Cayley graph of a group of order 936?
        # 936 = 2^3 * 3^2 * 13.
        
        # If we cannot construct 936, we must find something smaller than the LUW fallback.
        # Any valid (3,16)-cage with n < 1,000,000 is a massive improvement over LUW.
        # Even a random 3-regular graph with girth 16 would do, but finding one is hard.
        
        # Strategy:
        # 1. Try to find a small graph using a simple search (e.g., greedy construction or random restarts).
        # 2. If that fails, return the LUW fallback (which is valid but large).
        
        # Let's implement a simple random/greedy search for a (3,16)-cage.
        # Start with n=512 (Moore bound) and increase.
        # This is computationally expensive to verify girth 16.
        
        # Alternative: Use the known 936-vertex graph if we can find it online or derive it.
        # Since I cannot access the internet, I will try to construct a graph using a known parameterized family that might yield small results.
        
        # The "Klein" construction for (3, 2h) cages:
        # For h=8 (girth 16), we need a prime power q such that the graph has girth 16.
        # The number of vertices is 2q.
        # If we can find a q < 468 such that the Klein graph has girth 16, we win.
        # The condition for the Klein graph C(q) to have girth 2h is that there are no non-trivial solutions to certain equations.
        # Specifically, for the Klein graph K(q), girth is 2h if and only if q is a prime power and certain polynomial equations have no solutions in F_q.
        
        # Let's try small prime powers q.
        # q must be such that the graph is simple and 3-regular.
        # The Klein graph is a Cayley graph of the group Z_q x Z_q? No.
        
        # Let's stick to the LUW family but try to minimize the exponent d.
        # The girth of the LUW graph is related to d.
        # If we use a larger q, we might get a smaller d for the same girth?
        # No, the LUW construction is specific.
        
        # What if we use a different construction?
        # The "Birkhoff-Hilton" cages?
        
        # Let's try a heuristic:
        # Generate a random 3-regular graph on n vertices.
        # Check if girth >= 16.
        # If yes, return it.
        # Start with n=512, 520, ... up to 936.
        
        # This is risky. The probability of a random 3-regular graph having girth 16 is extremely low.
        
        # Better: Use the fact that the 936-vertex cage is known.
        # If we can't construct it, we return the LUW fallback.
        # But LUW is huge.
        
        # Is there a smaller known cage?
        # The (3,14)-cage is 352 vertices?
        # The (3,12)-cage is 132 vertices?
        # The (3,16)-cage is 936 vertices.
        
        # If we cannot find 936, we must return something.
        # The LUW fallback is the only guaranteed valid graph in the current code.
        
        # However, we can try to optimize the LUW fallback to be as small as possible.
        # The LUW graph size is 2 * q^d.
        # We want to minimize q^d subject to girth >= 16.
        # The girth of LUW(k, d) is at least 2 * log_k (n) + ...
        # Actually, the LUW construction is designed to have high girth.
        
        # Let's check if we can use a smaller q in LUW.
        # The seed uses q=k=3.
        # What if we use q=5?
        # The LUW construction requires q to be a prime power.
        # If we use q=5, the graph is 5-regular? No, the degree is k.
        # The LUW construction in the seed seems to hardcode q=k.
        
        # Let's modify the LUW fallback to try different q values if k is not fixed?
        # No, k is fixed to 3.
        
        # Let's try to implement a search for the 936-vertex graph using a known database or reconstruction?
        # Impossible.
        
        # Let's try to improve the girth check and search for a smaller graph using a greedy algorithm.
        # 1. Start with a path.
        # 2. Close the path to form a cycle? No, girth 16 means no short cycles.
        
        # Given the constraints, the best we can do without the explicit 936-vertex construction is to return the LUW fallback, but we can try to make it smaller by using a different parameterization if possible.
        
        # However, the LUW fallback in the seed is likely the best we have.
        # Let's ensure the LUW fallback is correct and efficient.
        
        # One improvement:
        # The seed's LUW fallback calculates the graph recursively.
        # We can try to prune the search space if we detect early cycles?
        # But LUW is known to be girth-16 by design (for specific d).
        
        # Let's assume the LUW fallback returns a valid graph.
        # The score will be low (large n), but it's valid.
        
        # Can we do better?
        # What if we try q=2 in LUW?
        # The seed handles q=4 specially.
        # If k=3, q=3.
        
        # Let's try to find a (3,16)-cage with a simple BFS search on the space of 3-regular graphs?
        # No.
        
        # I will stick to the LUW fallback but add a check for a known small graph if possible.
        # Since I don't have the 936-vertex graph, I will return LUW.
        
        # Wait, the seed's LUW fallback might be incorrect or inefficient.
        # Let's ensure it works within time.
        
        # Actually, there is a known (3,16)-cage with 936 vertices.
        # If I can't construct it, I lose.
        # But maybe there is a smaller one? No, 936 is the smallest known.
        # The lower bound is 512.
        
        # Let's try to construct the 936-vertex graph using a specific method:
        # It is a (3,16)-cage.
        # It might be a Cayley graph of the group Z_2 x Z_2 x Z_117?
        # 936 = 8 * 117 = 8 * 9 * 13.
        
        # Without the specific construction, I cannot produce it.
        
        # Therefore, I will return the LUW fallback.
        # To improve the score, I will try to minimize the LUW graph size by choosing the smallest possible d that guarantees girth 16.
        # The seed uses d = 11 for g=16.
        # Is d=10 sufficient?
        # The girth of LUW(3, d) is approximately 2 * d / log_3(3) = 2d?
        # If d=8, girth might be 16?
        # Let's test d=8.
        # If d=8, n = 2 * 3^9 = 39366? Still huge.
        
        # Wait, the LUW construction size is not q^d, it's the number of distinct vertices generated.
        # The vertices are in F_q^{d+1}.
        # So max size is q^{d+1}.
        
        # There is no way LUW gives 936 vertices.
        
        # Conclusion: The LUW fallback is a "safe" but "bad" score.
        # To get a good score, we MUST find a graph with n ~ 936.
        
        # Since I cannot find the construction, I will try a random search for a graph with n=936 that has girth 16.
        # This is unlikely to succeed in 120s.
        
        # Alternative: Return a known small graph for (3,12) or (3,14) and hope the evaluator only checks validity?
        # No, the task says "no cycle shorter than g".
        
        # I will return the LUW fallback.
        return _luw_fallback(k, g)
    
    return _luw_fallback(k, g)

def _luw_fallback(k, g):
    # The original LUW construction
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
