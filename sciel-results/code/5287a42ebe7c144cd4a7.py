"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0
    
    if k != 5 or g != 9:
        # Fallback to LUW for other parameters
        return _luw_construct(k, g, deadline)
    
    # For (5,9): try to construct a smaller graph
    # Strategy: Use a bipartite graph with structured edges based on finite field arithmetic
    # Target: ~430-600 vertices
    
    # Approach: Use a voltage graph over a base graph
    # Or: Direct construction using polynomial relations over GF(5)
    
    # Let's try: vertices are pairs (x, y) where x, y in some set
    # Edges defined by algebraic relations
    
    # Method: Try circulant graphs with 5 generators on Z_n
    # For n even: generators {a, -a, b, -b, n/2}
    # For n odd: need 5 generators, harder to make symmetric
    
    # Let's try n = 2 * p where p is prime, generators {a, -a, b, -b, p}
    # (since n/2 = p when n = 2p)
    
    # Girth verification: BFS from each vertex, check no cycle < 9
    
    best_n = None
    best_edges = None
    
    # Try several values of n
    for n in range(430, 700, 2):  # n must be even for the involution generator
        if time.time() > deadline - 10.0:
            break
        
        # Skip if n doesn't have good properties
        if n % 2 != 0:
            continue
        
        half = n // 2
        
        # Try different generator sets
        # Generators: {a, -a, b, -b, half} where a, b are coprime to n
        # Need to avoid short cycles
        
        found = False
        for a in range(1, n, 2):  # a must be odd for gcd(a, n) = 1 when n is even
            if time.time() > deadline - 10.0:
                break
            if gcd(a, n) != 1:
                continue
            
            for b in range(1, n, 2):
                if time.time() > deadline - 5.0:
                    break
                if gcd(b, n) != 1:
                    continue
                if b == a or b == -a % n:
                    continue
                
                gens = [a, (-a) % n, b, (-b) % n, half]
                
                # Quick check: no generator should be 0
                if any(g == 0 for g in gens):
                    continue
                
                # Check for very short cycles (length 4, 6, 8)
                # A 4-cycle exists if there exist distinct generators g1, g2, g3, g4
                # such that g1 + g2 + g3 + g4 = 0 mod n (with appropriate signs)
                # For bipartite-like structure, check if any pair sums to 0 (that's length 2, not allowed)
                # Actually in a Cayley graph, a 4-cycle from identity: g1, g2, -g1, -g2
                # exists if g1 + g2 - g1 - g2 = 0 (always true) - but we need distinct edges
                
                # Let me do a proper BFS girth check
                girth = _compute_girth(n, gens)
                
                if girth >= 9:
                    # Construct the graph
                    edges = []
                    for v in range(n):
                        for g in gens:
                            w = (v + g) % n
                            if v < w:
                                edges.append((v, w))
                    
                    if best_n is None or n < best_n:
                        best_n = n
                        best_edges = edges
                        found = True
                        break  # Found a valid graph for this n, try smaller n next
        
        if found:
            break  # Found a valid graph, stop searching
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    return _luw_construct(k, g, deadline)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)


def _compute_girth(n, gens):
    """Compute the girth of the Cayley graph using BFS from vertex 0."""
    # BFS from vertex 0, tracking parent to avoid backtracking
    # Actually, for girth we need to find the shortest cycle
    # Use BFS from each vertex is too expensive
    
    # Better: BFS from vertex 0, and for each vertex, check if we can reach it
    # via a different path of the same or shorter length
    
    # Standard approach: BFS from each vertex, but that's O(n^2)
    # For n ~ 500, this might be too slow
    
    # Alternative: use the fact that in a Cayley graph, the girth is the
    # length of the shortest non-trivial relation in the generators
    
    # A relation is a sequence g1, g2, ..., gl where g1+g2+...+gl = 0 mod n
    # and no proper prefix sums to 0 (to avoid backtracking)
    
    # For girth >= 9, we need no relation of length 4, 6, or 8
    # (odd lengths: 5, 7 also need checking)
    
    # Let's check for relations of length up to 8
    # This is a combinatorial search but bounded
    
    from itertools import product
    
    # Check for relations of length l = 4, 5, 6, 7, 8
    # A relation g1, g2, ..., gl is non-backtracking if:
    # - gi != -g(i-1) for all i (no immediate backtracking)
    # - No proper prefix sums to 0
    
    for l in range(4, 9):
        if l == 4:
            # Check all quadruples
            for g1 in gens:
                for g2 in gens:
                    if g2 == (-g1) % n:
                        continue
                    for g3 in gens:
                        if g3 == (-g2) % n:
                            continue
                        s23 = (g2 + g3) % n
                        for g4 in gens:
                            if g4 == (-g3) % n:
                                continue
                            if (s23 + g4) % n == 0 and g4 != (-g1) % n:
                                # Check no prefix sums to 0
                                s1 = g1
                                s12 = (s1 + g2) % n
                                s123 = (s12 + g3) % n
                                if s1 != 0 and s12 != 0 and s123 != 0:
                                    return l
        elif l == 5:
            for g1 in gens:
                for g2 in gens:
                    if g2 == (-g1) % n:
                        continue
                    s12 = (g1 + g2) % n
                    for g3 in gens:
                        if g3 == (-g2) % n:
                            continue
                        s123 = (s12 + g3) % n
                        if s123 == 0:
                            continue
                        for g4 in gens:
                            if g4 == (-g3) % n:
                                continue
                            s1234 = (s123 + g4) % n
                            if s1234 == 0:
                                continue
                            for g5 in gens:
                                if g5 == (-g4) % n:
                                    continue
                                if (s1234 + g5) % n == 0 and g5 != (-g1) % n:
                                    return l
        elif l == 6:
            for g1 in gens:
                for g2 in gens:
                    if g2 == (-g1) % n:
                        continue
                    s12 = (g1 + g2) % n
                    for g3 in gens:
                        if g3 == (-g2) % n:
                            continue
                        s123 = (s12 + g3) % n
                        if s123 == 0:
                            continue
                        for g4 in gens:
                            if g4 == (-g3) % n:
                                continue
                            s1234 = (s123 + g4) % n
                            if s1234 == 0:
                                continue
                            for g5 in gens:
                                if g5 == (-g4) % n:
                                    continue
                                s12345 = (s1234 + g5) % n
                                if s12345 == 0:
                                    continue
                                for g6 in gens:
                                    if g6 == (-g5) % n:
                                        continue
                                    if (s12345 + g6) % n == 0 and g6 != (-g1) % n:
                                        return l
        elif l == 7:
            # This gets expensive: 5^7 = 78125 combinations
            # Do it but with early termination
            for g1 in gens:
                for g2 in gens:
                    if g2 == (-g1) % n:
                        continue
                    s12 = (g1 + g2) % n
                    for g3 in gens:
                        if g3 == (-g2) % n:
                            continue
                        s123 = (s12 + g3) % n
                        if s123 == 0:
                            continue
                        for g4 in gens:
                            if g4 == (-g3) % n:
                                continue
                            s1234 = (s123 + g4) % n
                            if s1234 == 0:
                                continue
                            for g5 in gens:
                                if g5 == (-g4) % n:
                                    continue
                                s12345 = (s1234 + g5) % n
                                if s12345 == 0:
                                    continue
                                for g6 in gens:
                                    if g6 == (-g5) % n:
                                        continue
                                    s123456 = (s12345 + g6) % n
                                    if s123456 == 0:
                                        continue
                                    for g7 in gens:
                                        if g7 == (-g6) % n:
                                            continue
                                        if (s123456 + g7) % n == 0 and g7 != (-g1) % n:
                                            return l
        elif l == 8:
            # 5^8 = 390625, might be slow but let's try with pruning
            for g1 in gens:
                for g2 in gens:
                    if g2 == (-g1) % n:
                        continue
                    s12 = (g1 + g2) % n
                    for g3 in gens:
                        if g3 == (-g2) % n:
                            continue
                        s123 = (s12 + g3) % n
                        if s123 == 0:
                            continue
                        for g4 in gens:
                            if g4 == (-g3) % n:
                                continue
                            s1234 = (s123 + g4) % n
                            if s1234 == 0:
                                continue
                            for g5 in gens:
                                if g5 == (-g4) % n:
                                    continue
                                s12345 = (s1234 + g5) % n
                                if s12345 == 0:
                                    continue
                                for g6 in gens:
                                    if g6 == (-g5) % n:
                                        continue
                                    s123456 = (s12345 + g6) % n
                                    if s123456 == 0:
                                        continue
                                    for g7 in gens:
                                        if g7 == (-g6) % n:
                                            continue
                                        s1234567 = (s123456 + g7) % n
                                        if s1234567 == 0:
                                            continue
                                        for g8 in gens:
                                            if g8 == (-g7) % n:
                                                continue
                                            if (s1234567 + g8) % n == 0 and g8 != (-g1) % n:
                                                return l
    
    return 9  # No cycle of length < 9 found


def _luw_construct(k, g, deadline):
    """Fallback LUW construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        mul_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        mul_table = np.outer(np.arange(q, dtype=np.int64), np.arange(q, dtype=np.int64)) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    rule_a = np.array([rule[c][0] for c in range(2, d + 1)], dtype=np.int64)
    rule_b = np.array([rule[c][1] for c in range(2, d + 1)], dtype=np.int64)
    
    free_vals = np.arange(q, dtype=np.int64)
    
    def get_neighbours(side, x_arr):
        y_coords = np.empty((q, d + 1), dtype=np.int64)
        y_coords[:, 0] = 0
        y_coords[:, 1] = free_vals
        if side == 0:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                y_coords[:, c] = (x_arr[c] + mul_table[y_coords[:, a], x_arr[b]]) % q
        else:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                y_coords[:, c] = (x_arr[c] - mul_table[x_arr[a], y_coords[:, b]]) % q
        return y_coords
    
    start_key = (0, (0,) * (d + 1))
    index = {start_key: 0}
    order = [start_key]
    edges = []
    
    for i, vertex in enumerate(order):
        if time.time() > deadline - 5.0:
            break
        side, x_tuple = vertex
        x_arr = np.array(x_tuple, dtype=np.int64)
        y_coords = get_neighbours(side, x_arr)
        for j in range(q):
            other_key = (1 - side, tuple(y_coords[j]))
            if other_key not in index:
                index[other_key] = len(order)
                order.append(other_key)
            if side == 0:
                edges.append((index[vertex], index[other_key]))
    
    return len(order), edges
# EVOLVE-BLOCK-END
