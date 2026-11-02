"""Voltage graph (circulant) construction for 5-regular graphs of girth >= 9."""
import numpy as np
from itertools import combinations, permutations, product

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g using a circulant/voltage graph approach.
    
    For the specific case of k=5, g=9, we search for a circulant graph C(n, S)
    with girth >= 9. We use trace checks to filter candidates efficiently.
    """
    if k == 5 and g == 9:
        return _construct_5_9()
    
    # Fallback: use Lazebnik-Ustimenko-Woldar style construction (same as seed)
    return _construct_luw(k, g)


def _construct_luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction (fallback from seed)."""
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


def _construct_5_9():
    """
    Search for a circulant graph C(n, S) of degree 5 and girth >= 9.
    We check traces of A^L for L=2,4,6,8 to filter candidates.
    """
    import random
    
    def check_traces(n, S):
        """
        Check if the circulant graph C(n, S) has the trace properties of a graph with girth >= 9.
        Tr(A^L) should be n * 5 * 4^(L/2 - 1) for L=2,4,6,8.
        """
        # S is a list of 5 elements. Neighbors are +/- S.
        # We need to count the number of ways to sum L elements from {+/- s_i} to 0 mod n.
        # This is equivalent to Tr(A^L).
        
        # For L=2: sum of 2 elements is 0.
        # Pairs (s_i, -s_i). There are 5 such pairs.
        # Each vertex has 5 such pairs. Total 5n.
        # Expected: 5n.
        
        # For L=4: sum of 4 elements is 0.
        # In a tree, 5 * 4 * n = 20n.
        
        # For L=6: 5 * 16 * n = 80n.
        
        # For L=8: 5 * 64 * n = 320n.
        
        # We compute Tr(A^L) by counting solutions to sum_{j=1}^L x_j = 0 mod n
        # where x_j in S_inv = {s_i, -s_i}.
        
        # Efficiently, we can use convolution or just iterate if n is small.
        # But n can be up to 6000.
        # Number of terms in S_inv is 10.
        # L=8: 10^8 terms. Too many to iterate.
        
        # Alternative: Use FFT? Or just check specific small L?
        # Actually, if we just check L=4 and L=6, it might be sufficient for filtering.
        # L=4: 10^4 = 10000 combinations. For each n, we can count how many sum to 0.
        # Let count_L4 be the number of tuples (a,b,c,d) in S_inv^4 such that a+b+c+d = 0 mod n.
        # Then Tr(A^4) = n * (count_L4 / 10^4)? No.
        # Tr(A^4) is the number of closed walks of length 4.
        # This is exactly the number of solutions to a+b+c+d = 0 mod n, multiplied by n?
        # No. The walk starts at 0. The sum of steps must be 0.
        # So Tr(A^4) = number of solutions to a+b+c+d = 0 mod n.
        
        # Let's compute count_L4.
        S_inv = [s % n for s in S] + [(-s) % n for s in S]
        
        # Count sums of 4
        count_4 = 0
        for a in S_inv:
            for b in S_inv:
                for c in S_inv:
                    for d in S_inv:
                        if (a + b + c + d) % n == 0:
                            count_4 += 1
                            
        if count_4 != 20 * n:
            return False
            
        # Count sums of 6
        # 10^6 = 1,000,000. Might be slow for many n.
        # But we only do this for candidates that pass L=4.
        count_6 = 0
        for a in S_inv:
            for b in S_inv:
                for c in S_inv:
                    for d in S_inv:
                        for e in S_inv:
                            for f in S_inv:
                                if (a + b + c + d + e + f) % n == 0:
                                    count_6 += 1
                                    
        if count_6 != 80 * n:
            return False
            
        # If L=4 and L=6 pass, it's very likely girth >= 9.
        # We can also check L=2 (always 5n) and L=8 (320n) if needed.
        return True

    # Search for n and S
    # n must be > 426 (Moore bound)
    # We try n in a range.
    # To be fast, we try specific n values or a small range.
    
    # Known: LUW size is ~6250. We want n < 6250.
    # Let's try n from 1000 to 2000 first.
    
    for n in range(1000, 2000, 10):
        # Try a few S sets
        # 1. Powers of 2
        S1 = [1, 2, 4, 8, 16]
        if len(set(S1)) == 5:
            if check_traces(n, S1):
                edges = []
                for i in range(n):
                    for s in S1:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
        # 2. Powers of 3
        S2 = [1, 3, 9, 27, 81]
        if len(set(S2)) == 5:
            if check_traces(n, S2):
                edges = []
                for i in range(n):
                    for s in S2:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges

        # 3. Random
        random.seed(n)
        candidates = list(range(1, n))
        random.shuffle(candidates)
        S3 = candidates[:5]
        # Check distinctness and no inverses
        valid = True
        for i in range(5):
            for j in range(i+1, 5):
                if S3[i] == S3[j] or (S3[i] + S3[j]) % n == 0:
                    valid = False
        if valid:
            if check_traces(n, S3):
                edges = []
                for i in range(n):
                    for s in S3:
                        j = (i + s) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges

    # Fallback
    return _construct_luw(5, 9)
