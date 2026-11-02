import random
import time
import itertools

def construct(k, g):
    # Fallback: L-U-W graph
    def luw_construct(k, g):
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

    if k != 5 or g != 9:
        return luw_construct(k, g)
    
    start_time = time.time()
    time_limit = 115.0

    def has_4cycle(m, S):
        """Check for 4-cycles in bipartite circulant with connection set S."""
        S_list = list(S)
        for s1 in S_list:
            for s2 in S_list:
                p2 = (s1 + s2) % m
                if p2 == 0:
                    continue
                for s3 in S_list:
                    p3 = (p2 + s3) % m
                    if p3 == 0 or p3 == s1:
                        continue
                    for s4 in S_list:
                        if (p3 + s4) % m == 0:
                            return True
        return False

    def has_6cycle(m, S):
        """Check for 6-cycles."""
        S_list = list(S)
        # s1+s2+s3+s4+s5+s6 = 0
        # Partial sums: p1=s1, p2=s1+s2, p3=s1+s2+s3, p4=p3+s4, p5=p4+s5, p6=0
        # All partial sums p1..p5 must be distinct and non-zero.
        # This is expensive: 5^6 = 15625 iterations, each O(1)
        for s1 in S_list:
            for s2 in S_list:
                p2 = (s1 + s2) % m
                if p2 == 0:
                    continue
                for s3 in S_list:
                    p3 = (p2 + s3) % m
                    if p3 == 0 or p3 == s1:
                        continue
                    for s4 in S_list:
                        p4 = (p3 + s4) % m
                        if p4 == 0 or p4 == s1 or p4 == p2:
                            continue
                        for s5 in S_list:
                            p5 = (p4 + s5) % m
                            if p5 == 0 or p5 == s1 or p5 == p2 or p5 == p3:
                                continue
                            for s6 in S_list:
                                if (p5 + s6) % m == 0:
                                    return True
        return False

    def has_8cycle(m, S):
        """Check for 8-cycles. This is 5^8 = 390625 iterations, might be slow."""
        S_list = list(S)
        for s1 in S_list:
            for s2 in S_list:
                p2 = (s1 + s2) % m
                if p2 == 0:
                    continue
                for s3 in S_list:
                    p3 = (p2 + s3) % m
                    if p3 == 0 or p3 == s1:
                        continue
                for s4 in S_list:
                    p4 = (p3 + s4) % m
                    if p4 == 0 or p4 in (s1, p2):
                        continue
                    for s5 in S_list:
                        p5 = (p4 + s5) % m
                        if p5 == 0 or p5 in (s1, p2, p3):
                            continue
                        for s6 in S_list:
                            p6 = (p5 + s6) % m
                            if p6 == 0 or p6 in (s1, p2, p3, p4):
                                continue
                            for s7 in S_list:
                                p7 = (p6 + s7) % m
                                if p7 == 0 or p7 in (s1, p2, p3, p4, p5):
                                    continue
                                for s8 in S_list:
                                    if (p7 + s8) % m == 0:
                                        return True
        return False

    # Search for good (m, S) pairs
    for m in range(341, 700):
        if time.time() - start_time > time_limit:
            break
        
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
            
            S = random.sample(range(1, m), 5)
            S_set = set(S)
            
            if has_4cycle(m, S_set):
                continue
            if has_6cycle(m, S_set):
                continue
            if has_8cycle(m, S_set):
                continue
            
            n = 2 * m
            edges = []
            for i in range(m):
                for s in S:
                    j = (i + s) % m
                    edges.append((i, m + j))
            
            return n, edges

    return luw_construct(k, g)
