import math

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    if k != 4 or g != 10:
        return luw_construct(k, g)

    # Try to find a circulant graph C_N(a, b) with girth >= 10
    # Generators: a, -a, b, -b
    # Check for non-backtracking cycles of length 4, 6, 8
    
    # Precompute all non-backtracking walk index sequences
    inv_idx = [1, 0, 3, 2]
    
    # Length 4 walks: (i1, i2, i3, i4) where i_{j+1} != inv(i_j)
    walks4 = []
    for i1 in range(4):
        for i2 in range(4):
            if i2 == inv_idx[i1]: continue
            for i3 in range(4):
                if i3 == inv_idx[i2]: continue
                for i4 in range(4):
                    if i4 == inv_idx[i3]: continue
                    walks4.append((i1, i2, i3, i4))
    
    # Length 6 walks
    walks6 = []
    for i1 in range(4):
        for i2 in range(4):
            if i2 == inv_idx[i1]: continue
            for i3 in range(4):
                if i3 == inv_idx[i2]: continue
                for i4 in range(4):
                    if i4 == inv_idx[i3]: continue
                    for i5 in range(4):
                        if i5 == inv_idx[i4]: continue
                        for i6 in range(4):
                            if i6 == inv_idx[i5]: continue
                            walks6.append((i1, i2, i3, i4, i5, i6))
    
    # Length 8 walks
    walks8 = []
    for i1 in range(4):
        for i2 in range(4):
            if i2 == inv_idx[i1]: continue
            for i3 in range(4):
                if i3 == inv_idx[i2]: continue
                for i4 in range(4):
                    if i4 == inv_idx[i3]: continue
                    for i5 in range(4):
                        if i5 == inv_idx[i4]: continue
                        for i6 in range(4):
                            if i6 == inv_idx[i5]: continue
                            for i7 in range(4):
                                if i7 == inv_idx[i6]: continue
                                for i8 in range(4):
                                    if i8 == inv_idx[i7]: continue
                                    walks8.append((i1, i2, i3, i4, i5, i6, i7, i8))
    
    def check_girth_fast(N, a, b):
        gens = [a, -a, b, -b]
        gen_mod = [g % N for g in gens]
        
        # Check length 4
        for w in walks4:
            s = (gen_mod[w[0]] + gen_mod[w[1]] + gen_mod[w[2]] + gen_mod[w[3]]) % N
            if s == 0:
                return False
        
        # Check length 6
        for w in walks6:
            s = (gen_mod[w[0]] + gen_mod[w[1]] + gen_mod[w[2]] + gen_mod[w[3]] + gen_mod[w[4]] + gen_mod[w[5]]) % N
            if s == 0:
                return False
        
        # Check length 8
        for w in walks8:
            s = (gen_mod[w[0]] + gen_mod[w[1]] + gen_mod[w[2]] + gen_mod[w[3]] + gen_mod[w[4]] + gen_mod[w[5]] + gen_mod[w[6]] + gen_mod[w[7]]) % N
            if s == 0:
                return False
        
        return True

    # Search for smallest N with valid circulant graph
    for N in range(243, 500):
        for a in range(1, N // 2 + 1):
            if 2 * a % N == 0:
                continue
            for b in range(a + 1, N // 2 + 1):
                if 2 * b % N == 0:
                    continue
                if (a + b) % N == 0:
                    continue
                if (a - b) % N == 0:
                    continue
                # Connectivity: gcd(a, b, N) == 1
                if math.gcd(math.gcd(a, b), N) != 1:
                    continue
                
                if check_girth_fast(N, a, b):
                    edges = []
                    for i in range(N):
                        edges.append((i, (i + a) % N))
                        edges.append((i, (i + b) % N))
                    return N, edges

    return luw_construct(k, g)
