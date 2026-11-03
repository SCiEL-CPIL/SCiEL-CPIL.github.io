import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    def luw_construction():
        # Fallback construction
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

    if k == 4 and g == 10:
        n = 320
        # Generate non-backtracking words of length 4, 6, 8
        # Steps: 0:+a, 1:-a, 2:+b, 3:-b
        # Non-backtracking: next step cannot be the inverse of previous.
        # Inverses: 0<->1, 2<->3
        
        def get_valid_words(length):
            words = []
            def backtrack(current_word, last_step):
                if len(current_word) == length:
                    words.append(tuple(current_word))
                    return
                for step in range(4):
                    # Check if step is inverse of last_step
                    if last_step != -1:
                        if (last_step == 0 and step == 1) or (last_step == 1 and step == 0):
                            continue
                        if (last_step == 2 and step == 3) or (last_step == 3 and step == 2):
                            continue
                    current_word.append(step)
                    backtrack(current_word, step)
                    current_word.pop()
            backtrack([], -1)
            return words

        words_4 = get_valid_words(4)
        words_6 = get_valid_words(6)
        words_8 = get_valid_words(8)
        
        # We can precompute the "pattern" of a's and b's for each word
        # Sum = coeff_a * a + coeff_b * b
        # We just need to check if coeff_a * a + coeff_b * b == 0 mod n
        
        # To speed up, we can group words by (coeff_a, coeff_b)
        # But 108 + 972 + 8748 is small enough to just check each time.
        
        all_words = words_4 + words_6 + words_8
        # Precompute coefficients
        word_coeffs = []
        for w in all_words:
            ca = 0
            cb = 0
            for step in w:
                if step == 0: ca += 1
                elif step == 1: ca -= 1
                elif step == 2: cb += 1
                else: cb -= 1
            word_coeffs.append((ca, cb))

        found = False
        # a, b are odd, 1..159
        # a != b, a+b != 320
        for a in range(1, 160, 2):
            if time.time() > deadline - 10:
                break
            for b in range(1, 160, 2):
                if b <= a: continue # Symmetry? No, a and b are distinct roles in graph structure? 
                # Actually C(n, {a,b}) is isomorphic to C(n, {b,a}).
                # So we can assume a < b to halve the work.
                if a + b == 320: continue
                if a == b: continue
                
                valid = True
                for ca, cb in word_coeffs:
                    if (ca * a + cb * b) % 320 == 0:
                        valid = False
                        break
                
                if valid:
                    # Construct edges
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i - a) % n))
                        edges.append((i, (i + b) % n))
                        edges.append((i, (i - b) % n))
                    
                    # Remove duplicates and loops (though logic prevents them)
                    # And sort for consistency
                    unique_edges = set()
                    for u, v in edges:
                        if u == v: continue
                        if u < v:
                            unique_edges.add((u, v))
                        else:
                            unique_edges.add((v, u))
                    
                    return n, list(unique_edges)
        
        if not found:
             # Fallback
             return luw_construction()
             
    else:
        return luw_construction()
