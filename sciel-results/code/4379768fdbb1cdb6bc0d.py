# EVOLVE-BLOCK-START
def construct(k, g):
    # Use LUW construction with q = k (assuming k is prime power or we handle q=4 for k=5? 
    # Actually, if k=5, q=5 works for degree q. 
    # If k=3, q=3 works for degree q.
    # If k=4, q=4 works for degree q? No, q=4 degree is 4? 
    # Let's stick to q=k for prime k.
    q = k
    
    # Determine d for girth g
    # Girth is 2d + 1
    # 2d + 1 >= g  =>  d >= (g-1)/2
    d = (g - 1) // 2
    if 2 * d + 1 < g:
        d += 1

    # Precompute GF(q) tables for speed
    # For q=5, just use mod 5
    if q == 5:
        add_t = [[(i + j) % 5 for j in range(5)] for i in range(5)]
        sub_t = [[(i - j) % 5 for j in range(5)] for i in range(5)]
        mul_t = [[(i * j) % 5 for j in range(5)] for i in range(5)]
    elif q == 4:
        # GF(4) = {0, 1, 2, 3} with 00, 01, 10, 11
        # 1^2 = 1, 2^2 = 3, 3^2 = 2 (if primitive element is 2 or 3)
        # Standard GF(4) addition is XOR
        add_t = [[i ^ j for j in range(4)] for i in range(4)]
        sub_t = add_t # Since char 2
        # Multiplication table for GF(4) with primitive element 2 (0b10)
        # 0=0, 1=1, 2=2, 3=3
        # 1*1=1, 1*2=2, 1*3=3
        # 2*2=3, 2*3=1, 3*3=2
        mul_t = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
    else:
        # Fallback for other primes
        add_t = [[(i + j) % q for j in range(q)] for i in range(q)]
        sub_t = [[(i - j) % q for j in range(q)] for i in range(q)]
        mul_t = [[(i * j) % q for j in range(q)] for i in range(q)]

    def f_add(a, b): return add_t[a][b]
    def f_sub(a, b): return sub_t[a][b]
    def f_mul(a, b): return mul_t[a][b]

    # Recurrence rules for LUW
    # For d=4 (girth 9), we need specific rules.
    # The standard LUW rules for d=4 are:
    # x2 = x1 + y1
    # x3 = x2 + y2
    # x4 = x3 + y3
    # ... this is a simplified view.
    # The actual LUW construction uses a specific set of equations.
    # For the purpose of this task, we assume the rules from the seed are correct 
    # for the specific q and d.
    
    rules = {}
    rules[2] = (1, 1)
    rules[3] = (2, 1)
    rules[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        base = c - r
        candidates = [(1, base - 2), (base - 1, 1), (base, 1), (1, base + 1)]
        rules[c] = candidates[r]

    # BFS
    start_key = (0, tuple([0] * (d + 1)))
    label = {start_key: 0}
    queue = [start_key]
    edge_list = []

    for vkey in queue:
        side, coords = vkey
        vidx = label[vkey]
        for first_val in range(q):
            ncoords = [0] * (d + 1)
            ncoords[1] = first_val
            for c in range(2, d + 1):
                a, b = rules[c]
                if side == 0:
                    ncoords[c] = f_add(coords[c], f_mul(ncoords[a], coords[b]))
                else:
                    ncoords[c] = f_sub(coords[c], f_mul(coords[a], ncoords[b]))
            nkey = (1 - side, tuple(ncoords))
            if nkey not in label:
                label[nkey] = len(queue)
                queue.append(nkey)
            if side == 0:
                edge_list.append((vidx, label[nkey]))

    return len(queue), edge_list
# EVOLVE-BLOCK-END
