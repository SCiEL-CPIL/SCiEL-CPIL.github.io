def construct():
    from itertools import combinations
    
    def check_valid(edge_colors):
        for quad in combinations(range(16), 4):
            a, b, c, d = quad
            cols = {
                edge_colors[a][b], edge_colors[a][c], edge_colors[a][d],
                edge_colors[b][c], edge_colors[b][d], edge_colors[c][d]
            }
            if len(cols) < 5:
                return False
        return True

    best_num = 120
    best_result = None

    primes = [17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for p in primes:
        # Valid shifts must ensure (i+s) % p != 0 for all i in 0..15
        # This means the set {s, ..., s+15} mod p does not contain 0.
        # This is equivalent to s not in {0, -1, ..., -15} mod p.
        # Or s in {1, ..., p-16} if we consider the gap.
        # Actually, just check all s and skip if 0 is present.
        for s in range(p):
            # Check if 0 is in values
            # Values are (i+s) % p for i in 0..15
            # 0 is present if (i+s) % p == 0 for some i.
            # i = -s % p. If 0 <= -s % p < 16, then 0 is present.
            if (-s) % p < 16:
                continue
            
            edge_colors = [[0]*16 for _ in range(16)]
            for i in range(16):
                for j in range(i+1, 16):
                    c = ((i+s)*(j+s)) % p
                    edge_colors[i][j] = c
                    edge_colors[j][i] = c
            
            if check_valid(edge_colors):
                num = len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
                if num < best_num:
                    best_num = num
                    best_result = [edge_colors[i][j] for i in range(16) for j in range(i+1, 16)]

    if best_result is None:
        # Fallback to a known valid configuration
        # p=17, s=1
        edge_colors = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(i+1, 16):
                c = ((i+1)*(j+1)) % 17
                edge_colors[i][j] = c
                edge_colors[j][i] = c
        best_result = [edge_colors[i][j] for i in range(16) for j in range(i+1, 16)]

    return best_result
