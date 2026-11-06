def construct():
    from itertools import combinations
    
    def check_valid(edge_colors):
        """Check every K4 in K16 has at least 5 distinct colors."""
        for quad in combinations(range(16), 4):
            a, b, c, d = quad
            cols = {
                edge_colors[a][b], edge_colors[a][c], edge_colors[a][d],
                edge_colors[b][c], edge_colors[b][d], edge_colors[c][d]
            }
            if len(cols) < 5:
                return False
        return True
    
    # Try multiplicative schemes with various primes
    candidates = []
    
    for p in [17, 19, 23, 29, 31, 37, 41]:
        edge_colors = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(i+1, 16):
                c = ((i+1)*(j+1)) % p
                edge_colors[i][j] = c
                edge_colors[j][i] = c
        if check_valid(edge_colors):
            num = len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
            candidates.append((num, p, 'mult'))
    
    # Try additive schemes
    for p in [17, 19, 23, 29, 31]:
        edge_colors = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(i+1, 16):
                c = (i + j) % p
                edge_colors[i][j] = c
                edge_colors[j][i] = c
        if check_valid(edge_colors):
            num = len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
            candidates.append((num, p, 'add'))
    
    # Try mixed: (i*j + i + j) % p
    for p in [17, 19, 23]:
        edge_colors = [[0]*16 for _ in range(16)]
        for i in range(16):
            for j in range(i+1, 16):
                c = ((i+1)*(j+1) + i + j) % p
                edge_colors[i][j] = c
                edge_colors[j][i] = c
        if check_valid(edge_colors):
            num = len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
            candidates.append((num, p, 'mixed'))
    
    # Try (i*j) % p with shift
    for p in [17, 19, 23]:
        for shift in range(p):
            edge_colors = [[0]*16 for _ in range(16)]
            for i in range(16):
                for j in range(i+1, 16):
                    c = ((i+shift)*(j+shift)) % p
                    edge_colors[i][j] = c
                    edge_colors[j][i] = c
            if check_valid(edge_colors):
                num = len(set(edge_colors[i][j] for i in range(16) for j in range(i+1, 16)))
                candidates.append((num, p, f'mult_shift_{shift}'))
                break  # only need first valid shift for this p
    
    if candidates:
        candidates.sort()
        best_num = candidates[0][0]
        # Reconstruct best
        best_p = candidates[0][1]
        best_type = candidates[0][2]
        
        edge_colors = [[0]*16 for _ in range(16)]
        if best_type == 'mult':
            for i in range(16):
                for j in range(i+1, 16):
                    c = ((i+1)*(j+1)) % best_p
                    edge_colors[i][j] = c
                    edge_colors[j][i] = c
        elif best_type == 'add':
            for i in range(16):
                for j in range(i+1, 16):
                    c = (i + j) % best_p
                    edge_colors[i][j] = c
                    edge_colors[j][i] = c
        elif best_type == 'mixed':
            for i in range(16):
                for j in range(i+1, 16):
                    c = ((i+1)*(j+1) + i + j) % best_p
                    edge_colors[i][j] = c
                    edge_colors[j][i] = c
        elif best_type.startswith('mult_shift'):
            shift = int(best_type.split('_')[2])
            for i in range(16):
                for j in range(i+1, 16):
                    c = ((i+shift)*(j+shift)) % best_p
                    edge_colors[i][j] = c
                    edge_colors[j][i] = c
        
        return [edge_colors[i][j] for i in range(16) for j in range(i+1, 16)]
    
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
