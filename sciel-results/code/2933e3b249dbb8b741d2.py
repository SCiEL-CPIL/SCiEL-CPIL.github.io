import random
import itertools

def is_valid_tournament(n, orientation):
    # orientation is list of bits for i<j
    # Reconstruct adjacency matrix for speed?
    # Or just check 4-sets
    adj = [[0]*n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if orientation[idx] == 0:
                adj[i][j] = 1
                adj[j][i] = 0
            else:
                adj[i][j] = 0
                adj[j][i] = 1
            idx += 1
    
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        # Check if a beats b,c,d
        if adj[a][b] and adj[a][c] and adj[a][d]: continue
        if adj[b][a] and adj[b][c] and adj[b][d]: continue
        if adj[c][a] and adj[c][b] and adj[c][d]: continue
        if adj[d][a] and adj[d][b] and adj[d][c]: continue
        return False
    return True

def construct():
    # Try to find a smaller n using random cyclic tournaments
    for n in range(31, 67):
        if n % 2 == 0: continue # Cyclic tournaments usually defined for odd n? 
        # Actually, cyclic tournaments can be defined for any n, but "standard" ones are odd.
        # For even n, we can use a different definition, but let's stick to odd for simplicity.
        k = (n - 1) // 2
        for _ in range(500): # 500 trials per n
            S = set(random.sample(range(1, n), k))
            # Build orientation
            orient = []
            for i in range(n):
                for j in range(i+1, n):
                    if (j - i) % n in S:
                        orient.append(0)
                    else:
                        orient.append(1)
            if is_valid_tournament(n, orient):
                return {'n': n, 'orientation': orient}
    
    # Fallback: Paley 67
    p = 67
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
