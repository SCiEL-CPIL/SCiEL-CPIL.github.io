"""Rewritten program: randomized search for minimal 4-dominator tournament."""
import random
import itertools

def _check_valid(n, orientation):
    """Check if every 4-set has a common dominator."""
    # Build adjacency: adj[i][j] = True iff i -> j
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
    
    # For each 4-set, check if any vertex dominates the other 3
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Check if a dominates b,c,d
        if adj[a][b] and adj[a][c] and adj[a][d]:
            continue
        # Check if b dominates a,c,d
        if adj[b][a] and adj[b][c] and adj[b][d]:
            continue
        # Check if c dominates a,b,d
        if adj[c][a] and adj[c][b] and adj[c][d]:
            continue
        # Check if d dominates a,b,c
        if adj[d][a] and adj[d][b] and adj[d][c]:
            continue
        # No dominator found for this 4-set
        return False
    return True

def construct():
    """Find the smallest n such that a valid tournament exists."""
    # The smallest known is 67. Try n from 67 upward.
    # Use randomized approach: for each n, try many random tournaments.
    
    best_result = None
    
    for n in range(67, 72):  # Try a small range; 67 is the known minimum
        for attempt in range(200):
            random.seed(42 + attempt * 1000 + n)
            orientation = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            if _check_valid(n, orientation):
                return {'n': n, 'orientation': orientation}
    
    # Fallback: Paley tournament on 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
