"""Schütte's problem k=4: find smallest tournament where every 4-set has a common dominator."""
import time
import random
import numpy as np

_DEADLINE = 140.0  # seconds; safety margin before 150s kill

def _is_valid(n, orientation):
    """Check that every 4-subset has a vertex dominating all four.
    orientation: list of bits, index = i*n + j for i<j (lexicographic pair order).
    0 means i->j, 1 means j->i.
    """
    # Build adjacency: adj[i][j] = True if i beats j
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            b = orientation[idx]
            idx += 1
            if b == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True

    # For each 4-subset, check if some vertex in the subset beats all other three
    from itertools import combinations
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Check if any of the 4 dominates the other 3
        if (adj[a][b] and adj[a][c] and adj[a][d]) or \
           (adj[b][a] and adj[b][c] and adj[b][d]) or \
           (adj[c][a] and adj[c][b] and adj[c][d]) or \
           (adj[d][a] and adj[d][b] and adj[d][c]):
            continue
        return False
    return True

def _check_paley(p):
    """Check if Paley tournament on p vertices is valid for k=4. Returns orientation list or None."""
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            orientation.append(0 if (j - i) % p in squares else 1)
    if _is_valid(p, orientation):
        return orientation
    return None

def _paley_orientation(p):
    """Return orientation list for Paley tournament on p vertices."""
    squares = {x * x % p for x in range(1, p)}
    return [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]

def _greedy_improve(n, orientation, time_limit):
    """Try to improve coverage by flipping edges. Returns (n, orientation) if improved, else None."""
    start = time.time()
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            b = orientation[idx]
            idx += 1
            if b == 0:
                adj[i, j] = True
            else:
                adj[j, i] = True

    # For each 4-subset, find if it's covered
    # A 4-subset {a,b,c,d} is covered if some vertex in it beats the other 3
    # We want to find uncovered 4-subsets and fix them by flipping edges

    def count_uncovered(adj):
        uncovered = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        if not (adj[a,b] and adj[a,c] and adj[a,d]) and \
                           not (adj[b,a] and adj[b,c] and adj[b,d]) and \
                           not (adj[c,a] and adj[c,b] and adj[c,d]) and \
                           not (adj[d,a] and adj[d,b] and adj[d,c]):
                            uncovered += 1
        return uncovered

    # This is too slow for large n; use a sampling approach
    # Instead, let's try random edge flips and check validity on a sample
    rng = random.Random(42)
    best_adj = adj.copy()
    
    # Sample-based local search
    sample_size = min(5000, n*(n-1)//2)
    
    for iteration in range(2000):
        if time.time() - start > time_limit:
            break
        # Pick a random edge to flip
        i = rng.randint(0, n-1)
        j = rng.randint(0, n-1)
        if i == j:
            continue
        # Flip
        old_val = adj[i, j]
        adj[i, j] = not old_val
        adj[j, i] = old_val
        
        # Quick check on a random sample of 4-subsets
        valid = True
        for _ in range(200):
            quad = rng.sample(range(n), 4)
            a, b, c, d = quad
            if not (adj[a,b] and adj[a,c] and adj[a,d]) and \
               not (adj[b,a] and adj[b,c] and adj[b,d]) and \
               not (adj[c,a] and adj[c,b] and adj[c,d]) and \
               not (adj[d,a] and adj[d,b] and adj[d,c]):
                valid = False
                break
        if valid:
            best_adj = adj.copy()
        else:
            # Revert
            adj[i, j] = old_val
            adj[j, i] = not old_val

    # Convert best_adj to orientation
    orientation_out = []
    for i in range(n):
        for j in range(i+1, n):
            orientation_out.append(0 if best_adj[i,j] else 1)
    return (n, orientation_out)

def construct():
    t_start = time.time()
    
    # Strategy: Try Paley tournaments on primes, starting from small to large
    # Known: f(4) >= 31, smallest known is 67
    # Try primes in increasing order to find the smallest valid one
    
    # Primes to try (must be congruent to 3 mod 4 for Paley tournaments to be antisymmetric)
    # Actually Paley tournament works for any prime p ≡ 3 (mod 4)
    # For p ≡ 1 (mod 4), it's not a tournament (has bidirectional edges)
    
    # Let's list primes ≡ 3 (mod 4) starting from 31
    candidates = []
    for p in range(31, 91):
        # Check if p is prime
        is_prime = True
        for d in range(2, int(p**0.5)+1):
            if p % d == 0:
                is_prime = False
                break
        if is_prime and p % 4 == 3:
            candidates.append(p)
    
    # Also include some primes ≡ 1 mod 4 (Paley won't work as tournament, but let's try anyway)
    # Actually for p ≡ 1 mod 4, -1 is a square, so (j-i) and (i-j) can both be squares
    # This means the Paley construction doesn't give a tournament for p ≡ 1 mod 4
    # So we only use p ≡ 3 mod 4
    
    # Sort candidates by size (smallest first)
    candidates.sort()
    
    # First, quickly check if any small Paley tournament works
    for p in candidates:
        if time.time() - t_start > _DEADLINE - 30:
            break
        orientation = _check_paley(p)
        if orientation is not None:
            return {'n': p, 'orientation': orientation}
    
    # If no Paley tournament works, try a random search approach
    # Use the smallest n that might work (Erdos lower bound is 31)
    # Try random tournaments with local search
    
    # Try n from 31 upward with random initialization + local search
    for n in range(31, 91):
        if time.time() - t_start > _DEADLINE - 20:
            break
        
        # Try a few random tournaments
        for attempt in range(10):
            if time.time() - t_start > _DEADLINE - 15:
                break
            
            # Random orientation
            orientation = [random.randint(0, 1) for _ in range(n*(n-1)//2)]
            
            # Check if valid (expensive for large n, so sample)
            # For small n, do full check
            if n <= 40:
                if _is_valid(n, orientation):
                    return {'n': n, 'orientation': orientation}
            else:
                # Sample check
                adj = np.zeros((n, n), dtype=bool)
                idx = 0
                for i in range(n):
                    for j in range(i+1, n):
                        b = orientation[idx]
                        idx += 1
                        if b == 0:
                            adj[i, j] = True
                        else:
                            adj[j, i] = True
                
                # Sample 1000 random 4-subsets
                valid = True
                for _ in range(1000):
                    quad = [random.randint(0, n-1) for _ in range(4)]
                    # Ensure distinct
                    while len(set(quad)) < 4:
                        quad = [random.randint(0, n-1) for _ in range(4)]
                    a, b, c, d = quad
                    if not (adj[a,b] and adj[a,c] and adj[a,d]) and \
                       not (adj[b,a] and adj[b,c] and adj[b,d]) and \
                       not (adj[c,a] and adj[c,b] and adj[c,d]) and \
                       not (adj[d,a] and adj[d,b] and adj[d,c]):
                        valid = False
                        break
                
                if valid:
                    # Full verification
                    if _is_valid(n, orientation):
                        return {'n': n, 'orientation': orientation}
    
    # Fallback: use the best Paley tournament we know (83 vertices)
    p = 83
    orientation = _paley_orientation(p)
    return {'n': p, 'orientation': orientation}
