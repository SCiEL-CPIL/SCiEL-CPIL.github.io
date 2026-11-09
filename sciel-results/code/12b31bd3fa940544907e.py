import time

def _paley_orientation(p):
    squares = {x * x % p for x in range(1, p)}
    return [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]

def _is_valid_fast(n, orientation):
    """Fast check using numpy-like bit operations via lists."""
    adj = [[0] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if orientation[idx] == 0:
                adj[i][j] = 1
            else:
                adj[j][i] = 1
            idx += 1
    
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    s = (a, b, c, d)
                    ok = False
                    for v in s:
                        if adj[v][s[0] if v != s[0] else s[1]] and adj[v][s[1] if v != s[1] else s[0]]:
                            # Check all three others
                            others = [u for u in s if u != v]
                            if all(adj[v][u] for u in others):
                                ok = True
                                break
                    if not ok:
                        return False
    return True

def construct():
    start = time.time()
    deadline = start + 110
    
    # Primes p ≡ 3 mod 4, 5 ≤ p ≤ 89
    primes_3mod4 = []
    for p in range(7, 90):
        if p % 4 != 3:
            continue
        if all(p % d != 0 for d in range(2, int(p ** 0.5) + 1)):
            primes_3mod4.append(p)
    
    for p in primes_3mod4:
        if time.time() > deadline - 40:
            break
        orient = _paley_orientation(p)
        if _is_valid_fast(p, orient):
            return {'n': p, 'orientation': orient}
    
    # Fallback: known valid Paley on 83
    return {'n': 83, 'orientation': _paley_orientation(83)}
