import time

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    N = 180
    
    # Precompute for each a, the (b,c) pairs forming a forbidden triple with a as 'a'
    forbidden_as_a = {}
    for a in range(1, N+1):
        a2 = a * a
        pairs = []
        for x in range(1, a):
            if a2 % x == 0:
                y = a2 // x
                b = a - x
                c = a - y
                if 1 <= b <= N and 1 <= c <= N and b != c and b != a and c != a:
                    pairs.append((min(b, c), max(b, c)))
        if pairs:
            forbidden_as_a[a] = pairs
    
    S = set(range(91, 181))
    
    def check_add(x):
        if x in forbidden_as_a:
            for (b, c) in forbidden_as_a[x]:
                if b in S and c in S:
                    return False
        for a in S:
            if a == x:
                continue
            d = a - x
            if d == 0:
                continue
            if (a * a) % d != 0:
                continue
            c = a - (a * a) // d
            if c == x or c == a:
                continue
            if 1 <= c <= N and c in S:
                return False
        return True
    
    for x in range(90, 0, -1):
        if time.time() > deadline:
            break
        if x not in S and check_add(x):
            S.add(x)
    
    return sorted(S)
