import time

_N = 5040
_TARGET = 4979

def _divisors_of(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def construct():
    start = time.time()
    divisors = [d for d in _divisors_of(_N) if d <= _TARGET]
    
    INF = 10**9
    dp = [INF] * (_TARGET + 1)
    parent = [-1] * (_TARGET + 1)
    dp[0] = 0
    
    for d in divisors:
        for s in range(_TARGET, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
    
    if dp[_TARGET] == INF:
        # Fallback greedy
        remainder = _TARGET
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
    
    result = []
    s = _TARGET
    while s > 0:
        d = parent[s]
        if d == -1:
            break
        result.append(d)
        s -= d
    
    if sum(result) == _TARGET and len(set(result)) == len(result):
        return result
    
    # Final fallback
    remainder = _TARGET
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    return chosen
