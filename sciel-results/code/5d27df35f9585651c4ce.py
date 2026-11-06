import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80.0

    # Sieve for largest prime factor
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i

    best_val = float('inf')
    best_a = 1

    # Direct scan for the minimum max(P(a), P(N-a))
    # This is O(N) which is very fast for N=123457
    # We iterate a from 1 to N-1
    # To optimize, we can just iterate 1 to N//2 and check both a and N-a?
    # No, P(a) != P(N-a) generally. But max(P(a), P(N-a)) is symmetric.
    # So checking a and N-a is redundant if we cover the whole range.
    # We can iterate a from 1 to N//2 and consider the pair (a, N-a).
    # The value is max(lpf[a], lpf[N-a]).
    # This covers all unique pairs.
    
    for a in range(1, N // 2 + 1):
        pa = lpf[a]
        pb = lpf[N - a]
        current_max = pa if pa > pb else pb
        if current_max < best_val:
            best_val = current_max
            best_a = a
            
        if time.time() > deadline:
            break

    return [best_a, N - best_a]
