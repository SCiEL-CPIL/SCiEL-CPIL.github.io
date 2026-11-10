import time

def _valid(S):
    Sset = set(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            s_ab = a + b
            if s_ab not in Sset:
                continue
            for k in range(j + 1, n):
                c = S[k]
                if (a + c) in Sset and (b + c) in Sset:
                    return False
    return True

def construct():
    t0 = time.time()
    deadline = 90.0 - 2.0  # safety margin

    N = 128
    # Construction: S = {1..L} U {H..N} with H >= 2L+1  =>  no forbidden triple.
    # Size = L + (N - H + 1).  Minimize L to maximize size.
    # L=1 fails (triple 1,3,4).  L=2, H=5 works.
    L = 2
    H = 2 * L + 1
    S = list(range(1, L + 1)) + list(range(H, N + 1))

    # Verify (fast, bounded).  If somehow invalid, fall back to the seed.
    if _valid(S):
        return [int(x) for x in S]

    # Fallback: top half (always valid, size 64)
    return list(range(N // 2 + 1, N + 1))
