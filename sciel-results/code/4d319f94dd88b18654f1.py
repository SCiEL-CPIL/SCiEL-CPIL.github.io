import time
from math import gcd

def _lcm(a,b):
    return a*b//gcd(a,b)

def construct():
    import time, random
    from math import gcd
    start = time.time()
    DEADLINE = start + 85.0
    N = 128
    nums = list(range(1, N+1))
    def lcm(a,b):
        return a*b//gcd(a,b)
    # precompute bad triple table as set of (a,b,c) with a<b<c
    bad = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            lab = lcm(a,b)
            for c in range(b+1, N+1):
                if lab == lcm(a,c) == lcm(b,c):
                    bad.add((a,b,c))
    # map for quick: for a pair (a,b) and x, is (a,b,x) bad?
    # build bad_with[x] = set of (a,b) pairs (a<b) such that {a,b,x} bad
    bad_with = [set() for _ in range(N+1)]
    for (a,b,c) in bad:
        bad_with[c].add((a,b))
        bad_with[b].add((a,c))
        bad_with[a].add((b,c))
    def can_add(S, x):
        # S is a set
        # check all pairs in S against bad_with[x]
        bw = bad_with[x]
        if not bw:
            return True
        # iterate over S pairs; but faster: for each pair in bw, check both in S
        for (a,b) in bw:
            if a in S and b in S:
                return False
        return True
    def greedy(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    orders = []
    orders.append(nums)
    orders.append(nums[::-1])
    orders.append(sorted(nums, key=lambda v: (v.bit_count(), v)))
    orders.append(sorted(nums, key=lambda v: -v.bit_count()))
    random.seed(12345)
    for _ in range(6):
        o = nums[:]; random.shuffle(o); orders.append(o)
    best = set()
    for o in orders:
        S = greedy(o)
        if len(S) > len(best):
            best = S
    # local search polish
    def polish(S):
        S = set(S)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # try additions
            for x in range(1, N+1):
                if x not in S and can_add(S, x):
                    S.add(x); improved = True
            # try 2-for-1 swaps
            for i in list(S):
                S2 = S - {i}
                # find two excluded that can be added to S2
                added = 0
                for x in range(1, N+1):
                    if x not in S2 and x != i and can_add(S2, x):
                        S2.add(x); added += 1
                        if added == 2:
                            break
                if added == 2:
                    S = S2; improved = True
        return S
    best = polish(best)
    # try polishing a couple other good greedy starts too
    for o in orders[:4]:
        if time.time() > DEADLINE - 5:
            break
        S = greedy(o)
        S = polish(S)
        if len(S) > len(best):
            best = S
    if not best:
        best = {1}
    return sorted(best)
