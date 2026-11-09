"""Program: large 3-sum-free-pair-sum set in 1..128 via bit-parallel greedy independent set."""
# EVOLVE-BLOCK-START
import random

N = 128

def construct():
    # Model as 3-uniform hypergraph on vertices 1..128.
    # Hyperedge {a,b,c} (a<b<c) exists iff a+b, a+c, b+c are all in [1,128].
    # We seek a large independent set (no hyperedge fully contained in S).
    #
    # Key insight: a+b, a+c, b+c are all >= 1+2=3 and <= 126+127=253, but we need
    # them <= 128 to possibly be in S. So a+b<=128, a+c<=128, b+c<=128.
    # Since a<b<c, b+c is the largest pair sum, so we need b+c<=128.
    # This restricts hyperedges to triples where the two largest sum to <=128.
    #
    # Bit-parallel greedy: represent S as a bitmask. When considering adding x,
    # check if there exist a,b in S with a<b such that {x,a,b} forms a hyperedge
    # AND all three pair sums are in S.
    #
    # Precompute for each x: list of (a,b) with a<b, a,b != x, such that
    # {min(x,a,b), mid(x,a,b), max(x,a,b)} has all pair sums <= 128.
    # Then when checking if x can be added, for each (a,b) in that list:
    #   if a in S and b in S:
    #     let t = sorted([x,a,b]) -> (u,v,w)
    #     if (u+v in S) and (u+w in S) and (v+w in S): cannot add x.

    # Precompute dangerous pairs for each x
    # For vertex x, dangerous pairs (a,b) where a<b, a!=x, b!=x, and
    # when we form triple {x,a,b}, all pair sums are <= 128.
    dangerous_pairs = [[] for _ in range(N+1)]
    for x in range(1, N+1):
        for a in range(1, N+1):
            if a == x:
                continue
            for b in range(a+1, N+1):
                if b == x:
                    continue
                # Form triple from {x, a, b}
                triple = sorted([x, a, b])
                u, v, w = triple[0], triple[1], triple[2]
                # Check all pair sums <= 128
                if u+v <= 128 and u+w <= 128 and v+w <= 128:
                    # Store the pair (a,b) with a<b for this x
                    dangerous_pairs[x].append((a, b))

    def can_add(x, S_mask, S_set):
        for (a, b) in dangerous_pairs[x]:
            if (1 << a) & S_mask and (1 << b) & S_mask:
                # Both a and b are in S. Check if all pair sums are in S.
                triple = sorted([x, a, b])
                u, v, w = triple[0], triple[1], triple[2]
                if ((1 << (u+v)) & S_mask) and ((1 << (u+w)) & S_mask) and ((1 << (v+w)) & S_mask):
                    return False
        return True

    best = None
    best_len = 0

    rng = random.Random(12345)

    for trial in range(3000):
        perm = list(range(1, N+1))
        rng.shuffle(perm)

        S_mask = 0
        S_set = set()

        for x in perm:
            if can_add(x, S_mask, S_set):
                S_mask |= (1 << x)
                S_set.add(x)

        if len(S_set) > best_len:
            best_len = len(S_set)
            best = list(S_set)

        if best_len >= 95:
            break

    # Verify best
    def is_valid(S):
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                for k in range(j+1, len(S)):
                    a, b, c = S[i], S[j], S[k]
                    if (a+b in S_set) and (a+c in S_set) and (b+c in S_set):
                        return False
        return True

    if best is None:
        best = list(range(65, N+1))

    if not is_valid(best):
        best = list(range(65, N+1))

    return list(best)
# EVOLVE-BLOCK-END
