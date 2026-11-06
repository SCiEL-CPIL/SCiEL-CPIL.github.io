"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    # Precompute the "forbidden" bitset for each a:
    # forbid[a] is a bitset of values v in [a+1, n] such that
    # there exists b > a with b in set and v = b + c for some c >= b in set
    # that makes a | (b+c).
    # More precisely: for a given a, and a given b > a, the forbidden c values
    # are those c >= b such that a | (b+c).
    # If we include b in the set, then we cannot include any c >= b where a | (b+c).
    # This is a complex constraint. Let's think differently.
    
    # A set S is valid if for all a < b <= c in S, a does not divide b+c.
    # Equivalent: for all a in S, for all b in S with b > a, for all c in S with c >= b,
    # a does not divide b+c.
    
    # Let's use a greedy approach with lookahead, or simply try to build the largest set.
    # Given the structure, let's try a different approach:
    # The constraint is a | (b+c) is forbidden.
    # For large a, the multiples of a in the range [2a+2, 512] are sparse.
    
    # Let's try a construction based on intervals.
    # If we take S = {k, k+1, ..., n}, the condition is:
    # For all a in [k, n], for all b in [a+1, n], for all c in [b, n]:
    # a does not divide b+c.
    # b+c ranges from 2a+2 to 2n.
    # The multiples of a in this range: ma where ma in [2a+2, 2n].
    # For a > n/2, 2a > n, so 2a+2 > n+2, and 3a > 3n/2.
    # If a > n/3 = 85.33, then 3a > n, so the only multiple could be 2a, but 2a < 2a+2.
    # Wait, b+c >= 2a+2 (since b >= a+1, c >= b >= a+1).
    # So the smallest sum is 2a+2.
    # For a > 170, 2a+2 > 342, and 3a > 510.
    # Actually, let's just check: for a in [171, 256], the sums b+c are in [2a+2, 512].
    # Multiples of a in this range: 3a, 4a, ...
    # 3a >= 3*171 = 513 > 512. So no multiples.
    # For a = 170: 3*170 = 510 <= 512. So if 510 = b+c is possible, it's forbidden.
    # b+c = 510 with b > 170, c >= b: b+c=510, b >= 171, c >= b => b >= 171, c = 510-b <= 510-171 = 339.
    # Also c >= b means 510-b >= b => b <= 255.
    # So b in [171, 255], c = 510-b in [255, 339].
    # If both b and c are in S, then 170 divides b+c=510, which is forbidden.
    # So including 170 requires care.
    
    # Let's try S = {171, 172, ..., 256}. Size = 86.
    # This is valid as shown above.
    
    # Can we do better? Let's try to include some numbers below 171.
    # Let's try a heuristic: start with the full set {171, ..., 256} and try to add numbers from below.
    
    base = set(range(171, 257))
    result = list(base)
    
    # Try adding numbers from 1 to 170, one by one, if valid.
    # For efficiency, check validity against current set.
    for a in range(1, 171):
        # Check if adding a to the set keeps it valid.
        # For each b in set with b > a, and each c in set with c >= b:
        # a must not divide b+c.
        ok = True
        s = sorted(result)
        for i, b in enumerate(s):
            if b <= a:
                continue
            for c in s[i:]:
                if c < b:
                    continue
                if (b + c) % a == 0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            result.append(a)
    
    return result
# EVOLVE-BLOCK-END
