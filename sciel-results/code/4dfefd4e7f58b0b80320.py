"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

MAX_D = 10_000_000
MIN_D = 5
TIME_LIMIT = 80.0  # leave a safety margin before 90s kill


def _valid(denoms):
    if not (1 <= len(denoms) <= 256):
        return False
    s = set(denoms)
    if len(s) != len(denoms):
        return False
    for d in s:
        if not (MIN_D <= d <= MAX_D):
            return False
    return sum(Fraction(1, d) for d in s) == 1


def _try_reduce(denoms, deadline):
    """
    Try to reduce the count of `denoms` (which sums to 1) by local moves.
    Moves:
      (A) Replace 1/x with 1/a + 1/b where a,b > x, distinct, a,b not in set,
          a,b in range. Sum preserved. Count +1 (we don't want this for reduction).
      (B) Replace two terms 1/x + 1/y (x<y) with 1/z where z integer,
          1/z = 1/x + 1/y => z = xy/(x+y). Must be integer, z >= MIN_D, z <= MAX_D,
          z not in set \\ {x,y}. Count -1.
      (C) Replace one term 1/x with 1/a + 1/b where a, b can be anything distinct
          and not in set (could be < x or > x). Count +1 (skip for reduction).
    For reduction, only (B) helps. Also, we can try (B) on pairs, and if stuck,
    try swapping: remove one term t, add a new term u such that sum changes by
    1/u - 1/t and then fix with one merge. This is complex; we'll stick to (B).
    """
    ds = sorted(set(denoms))
    used = set(ds)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs (i<j) for merge
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                x, y = ds[i], ds[j]
                if y % (x + y) != 0:
                    # z = x*y/(x+y); for z integer, (x+y) | x*y.
                    # Since gcd(x, x+y)=gcd(x,y), this is equivalent to
                    # (x+y)/gcd(x,y) | x/gcd(x,y) ... simpler: just check divisibility.
                    continue
                z = x * y // (x + y)
                if z < MIN_D or z > MAX_D:
                    continue
                if z in used and z != x and z != y:
                    continue
                if z == x or z == y:
                    # degenerate: merging gives same denominator; skip
                    continue
                # Build new set
                new_used = used - {x, y} | {z}
                if len(new_used) != len(used) - 1:
                    continue
                new_ds = sorted(new_used)
                # Verify sum (should be 1)
                if sum(Fraction(1, d) for d in new_ds) == 1:
                    ds = new_ds
                    used = new_used
                    improved = True
                    break
            if improved:
                break
    return ds


def _known_13():
    # Known 13-term Egyptian fraction for 1 with denominators >= 4:
    # 1/4 + 1/5 + 1/6 + 1/10 + 1/12 + 1/15 + 1/20 + 1/30 + 1/60 + 1/70 + 1/105 + 1/140 + 1/420
    # Check:
    # 1/4 + 1/5 = 9/20
    # + 1/6 = 27/60 + 12/60 = 39/60
    # + 1/10 = 39/60 + 6/60 = 45/60
    # + 1/12 = 45/60 + 5/60 = 50/60
    # + 1/15 = 50/60 + 4/60 = 54/60
    # + 1/20 = 54/60 + 3/60 = 57/60
    # + 1/30 = 57/60 + 2/60 = 59/60
    # + 1/60 = 60/60 = 1
    # So actually 1/4+1/5+1/6+1/10+1/12+1/15+1/20+1/30+1/60 = 1 (9 terms).
    # Let me verify:
    # 1/4 + 1/5 = 0.25 + 0.2 = 0.45
    # + 1/6 ≈ 0.616667
    # + 1/10 = 0.716667
    # + 1/12 ≈ 0.800000
    # + 1/15 ≈ 0.866667
    # + 1/20 = 0.916667
    # + 1/30 ≈ 0.950000
    # + 1/60 ≈ 1.000000  Yes!
    # So 9 terms sum to 1: {4,5,6,10,12,15,20,30,60}
    # But 4 is not allowed (min 5). Remove 4: remaining sum = 1 - 1/4 = 3/4.
    # So we need to find denominators >= 5 summing to 3/4, plus 4 is not allowed.
    # Actually we need ALL denominators >= 5. So we need a representation of 1
    # with all denominators >= 5.
    # Let's find one: start with 1/5, need 4/5 more.
    # 1/5 + 1/6 = 11/30, need 19/30.
    # 1/5 + 1/6 + 1/7 = 71/210, need 139/210.
    # 1/5+1/6+1/7+1/10 = 71/210+21/210=92/210=46/105, need 59/105.
    # 1/5+1/6+1/7+1/10+1/14 = 46/105+15/105=61/105, need 44/105.
    # 1/5+1/6+1/7+1/10+1/14+1/35 = 61/105+3/105=64/105, need 41/105.
    # 1/5+1/6+1/7+1/10+1/14+1/35+1/70 = 64/105+15/105=79/105, need 26/105.
    # 1/5+1/6+1/7+1/10+1/14+1/35+1/70+1/105 = 79/105+1/105=80/105=16/21, need 5/21.
    # 1/5+1/6+1/7+1/10+1/14+1/35+1/70+1/105+1/42 = 16/21+1/6=33/42=11/14, need 3/14.
    # 1/5+1/6+1/7+1/10+1/14+1/35+1/70+1/105+1/42+1/14 -- duplicate
    # Let me just use a systematic approach.
    # Known: 1 = 1/2+1/3+1/6. Split each into >=5:
    # 1/2 = 1/5+1/10+1/20+1/20+1/20 -- not distinct
    # 1/2 = 1/5+1/6+1/20+1/30+1/60+1/60 -- not distinct
    # 1/2 = 1/5+1/6+1/7+1/20+1/420+1/... 
    # Let me just return a valid 14-term solution I can verify.
    # 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13+1/14+1/15+1/16+1/17+1/18+1/19+1/20+...
    # Sum of 1/5 to 1/20 ≈ 1.253, too much.
    # Let me just hardcode a verified solution.
    # I'll compute one now:
    # Start: need sum 1, denominators >= 5, distinct.
    # 1/5, need 4/5.
    # 1/6, need 19/30.
    # 1/7, need 71/210 - 1/7 = 71/210 - 30/210 = 41/210. Wait, 1/5+1/6+1/7 = 42/210+35/210+30/210=107/210. Need 103/210.
    # 1/10, need 103/210-21/210=82/210=41/105.
    # 1/14, need 41/105-15/105=26/105.
    # 1/30, need 26/105-1/30. 26/105=52/210, 1/30=7/210, need 45/210=3/14.
    # 1/15, need 3/14-1/15=45/210-14/210=31/210.
    # 1/35, need 31/210-1/35=31/210-6/210=25/210=5/42.
    # 1/42, need 5/42-1/42=4/42=2/21.
    # 1/21, need 2/21-1/21=1/21.
    # 1/21 duplicate.
    # Let me try: 1/5+1/6+1/7+1/10+1/14+1/30+1/15+1/35+1/42 = ?
    # = 1/5+1/6+1/7+1/10+1/14+1/15+1/30+1/35+1/42
    # LCM of 5,6,7,10,14,15,30,35,42 = 210
    # 42+35+30+21+15+14+7+6+5 = 175/210 = 5/6. Need 1/6 more.
    # Add 1/30? Already have it. Add 1/42? Have it.
    # Need 1/6 with new denominators not in {5,6,7,10,14,15,30,35,42}.
    # 1/6 = 1/12+1/24+1/48+1/96+... or 1/7+... no.
    # 1/6 = 1/8+1/24. 8 and 24 not in set. Add 1/8+1/24.
    # Total: 1/5+1/6+1/7+1/10+1/14+1/15+1/30+1/35+1/42+1/8+1/24
    # = 5/6 + 1/8+1/24 = 5/6+1/6 = 1. 11 terms!
    # Denominators: 5,6,7,8,10,14,15,24,30,35,42. All >= 5, distinct.
    # Let me verify: sum = 1/5+1/6+1/7+1/8+1/10+1/14+1/15+1/24+1/30+1/35+1/42
    # LCM = 2520
    # 504+420+360+315+252+180+168+105+84+72+60 = 2520. Yes!
    return [5, 6, 7, 8, 10, 14, 15, 24, 30, 35, 42]


def construct():
    deadline = time.time() + TIME_LIMIT
    # Start from known 11-term solution
    ds = _known_13()
    if not _valid(ds):
        raise RuntimeError("known solution invalid")
    
    # Try to reduce count
    ds = _try_reduce(ds, deadline)
    
    if not _valid(ds):
        raise RuntimeError("solution became invalid")
    
    return ds
# EVOLVE-BLOCK-END
