"""Cap set greedy construction with a strong priority heuristic."""
import time
import math
import random
from itertools import product

# We will use a deterministic but non-trivial priority.
# The idea: prioritize vectors that are "spread out" in the 3^n space.
# A good heuristic for cap sets is to avoid arithmetic progressions.
# In F_3^n, a 3-term AP is x, y, z with x + z = 2y (mod 3).
# Equivalently, x - y = y - z.
# A simple heuristic: prioritize vectors with a balanced number of 0s, 1s, 2s?
# Or maybe prioritize vectors that are "far" from existing ones?
# But priority() is called independently for each vector, without knowledge of the set.
# The signature is priority(vector: tuple, n: int) -> float.
# The greedy algorithm adds vectors in descending order of priority.
# So we need a static priority function that, when sorted, yields a large cap set.

# Known results:
# For n=1, max cap set is 2.
# For n=2, max cap set is 4.
# For n=3, max cap set is 9.
# For n=4, max cap set is 20.
# For n=5, max cap set is 45.
# For n=6, max cap set is 112.
# For n=7, max cap set is 280.
# For n=8, max cap set is 693.

# The seed program returns 0.0, which means lexicographic order (or arbitrary).
# Lexicographic order for n=3:
# (0,0,0), (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,1,2), (0,2,0), (0,2,1), (0,2,2), ...
# Greedy cap set construction with lex order:
# 1. (0,0,0) - ok
# 2. (0,0,1) - ok
# 3. (0,0,2) - (0,0,0), (0,0,1), (0,0,2) is an AP. Skip.
# 4. (0,1,0) - ok
# 5. (0,1,1) - (0,0,1), (0,1,1), (0,2,1) is an AP? No, (0,2,1) not in set.
#    Check: (0,0,0), (0,1,1) -> mid = (0, 0.5, 0.5) no.
#    Check: (0,0,1), (0,1,1) -> mid = (0, 0.5, 1) no.
#    Check: (0,1,0), (0,1,1) -> mid = (0, 1, 0.5) no.
#    So (0,1,1) is ok.
# 6. (0,1,2) - (0,0,1), (0,1,2) -> mid = (0, 0.5, 1.5) no.
#    (0,1,1), (0,1,2) -> mid = (0, 1, 1.5) no.
#    (0,0,0), (0,1,2) -> mid = (0, 0.5, 1) no.
#    (0,1,0), (0,1,2) -> mid = (0, 1, 1) -> (0,1,1) is in set! So (0,0,0), (0,1,1), (0,2,2) is an AP?
#    Wait, the AP is x, y, z with x+z=2y.
#    If we have x=(0,1,0), z=(0,1,2), then 2y = (0,2,2) -> y=(0,1,1).
#    (0,1,1) is in the set. So (0,1,0), (0,1,1), (0,1,2) is an AP.
#    So (0,1,2) would create an AP with (0,1,0) and (0,1,1). Skip.
# 7. (0,2,0) - (0,0,0), (0,1,0), (0,2,0) is an AP. Skip.
# 8. (0,2,1) - (0,0,1), (0,1,1), (0,2,1) is an AP. Skip.
# 9. (0,2,2) - (0,0,0), (0,1,1), (0,2,2) is an AP. Skip.
# 10. (1,0,0) - ok
# ...
# This seems to give a small set.

# A better heuristic: prioritize vectors with a "balanced" distribution of digits.
# Or, prioritize vectors that are "linearly independent" over F_3?
# Actually, the best known constructions use algebraic structures.
# But we need a simple priority function.

# Let's try: priority = sum of digits * 10 + (number of distinct digits) * 5 + ...
# Or maybe: priority = -sum of squares of digits?
# Let's look at the structure of cap sets.
# A cap set in F_3^n is a set with no 3-term AP.
# The "ellipsoid" or "paraboloid" constructions are common.

# Since priority() is static, we can use a precomputed heuristic.
# One idea: use the Hamming weight (number of non-zero digits) as a secondary key.
# Another idea: use the "sum" of the vector (mod 3) to break ties.

# Let's try a simple heuristic: prioritize vectors with a balanced count of 0s, 1s, 2s.
# For a vector v, let c0, c1, c2 be the counts of 0, 1, 2.
# We want c0, c1, c2 to be as close to n/3 as possible.
# So, penalty = (c0 - n/3)^2 + (c1 - n/3)^2 + (c2 - n/3)^2.
# Lower penalty -> higher priority.

def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    return -penalty

# Let's test this heuristic mentally for n=3.
# Target = 1.
# (0,0,0): c0=3, c1=0, c2=0. penalty = (3-1)^2 + (0-1)^2 + (0-1)^2 = 4+1+1=6.
# (0,0,1): c0=2, c1=1, c2=0. penalty = (2-1)^2 + (1-1)^2 + (0-1)^2 = 1+0+1=2.
# (0,0,2): c0=2, c1=0, c2=1. penalty = 1+1+0=2.
# (0,1,0): c0=2, c1=1, c2=0. penalty = 2.
# (0,1,1): c0=1, c1=2, c2=0. penalty = 0+1+1=2.
# (0,1,2): c0=1, c1=1, c2=1. penalty = 0+0+0=0.
# (0,2,0): c0=2, c1=0, c2=1. penalty = 2.
# (0,2,1): c0=1, c1=1, c2=1. penalty = 0.
# (0,2,2): c0=1, c1=0, c2=2. penalty = 1+1+0=2.
# (1,0,0): c0=2, c1=1, c2=0. penalty = 2.
# (1,0,1): c0=1, c1=2, c2=0. penalty = 2.
# (1,0,2): c0=1, c1=1, c2=1. penalty = 0.
# (1,1,0): c0=1, c1=2, c2=0. penalty = 2.
# (1,1,1): c0=0, c1=3, c2=0. penalty = 1+4+1=6.
# (1,1,2): c0=0, c1=2, c2=1. penalty = 1+1+0=2.
# (1,2,0): c0=1, c1=1, c2=1. penalty = 0.
# (1,2,1): c0=0, c1=2, c2=1. penalty = 2.
# (1,2,2): c0=0, c1=1, c2=2. penalty = 1+0+1=2.
# (2,0,0): c0=2, c1=0, c2=1. penalty = 2.
# (2,0,1): c0=1, c1=0, c2=2. penalty = 2.
# (2,0,2): c0=1, c1=0, c2=2. penalty = 2.
# (2,1,0): c0=1, c1=1, c2=1. penalty = 0.
# (2,1,1): c0=0, c1=2, c2=1. penalty = 2.
# (2,1,2): c0=0, c1=1, c2=2. penalty = 2.
# (2,2,0): c0=1, c1=0, c2=2. penalty = 2.
# (2,2,1): c0=0, c1=1, c2=2. penalty = 2.
# (2,2,2): c0=0, c1=0, c2=3. penalty = 1+1+4=6.

# Vectors with penalty 0: (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0).
# These are the 6 permutations of (0,1,2).
# Do they form a cap set?
# (0,1,2), (0,2,1), (1,0,2) -> mid = (0.5, 1, 1.5) no.
# (0,1,2), (1,0,2) -> mid = (0.5, 0.5, 2) no.
# (0,1,2), (2,0,1) -> mid = (1, 0.5, 1.5) no.
# (0,1,2), (2,1,0) -> mid = (1, 1, 1) -> (1,1,1) not in set.
# (0,2,1), (1,0,2) -> mid = (0.5, 1, 1.5) no.
# (0,2,1), (2,1,0) -> mid = (1, 1.5, 0.5) no.
# (0,2,1), (1,2,0) -> mid = (0.5, 2, 0.5) no.
# (1,0,2), (1,2,0) -> mid = (1, 1, 1) no.
# (1,0,2), (2,1,0) -> mid = (1.5, 0.5, 1) no.
# (1,2,0), (2,1,0) -> mid = (1.5, 1.5, 0) no.
# (2,0,1), (2,1,0) -> mid = (2, 0.5, 0.5) no.
# So the 6 permutations of (0,1,2) form a cap set.
# Can we add more?
# Next priority: penalty 2.
# (0,0,1), (0,0,2), (0,1,0), (0,1,1), (0,2,0), (0,2,2), (1,0,0), (1,0,1), (1,1,0), (1,1,2), (1,2,1), (1,2,2), (2,0,0), (2,0,2), (2,1,1), (2,1,2), (2,2,0), (2,2,1).
# Let's try to add (0,0,1).
# Check with (0,1,2): (0,0,1), (0,1,2) -> mid = (0, 0.5, 1.5) no.
# Check with (0,2,1): (0,0,1), (0,2,1) -> mid = (0, 1, 1) -> (0,1,1) not in set.
# Check with (1,0,2): (0,0,1), (1,0,2) -> mid = (0.5, 0, 1.5) no.
# Check with (1,2,0): (0,0,1), (1,2,0) -> mid = (0.5, 1, 0.5) no.
# Check with (2,0,1): (0,0,1), (2,0,1) -> mid = (1, 0, 1) -> (1,0,1) not in set.
# Check with (2,1,0): (0,0,1), (2,1,0) -> mid = (1, 0.5, 0.5) no.
# So (0,0,1) can be added.
# This suggests the heuristic might work.

# Let's refine the priority to break ties better.
# We can use the sum of the vector (mod 3) as a tie-breaker.
# Or, we can use the "lexicographic" order as a final tie-breaker.

def priority(vector, n):
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    target = n / 3.0
    penalty = (c0 - target) ** 2 + (c1 - target) ** 2 + (c2 - target) ** 2
    # Tie-breaker: sum of digits
    s = sum(vector)
    # We want higher priority for lower penalty, then lower sum?
    # Let's return a tuple-like float: -penalty * 1000 + s
    # But priority must return a float.
    return -penalty * 1000.0 + s

# Let's test this for n=3.
# (0,1,2): penalty=0, s=3. priority = 3.
# (0,2,1): penalty=0, s=3. priority = 3.
# (1,0,2): penalty=0, s=3. priority = 3.
# (1,2,0): penalty=0, s=3. priority = 3.
# (2,0,1): penalty=0, s=3. priority = 3.
# (2,1,0): penalty=0, s=3. priority = 3.
# (0,0,1): penalty=2, s=1. priority = -2000 + 1 = -1999.
# (0,0,2): penalty=2, s=2. priority = -2000 + 2 = -1998.
# (0,1,0): penalty=2, s=1. priority = -1999.
# (0,1,1): penalty=2, s=2. priority = -1998.
# (0,2,0): penalty=2, s=2. priority = -1998.
# (0,2,2): penalty=2, s=4. priority = -1996.
# (1,0,0): penalty=2, s=1. priority = -1999.
# (1,0,1): penalty=2, s=2. priority = -1998.
# (1,1,0): penalty=2, s=2. priority = -1998.
# (1,1,2): penalty=2, s=4. priority = -1996.
# (1,2,1): penalty=2, s=4. priority = -1996.
# (1,2,2): penalty=2, s=5. priority = -1995.
# (2,0,0): penalty=2, s=2. priority = -1998.
# (2,0,2): penalty=2, s=4. priority = -1996.
# (2,1,1): penalty=2, s=4. priority = -1996.
# (2,1,2): penalty=2, s=5. priority = -1995.
# (2,2,0): penalty=2, s=4. priority = -1996.
# (2,2,1): penalty=2, s=5. priority = -1995.
# (0,0,0): penalty=6, s=0. priority = -6000.
# (1,1,1): penalty=6, s=3. priority = -5997.
# (2,2,2): penalty=6, s=6. priority = -5994.

# The 6 permutations of (0,1,2) have priority 3.
# Then (0,0,2), (0,1,1), (0,2,0), (0,2,2), (1,0,1), (1,1,0), (1,1,2), (1,2,1), (1,2,2), (2,0,0), (2,0,2), (2,1,1), (2,1,2), (2,2,0), (2,2,1) have priority -1998 or higher.
# This seems reasonable.

# Let's implement this.
