"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import product

# EVOLVE-BLOCK-START
# Cap set in {0,1,2}^n: no three distinct elements form an affine line.
# An affine line is {a, b, c} where b = (a+c)/2 coordinate-wise, i.e.,
# for each coordinate, a_i + c_i = 2*b_i. Equivalently, a, b, c are distinct
# and a + c = 2b.

# The greedy algorithm: sort all vectors by priority (descending), then add
# each vector if it doesn't create a cap set violation with already-selected vectors.

# A violation occurs when there exist a, b in the current set such that adding v
# creates a line. Specifically, for any two existing vectors u, w, we need to check
# if {u, w, v} or {u, v, w} forms an affine line.

# Actually, the standard check: a set S is a cap set if for any distinct a, b in S,
# the third point c on the line through a and b (i.e., c = 2b - a or c = 2a - b)
# is not in S. When adding v to S, we need to check: for each u in S, does the
# line through u and v contain another point w in S? Or equivalently, for each pair
# (u, w) in S, is v on the line through u and w?

# More efficiently: when adding v, for each u in S, compute the "midpoint" m = (u+v)/2
# (which must have all coordinates in {0,1,2}, i.e., u_i + v_i must be even for all i,
# and m_i = (u_i+v_i)//2 must be in {0,1,2}). If m is in S and m != u and m != v,
# then {u, m, v} is a line. Also check: is there w in S such that w = 2u - v or w = 2v - u?

# Actually the cleanest check: v creates a violation iff there exist u, w in S (u != w)
# such that u + w = 2v (v is midpoint) OR u + v = 2w (w is midpoint of u,v) OR v + w = 2u (u is midpoint of v,w).

# Let me think again. Three points a, b, c form an affine line if a + c = 2b (b is midpoint).
# So when adding v to S, a violation exists iff:
# 1. There exist u, w in S with u != w such that u + w = 2v (v is the midpoint of u and w)
# 2. There exist u in S such that u + v = 2w for some w in S, w != u, w != v (w is midpoint of u, v)
# 3. There exist w in S such that v + w = 2u for some u in S, u != v, u != w (u is midpoint of v, w)

# Cases 2 and 3 are symmetric. Let me combine: for each u in S, check if midpoint(u, v) is in S (and != u, != v).
# For case 1: for each pair (u, w) in S, check if v is midpoint. But that's O(|S|^2) per insertion.

# Better approach: precompute for each pair in S whether their midpoint is defined and track it.
# Or: maintain a set S. When adding v:
#   - For each u in S: compute mid = (u+v)/2. If all coords of u+v are even, mid is valid. If mid in S and mid != u and mid != v: violation.
#   - Also: is v the midpoint of any pair in S? That means for some u, w in S: u + w = 2v. 
#     Equivalently, for each u in S: w = 2v - u. Check if w in S and w != u and w != v.

# So the check per insertion is O(|S|) if we use a set for membership.

def _can_add(v, S):
    """Check if v can be added to cap set S without creating an affine line."""
    n = len(v)
    v_tuple = tuple(v)
    
    # Check 1: v is not already in S
    if v_tuple in S:
        return False
    
    # Check 2: for each u in S, midpoint(u, v) should not be in S (and distinct from u, v)
    for u in S:
        # Compute u + v
        s = tuple(u[i] + v[i] for i in range(n))
        # Check if all even
        all_even = all(s[i] % 2 == 0 for i in range(n))
        if all_even:
            mid = tuple(s[i] // 2 for i in range(n))
            if mid != u and mid != v_tuple and mid in S:
                return False
    
    # Check 3: v is midpoint of some pair (u, w) in S: u + w = 2v
    # For each u in S: w = 2v - u
    two_v = tuple(2 * v[i] for i in range(n))
    for u in S:
        w = tuple(two_v[i] - u[i] for i in range(n))
        # w must be in {0,1,2}^n
        if all(0 <= w[i] <= 2 for i in range(n)):
            w_tuple = tuple(w)
            if w_tuple != u and w_tuple != v_tuple and w_tuple in S:
                return False
    
    return True


def _priority_default(vector, n):
    """Default priority: based on number of 2s (or some heuristic)."""
    # Try to maximize diversity: prefer vectors with more 2s? Or fewer?
    # Actually, let's use a random-ish but deterministic priority based on the vector itself.
    # A good heuristic: prefer vectors that are "far" from existing ones in some sense.
    # For a static priority (computed before greedy), we can use:
    # - Sum of coordinates (prefer medium sums)
    # - Or: number of distinct values
    # Let's try: prefer vectors with sum close to n (balanced)
    s = sum(vector)
    return -abs(s - n)


def priority(vector, n):
    """
    Priority function for greedy cap set construction.
    
    vector: tuple of n elements, each in {0, 1, 2}
    n: dimension
    
    Returns: float priority (higher = added earlier in greedy order)
    """
    # Deadline guard
    deadline = time.time() + 280  # Leave 20s margin for 300s limit
    
    # For small n, we can do an exact/optimized search
    # For larger n, use heuristic priority
    
    # Strategy: Use a priority that encourages algebraic independence.
    # The directive suggests: maximize magnitude of projection onto orthogonal complement
    # of span of low-degree monomial basis vectors.
    
    # Let's implement a simpler but effective heuristic:
    # 1. Convert vector to a feature vector using monomials x_i, x_i*x_j
    # 2. Use a hash-like function based on these features to create diverse priorities
    
    # Actually, for the greedy cap set problem, a well-known approach is:
    # - Use a random or structured ordering that avoids creating lines early
    
    # Let me think about what makes a good cap set ordering:
    # - Vectors that are "far apart" in the {0,1,2}^n space should be prioritized
    # - Avoid vectors that are midpoints of other vectors
    
    # Simple effective heuristic: use the sum of squared coordinates or a weighted sum
    # with pseudo-random weights to break ties
    
    # Let's use: priority = sum of (vector[i] * weight[i]) where weights are derived from i
    # This creates a deterministic but non-trivial ordering
    
    # Better: use the number of 2s as primary, then lexicographic
    num_2s = sum(1 for x in vector if x == 2)
    num_0s = sum(1 for x in vector if x == 0)
    
    # Prefer balanced vectors (mix of 0s, 1s, 2s)
    # A vector with all same values is bad (creates lines easily)
    variety = min(num_0s, num_2s)  # more variety = better
    
    # Use a hash-like function for tie-breaking
    # Compute a pseudo-random value from the vector
    h = 0
    for i, x in enumerate(vector):
        h = (h * 31 + x * (i + 1)) % (2**31)
    
    # Combine: primary key is variety, secondary is hash
    # Scale to make variety dominant
    return variety * 1000000 + (h / (2**31)) * 1000


# EVOLVE-BLOCK-END
