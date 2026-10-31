import time
import itertools

_START_TIME = time.time()
_DEADLINE = 320.0  # Seconds, safety margin before 330s kill


def _capset_conflict(a, b):
    """Check if a and b form a line (with any third point).
    Three points form a line in F_3^n iff a_i + b_i + c_i == 0 mod 3 for all i.
    For two points a, b, the third point is c_i = (0 - a_i - b_i) mod 3.
    But actually, two points alone don't form a line. Three points a, b, c form a line
    iff a + b + c ≡ 0 (mod 3) component-wise.
    So a and b conflict if there exists c in the set such that a + b + c ≡ 0.
    But in the greedy algorithm, we only need to check: for a new point v, does it
    form a line with any two existing points a, b? That is, does a + b + v ≡ 0 mod 3?
    Equivalently, is v ≡ -(a+b) mod 3?
    
    Actually, the standard cap set greedy: add v if for all pairs (a,b) in current set,
    a + b + v ≠ 0 mod 3, AND for all a in current set, v ≠ a (trivially true since we
    check distinctness), and a + a + v ≠ 0 mod 3 (i.e., v ≠ -2a mod 3 = a mod 3... wait).
    
    Let me be precise: A cap set is a set with no three distinct points a,b,c such that
    a + b + c = 0 mod 3. In F_3^n, a line is {a, b, c} where a+b+c=0 and a,b,c distinct.
    Note: if a=b, then 2a+c=0 => c = -2a = a (since 2 ≡ -1 mod 3, so -2 ≡ 1), meaning
    c=a, so not three distinct points. So the condition is: no three DISTINCT points sum to 0.
    
    So when adding v to set S, we need: for all pairs a, b in S (a≠b), a + b + v ≠ 0 mod 3.
    Also, v should not be in S already.
    """
    pass


def priority(vector, n):
    """
    Greedy cap set builder.
    
    The evaluator calls priority(vector, n) for each vector in {0,1,2}^n in some order,
    and the vector is added if it keeps the set a cap set. We need to return a priority
    value such that when sorted in descending order, the greedy selection yields a large cap set.
    
    Key insight: The priority function is called once per vector, and the evaluator handles
    the greedy selection. We can maintain state between calls to guide the selection.
    
    Strategy: Use a potential function that estimates how "conflict-free" a vector is
    with respect to already-selected vectors. But we don't know which vectors were selected
    - the evaluator does that. However, we can maintain a set of vectors we've seen and
    track which ones were likely accepted.
    
    Alternative: Pre-compute a good ordering based on structural properties.
    For cap sets in F_3^n, known constructions use algebraic structures.
    
    Let's use a different approach: maintain a running set of vectors we've been asked
    about, and for each new vector, compute its "distance" from already-seen vectors
    in a way that avoids lines.
    
    Actually, re-reading the task: "vectors of {0,1,2}^n are added greedily in descending
    priority when they keep the set a cap set". So the evaluator:
    1. Gets all vectors in {0,1,2}^n
    2. Sorts them by priority descending
    3. Iterates in that order, adding each if it doesn't create a line
    
    So we need to return a priority for each vector such that the resulting greedy
    order produces a large cap set.
    
    Key realization: We can use the fact that we're called for ALL vectors. We can
    maintain state and use a smarter heuristic.
    
    Let me think about what makes a good greedy order for cap sets:
    - Vectors that are "far" from each other (in terms of not forming lines) should
      be prioritized.
    - A vector v forms a line with a and b iff a+b+v ≡ 0 mod 3.
    - So v is "bad" if there exist a,b already chosen such that v ≡ -(a+b) mod 3.
    
    Since we don't know which are chosen, but we can approximate: if we process vectors
    in an order where early vectors are "spread out", later vectors that conflict with
    early ones will be skipped, and we get a larger set.
    
    New strategy: Use a space-filling curve / ordering that maximizes the minimum
    distance between consecutive vectors in the ordering. This way, early vectors in
    the sorted order are far apart, reducing conflicts.
    
    Actually, let's try a completely different approach:
    - Precompute all vectors
    - Use a known cap set construction as a guide
    - Assign high priority to vectors in known large cap sets
    
    For F_3^n, the largest known cap sets have size roughly 3^(n * c) for some c < 1.
    Explicit constructions exist.
    
    Simpler approach: Use the Hamming weight and a structured ordering.
    Vectors with more 1s and 2s (i.e., fewer 0s) might be more "spread out".
    
    Let me try: priority = weighted sum where we prefer vectors that are "orthogonal"
    in some sense to a basis.
    
    Actually, the most promising structural change: use a potential-based approach
    where we maintain a set of "representative" vectors and score new vectors by
    their minimum distance to these representatives.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: simple deterministic score
        return -sum(v * v for v in vector)
    
    # Use a hash-based approach with structural awareness
    # Convert to base-3 integer for canonical representation
    val = 0
    for v in vector:
        val = val * 3 + v
    
    # Structural component: prefer vectors with balanced symbol distribution
    # Count 0s, 1s, 2s
    c0 = c1 = c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
    
    # Variance from uniform distribution (n/3 each)
    # Lower variance = more balanced = potentially better spread
    variance = (c0 - n/3)**2 + (c1 - n/3)**2 + (c2 - n/3)**2
    
    # Hamming weight (number of non-zero coordinates)
    hw = c1 + c2
    
    # Primary: prefer balanced vectors (lower variance)
    # Secondary: prefer higher Hamming weight (more spread)
    # Tertiary: hash for tie-breaking
    h = val % (10**9 + 7)
    h_norm = h / (10**9 + 7)
    
    # Score: higher is better
    # Balanced vectors should be prioritized
    score = -variance + 0.1 * hw + h_norm
    
    return float(score)
