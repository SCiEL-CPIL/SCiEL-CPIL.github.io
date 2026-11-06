"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import itertools

# EVOLVE-BLOCK-START

def _is_cap_set(triples):
    """
    Check if a set of vectors in {0,1,2}^n is a cap set.
    A cap set has no three distinct elements a, b, c such that a+b+c = 0 mod 3.
    Equivalently, no three distinct elements form an affine line: a, a+d, a+2d.
    """
    s = set(triples)
    for a in s:
        for b in s:
            if a == b:
                continue
            # c = -a - b mod 3
            c = tuple((-a[i] - b[i]) % 3 for i in range(len(a)))
            if c in s and c != a and c != b:
                return False
    return True

def _check_cap_set_incremental(existing_set, new_vec):
    """
    Check if adding new_vec to existing_set (a set of tuples) keeps it a cap set.
    We only need to check lines involving new_vec.
    For any a in existing_set, b = -a - new_vec mod 3 must not be in existing_set 
    (and distinct from a and new_vec).
    Also for any a, b in existing_set, c = -a-b mod 3 must not equal new_vec.
    But actually, it's simpler: a line is {x, x+d, x+2d}. 
    If we add v, we need to ensure no line contains v and two other elements from existing_set.
    A line containing v has the form {v, v+d, v+2d} for some direction d.
    So for each direction d (non-zero vector in {0,1,2}^n, but we can just check the two other points),
    the other two points are v+d and v+2d.
    We need to ensure that we don't have both v+d and v+2d in existing_set.
    Also, we need to ensure that for any a in existing_set, the third point on the line through a and v is not in existing_set.
    The third point c satisfies a + v + c = 0 mod 3 => c = -a - v mod 3.
    So for each a in existing_set, let c = (-a[0]-v[0], -a[1]-v[1], ...).
    If c is in existing_set and c != a and c != v, then we have a line {a, v, c}.
    """
    s = existing_set
    for a in s:
        c = tuple((-a[i] - new_vec[i]) % 3 for i in range(len(a)))
        if c in s and c != a and c != new_vec:
            return False
    return True

def priority(vector, n):
    """
    Greedy cap set construction.
    We need to return a priority such that when vectors are sorted in descending order,
    the greedy algorithm produces a large cap set.
    
    Since the evaluator calls priority(vector, n) for each vector and sorts them,
    we can use a heuristic that favors vectors that are "independent" in some sense.
    
    However, the key insight is that the greedy algorithm's outcome depends on the order.
    We want to find an order that yields a large cap set.
    
    A common heuristic for cap sets is to use a weight function based on the coordinates.
    For example, vectors with more 0s might be "simpler" or vectors with balanced coordinates.
    
    Let's try a heuristic: prioritize vectors that have a high "distance" from the origin in some sense,
    or vectors that are less likely to form lines with previously added vectors.
    
    Since we don't know the previously added vectors at the time of priority assignment,
    we need a static heuristic.
    
    One idea: use a random seed based priority, but that's not deterministic.
    Another idea: use a hash of the vector to create a pseudo-random priority.
    
    Actually, the task says "vectors of {0,1,2}^n are added greedily in descending priority".
    So the evaluator will:
    1. Generate all vectors in {0,1,2}^n.
    2. Sort them by priority(vector, n) in descending order.
    3. Iterate through the sorted list, adding a vector to the set if it keeps the set a cap set.
    
    So we need to assign priorities to all 3^n vectors such that the greedy algorithm yields a large cap set.
    
    A known result is that the largest cap set in {0,1,2}^n grows like c^n where c < 3.
    For small n, we can compute the maximum cap set size.
    
    Let's try a heuristic based on the sum of coordinates or the number of distinct coordinates.
    Vectors with all coordinates equal (000...0, 111...1, 222...2) are likely to be in many lines, so maybe deprioritize them.
    Vectors with a mix of coordinates might be better.
    
    Let's try: priority = -sum of squares of coordinates. This would favor vectors with more 0s.
    Or priority = sum of coordinates. This would favor vectors with more 2s.
    
    Actually, let's think about it differently. The greedy algorithm's performance depends on the order.
    A good order might be one that adds vectors in a way that they are "spread out" in the space.
    
    Let's use a heuristic based on a linear function of the coordinates with random-looking coefficients.
    For example, priority = sum(vector[i] * weight[i] for i in range(n)) where weight is a fixed sequence.
    
    Let's use the golden ratio conjugate or some irrational number to create a "random" ordering.
    weight[i] = (i * 0.6180339887498949) % 1.0
    
    Or simply use a hash of the vector.
    """
    # Use a deterministic hash-based priority
    # Convert vector to a base-3 number and use its hash
    val = 0
    for i, v in enumerate(vector):
        val = val * 3 + v
    
    # Use a simple hash function
    h = val
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    
    return float(h)

# EVOLVE-BLOCK-END
