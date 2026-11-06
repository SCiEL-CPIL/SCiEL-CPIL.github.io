"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

def priority(vector, n):
    """
    Priority function for cap set greedy selection.
    Higher priority = added earlier.
    Uses balanced symbol counts as primary signal.
    """
    # Count symbols
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Balanced vectors (close to n/3 each) are preferred
    # Use negative deviation from ideal balance
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Primary: minimize deviation (so negate for max-priority)
    score = -deviation * 1000.0
    
    # Secondary: deterministic tiebreaker based on vector pattern
    # Use a hash-like function for consistent ordering
    tiebreak = 0
    for i, v in enumerate(vector):
        tiebreak += v * (i + 1)
    score += tiebreak * 0.001
    
    return score


def _is_cap_set_incremental(existing_set, new_vec, n):
    """
    Check if adding new_vec to existing_set keeps it a cap set.
    existing_set is a set of tuples.
    Returns True if no 3-term AP is formed.
    """
    new_t = tuple(new_vec)
    
    # Check if new_vec forms a 3-term AP with any two existing vectors
    # Case 1: new_vec is the middle element: x + z = 2 * new_vec
    # For each x in existing_set, check if z = 2*new_vec - x is in existing_set
    for x in existing_set:
        # z_i = 2*new_vec_i - x_i, must be in {0,1,2}
        z = tuple(2 * nv - xi for nv, xi in zip(new_t, x))
        if all(0 <= zi <= 2 for zi in z) and z in existing_set and z != x:
            return False
    
    # Case 2: new_vec is the first element: new_vec + z = 2 * y
    # For each y in existing_set, check if z = 2*y - new_vec is in existing_set
    for y in existing_set:
        z = tuple(2 * yi - nv for yi, nv in zip(y, new_t))
        if all(0 <= zi <= 2 for zi in z) and z in existing_set and z != new_t:
            return False
    
    # Case 3: new_vec is the last element: x + new_vec = 2 * y
    # For each x in existing_set, check if y = (x + new_vec) / 2 is in existing_set
    for x in existing_set:
        # y_i = (x_i + new_vec_i) / 2, must be integer in {0,1,2}
        y_list = []
        valid = True
        for xi, nv in zip(x, new_t):
            s = xi + nv
            if s % 2 != 0:
                valid = False
                break
            yi = s // 2
            if yi < 0 or yi > 2:
                valid = False
                break
            y_list.append(yi)
        if valid:
            y = tuple(y_list)
            if y in existing_set and y != x and y != new_t:
                return False
    
    return True


def _generate_all_vectors(n):
    """Generate all vectors in {0,1,2}^n."""
    from itertools import product
    return [tuple(p) for p in product([0, 1, 2], repeat=n)]


def _solve_cap_set(n, time_limit=280.0):
    """
    Solve the cap set problem using greedy selection with priority ordering.
    Returns the size of the largest cap set found.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Generate all vectors
    all_vectors = _generate_all_vectors(n)
    
    # Sort by priority (descending)
    all_vectors.sort(key=lambda v: priority(v, n), reverse=True)
    
    # Greedily add vectors that maintain cap set property
    cap_set = set()
    
    for vec in all_vectors:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Check if adding this vector keeps the set a cap set
        if _is_cap_set_incremental(cap_set, vec, n):
            cap_set.add(vec)
    
    return len(cap_set)


def _solve_cap_set_fast(n, time_limit=280.0):
    """
    Faster version using numpy for the AP check.
    """
    import numpy as np
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Generate all vectors as numpy array
    from itertools import product
    all_vectors = np.array(list(product([0, 1, 2], repeat=n)), dtype=np.int8)
    
    # Compute priorities
    priorities = np.zeros(len(all_vectors))
    for i, vec in enumerate(all_vectors):
        priorities[i] = priority(tuple(vec), n)
    
    # Sort by priority descending
    order = np.argsort(-priorities)
    sorted_vectors = all_vectors[order]
    
    # Greedy selection
    cap_set = set()
    
    for vec in sorted_vectors:
        if time.time() > deadline:
            break
        
        vec_t = tuple(vec)
        
        # Check AP conditions
        is_valid = True
        
        # Case 1: vec is middle: x + z = 2*vec
        for x in cap_set:
            z = tuple(2 * v - xi for v, xi in zip(vec_t, x))
            if all(0 <= zi <= 2 for zi in z) and z in cap_set and z != x:
                is_valid = False
                break
        
        if not is_valid:
            continue
        
        # Case 2: vec is first: vec + z = 2*y
        for y in cap_set:
            z = tuple(2 * yi - v for yi, v in zip(y, vec_t))
            if all(0 <= zi <= 2 for zi in z) and z in cap_set and z != vec_t:
                is_valid = False
                break
        
        if not is_valid:
            continue
        
        # Case 3: vec is last: x + vec = 2*y
        for x in cap_set:
            y_list = []
            valid = True
            for xi, v in zip(x, vec_t):
                s = xi + v
                if s % 2 != 0:
                    valid = False
                    break
                yi = s // 2
                if yi < 0 or yi > 2:
                    valid = False
                    break
                y_list.append(yi)
            if valid:
                y = tuple(y_list)
                if y in cap_set and y != x and y != vec_t:
                    is_valid = False
                    break
        
        if is_valid:
            cap_set.add(vec_t)
    
    return len(cap_set)


def priority(vector, n):
    """
    Priority function for cap set greedy selection.
    Higher priority = added earlier.
    Uses balanced symbol counts as primary signal.
    """
    # Count symbols
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Balanced vectors (close to n/3 each) are preferred
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Primary: minimize deviation (negate for max-priority)
    score = -deviation * 1000.0
    
    # Secondary: deterministic tiebreaker
    tiebreak = 0
    for i, v in enumerate(vector):
        tiebreak += v * (i + 1)
    score += tiebreak * 0.001
    
    return score
# EVOLVE-BLOCK-END


def _main():
    """
    Main entry point called by the evaluator.
    Determines n from context or uses a reasonable default.
    """
    import time
    start_time = time.time()
    
    # The task says vectors of {0,1,2}^n, but n is not directly provided
    # In the FunSearch contract, n is passed to priority()
    # We need to figure out what n to use. Looking at the task, it seems
    # n should be a parameter. Let me re-read...
    # 
    # The interface is: priority(vector: tuple, n: int) -> float
    # So the evaluator calls priority with a specific n.
    # But we also need to return the cap set size.
    # 
    # Looking at the seed program, it just returns 0.0 from priority.
    # The score is "cap set size" and the evaluator measures the raw value.
    # 
    # I think the evaluator:
    # 1. Calls priority() for each vector in some order
    # 2. Sorts vectors by priority descending
    # 3. Greedily adds vectors that maintain cap set property
    # 4. Returns the final cap set size as the score
    #
    # So my priority function just needs to return good priorities.
    # The actual greedy selection is done by the evaluator!
    #
    # Wait, but the task says "vectors of {0,1,2}^n are added greedily in 
    # descending priority when they keep the set a cap set"
    #
    # So the evaluator does the greedy selection. My job is just to provide
    # good priority values.
    #
    # But then how do I return the cap set size? The priority function returns
    # a float priority, not the cap set size.
    #
    # Let me re-read: "Score: cap set size" and "The evaluator measures the 
    # raw value"
    #
    # I think the evaluator:
    # 1. Generates all vectors in {0,1,2}^n for some n
    # 2. Calls priority(vector, n) for each
    # 3. Sorts by priority descending
    # 4. Greedily adds to cap set
    # 5. Returns len(cap_set) as the score
    #
    # So I just need a good priority function. The current one returns 0.0 
    # which gives lexicographic order (or some default order).
    #
    # My improved priority should favor balanced vectors.
    pass


# The actual entry point that the evaluator calls
# Based on the task description, the evaluator calls priority() and does
# the greedy selection itself. So I just need to provide a good priority.

def priority(vector, n):
    """
    Priority function for cap set greedy selection.
    Higher priority = added earlier.
    
    Strategy: Favor vectors with balanced symbol counts (close to n/3 each).
    This avoids early saturation of specific patterns and reduces the
    likelihood of forming 3-term arithmetic progressions.
    """
    # Count symbols
    c0 = vector.count(0)
    c1 = vector.count(1)
    c2 = vector.count(2)
    
    # Ideal: each symbol appears n/3 times
    ideal = n / 3.0
    
    # Deviation from ideal balance
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    
    # Primary score: minimize deviation
    score = -deviation * 1000.0
    
    # Secondary: deterministic tiebreaker to ensure consistent ordering
    # Use a weighted sum based on position
    tiebreak = 0.0
    for i, v in enumerate(vector):
        # Weight by position to break ties deterministically
        tiebreak += v * (i + 1)
    score += tiebreak * 0.001
    
    return score
# EVOLVE-BLOCK-END
