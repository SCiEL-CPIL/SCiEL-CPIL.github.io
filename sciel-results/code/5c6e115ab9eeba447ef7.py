"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import hashlib
import time

# Global state to track which vectors have been added
_added_vectors = []
_start_time = None

def _init():
    global _added_vectors, _start_time
    _added_vectors = []
    _start_time = time.time()

def _forms_line_with_new(vector, existing):
    """Check if adding vector to existing set would create an affine line."""
    v = vector
    for e in existing:
        # For each direction d in {0,1,2}^n, check if v, e, and a third point form a line
        # An affine line in F_3^n: a, a+d, a+2d
        # Given two points a and b, the third point is 2a - b (mod 3) if b-a is the direction
        # Or: if v and e are on a line, then for some position, the third point must also be in the set
        # Direction from e to v: d = (v_i - e_i) mod 3
        # If d_i == 0 for all i, they're the same point (won't happen)
        # Third point: e - d = 2e - v (mod 3)
        # Or: v + d = 2v - e (mod 3)
        # Check if either third point is in the existing set
        third1 = tuple((2*e[i] - v[i]) % 3 for i in range(len(v)))
        third2 = tuple((2*v[i] - e[i]) % 3 for i in range(len(v)))
        if third1 in existing or third2 in existing:
            return True
    return False

def _compute_conflict_score(vector, existing_set, n):
    """Compute how many potential lines the vector would form with existing vectors."""
    conflict_count = 0
    v = vector
    for e in existing_set:
        # Compute the two possible third points
        third1 = tuple((2*e[i] - v[i]) % 3 for i in range(n))
        third2 = tuple((2*v[i] - e[i]) % 3 for i in range(n))
        if third1 in existing_set:
            conflict_count += 1
        if third2 in existing_set:
            conflict_count += 1
    return conflict_count

def _accept_vector(vector, n):
    """Add vector to the set if it doesn't create a line."""
    existing_set = set(_added_vectors)
    if not _forms_line_with_new(vector, existing_set):
        _added_vectors.append(vector)
        existing_set.add(vector)
        return True
    return False

def priority(vector, n):
    """
    Priority function for cap set greedy insertion.
    
    Uses a combination of:
    1. A deterministic hash-based randomization to break lexicographic clustering
    2. A geometric conflict score estimating line formation risk
    3. A structural bias preferring balanced vectors
    
    Higher priority = inserted earlier.
    """
    global _added_vectors, _start_time
    
    # Initialize on first call
    if _start_time is None:
        _init()
    
    # Check time limit - if we're close to the deadline, just use hash
    elapsed = time.time() - _start_time
    if elapsed > 280:
        vec_str = ''.join(str(v) for v in vector)
        h = int(hashlib.md5(vec_str.encode()).hexdigest(), 16)
        return (h % 1000000) / 1000000.0
    
    # Convert vector to a string for hashing
    vec_str = ''.join(str(v) for v in vector)
    # Use MD5 hash for deterministic pseudo-random priority
    h = int(hashlib.md5(vec_str.encode()).hexdigest(), 16)
    random_part = (h % 1000000) / 1000000.0  # [0, 1)
    
    # Compute geometric conflict score
    existing_set = set(_added_vectors)
    conflict = _compute_conflict_score(vector, existing_set, n)
    
    # Normalize conflict score (cap at a reasonable max)
    conflict_normalized = min(conflict / 10.0, 1.0)
    
    # Structural bias: prefer balanced vectors (closer to n/3 of each symbol)
    n0 = sum(1 for v in vector if v == 0)
    n1 = sum(1 for v in vector if v == 1)
    n2 = sum(1 for v in vector if v == 2)
    target = n / 3.0
    deviation = (n0 - target)**2 + (n1 - target)**2 + (n2 - target)**2
    structural = -deviation / (n * n)  # Normalize to small range
    
    # Combine: random part dominates, conflict and structural are biases
    # Higher conflict means lower priority
    return random_part - 0.5 * conflict_normalized + 0.1 * structural
# EVOLVE-BLOCK-END
