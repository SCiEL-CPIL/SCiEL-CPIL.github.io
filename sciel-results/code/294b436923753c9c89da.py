"""Optimize over geometric-progressions U = {floor(b^k)} to maximize the
sumset/difference-set exponent lower bound."""
import math
import time
import itertools

# EVOLVE-BLOCK-START
def _score_of(U):
    """Return the raw lower-bound score for a concrete set U (must contain 0)."""
    if not U:
        return 0.0
    u = sorted(set(U))
    if 0 not in u:
        u.append(0)
    n = len(u)
    mx = u[-1]
    if mx <= 0:
        return 0.0
    # difference set U - U
    diffs = set()
    for a in u:
        for b in u:
            diffs.add(a - b)
    # sum set U + U
    sums = set()
    for a in u:
        for b in u:
            sums.add(a + b)
    denom = 2.0 * mx + 1.0
    if denom <= 1.0:
        return 0.0
    ratio = len(diffs) / float(len(sums))
    if ratio <= 0.0:
        return 0.0
    return 1.0 + (math.log(ratio) / math.log(denom))

def _build_gp(b, m):
    """Construct U = {floor(b^k) | k=0..m} as a list containing 0."""
    U = [0]
    v = 1.0
    for k in range(1, m + 1):
        v *= b
        U.append(int(v))
    # Ensure uniqueness and sorting
    U = sorted(set(U))
    return U

def construct():
    deadline = time.time() + 100.0
    best_score = 0.0
    best_set = [0, 1, 2, 4, 8]
    
    # Search over geometric progressions with varying base b and length m
    # b can be non-integer
    # m is integer length (number of elements minus 1)
    
    # We'll sweep b in a range and m in a range
    # Based on manual analysis, small m (3-6) and moderate b might be good
    # Let's try b from 1.1 to 10.0 and m from 2 to 10
    
    # To be efficient, we can step b
    b_vals = []
    # Logarithmic sweep for b
    for i in range(1, 100):
        b = 1.0 + i * 0.1
        if b > 20.0: break
        b_vals.append(b)
        
    # Also add some integer bases
    for b in range(2, 20):
        b_vals.append(float(b))
        
    b_vals = sorted(set(b_vals))
    
    m_vals = range(2, 12) # m=2 means size 3 (0, 1, b)
    
    for m in m_vals:
        for b in b_vals:
            if time.time() > deadline:
                break
            U = _build_gp(b, m)
            if len(U) < 2:
                continue
            s = _score_of(U)
            if s > best_score:
                best_score = s
                best_set = U

    return best_set
# EVOLVE-BLOCK-END
