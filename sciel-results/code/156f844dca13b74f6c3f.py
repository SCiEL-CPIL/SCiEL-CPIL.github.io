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

def _build(b, m):
    """Construct U = {floor(b^k) | k=0..m} as a list containing 0."""
    U = [0]
    v = 1.0
    for k in range(1, m + 1):
        v *= b
        U.append(int(v))
    return U

def construct():
    deadline = time.time() + 100.0
    best_score = 0.0
    best_set = [0, 1, 2, 4, 8]
    
    # Search small sets
    # We know {0, 1, 2, 4, 8} gives ~1.20
    # Let's verify and search neighbors
    candidates = []
    
    # Add the known good set
    candidates.append([0, 1, 2, 4, 8])
    
    # Generate small sets
    # Size 3 to 6, max element up to 20
    for n in range(3, 7):
        for comb in itertools.combinations(range(1, 21), n-1):
            U = [0] + list(comb)
            if U[-1] > 20:
                continue
            candidates.append(U)
            
    # Also check some geometric progressions
    for b in [1.5, 2.0, 2.5, 3.0]:
        for m in range(3, 8):
            U = _build(b, m)
            if U[-1] > 50:
                break
            candidates.append(U)

    # Evaluate all candidates
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score_of(U)
        if s > best_score:
            best_score = s
            best_set = U

    return best_set
# EVOLVE-BLOCK-END
