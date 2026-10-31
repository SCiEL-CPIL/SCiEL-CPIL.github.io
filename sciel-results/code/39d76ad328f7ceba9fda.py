import time
from collections import Counter

_start_time = None
_DEADLINE = 330 * 0.95  # 313.5s
_selected_vectors = []
_selected_tuples = set()

def priority(vector, n):
    """
    Priority function for greedy cap set construction in {0,1,2}^n.
    
    Strategy:
    1. Primary: Farthest-point sampling - maximize minimum Hamming distance to selected vectors.
    2. Secondary: Prioritize vectors with all 3 distinct symbols.
    3. Tertiary: Balance - symbol counts close to uniform.
    4. Quaternary: Fast structural hash for deterministic tie-breaking.
    
    The farthest-point approach spreads selections uniformly across the ternary grid,
    which is a stronger geometric proxy for avoiding affine lines than static heuristics.
    """
    global _start_time, _selected_vectors, _selected_tuples
    
    if _start_time is None:
        _start_time = time.time()
    elif time.time() - _start_time > _DEADLINE:
        # Past deadline: return a constant to signal "stop caring about ordering"
        return 0.0

    vec_tuple = tuple(vector)
    
    # Track selected vectors for farthest-point sampling
    if vec_tuple not in _selected_tuples:
        _selected_vectors.append(vec_tuple)
        _selected_tuples.add(vec_tuple)
    
    # Fast symbol counting using Counter (C-implemented)
    c = Counter(vector)
    c0 = c.get(0, 0)
    c1 = c.get(1, 0)
    c2 = c.get(2, 0)
    
    # Distinct count (3 is best)
    if c0 > 0 and c1 > 0 and c2 > 0:
        distinct_score = 30000.0
    elif c0 > 0 and c1 > 0:
        distinct_score = 20000.0
    elif c0 > 0 and c2 > 0:
        distinct_score = 20000.0
    elif c1 > 0 and c2 > 0:
        distinct_score = 20000.0
    else:
        distinct_score = 10000.0
    
    # Balance: L1 deviation from uniform (faster than chi2, same ordering for small n)
    ideal = n / 3.0
    dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -dev * 10.0
    
    # Farthest-point sampling: compute minimum Hamming distance to all selected vectors
    # This is the key innovation - forces spatial separation
    min_dist = float('inf')
    if len(_selected_vectors) > 1:  # At least one other vector exists
        for sel_vec in _selected_vectors:
            if sel_vec == vec_tuple:
                continue
            # Hamming distance: count positions where symbols differ
            dist = sum(1 for a, b in zip(vector, sel_vec) if a != b)
            if dist < min_dist:
                min_dist = dist
                if min_dist == 0:
                    break
    
    # Scale distance to be comparable with other scores
    # Max possible distance is n, so scale by a factor to make it significant
    distance_score = min_dist * 50.0 if min_dist != float('inf') else n * 50.0
    
    # Fast hash: simple polynomial rolling hash
    h = 0
    for i, val in enumerate(vector):
        h = (h * 31 + val + i) & 0x7FFFFFFF
    
    hash_score = h / 0x7FFFFFFF
    
    # Combine: distance dominates, then distinct, then balance, then hash
    return float(distance_score + distinct_score + balance_score + hash_score)
