import math
import time

def _score(U):
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    n_sum = len(sumset)
    n_diff = len(diffset)
    if n_sum == 0:
        return 0.0
    ratio = n_diff / n_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Seed set
    best_set = [0, 1, 3, 9]
    best_score = _score(best_set)
    
    # Try to improve by adding one element
    # Candidate elements: powers of 3, or numbers near existing ones
    candidates = list(range(0, 100))
    # Sort candidates by potential to improve (heuristic: small numbers first)
    # Actually, just try all small numbers
    
    current = set(best_set)
    improved = False
    
    # Local search: add one element
    for x in range(1, 50):
        if time.time() > deadline:
            break
        if x in current:
            continue
        new_set = list(current) + [x]
        s = _score(new_set)
        if s > best_score:
            best_score = s
            best_set = new_set
            current = set(new_set)
            improved = True
            # Reset candidates or continue?
            # If we add one, we might add another.
            # But keep it simple: one pass.
            # Restart loop to allow multiple additions?
            # Given time constraints, maybe just one addition is safe.
            # Let's do a simple hill climbing with a few restarts if time permits.
            pass

    # Try removing elements? Unlikely to help if we started small.
    
    # Try a different base?
    # Powers of 4?
    # U = {0, 1, 4, 16}
    # M = 33
    # Sum: 0,1,2,4,5,8,16,17,20,32 (10)
    # Diff: 0, +/-1, +/-4, +/-16, +/-3, +/-15, +/-12 (13)
    # R = 1.3. Worse.
    
    return best_set
