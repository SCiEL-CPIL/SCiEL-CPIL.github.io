import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _score(U):
    """Compute the score for set U."""
    U = sorted(set(U))
    if not U or 0 not in U:
        return 0.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute |U+U| and |U-U|
    sum_set = set()
    diff_set = set()
    for i in range(n):
        for j in range(n):
            sum_set.add(U[i] + U[j])
            diff_set.add(U[i] - U[j])
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1.0:
        return 0.0
    import math
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _local_search(U, max_iters=5000, deadline=None):
    """Local search to improve the set."""
    best = list(U)
    best_score = _score(best)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Try adding a random element
        if random.random() < 0.5:
            max_u = max(best)
            # Try adding an element in a promising range
            candidates = [random.randint(1, max_u * 2)]
            # Also try structured additions
            if best:
                candidates.append(max_u + random.randint(1, max_u))
            for c in candidates:
                if c not in best:
                    new_set = sorted(best + [c])
                    s = _score(new_set)
                    if s > best_score:
                        best = new_set
                        best_score = s
                        break
        else:
            # Try removing an element (keep 0)
            if len(best) > 2:
                removable = [x for x in best if x != 0]
                if removable:
                    idx = random.choice(removable)
                    new_set = [x for x in best if x != idx]
                    s = _score(new_set)
                    if s > best_score:
                        best = new_set
                        best_score = s
    return best, best_score

def _generate_initial_candidates():
    """Generate diverse initial candidate sets."""
    candidates = []
    
    # Powers of 2 minus 1
    for k in range(2, 12):
        U = [2**i - 1 for i in range(k)]
        candidates.append(U)
    
    # Powers of 3
    for k in range(2, 8):
        U = [3**i for i in range(k)]
        candidates.append(sorted(set(U)))
    
    # Mixed: powers of 2 with some perturbations
    for k in range(3, 10):
        base = [2**i - 1 for i in range(k)]
        # Perturb last few
        for perturb in range(3):
            U = base[:]
            if len(U) > 2:
                U[-1] += random.randint(-2, 2)
                if U[-1] > 0 and U[-1] not in U[:-1]:
                    U = sorted(set(U))
                    candidates.append(U)
    
    # Arithmetic progressions with growing steps
    for base in [2, 3, 4, 5]:
        U = [0]
        current = 0
        for i in range(8):
            current += base ** i
            U.append(current)
        candidates.append(U)
    
    # Random structured sets
    for _ in range(20):
        U = [0]
        current = 0
        for i in range(random.randint(3, 10)):
            step = random.randint(1, 3) ** random.randint(1, 4)
            current += step
            U.append(current)
        candidates.append(sorted(set(U)))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s
    
    random.seed(42)
    
    # Generate initial candidates
    candidates = _generate_initial_candidates()
    
    best_set = None
    best_score = 0.0
    
    # Evaluate all candidates
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Local search on best candidates
    # Sort candidates by score and try local search on top ones
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline:
            break
        s = _score(U)
        scored_candidates.append((s, U))
    
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Try local search on top 5 candidates
    for s, U in scored_candidates[:5]:
        if time.time() > deadline:
            break
        if s <= best_score:
            continue
        new_set, new_score = _local_search(U, max_iters=2000, deadline=deadline)
        if new_score > best_score:
            best_score = new_score
            best_set = new_set
    
    # Also do a broader local search from the current best
    if best_set is not None and time.time() < deadline:
        new_set, new_score = _local_search(best_set, max_iters=5000, deadline=deadline)
        if new_score > best_score:
            best_score = new_score
            best_set = new_set
    
    if best_set is None:
        return [0, 1, 3]
    
    return sorted(set(best_set))
# EVOLVE-BLOCK-END
