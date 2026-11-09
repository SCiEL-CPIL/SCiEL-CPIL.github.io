"""Search-based program: construct a finite set to maximize the sumset/difference-set exponent."""
# EVOLVE-BLOCK-START
import time
import math
import random

_START_TIME = time.time()
TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _sum_diff_counts(U):
    """Return (|U+U|, |U-U|) for a finite set U of ints."""
    Us = list(U)
    n = len(Us)
    if n <= 1:
        return 1, 1
    # Use sets for sum and difference
    plus = set()
    minus = set()
    for i in range(n):
        a = Us[i]
        for j in range(n):
            plus.add(a + Us[j])
            minus.add(a - Us[j])
    return len(plus), len(minus)

def _score_set(U):
    """Compute the score: 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)."""
    U = sorted(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    if not U or max(U) <= 0:
        return 0.0
    nplus, nminus = _sum_diff_counts(U)
    mx = max(U)
    base = 2 * mx + 1
    if nplus == 0 or base <= 1:
        return 0.0
    ratio = nminus / nplus
    if ratio <= 0:
        return 0.0
    val = 1.0 + math.log(ratio) / math.log(base)
    return val

def _candidate_lacunary(r, n_terms):
    """Generate a lacunary sequence: 0, 1, r, r^2, ..., r^(n_terms-1)."""
    if r <= 1:
        r = 2
    elems = [0]
    val = 1
    for i in range(1, n_terms):
        elems.append(val)
        val *= r
    return elems

def _candidate_geometric(start, ratio, n_terms):
    """Geometric-like sequence starting from 'start'."""
    elems = [0, start]
    val = start
    for i in range(1, n_terms):
        val = val * ratio
        elems.append(val)
    return elems

def _candidate_arithmetic(step, n_terms):
    """Arithmetic progression: 0, step, 2*step, ..., (n_terms-1)*step."""
    return [i * step for i in range(n_terms)]

def _local_search(U, max_iters=5000):
    """Local search: try adding/removing elements to improve score."""
    best = list(U)
    best_score = _score_set(best)
    if best_score <= 0:
        return best, best_score
    
    mx = max(best)
    # Candidate pool: elements near existing ones and some random ones
    pool = set()
    for x in best:
        for d in range(-3, 4):
            if x + d >= 0:
                pool.add(x + d)
    # Add some random candidates up to 2*mx
    for _ in range(50):
        pool.add(random.randint(0, min(2 * mx, mx + 20)))
    pool.discard(0)  # 0 must stay
    pool = sorted(pool)
    
    for it in range(max_iters):
        if _elapsed() > TIME_LIMIT - 10:
            break
        improved = False
        # Try adding an element
        for cand in pool:
            if cand in best:
                continue
            trial = best + [cand]
            s = _score_set(trial)
            if s > best_score + 1e-12:
                best = trial
                best_score = s
                improved = True
                # Update pool
                for d in range(-3, 4):
                    if cand + d >= 0:
                        pool.add(cand + d)
                break
        if improved:
            continue
        # Try removing an element (not 0)
        removed = False
        for i in range(len(best) - 1, 0, -1):  # don't remove 0 (index 0 after sort)
            if best[i] == 0:
                continue
            trial = best[:i] + best[i+1:]
            s = _score_set(trial)
            if s > best_score + 1e-12:
                best = trial
                best_score = s
                removed = True
                break
        if not improved and not removed:
            # Stuck, try random perturbation
            if random.random() < 0.3:
                # Random add
                cand = random.randint(1, max(2 * mx, 50))
                if cand not in best:
                    trial = best + [cand]
                    s = _score_set(trial)
                    if s > best_score:
                        best = trial
                        best_score = s
                        mx = max(best)
                        continue
            break
    return best, best_score

def construct():
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Try various parametric families
    candidates = []
    
    # Lacunary sequences with various ratios and lengths
    for r in [2, 3, 4, 5, 6, 7, 8, 10]:
        for n in range(3, 12):
            cand = _candidate_lacunary(r, n)
            candidates.append(cand)
    
    # Geometric-like
    for start in [1, 2, 3, 4, 5]:
        for ratio in [2, 3, 4, 5]:
            for n in range(3, 10):
                cand = _candidate_geometric(start, ratio, n)
                candidates.append(cand)
    
    # Arithmetic progressions
    for step in [1, 2, 3, 4, 5]:
        for n in range(3, 15):
            cand = _candidate_arithmetic(step, n)
            candidates.append(cand)
    
    # Mixed: start with lacunary, add some intermediate elements
    for r in [2, 3, 4]:
        for n in range(4, 8):
            base = _candidate_lacunary(r, n)
            # Add midpoints
            for i in range(len(base) - 1):
                mid = (base[i] + base[i+1]) // 2
                if mid not in base and mid > 0:
                    candidates.append(base + [mid])
    
    # Evaluate all candidates
    for cand in candidates:
        if _elapsed() > TIME_LIMIT - 30:
            break
        s = _score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Local search from the best candidate
    if _elapsed() < TIME_LIMIT - 20:
        best_set, best_score = _local_search(best_set, max_iters=20000)
    
    # Try local search from a few other good candidates
    if _elapsed() < TIME_LIMIT - 15:
        # Find second-best candidate
        scored = []
        for cand in candidates[:50]:
            s = _score_set(cand)
            scored.append((s, cand))
        scored.sort(reverse=True)
        for s, cand in scored[1:4]:
            if _elapsed() > TIME_LIMIT - 10:
                break
            if s < best_score:
                continue
            trial_set, trial_score = _local_search(cand, max_iters=5000)
            if trial_score > best_score:
                best_score = trial_score
                best_set = trial_set
    
    return sorted(set(int(x) for x in best_set))
# EVOLVE-BLOCK-END
