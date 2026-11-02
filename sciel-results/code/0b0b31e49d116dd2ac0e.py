import time
import random
import math
from collections import Counter

_BEST = None
_DEADLINE = time.time() + 110.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _fast_greedy(k, n_rows, rng=None, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _sa_optimize(rows, k, iters, rng, deadline):
    """
    Simulated Annealing on the scope.
    State: list of rows (each row is list of k+1 ints, row[0]=0).
    Objective: scope (max entry). We want to minimize it.
    Moves: 
      1. Pick a random row i and a random index j (1 <= j <= k).
      2. Propose a new value for rows[i][j] in a local range or global range.
      3. Ensure strict increase within the row.
      4. Calculate new scope and collision penalty.
      5. Accept based on Metropolis criterion.
    """
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [list(r) for r in current_rows]
    best_scope = current_scope
    
    # Precompute some constants for speed
    # We track collisions via a global set of differences.
    # For SA, we need to evaluate the change in "cost".
    # Cost = scope + lambda * (number of collisions)
    # But we want to minimize scope primarily.
    # Let's define cost = scope + 1000 * collisions.
    
    def get_cost(rows):
        scope = _scope(rows)
        diffs = set()
        collisions = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
                    d = r[j] - r[i]
                    if d in diffs:
                        collisions += 1
                    else:
                        diffs.add(d)
        return scope + 1000 * collisions

    current_cost = get_cost(current_rows)
    best_cost = current_cost
    
    T = 10.0
    cooling = 0.999
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        # Pick a random element to change
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)  # index in row, 1..k
        
        old_val = current_rows[i][j]
        
        # Determine range for new value
        # Must be > prev and < next
        prev_val = current_rows[i][j-1]
        next_val = current_rows[i][j+1] if j < k else current_scope + 10
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
            
        # Propose new value
        # Sometimes small jump, sometimes larger
        if rng.random() < 0.7:
            span = hi - lo
            # Gaussian-like jump
            delta = int(rng.gauss(0, max(1, span * 0.2)))
            new_val = old_val + delta
        else:
            new_val = rng.randint(lo, hi)
            
        if new_val == old_val:
            continue
            
        # Apply change
        current_rows[i][j] = new_val
        
        # Check validity of the row (strict increase)
        row = current_rows[i]
        if not all(row[t] < row[t+1] for t in range(k)):
            current_rows[i][j] = old_val
            continue
            
        new_cost = get_cost(current_rows)
        
        # Metropolis
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, 0.1)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows = [list(r) for r in current_rows]
        else:
            current_rows[i][j] = old_val
            
        T *= cooling
        
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 110.0
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    # 1. Generate several initial candidates using different greedy strategies
    candidates = []
    
    # Standard greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    # Spread greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Randomized greedy (shuffle candidates)
    for _ in range(5):
        used = 0
        rows = []
        ok = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a range of candidates
                found = False
                # Randomly decide step size
                max_step = rng.randint(1, 10)
                for t in range(x + 1, x + 1 + max_step * 3):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    # Fallback: find smallest valid
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
        
    # 2. Run SA on the best few candidates
    # Sort candidates by scope
    candidates.sort(key=_scope)
    
    # Run SA on the top 3 candidates
    for cand in candidates[:3]:
        if time.time() > _DEADLINE:
            break
        # Run SA for a while
        iters = 5000
        s, rows = _sa_optimize(cand, k, iters, rng, _DEADLINE)
        consider(rows)
        
    # 3. If time remains, run SA on the best result found so far
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        # Estimate iterations based on remaining time
        # Roughly 10000 iters per second is optimistic, let's be safe
        extra_iters = int(remaining_time * 5000)
        if extra_iters > 1000:
            s, rows = _sa_optimize(_BEST[1], k, extra_iters, rng, _DEADLINE)
            consider(rows)
            
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
