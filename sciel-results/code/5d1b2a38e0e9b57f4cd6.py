"""Rewritten program: randomized scored greedy for B2[2] sets in 1..1000."""
import time
import random

N = 1000
MAX_REP = 2
DEADLINE = 150.0
SAFETY = 8.0  # seconds margin

def _build(score_mode, rng=None):
    """Build a B2[2] set using a scored greedy. Returns (size, chosen)."""
    chosen = []
    sum_counts = {}  # sum value -> count of representations
    # We'll iterate candidate x from 1..N and decide order by score.
    # To allow different strategies, we precompute a candidate ordering.
    # Simplest: scan x from 1..N, but use score to *choose* among a window.
    
    # We'll do a global scan but at each step pick the best feasible x by score,
    # which can be far ahead. To keep it efficient, we scan 1..N each step is O(n^2 * k).
    # Better: maintain a list of not-yet-considered and evaluate incrementally.
    # For clarity and given N=1000, O(n^2) per full build is ~1e6 * small, fine.
    
    # We'll just scan x=1..N in order but allow skipping: pick smallest x that is
    # feasible AND has best score among a lookahead window.
    # Simpler robust approach: repeatedly find best (score, x) among all feasible x.
    
    # Precompute nothing; loop.
    remaining = list(range(1, N + 1))
    while remaining:
        best = None
        best_key = None
        for x in remaining:
            # compute affected sums and check feasibility + score
            # new sums: x + y for y in chosen, plus 2x
            # but we only need to check the max count among affected
            # For speed, compute the worst count.
            ok = True
            penalty = 0  # number of sums that would reach exactly MAX_REP
            # iterate chosen
            # To early-exit, track max.
            for y in chosen:
                s = x + y
                c = sum_counts.get(s, 0) + 1
                if c > MAX_REP:
                    ok = False
                    break
                if c == MAX_REP:
                    penalty += 1
            if not ok:
                continue
            s2 = 2 * x
            c2 = sum_counts.get(s2, 0) + 1
            if c2 > MAX_REP:
                continue
            if c2 == MAX_REP:
                penalty += 1
            # also count sums that would reach 1 (neutral) - not penalized
            # score key: minimize penalty, then minimize x (to keep density) 
            # but sometimes larger x spreads better; use mode.
            if score_mode == 0:
                key = (penalty, x)
            elif score_mode == 1:
                key = (penalty, -x)  # prefer larger x to spread
            elif score_mode == 2:
                # random tie-break
                key = (penalty, rng.random(), x)
            else:
                key = (penalty, x)
            if best_key is None or key < best_key:
                best_key = key
                best = x
        if best is None:
            break
        # commit
        for y in chosen:
            s = best + y
            sum_counts[s] = sum_counts.get(s, 0) + 1
        s2 = 2 * best
        sum_counts[s2] = sum_counts.get(s2, 0) + 1
        chosen.append(best)
        remaining.remove(best)
    return len(chosen), chosen

def construct():
    start = time.time()
    best_size = 0
    best_set = []
    rng = random.Random(12345)
    # Try several modes and restarts within budget.
    attempts = 0
    # First, a deterministic scored greedy (mode 0) as baseline.
    while True:
        elapsed = time.time() - start
        if elapsed > DEADLINE - SAFETY:
            break
        # choose a mode: cycle through 0,1,2 with random restarts
        mode = attempts % 3
        size, chosen = _build(mode, rng)
        if size > best_size:
            best_size = size
            best_set = chosen
        attempts += 1
        # After a few deterministic-ish, rely on random (mode 2) which gives variety.
        if attempts >= 60:
            break
    # Ensure we have at least the seed-quality result if something went wrong.
    if best_size == 0:
        # fallback: simple greedy
        chosen = []
        sum_counts = {}
        for x in range(1, N + 1):
            ok = True
            for y in chosen:
                if sum_counts.get(x + y, 0) + 1 > MAX_REP:
                    ok = False
                    break
            if ok and sum_counts.get(2 * x, 0) + 1 <= MAX_REP:
                for y in chosen:
                    sum_counts[x + y] = sum_counts.get(x + y, 0) + 1
                sum_counts[2 * x] = sum_counts.get(2 * x, 0) + 1
                chosen.append(x)
        best_size = len(chosen)
        best_set = chosen
    return best_set
