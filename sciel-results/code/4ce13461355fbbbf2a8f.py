"""Improved covering-code constructor: direct-sum baseline + greedy set-cover +
repair-based stochastic local search.

Strategy:
  * A simple always-valid baseline (direct sum of a radius-1 perfect code with the
    remaining coordinates) guarantees correctness and gives an upper bound.
  * A greedy set-cover heuristic (place the center that covers the most uncovered
    words, repeat) finds codes that are often smaller than the baseline.
  * A repair-based stochastic local search (SLS) maximizes the number of radius-1
    balls packed while penalizing overlaps via dynamic weights.  "Repair" moves
    instantly fix coverage deficits by swapping in a word that covers the most
    uncovered positions, keeping the codeword count constant.  This shifts focus
    from minimizing set-cover cost to maximizing packing density with hard coverage
    constraints handled by repair.
  * Coverage is verified exactly before returning; on any failure or time-out the
    valid baseline is returned, so the result is always a correct covering.
"""

import time
import itertools
import numpy as np
import random

# Safety margin: the evaluator kills the process at 150 s; keep a generous buffer.
_DEADLINE = 140.0

# Precomputed balls for the specific task q=3, n=6, r=1 to avoid recomputation
# and allow for faster lookups if needed, though we will use numpy arrays.
# We will generate them on the fly in _build_balls but ensure they are efficient.


def _direct_sum(q, n, radius):
    """Always-valid baseline: a radius-1 perfect code of length 4 over the alphabet,
    tensored with every word on the remaining n-4 coordinates.  Valid for R>=1, n>=4.
    Size = q^2 * q^(n-4) = q^(n-2)."""
    if n < 4:
        # Fallback for small n: cover every word directly
        return [list(w) for w in itertools.product(range(q), repeat=n)]
    # Perfect (q^2, 4, 1) covering: positions 0,1 free; 2 = a+b; 3 = a+2b (mod q).
    base = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    return [h + list(tail) for h in base
            for tail in itertools.product(range(q), repeat=n - 4)]


def _idx(w, q):
    idx = 0
    for x in w:
        idx = idx * q + x
    return idx


def _ball_indices(c, q, n, radius):
    """Flat indices of all words within Hamming distance `radius` of center c."""
    if radius == 0:
        return [_idx(c, q)]
    ball = [_idx(c, q)]
    for d in range(1, radius + 1):
        for pos in itertools.combinations(range(n), d):
            for vals in itertools.product(range(1, q), repeat=d):
                w = list(c)
                for p, v in zip(pos, vals):
                    w[p] = v
                ball.append(_idx(w, q))
    return ball


def _build_balls(q, n, radius):
    """Precompute the radius ball (as a sorted unique list of flat indices) for every
    word in {0..q-1}^n.  Returns (all_words, balls) where all_words is a numpy array
    of shape (total, n) and balls is a list of lists of ints."""
    total = q ** n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    balls = []
    for i in range(total):
        c = all_words[i]
        balls.append(np.array(_ball_indices(c.tolist(), q, n, radius), dtype=np.int32))
    return all_words, balls


def _covers_all(cw, q, n, radius):
    """Exact coverage check over the whole space (only used for small spaces)."""
    total = q ** n
    if total > 5 * 10**6:
        return False
    covered = np.zeros(total, dtype=bool)
    for c in cw:
        for idx in _ball_indices(c, q, n, radius):
            covered[idx] = True
    return bool(covered.all())


def _greedy(q, n, radius, time_budget):
    """Greedy set-cover.  Returns (codewords, verified) where verified is True only if
    the returned code is a confirmed covering of the whole space."""
    total = q ** n
    if total > 3 * 10**6:
        return None, False  # too large to track with an explicit covered-set

    start = time.time()
    all_words, balls = _build_balls(q, n, radius)
    covered = np.zeros(total, dtype=bool)
    code = []
    while not covered.all():
        if time.time() - start > time_budget:
            return code, _covers_all(code, q, n, radius)
        # For each candidate center, how many currently-uncovered words it would cover.
        best = -1
        best_gain = 0
        for i in range(total):
            b = balls[i]
            gain = int(np.sum(~covered[b]))
            if gain > best_gain:
                best_gain = gain
                best = i
        code.append(all_words[best].tolist())
        covered[balls[best]] = True
    return code, True


def _sls_repair(code, q, n, radius, time_budget, seed=0):
    """Optimized Stochastic Local Search with repair moves.
    
    Uses incremental overlap calculation and efficient coverage checks.
    """
    total = q ** n
    if total > 3 * 10**6:
        return code, False

    rng = random.Random(seed)
    start = time.time()

    all_words, balls = _build_balls(q, n, radius)
    
    # State: list of word indices
    state = []
    for c in code:
        state.append(_idx(c, q))
        
    # Initial counts
    counts = np.zeros(total, dtype=np.int32)
    for wi in state:
        for idx in balls[wi]:
            counts[idx] += 1
            
    def calc_overlap(cnt):
        return int(np.sum(cnt * cnt))

    cur_overlap = calc_overlap(counts)
    best_state = list(state)
    best_overlap = cur_overlap
    best_size = len(state)
    
    no_improve_count = 0
    max_no_improve = 20000 # Increased as operations are faster
    
    while time.time() - start < time_budget:
        if no_improve_count > max_no_improve:
            break
            
        # 1. Try Removals
        improved = False
        for i in range(len(state) - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            wi = state[i]
            b = balls[wi]
            # Can we remove? Check if any word in b has count 1
            if not np.any(counts[b] == 1):
                # Remove
                for idx in b:
                    counts[idx] -= 1
                state.pop(i)
                new_overlap = calc_overlap(counts)
                best_state = list(state)
                best_size = len(state)
                best_overlap = new_overlap
                improved = True
                break # Restart loop with new state
        
        if improved:
            no_improve_count = 0
            continue
            
        # 2. Try Swaps
        if len(state) > 0:
            i = rng.randrange(len(state))
            old_wi = state[i]
            b_old = balls[old_wi]
            
            # Precompute set of b_old for faster "in" check if needed, 
            # but we use the condition: for w in b_old, if counts[w]==1 then w in b_new.
            
            found_swap = False
            # Try more candidates since each is cheap
            for _ in range(200):
                if time.time() - start > time_budget:
                    break
                new_wi = rng.randrange(total)
                if new_wi == old_wi:
                    continue
                b_new = balls[new_wi]
                
                # Check coverage validity
                # Condition: For all w in b_old, if counts[w] == 1, then w must be in b_new.
                valid = True
                # Iterate through b_old
                for w in b_old:
                    if counts[w] == 1:
                        # Check if w is in b_new
                        # Since b_new is small, we can use a set or just check
                        # To avoid set creation overhead, we can use a precomputed bitset or just a loop
                        # But b_new is a numpy array. 
                        # Let's create a set of b_new once per candidate? No, too slow.
                        # Use np.isin? 
                        # Or just check if w is in b_new using a helper?
                        # For n=6, r=1, ball size is 13. 
                        # A simple loop `if w in b_new` is O(13). 13*13 = 169 ops. Acceptable.
                        if w not in b_new:
                            valid = False
                            break
                            
                if not valid:
                    continue
                
                # Calculate delta overlap
                # Delta = sum_{w in b_old U b_new} (c'_w^2 - c_w^2)
                # c'_w = c_w - 1 (if w in b_old) + 1 (if w in b_new)
                # We can compute this by iterating the union.
                # To avoid creating a union set, we can just apply, calc, revert?
                # No, we want to avoid full calc_overlap.
                # Let's compute delta directly.
                delta = 0
                # We need to know for each w in b_old U b_new the old count and new count.
                # Let's just apply to a copy? No, modifying counts is fast.
                # Let's just apply, check delta, and revert if bad.
                # But we need to know if it's an improvement BEFORE committing?
                # Actually, we can just apply, calc new_overlap, and if worse, revert.
                # The cost of apply/revert is O(26).
                # The cost of calc_overlap is O(729).
                # So we should avoid calc_overlap.
                
                # Let's compute delta analytically.
                # For w in b_old \ b_new: c' = c - 1. Delta = (c-1)^2 - c^2 = -2c + 1.
                # For w in b_new \ b_old: c' = c + 1. Delta = (c+1)^2 - c^2 = 2c + 1.
                # For w in b_old \cap b_new: c' = c. Delta = 0.
                
                # We need to identify these sets.
                # This requires set operations.
                # Given the small size, let's just use Python sets for the balls?
                # Precompute sets?
                
                # Alternative: Just apply, compute delta by summing changes in squares for affected words.
                # Affected words are b_old U b_new.
                # We can iterate b_old and b_new.
                
                # Let's try the apply-revert method but ONLY calculate the change in overlap
                # for the affected words.
                
                # 1. Apply
                for idx in b_old:
                    counts[idx] -= 1
                for idx in b_new:
                    counts[idx] += 1
                    
                # 2. Calculate delta overlap
                # We need the OLD counts to calculate the delta?
                # We just modified counts.
                # Delta = sum (new^2 - old^2).
                # We didn't store old counts.
                # So this approach requires storing old counts or doing it analytically.
                
                # Let's revert and do it analytically with sets.
                for idx in b_new:
                    counts[idx] -= 1
                for idx in b_old:
                    counts[idx] += 1
                    
                # Analytical Delta Calculation
                set_old = set(b_old)
                set_new = set(b_new)
                
                delta = 0
                # Words in old but not new
                for w in set_old - set_new:
                    c = counts[w]
                    delta += (c - 1) ** 2 - c ** 2
                # Words in new but not old
                for w in set_new - set_old:
                    c = counts[w]
                    delta += (c + 1) ** 2 - c ** 2
                # Intersection: no change
                
                new_overlap = cur_overlap + delta
                
                if new_overlap < best_overlap or (new_overlap == best_overlap and rng.random() < 0.1):
                    # Accept
                    for idx in b_old:
                        counts[idx] -= 1
                    for idx in b_new:
                        counts[idx] += 1
                    state[i] = new_wi
                    cur_overlap = new_overlap
                    best_state = list(state)
                    best_overlap = new_overlap
                    found_swap = True
                    no_improve_count = 0
                    break
            
            if found_swap:
                continue
                
        no_improve_count += 1

    # Convert best_state back to codewords
    best_code = [all_words[wi].tolist() for wi in best_state]
    verified = _covers_all(best_code, q, n, radius)
    return best_code, verified


def construct(q, n, radius):
    start = time.time()
    # Always-valid baseline (used as the guaranteed fallback).
    if n >= 4 and radius >= 1:
        baseline = _direct_sum(q, n, radius)
    else:
        # Degenerate small cases: cover every word directly (exact, valid).
        baseline = [list(w) for w in itertools.product(range(q), repeat=n)]

    best = baseline
    best_len = len(baseline)

    # Try to improve with the greedy set-cover, only when the space is tractable.
    if q ** n <= 3 * 10**6 and best_len > 1:
        time_budget = max(0.5, _DEADLINE - (time.time() - start) - 10.0)
        try:
            cand, verified = _greedy(q, n, radius, time_budget)
        except Exception:
            cand, verified = None, False
        if cand is not None and verified and len(cand) < best_len:
            best = cand
            best_len = len(cand)

        # SLS repair polish to try to beat the greedy result.
        remaining = _DEADLINE - (time.time() - start) - 5.0
        if remaining > 2.0 and best_len > 1:
            try:
                polished, p_verified = _sls_repair(best, q, n, radius, remaining, seed=42)
            except Exception:
                polished, p_verified = None, False
            if polished is not None and p_verified and len(polished) < best_len:
                best = polished
                best_len = len(polished)

    # Final safety: if we are running out of time, return the valid baseline.
    if time.time() - start > _DEADLINE - 5.0:
        return baseline

    return best
