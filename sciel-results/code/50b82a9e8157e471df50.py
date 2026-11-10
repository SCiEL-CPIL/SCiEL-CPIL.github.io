"""Simulated annealing for maximum mean-free set in [1,512].

Structural change: Instead of greedy+local search, use SA with an efficient
incremental violation counter. The violation count tells us exactly how many
pairs (a, subset S) violate the mean-free property, enabling O(1) acceptance
decisions during SA moves.
"""
import time
import random
import math

_DEADLINE = 85.0
_start_t = time.time()

def _elapsed():
    return time.time() - _start_t

def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _count_violations(selected):
    """Count total mean violations in selected.
    A violation: element a in selected, subset S of other elements (|S|>=2)
    such that sum(S) == |S| * a.
    Returns the count of such (a, S) pairs.
    """
    n = len(selected)
    if n < 3:
        return 0
    total = 0
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        if m < 2:
            continue
        max_sum = sum(others)
        # dp[k] = bitset of achievable sums with exactly k elements
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
        for k in range(2, m + 1):
            target = k * a
            if target <= max_sum and ((dp[k] >> target) & 1):
                # Count how many subsets of size k sum to k*a
                # For efficiency, just count 1 per (a,k) pair
                total += 1
    return total


def _is_valid(selected):
    """Check if selected is mean-free (no violations)."""
    return _count_violations(selected) == 0


def _fast_check_add(selected, candidate):
    """Check if adding candidate keeps the set mean-free.
    Only need to check:
    1. candidate is not the mean of any subset of selected
    2. each element in selected is not the mean of any subset of (selected \ {that elem}) + {candidate}
    
    For (1): check if candidate is mean of subset of selected
    For (2): the new subsets involving candidate have size >= 2, so they include candidate
             and at least one other element from selected.
             mean = (sum(S) + candidate) / (|S| + 1) for some S subset of selected
             We need: for each a in selected, no subset T of (selected \ {a}) U {candidate}
             with |T| >= 2 and sum(T) = |T| * a.
    """
    n = len(selected)
    if n == 0:
        return True
    
    # Check 1: candidate is not mean of any subset of selected (size >= 2)
    max_sum_sel = sum(selected)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in selected:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k-1] << x)
    for k in range(2, n + 1):
        target = k * candidate
        if target <= max_sum_sel and ((dp[k] >> target) & 1):
            return False
    
    # Check 2: for each a in selected, no subset of (selected \ {a}) U {candidate} 
    #           of size >= 2 has mean a
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        # Subsets of (others + [candidate]) of size >= 2
        # Case A: subset doesn't include candidate -> already valid (selected was valid)
        # Case B: subset includes candidate -> subset = {candidate} U S where S subset of others, |S| >= 1
        #   mean = (sum(S) + candidate) / (|S| + 1) = a
        #   => sum(S) = (|S|+1)*a - candidate = (|S|+1)*a - candidate
        #   Let k = |S| + 1, then k >= 2, and sum(S) = k*a - candidate, |S| = k-1 >= 1
        #   So: for k in [2, m+1], is there a subset of others of size (k-1) summing to k*a - candidate?
        
        if m == 0:
            # others is empty, only subset is {candidate} which has size 1 < 2, skip
            continue
        
        max_sum_others = sum(others)
        dp2 = [0] * (m + 1)
        dp2[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                dp2[k] |= (dp2[k-1] << x)
        
        for k in range(2, m + 2):
            # k = size of full subset (including candidate)
            # S has size k-1, sum = k*a - candidate
            s_size = k - 1
            s_sum = k * a - candidate
            if s_size < 1 or s_size > m:
                continue
            if s_sum < 0 or s_sum > max_sum_others:
                continue
            if ((dp2[s_size] >> s_sum) & 1):
                return False
    
    return True


def _greedy_construct(limit, max_size, order=None):
    """Greedy construction with given order (or natural order)."""
    selected = []
    in_set = set()
    if order is None:
        order = list(range(1, limit + 1))
    for x in order:
        if len(selected) >= max_size:
            break
        if x in in_set:
            continue
        if _fast_check_add(selected, x):
            selected.append(x)
            in_set.add(x)
    return selected


def _simulated_annealing(initial_set, limit, max_size, rng, time_budget=40.0):
    """SA to grow/refine the set.
    State: a set of elements.
    Moves: 
      - Add a random element not in set
      - Remove a random element from set
      - Swap: remove one, add another
    Objective: maximize size, subject to validity (violations == 0).
    We use a two-level objective: primary = -size, secondary = violations.
    Actually, let's use: fitness = size * 1000 - violations. Higher is better.
    But we must ensure final result is valid. So we track best valid set separately.
    
    Alternative: only accept moves that keep violations == 0 (constrained SA),
    or use penalty method.
    
    Let's use: if violations > 0, penalty = violations * 100.
    fitness = len(selected) - violations * 100
    We want to maximize fitness.
    """
    current = list(initial_set)
    current_viol = _count_violations(current)
    current_fit = len(current) - current_viol * 100
    
    best = list(current)
    best_viol = current_viol
    best_fit = current_fit
    
    T_start = 50.0
    T_end = 0.1
    n_iter = 50000
    t0 = time.time()
    
    in_set = set(current)
    
    for it in range(n_iter):
        if _deadline_exceeded():
            break
        # Adaptive time check
        if it % 500 == 0:
            elapsed = time.time() - t0
            if elapsed > time_budget:
                break
            # Cool down
            progress = it / n_iter
            T = T_start * (T_end / T_start) ** progress
        
        # Choose move
        move_type = rng.random()
        trial = list(current)
        trial_viol = current_viol  # will recompute
        
        if move_type < 0.35:
            # Add
            if len(trial) < max_size:
                candidates = [x for x in range(1, limit + 1) if x not in in_set]
                if candidates:
                    x = rng.choice(candidates)
                    trial.append(x)
                    in_set.add(x)
                    # Check if still valid
                    trial_viol = _count_violations(trial)
        elif move_type < 0.55:
            # Remove
            if len(trial) > 1:
                idx = rng.randrange(len(trial))
                x = trial[idx]
                trial.pop(idx)
                in_set.discard(x)
                trial_viol = _count_violations(trial)
        else:
            # Swap
            if len(trial) >= 1:
                idx = rng.randrange(len(trial))
                x_remove = trial[idx]
                candidates = [x for x in range(1, limit + 1) if x not in in_set]
                if candidates:
                    x_add = rng.choice(candidates)
                    trial[idx] = x_add
                    in_set.discard(x_remove)
                    in_set.add(x_add)
                    trial_viol = _count_violations(trial)
        
        trial_fit = len(trial) - trial_viol * 100
        delta_fit = trial_fit - current_fit
        
        if delta_fit >= 0:
            current = trial
            current_viol = trial_viol
            current_fit = trial_fit
            if current_viol == 0 and len(current) > len(best):
                best = list(current)
                best_viol = 0
                best_fit = current_fit
        else:
            # Accept with probability exp(delta_fit / T)
            if T > 0:
                accept_prob = math.exp(delta_fit / T)
                if rng.random() < accept_prob:
                    current = trial
                    current_viol = trial_viol
                    current_fit = trial_fit
                    if current_viol == 0 and len(current) > len(best):
                        best = list(current)
                        best_viol = 0
                        best_fit = current_fit
    
    # Return best valid set found
    if best_viol == 0:
        return best
    # If no valid set found, return current if valid, else initial
    if current_viol == 0:
        return current
    return initial_set


def _targeted_growth(selected, limit, max_size, rng):
    """Try to add elements one by one, checking validity."""
    selected = list(selected)
    in_set = set(selected)
    # Try all possible additions in random order
    candidates = [x for x in range(1, limit + 1) if x not in in_set]
    rng.shuffle(candidates)
    for x in candidates:
        if len(selected) >= max_size:
            break
        if _deadline_exceeded():
            break
        if _fast_check_add(selected, x):
            selected.append(x)
            in_set.add(x)
    return selected


def _multi_greedy(limit, max_size, rng, n_trials=20):
    """Multiple greedy constructions with different orderings."""
    best = []
    for t in range(n_trials):
        if _deadline_exceeded():
            break
        if t == 0:
            order = list(range(1, limit + 1))
        elif t == 1:
            order = list(range(limit, 0, -1))
        else:
            order = list(range(1, limit + 1))
            rng.shuffle(order)
        sel = _greedy_construct(limit, max_size, order)
        # Try to grow
        sel = _targeted_growth(sel, limit, max_size, rng)
        if len(sel) > len(best):
            best = sel
        if len(best) >= max_size:
            break
    return best


def construct():
    global _start_t
    _start_t = time.time()
    rng = random.Random(42)
    limit = 512
    max_size = 48
    
    # Phase 1: Multi-greedy initialization
    best = _multi_greedy(limit, max_size, rng, n_trials=30)
    
    if len(best) >= max_size:
        if _is_valid(best):
            return best[:max_size]
    
    # Phase 2: Simulated annealing from best greedy result
    if not _deadline_exceeded():
        time_budget = min(35.0, _DEADLINE - _elapsed() - 5.0)
        if time_budget > 2.0:
            sa_result = _simulated_annealing(best, limit, max_size, rng, time_budget)
            if len(sa_result) > len(best):
                best = sa_result
    
    # Phase 3: Targeted growth from SA result
    if not _deadline_exceeded():
        best = _targeted_growth(best, limit, max_size, rng)
    
    # Phase 4: More SA if time permits
    if not _deadline_exceeded():
        remaining = _DEADLINE - _elapsed() - 5.0
        if remaining > 5.0 and len(best) < max_size:
            sa_result2 = _simulated_annealing(best, limit, max_size, rng, min(remaining - 2.0, 20.0))
            if len(sa_result2) > len(best):
                best = sa_result2
            best = _targeted_growth(best, limit, max_size, rng)
    
    # Phase 5: Additional random restarts if time permits
    if not _deadline_exceeded():
        remaining = _DEADLINE - _elapsed() - 5.0
        if remaining > 3.0:
            for attempt in range(10):
                if _deadline_exceeded():
                    break
                rng2 = random.Random(1000 + attempt)
                order = list(range(1, limit + 1))
                rng2.shuffle(order)
                sel = _greedy_construct(limit, max_size, order)
                sel = _targeted_growth(sel, limit, max_size, rng2)
                if len(sel) > len(best):
                    best = sel
                if len(best) >= max_size:
                    break
    
    # Final: ensure validity
    if len(best) > max_size:
        best = best[:max_size]
    
    if not _is_valid(best):
        # Trim to largest valid prefix
        for k in range(len(best), 0, -1):
            if _is_valid(best[:k]):
                best = best[:k]
                break
        else:
            best = []
    
    return best
