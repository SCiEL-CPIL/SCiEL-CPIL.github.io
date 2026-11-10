"""Simulated annealing over binary selection vector with mean-violation penalty."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 82.0
_start_t = time.time()

def _elapsed():
    return time.time() - _start_t

def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _count_violations(selected):
    """Count total mean violations in the selected set.
    A violation: element a in selected, and a subset S of other selected elements
    (|S| >= 2) such that sum(S) == |S| * a.
    We count each violating (a, S) pair.
    For speed, we only check subsets of size 2 and 3 (most common violations).
    Size 2: a = (b+c)/2, i.e., b+c = 2a, b≠c, b≠a, c≠a
    Size 3: a = (b+c+d)/3, i.e., b+c+d = 3a
    """
    if len(selected) < 3:
        return 0
    sel_set = set(selected)
    violations = 0
    # Size-2 violations: for each pair (b,c) with b<c, check if (b+c)/2 is in set and != b,c
    # b+c must be even, and (b+c)//2 in sel_set
    n = len(selected)
    # Precompute for size-2
    for i in range(n):
        for j in range(i+1, n):
            s = selected[i] + selected[j]
            if s % 2 == 0:
                mid = s // 2
                if mid in sel_set and mid != selected[i] and mid != selected[j]:
                    violations += 1
    # Size-3 violations: for each triple (b,c,d), check if (b+c+d)/3 is integer and in set
    # Only check if sum is divisible by 3
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = selected[i] + selected[j] + selected[k]
                if s % 3 == 0:
                    mid = s // 3
                    if mid in sel_set and mid != selected[i] and mid != selected[j] and mid != selected[k]:
                        violations += 1
    return violations


def _is_valid(selected):
    """Check if selected set is fully mean-free (no violations of any subset size)."""
    if len(selected) < 3:
        return True
    n = len(selected)
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        m = len(others)
        if m < 2:
            continue
        max_sum = sum(others)
        # Bitset DP: dp[k] is integer where bit s set means sum s achievable with k elements
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << x)
        for k in range(2, m + 1):
            target = k * a
            if target <= max_sum and ((dp[k] >> target) & 1):
                return False
    return True


def _fast_add_check(selected, candidate):
    """Quick check if adding candidate creates a size-2 or size-3 violation.
    Only checks new violations involving candidate or where candidate is the mean.
    """
    sel_set = set(selected)
    n = len(selected)
    # Case 1: candidate is the mean of two others
    # For each pair (b,c) in selected, check if (b+c)/2 == candidate
    for i in range(n):
        for j in range(i+1, n):
            s = selected[i] + selected[j]
            if s % 2 == 0 and s // 2 == candidate:
                return False
    # Case 2: candidate is the mean of three others
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = selected[i] + selected[j] + selected[k]
                if s % 3 == 0 and s // 3 == candidate:
                    return False
    # Case 3: some a in selected is the mean of candidate and one other
    # a = (candidate + b)/2 => candidate + b = 2a => b = 2a - candidate
    for a in selected:
        b = 2 * a - candidate
        if b != candidate and b in sel_set:
            return False
    # Case 4: some a in selected is the mean of candidate and two others
    # a = (candidate + b + c)/3 => candidate + b + c = 3a => b + c = 3a - candidate
    for a in selected:
        target = 3 * a - candidate
        # Check if any pair in selected sums to target
        for i in range(n):
            for j in range(i+1, n):
                if selected[i] + selected[j] == target:
                    return False
    return True


def _greedy_init(limit, max_size, rng):
    """Greedy initialization: add elements in random order if they don't create violations."""
    selected = []
    in_set = set()
    order = list(range(1, limit + 1))
    rng.shuffle(order)
    for x in order:
        if len(selected) >= max_size:
            break
        if _fast_add_check(selected, x):
            selected.append(x)
            in_set.add(x)
    return selected


def _simulated_annealing(initial, limit, max_size, rng, time_budget):
    """Simulated annealing to maximize |selected| while minimizing violations.
    
    State: frozenset of selected numbers
    Energy: -count + λ * violations
    Move: add or remove a random element
    """
    selected = list(initial)
    in_set = set(selected)
    current_count = len(selected)
    current_viol = _count_violations(selected)
    current_energy = -current_count + 5.0 * current_viol
    
    best_selected = list(selected)
    best_count = current_count
    best_viol = current_viol
    best_energy = current_energy
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    # Precompute candidate pool
    candidates = list(range(1, limit + 1))
    
    step = 0
    while not _deadline_exceeded() and T > T_min:
        step += 1
        if step % 1000 == 0 and _elapsed() > time_budget:
            break
        
        # Choose move: add or remove
        if current_count < max_size and rng.random() < 0.6:
            # Add a random element not in set
            x = rng.choice(candidates)
            if x in in_set:
                continue
            new_selected = selected + [x]
            new_in_set = in_set | {x}
            new_count = current_count + 1
            # Quick violation count: only check new violations
            # For speed, use _count_violations on the new set (approximate)
            new_viol = _count_violations(new_selected)
            new_energy = -new_count + 5.0 * new_viol
        else:
            # Remove a random element
            if current_count == 0:
                continue
            idx = rng.randrange(current_count)
            x = selected[idx]
            new_selected = selected[:idx] + selected[idx+1:]
            new_in_set = in_set - {x}
            new_count = current_count - 1
            new_viol = _count_violations(new_selected)
            new_energy = -new_count + 5.0 * new_viol
        
        delta = new_energy - current_energy
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            selected = new_selected
            in_set = new_in_set
            current_count = new_count
            current_viol = new_viol
            current_energy = new_energy
            
            if current_count > best_count or (current_count == best_count and current_viol < best_viol):
                best_selected = list(selected)
                best_count = current_count
                best_viol = current_viol
                best_energy = current_energy
        
        T *= cooling
    
    return best_selected


def construct():
    global _start_t
    _start_t = time.time()
    rng = random.Random(12345)
    limit = 512
    max_size = 48
    
    # Phase 1: Greedy initialization
    selected = _greedy_init(limit, max_size, rng)
    best = list(selected)
    best_size = len(selected)
    
    # Phase 2: Simulated annealing with multiple restarts
    sa_time_budget = 60.0
    restarts = 0
    while not _deadline_exceeded() and restarts < 10:
        restarts += 1
        # New initialization
        init = _greedy_init(limit, max_size, rng)
        if len(init) <= best_size:
            # Perturb best solution for SA
            perturbed = list(best)
            for _ in range(min(5, len(perturbed))):
                if rng.random() < 0.5 and perturbed:
                    idx = rng.randrange(len(perturbed))
                    perturbed.pop(idx)
                else:
                    x = rng.randint(1, limit)
                    if x not in perturbed:
                        perturbed.append(x)
            init = perturbed
        
        result = _simulated_annealing(init, limit, max_size, rng, sa_time_budget)
        if len(result) > best_size:
            best = result
            best_size = len(result)
        
        if best_size >= max_size:
            break
    
    # Phase 3: Final validation and trimming
    if len(best) > max_size:
        best = best[:max_size]
    
    # Validate with exact check
    if not _is_valid(best):
        # Trim from end until valid
        for k in range(len(best), 0, -1):
            if _is_valid(best[:k]):
                best = best[:k]
                break
    
    return best
# EVOLVE-BLOCK-END
