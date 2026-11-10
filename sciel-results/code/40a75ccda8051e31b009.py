"""Rewritten program with bitmask-based conflict tracking and efficient search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
MAX_SIZE = 48


def _is_valid_subset(S):
    """Check if subset S satisfies the constraint."""
    S_list = list(S)
    n = len(S_list)
    for i, a in enumerate(S_list):
        others = [S_list[j] for j in range(n) if j != i]
        m = len(others)
        if m < 3:
            continue
        for b_idx in range(m - 2):
            b = others[b_idx]
            for c_idx in range(b_idx + 1, m - 1):
                c = others[c_idx]
                for d_idx in range(c_idx + 1, m):
                    d = others[d_idx]
                    if (b + c + d) % a == 0:
                        return False
    return True


def _count_conflicts(S, x):
    """Count how many constraint violations x would cause if added to S."""
    S_list = list(S)
    n = len(S_list)
    violations = 0
    
    # Check 1: Does x divide sum of any 3 elements in S?
    if n >= 3:
        for i in range(n - 2):
            a = S_list[i]
            for j in range(i + 1, n - 1):
                b = S_list[j]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if (a + b + c) % x == 0:
                        violations += 1
                        if violations > 100:
                            return violations
    
    # Check 2: For each a in S, does a divide (x + b + c) for any b,c in S\{a}?
    if n >= 2:
        for idx, a in enumerate(S_list):
            others = [S_list[j] for j in range(n) if j != idx]
            m = len(others)
            if m >= 2:
                for p in range(m - 1):
                    b = others[p]
                    for q in range(p + 1, m):
                        c = others[q]
                        if (x + b + c) % a == 0:
                            violations += 1
                            if violations > 100:
                                return violations
    
    return violations


def _can_add(x, S_set, S_list):
    """Fast check if x can be added to S without violating constraints."""
    n = len(S_list)
    
    # Check 1: Does x divide sum of any 3 elements in S?
    if n >= 3:
        for i in range(n - 2):
            a = S_list[i]
            for j in range(i + 1, n - 1):
                b = S_list[j]
                for k in range(j + 1, n):
                    c = S_list[k]
                    if (a + b + c) % x == 0:
                        return False
    
    # Check 2: For each a in S, does a divide (x + b + c) for any b,c in S\{a}?
    if n >= 2:
        for idx, a in enumerate(S_list):
            others = [S_list[j] for j in range(n) if j != idx]
            m = len(others)
            if m >= 2:
                for p in range(m - 1):
                    b = others[p]
                    for q in range(p + 1, m):
                        c = others[q]
                        if (x + b + c) % a == 0:
                            return False
    
    return True


def _greedy_construct(order, deadline):
    """Greedy construction with conflict checking."""
    selected = []
    selected_set = set()
    
    for x in order:
        if time.time() > deadline:
            break
        if x in selected_set:
            continue
        if _can_add(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    
    return selected


def _local_search(S, deadline, max_swaps=3):
    """Local search with 1-swap and 2-swap moves."""
    best = list(S)
    best_set = set(best)
    best_size = len(best)
    
    while time.time() < deadline:
        improved = False
        
        # Try 1-swap: remove one, add one
        for i in range(len(best)):
            if time.time() > deadline:
                break
            removed = best[i]
            temp = [best[j] for j in range(len(best)) if j != i]
            temp_set = set(temp)
            
            # Try adding each candidate
            for x in range(N, 0, -1):
                if x in temp_set or x == removed:
                    continue
                if _can_add(x, temp_set, temp):
                    new_set = temp + [x]
                    if len(new_set) > best_size:
                        best = new_set
                        best_set = set(best)
                        best_size = len(best)
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-swap: remove two, add two
        if len(best) >= 2:
            for i in range(len(best) - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best)):
                    if time.time() > deadline:
                        break
                    removed1 = best[i]
                    removed2 = best[j]
                    temp = [best[k] for k in range(len(best)) if k != i and k != j]
                    temp_set = set(temp)
                    
                    # Try adding two candidates
                    added = []
                    for x in range(N, 0, -1):
                        if len(added) >= 2:
                            break
                        if x in temp_set or x == removed1 or x == removed2:
                            continue
                        if _can_add(x, temp_set, temp + added):
                            added.append(x)
                    
                    if len(added) == 2:
                        new_set = temp + added
                        if len(new_set) > best_size:
                            best = new_set
                            best_set = set(best)
                            best_size = len(best)
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return best


def _random_order(seed):
    """Generate a randomized order for greedy."""
    rng = random.Random(seed)
    order = list(range(N, 0, -1))
    rng.shuffle(order)
    return order


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best = []
    best_size = 0
    
    # Phase 1: Multiple greedy starts with different orderings
    # Descending order
    order_desc = list(range(N, 0, -1))
    cand = _greedy_construct(order_desc, deadline)
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)
    
    # Primes first
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(N, 0, -1) if is_prime(x)]
    non_primes = [x for x in range(N, 0, -1) if not is_prime(x)]
    order_primes = primes + non_primes
    cand = _greedy_construct(order_primes, deadline)
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)
    
    # Randomized greedy starts
    for trial in range(30):
        if time.time() > deadline - 10.0:
            break
        order = _random_order(trial * 7 + 1)
        cand = _greedy_construct(order, deadline)
        if len(cand) > best_size:
            best = cand
            best_size = len(cand)
    
    # Phase 2: Local search on best
    if time.time() < deadline - 5.0:
        improved = _local_search(best, deadline - 2.0)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Phase 3: More local search iterations
    if time.time() < deadline - 2.0:
        improved = _local_search(best, deadline - 1.0)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Final cleanup
    if not best:
        best = [N]
    
    best = list(set(best))
    best = [x for x in best if 1 <= x <= N]
    best.sort(reverse=True)
    
    if len(best) > MAX_SIZE:
        best = best[:MAX_SIZE]
    
    return best
# EVOLVE-BLOCK-END
