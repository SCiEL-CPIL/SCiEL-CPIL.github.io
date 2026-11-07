"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import isqrt, gcd
from itertools import combinations

NMAX = 256

def _conflict_degree(x, S_list):
    """Count how many pairs (b,c) in S_list have x | b*c or b | x*c or c | x*b."""
    n = len(S_list)
    count = 0
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if (b * c) % x == 0:
                count += 1
            if (x * c) % b == 0:
                count += 1
            if (x * b) % c == 0:
                count += 1
    return count

def _valid_addition(x, S_list):
    """Check if x can be added to set S_list without violating the constraint."""
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if (b * c) % x == 0:
                return False
            if (x * c) % b == 0:
                return False
            if (x * b) % c == 0:
                return False
    return True

def _is_valid_set(S):
    """Verify that a set S satisfies the constraint for all triples."""
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(n):
            if i == j:
                continue
            b = S[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S[k]
                if (b * c) % a == 0:
                    return False
    return True

def _greedy_conflict_removal(deadline):
    """Start with all numbers 2..256, iteratively remove the element with highest conflict degree."""
    S = list(range(2, NMAX + 1))
    
    while time.time() < deadline:
        if len(S) <= 1:
            break
        
        # Compute conflict degree for each element
        degrees = []
        for i, x in enumerate(S):
            deg = 0
            for j in range(len(S)):
                if j == i:
                    continue
                b = S[j]
                for k in range(len(S)):
                    if k == i or k == j:
                        continue
                    c = S[k]
                    if (b * c) % x == 0:
                        deg += 1
                    if (x * c) % b == 0:
                        deg += 1
                    if (x * b) % c == 0:
                        deg += 1
            degrees.append(deg)
        
        if max(degrees) == 0:
            break
        
        # Remove the element with highest degree
        max_idx = degrees.index(max(degrees))
        S.pop(max_idx)
    
    return S

def _greedy_largest_first(deadline):
    """Greedy: try numbers from largest to smallest."""
    chosen = []
    chosen_set = set()
    
    for x in range(NMAX, 1, -1):
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _greedy_smallest_first(deadline):
    """Greedy: try numbers from smallest to largest."""
    chosen = []
    chosen_set = set()
    
    for x in range(2, NMAX + 1):
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _greedy_randomized(deadline, seed=42):
    """Greedy with randomized order."""
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    rng.shuffle(candidates)
    
    chosen = []
    chosen_set = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _greedy_prime_heavy(deadline, seed=42):
    """Greedy that heavily favors primes and prime powers."""
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    
    # Score each candidate: primes get high score
    def score(x):
        # Check if prime
        if x < 2:
            return 0
        for d in range(2, isqrt(x) + 1):
            if x % d == 0:
                # Count prime factors
                pf = 0
                n = x
                for d2 in range(2, isqrt(n) + 1):
                    if n % d2 == 0:
                        pf += 1
                        while n % d2 == 0:
                            n //= d2
                if n > 1:
                    pf += 1
                return max(0, 10 - pf * 3)
        return 10  # prime
    
    # Weighted selection
    weights = [score(x) + 1 for x in candidates]
    total_w = sum(weights)
    
    chosen = []
    chosen_set = set()
    remaining = candidates[:]
    remaining_weights = weights[:]
    
    while remaining:
        if time.time() > deadline:
            break
        r = rng.random() * sum(remaining_weights)
        cumulative = 0
        idx = 0
        for k in range(len(remaining)):
            cumulative += remaining_weights[k]
            if r <= cumulative:
                idx = k
                break
        x = remaining[idx]
        remaining.pop(idx)
        remaining_weights.pop(idx)
        
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _local_search(S, deadline, max_iters=5000):
    """Local search: try adding, removing, swapping elements."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(99999)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        op = rng.random()
        
        if op < 0.4 and len(S) < NMAX:
            # Try adding a random element
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                rng.shuffle(candidates)
                added = False
                for x in candidates[:30]:
                    if _valid_addition(x, S):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        added = True
                        break
                if added:
                    continue
        
        elif op < 0.7 and S:
            # Try swapping: remove one, add another
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                rng.shuffle(candidates)
                improved = False
                for x in candidates[:50]:
                    if _valid_addition(x, S_temp):
                        S = S_temp + [x]
                        S_set = set(S)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        improved = True
                        break
                if not improved:
                    S = S_temp + [removed]
                    S_set = set(S)
            else:
                S = S_temp + [removed]
                S_set = set(S)
        
        else:
            # Try 2-swap: remove two, add two
            if len(S) >= 2:
                i, j = rng.sample(range(len(S)), 2)
                i, j = min(i, j), max(i, j)
                S_temp = S[:i] + S[i+1:j] + S[j+1:]
                S_temp_set = set(S_temp)
                candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
                if candidates:
                    rng.shuffle(candidates)
                    improved = False
                    for x in candidates[:40]:
                        if not _valid_addition(x, S_temp):
                            continue
                        S_temp2 = S_temp + [x]
                        S_temp2_set = S_temp_set | {x}
                        for y in candidates[:40]:
                            if y == x or y in S_temp2_set:
                                continue
                            if _valid_addition(y, S_temp2):
                                S = S_temp2 + [y]
                                S_set = set(S)
                                if len(S) > best_size:
                                    best = S[:]
                                    best_size = len(S)
                                improved = True
                                break
                        if improved:
                            break
                    if not improved:
                        S = S_temp + [S[i], S[j]]
                        S_set = set(S)
    
    return best

def _simulated_annealing(S, deadline, max_iters=5000, T_start=50.0, T_end=0.5):
    """Simulated annealing for optimization."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(55555)
    
    current_temp = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        current_temp *= cooling
        
        op = rng.random()
        if op < 0.3 and len(S) < NMAX:
            # Add
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    continue
        elif op < 0.6 and S:
            # Remove and try to re-add
            i = rng.randrange(len(S))
            removed = S[i]
            S.pop(i)
            S_set.discard(removed)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                else:
                    S.append(removed)
                    S_set.add(removed)
        elif S:
            # Swap
            i = rng.randrange(len(S))
            removed = S[i]
            S.pop(i)
            S_set.discard(removed)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                else:
                    if current_temp > 0.1 and rng.random() < 0.05:
                        S.append(x)
                        S_set.add(x)
                    else:
                        S.append(removed)
                        S_set.add(removed)
    
    return best

def _conflict_removal_with_recovery(S, deadline):
    """Remove high-conflict elements, then try to recover by adding others."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(42424)
    
    # Identify and remove high-conflict elements
    for _ in range(20):
        if time.time() > deadline:
            break
        if len(S) <= 2:
            break
        
        # Find element with highest conflict
        max_conf = -1
        max_idx = -1
        for i, x in enumerate(S):
            conf = _conflict_degree(x, S)
            if conf > max_conf:
                max_conf = conf
                max_idx = i
        
        if max_conf <= 0:
            break
        
        # Remove it
        removed = S.pop(max_idx)
        S_set.discard(removed)
        
        # Try to add something else
        candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
        rng.shuffle(candidates)
        added = False
        for x in candidates[:50]:
            if _valid_addition(x, S):
                S.append(x)
                S_set.add(x)
                added = True
                break
        
        if len(S) > best_size:
            best = S[:]
            best_size = len(S)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best = None
    best_size = 0
    
    def update_best(c):
        nonlocal best, best_size
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Phase 1: Conflict removal heuristic (new strategy)
    if time.time() < deadline - 70:
        c = _greedy_conflict_removal(deadline - 70)
        update_best(c)
    
    # Phase 2: Multiple greedy initializations
    greedy_results = []
    
    if time.time() < deadline - 60:
        c = _greedy_largest_first(deadline - 60)
        greedy_results.append(c)
    
    if time.time() < deadline - 55:
        c = _greedy_smallest_first(deadline - 55)
        greedy_results.append(c)
    
    for seed in [1, 7, 13, 42, 99, 128, 200, 256, 313, 456]:
        if time.time() > deadline - 45:
            break
        c = _greedy_randomized(deadline - 45, seed=seed)
        greedy_results.append(c)
    
    for seed in [10, 20, 30]:
        if time.time() > deadline - 40:
            break
        c = _greedy_prime_heavy(deadline - 40, seed=seed)
        greedy_results.append(c)
    
    for c in greedy_results:
        update_best(c)
    
    if best is None:
        best = list(range(129, 257))
        best_size = len(best)
    
    # Phase 3: Local search on best
    if time.time() < deadline - 25:
        c = _local_search(best, deadline - 25, max_iters=5000)
        update_best(c)
    
    # Phase 4: Conflict removal with recovery
    if time.time() < deadline - 20:
        c = _conflict_removal_with_recovery(best, deadline - 20)
        update_best(c)
    
    # Phase 5: Simulated annealing
    if time.time() < deadline - 15:
        c = _simulated_annealing(best, deadline - 15, max_iters=5000)
        update_best(c)
    
    # Phase 6: More local search from perturbed states
    rng = random.Random(31337)
    for restart in range(15):
        if time.time() > deadline - 5:
            break
        
        S_perturbed = best[:]
        n_remove = rng.randint(1, min(5, len(S_perturbed)))
        indices = rng.sample(range(len(S_perturbed)), n_remove)
        indices.sort(reverse=True)
        for idx in indices:
            S_perturbed.pop(idx)
        
        c = _local_search(S_perturbed, deadline - 5, max_iters=1000)
        update_best(c)
    
    # Phase 7: Final local search
    if time.time() < deadline - 2:
        c = _local_search(best, deadline - 2, max_iters=3000)
        update_best(c)
    
    # Verify and return
    if best is None:
        best = [2]
    
    # Ensure all are Python ints
    result = [int(x) for x in best]
    
    # Final validity check
    if not _is_valid_set(result):
        # Fallback: use primes
        primes = [x for x in range(2, NMAX + 1) if all(x % d != 0 for d in range(2, isqrt(x) + 1))]
        result = primes
    
    return result
