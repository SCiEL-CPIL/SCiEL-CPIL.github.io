"""Construct a set of 20 distinct positive integers minimizing max(|A+A|, |A*A|)."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def _objective(A):
    """
    Compute max(|A+A|, |A*A|) using vectorized operations.
    A is a list of 20 distinct positive integers.
    """
    arr = np.array(A, dtype=np.int64)
    i_idx, j_idx = np.triu_indices(20)
    sums = arr[i_idx] + arr[j_idx]
    prods = arr[i_idx] * arr[j_idx]
    return max(len(np.unique(sums)), len(np.unique(prods)))

def _objective_fast(A):
    """
    Compute max(|A+A|, |A*A|) using set operations (Python sets).
    A is a list of 20 distinct positive integers.
    """
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _is_valid(A):
    """Check 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _generate_structured_candidates():
    """
    Generate initial candidates from structured sets to exploit collisions.
    Focus on unions of arithmetic and geometric progressions with thorough parameter search.
    """
    candidates = []
    seen = set()
    
    def add_candidate(lst):
        if len(lst) != 20 or len(set(lst)) != 20:
            return
        if any(x < 1 or x > 10000 for x in lst):
            return
        key = tuple(sorted(lst))
        if key not in seen:
            seen.add(key)
            candidates.append(sorted(lst))
    
    # Strategy 1: Union of arithmetic progression (length n1) and geometric progression (length n2)
    # More exhaustive parameter ranges
    for n1 in range(2, 19):
        n2 = 20 - n1
        if n2 < 1:
            continue
        # Arithmetic part: a + i*d for i in range(n1)
        for a in range(1, 50):
            for d in range(1, 20):
                arith = [a + i*d for i in range(n1)]
                if max(arith) > 10000:
                    break
                # Geometric part: b * r^j for j in range(n2)
                for b in range(1, 50):
                    for r in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
                        geo = []
                        val = b
                        for j in range(n2):
                            if val > 10000:
                                break
                            geo.append(val)
                            val *= r
                        if len(geo) != n2:
                            continue
                        combined = arith + geo
                        if len(set(combined)) == 20:
                            add_candidate(combined)
    
    # Strategy 2: Two geometric progressions with different bases
    for base1 in [2, 3, 4, 5, 6, 7]:
        for base2 in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
            if base1 >= base2:
                continue
            s1 = set()
            val = 1
            for _ in range(20):
                if val > 10000:
                    break
                s1.add(val)
                val *= base1
            s2 = set()
            val = 1
            for _ in range(20):
                if val > 10000:
                    break
                s2.add(val)
                val *= base2
            union = sorted(s1 | s2)
            if len(union) >= 20:
                add_candidate(union[:20])
                if len(union) > 20:
                    for start in range(0, len(union) - 20 + 1, 3):
                        add_candidate(union[start:start+20])
    
    # Strategy 3: Powers of 2 and 3 mixed
    s2 = set()
    val = 1
    for _ in range(20):
        if val > 10000:
            break
        s2.add(val)
        val *= 2
    s3 = set()
    val = 1
    for _ in range(20):
        if val > 10000:
            break
        s3.add(val)
        val *= 3
    union23 = sorted(s2 | s3)
    if len(union23) >= 20:
        add_candidate(union23[:20])
    
    # Strategy 4: Arithmetic progression only - wider range
    for a in range(1, 50):
        for d in range(1, 25):
            seq = [a + i*d for i in range(20)]
            if max(seq) <= 10000:
                add_candidate(seq)
    
    # Strategy 5: Geometric progression only - wider range
    for a in range(1, 30):
        for r in range(2, 20):
            seq = []
            val = a
            for _ in range(20):
                if val > 10000:
                    break
                seq.append(val)
                val *= r
            if len(seq) == 20:
                add_candidate(seq)
    
    # Strategy 6: Quadratic sequences - wider range
    for a in range(1, 12):
        for b in range(1, 12):
            for c in range(1, 12):
                seq = [a*i*i + b*i + c for i in range(20)]
                if max(seq) <= 10000 and min(seq) >= 1:
                    add_candidate(seq)
    
    # Strategy 7: Two arithmetic progressions with different steps
    for n1 in range(5, 16):
        n2 = 20 - n1
        for a1 in range(1, 30):
            for d1 in range(1, 10):
                arith1 = [a1 + i*d1 for i in range(n1)]
                if max(arith1) > 10000:
                    break
                for a2 in range(1, 30):
                    for d2 in range(1, 10):
                        if d1 == d2:
                            continue
                        arith2 = [a2 + i*d2 for i in range(n2)]
                        if max(arith2) > 10000:
                            continue
                        combined = arith1 + arith2
                        if len(set(combined)) == 20:
                            add_candidate(combined)
    
    # Strategy 8: Random structured candidates - more iterations
    for _ in range(200):
        n1 = random.randint(3, 17)
        n2 = 20 - n1
        a = random.randint(1, 40)
        d = random.randint(1, 15)
        arith = [a + i*d for i in range(n1)]
        if max(arith) > 10000:
            continue
        b = random.randint(1, 40)
        r = random.choice([2, 3, 4, 5, 6, 7, 8, 10, 12])
        geo = []
        val = b
        for j in range(n2):
            if val > 10000:
                break
            geo.append(val)
            val *= r
        if len(geo) == n2 and len(set(arith + geo)) == 20:
            add_candidate(arith + geo)
    
    return candidates

def _nudge_move(A):
    """Nudge one element by a small amount."""
    idx = random.randrange(20)
    delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5, 10, -10])
    new_val = A[idx] + delta
    if new_val < 1 or new_val > 10000:
        return None
    if new_val in A:
        return None
    A[idx] = new_val
    return sorted(A)

def _swap_move(A):
    """Swap one element with a random new value, biased towards small numbers."""
    idx = random.randrange(20)
    old = A[idx]
    while True:
        if random.random() < 0.7:
            new = random.randint(1, 50)
        else:
            new = random.randint(1, 10000)
        if new != old and new not in A:
            break
    A[idx] = new
    return sorted(A)

def _anneal(A, start_temp, end_temp, steps, deadline):
    """Simulated annealing with nudge and swap moves, adaptive temperature."""
    current = list(A)
    current_cost = _objective_fast(current)
    best = list(current)
    best_cost = current_cost
    temp = start_temp
    no_improve_count = 0
    
    for step in range(steps):
        if time.time() > deadline - 1:
            break
        r = random.random()
        if r < 0.55:
            new_A = _nudge_move(list(current))
            if new_A is None:
                continue
        elif r < 0.85:
            new_A = _swap_move(list(current))
        else:
            # Larger move: replace 2 elements
            new_A = list(current)
            for _ in range(2):
                idx = random.randrange(20)
                old = new_A[idx]
                while True:
                    if random.random() < 0.7:
                        new_val = random.randint(1, 60)
                    else:
                        new_val = random.randint(1, 10000)
                    if new_val != old and new_val not in new_A:
                        break
                new_A[idx] = new_val
            new_A = sorted(new_A)
        
        new_cost = _objective_fast(new_A)
        delta = new_cost - current_cost
        
        if delta < 0:
            accept = True
            no_improve_count = 0
        else:
            if temp > 1e-9:
                prob = math.exp(-delta / temp)
            else:
                prob = 0.0
            accept = random.random() < prob
            if accept:
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        if accept:
            current = new_A
            current_cost = new_cost
            if new_cost < best_cost:
                best = list(current)
                best_cost = new_cost
        
        # Adaptive cooling: cool faster if stuck
        if no_improve_count > 50:
            temp *= 0.95
        else:
            temp = start_temp * (end_temp / start_temp) ** (step / max(steps - 1, 1))
    
    return best, best_cost

def _local_refinement(A, deadline):
    """
    Local refinement: try replacing each element with nearby values to reduce objective.
    Uses a broader neighborhood and more aggressive search.
    """
    current = list(A)
    current_cost = _objective_fast(current)
    best = list(current)
    best_cost = current_cost
    
    improved = True
    iter_count = 0
    max_iters = 100
    while improved and time.time() < deadline - 2 and iter_count < max_iters:
        improved = False
        iter_count += 1
        indices = list(range(20))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline - 2:
                break
            old = current[idx]
            tried = set()
            # Try a broader range of replacements
            deltas = list(range(-60, 61, 1))
            random.shuffle(deltas)
            # Also add some larger jumps
            deltas.extend([random.randint(-100, 100) for _ in range(20)])
            for delta in deltas:
                if time.time() > deadline - 2:
                    break
                new_val = old + delta
                if new_val < 1 or new_val > 10000 or new_val in current or new_val in tried:
                    continue
                tried.add(new_val)
                current[idx] = new_val
                new_cost = _objective_fast(current)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = list(current)
                    break
                else:
                    current[idx] = old
            # Random replacement - more attempts
            for _ in range(5):
                if time.time() > deadline - 2:
                    break
                # Bias towards small numbers
                if random.random() < 0.6:
                    new_val = random.randint(1, 80)
                else:
                    new_val = random.randint(1, 10000)
                if new_val in current or new_val in tried:
                    continue
                tried.add(new_val)
                current[idx] = new_val
                new_cost = _objective_fast(current)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = list(current)
                    break
                else:
                    current[idx] = old
    
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    random.seed(42)
    
    # Phase 1: Generate structured initial solutions
    best_set = None
    best_cost = float('inf')
    
    candidates = _generate_structured_candidates()
    for A in candidates:
        if time.time() > deadline - 15:
            break
        if not _is_valid(A):
            continue
        cost = _objective_fast(A)
        if cost < best_cost:
            best_cost = cost
            best_set = list(A)
            
    if best_set is None:
        best_set = list(range(1, 21))
        best_cost = _objective_fast(best_set)

    # Phase 2: Multiple simulated annealing runs
    remaining = deadline - time.time()
    if remaining > 15:
        # Run annealing with multiple restarts
        num_restarts = min(10, max(3, int(remaining / 10)))
        for restart in range(num_restarts):
            if time.time() > deadline - 8:
                break
            if restart == 0:
                start_A = list(best_set)
            else:
                # Perturb best solution with varying strength
                start_A = list(best_set)
                num_perturb = random.randint(2, 6)
                for _ in range(num_perturb):
                    idx = random.randrange(20)
                    old = start_A[idx]
                    while True:
                        if random.random() < 0.6:
                            new_val = random.randint(1, 60)
                        else:
                            new_val = random.randint(1, 10000)
                        if new_val != old and new_val not in start_A:
                            break
                    start_A[idx] = new_val
                start_A = sorted(start_A)
            
            start_temp = 20.0
            end_temp = 0.01
            steps = 4000
            result, cost = _anneal(start_A, start_temp, end_temp, steps, deadline)
            if cost < best_cost:
                best_cost = cost
                best_set = list(result)
        
        # Phase 3: Local refinement - multiple passes
        if time.time() < deadline - 5:
            for _ in range(3):
                if time.time() < deadline - 3:
                    result, cost = _local_refinement(best_set, deadline)
                    if cost < best_cost:
                        best_cost = cost
                        best_set = result
                    else:
                        break
    
    return best_set

# EVOLVE-BLOCK-END
