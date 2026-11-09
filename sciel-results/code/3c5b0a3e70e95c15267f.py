"""Construct a set of 20 distinct positive integers minimizing max(|A+A|, |A*A|)."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

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
    Generate initial candidates from structured sets: unions of arithmetic
    and geometric progressions with overlapping terms to maximize collisions.
    Direct parameter enumeration of start, step, ratio, and lengths.
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
    
    # Strategy 1: Union of AP (length n1) and GP (length n2) with overlap
    # AP: a + i*d for i in range(n1)
    # GP: b * r^j for j in range(n2)
    # Allow overlap: some terms can be shared
    for n1 in range(4, 17):
        n2 = 20 - n1 + 1  # +1 to allow for overlap
        if n2 < 2:
            continue
        for a in range(1, 25):
            for d in range(1, 15):
                arith = [a + i * d for i in range(n1)]
                if max(arith) > 10000 or min(arith) < 1:
                    break
                for b in range(1, 25):
                    for r in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
                        geo = []
                        val = b
                        for j in range(n2):
                            if val > 10000:
                                break
                            geo.append(val)
                            val *= r
                        if len(geo) < n2:
                            continue
                        combined = list(set(arith + geo))
                        if len(combined) >= 20:
                            # Take first 20 sorted
                            add_candidate(sorted(combined)[:20])
                        elif len(combined) == 20:
                            add_candidate(combined)
    
    # Strategy 2: Two geometric progressions
    for base1 in [1, 2, 3]:
        for r1 in [2, 3, 4, 5, 6]:
            for base2 in [1, 2, 3]:
                for r2 in [2, 3, 4, 5, 6, 7, 8, 10]:
                    if r1 == r2 and base1 == base2:
                        continue
                    s1 = set()
                    val = base1
                    for _ in range(20):
                        if val > 10000:
                            break
                        s1.add(val)
                        val *= r1
                    s2 = set()
                    val = base2
                    for _ in range(20):
                        if val > 10000:
                            break
                        s2.add(val)
                        val *= r2
                    union = sorted(s1 | s2)
                    if len(union) >= 20:
                        add_candidate(union[:20])
                        if len(union) > 20:
                            for start in range(0, min(5, len(union) - 20 + 1)):
                                add_candidate(union[start:start+20])
    
    # Strategy 3: Powers of 2 and 3
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
    
    # Strategy 4: Arithmetic progression only
    for a in range(1, 25):
        for d in range(1, 12):
            seq = [a + i * d for i in range(20)]
            if max(seq) <= 10000 and min(seq) >= 1:
                add_candidate(seq)
    
    # Strategy 5: Geometric progression only
    for a in range(1, 15):
        for r in range(2, 15):
            seq = []
            val = a
            for _ in range(20):
                if val > 10000:
                    break
                seq.append(val)
                val *= r
            if len(seq) == 20:
                add_candidate(seq)
    
    # Strategy 6: Quadratic sequences
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                seq = [a * i * i + b * i + c for i in range(20)]
                if max(seq) <= 10000 and min(seq) >= 1 and len(set(seq)) == 20:
                    add_candidate(seq)
    
    # Strategy 7: Random structured candidates with AP+GP
    for _ in range(200):
        n1 = random.randint(6, 14)
        n2 = 20 - n1
        a = random.randint(1, 20)
        d = random.randint(1, 10)
        arith = [a + i * d for i in range(n1)]
        if max(arith) > 10000 or min(arith) < 1:
            continue
        b = random.randint(1, 20)
        r = random.choice([2, 3, 4, 5, 6, 7, 8, 10])
        geo = []
        val = b
        for j in range(n2):
            if val > 10000:
                break
            geo.append(val)
            val *= r
        if len(geo) == n2:
            combined = arith + geo
            if len(set(combined)) == 20:
                add_candidate(combined)
    
    return candidates

def _exhaustive_local_refinement(A, deadline):
    """
    Exhaustive deterministic single-element replacement search.
    For each element, try ALL valid replacements in a targeted range,
    keep the best improvement. Repeat until no improvement or deadline.
    This escapes local minima that stochastic SA gets stuck in.
    """
    current = list(A)
    current_cost = _objective_fast(current)
    best = list(current)
    best_cost = current_cost
    
    improved = True
    iter_count = 0
    while improved and time.time() < deadline - 2 and iter_count < 80:
        improved = False
        iter_count += 1
        indices = list(range(20))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 2:
                break
            old = current[idx]
            old_set = set(current)
            best_for_idx = old
            best_cost_for_idx = current_cost
            
            # Build a targeted list of candidate values to try
            # Near old value, plus small values, plus some random
            candidates_to_try = set()
            # Near old
            for delta in range(-40, 41):
                nv = old + delta
                if 1 <= nv <= 10000 and nv != old:
                    candidates_to_try.add(nv)
            # Small values (1-100)
            for nv in range(1, 101):
                if nv != old:
                    candidates_to_try.add(nv)
            # Some medium values
            for nv in [100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000]:
                if nv != old:
                    candidates_to_try.add(nv)
            # Random samples
            for _ in range(30):
                nv = random.randint(1, 10000)
                if nv != old:
                    candidates_to_try.add(nv)
            
            # Filter out values already in set
            valid_candidates = [nv for nv in candidates_to_try if nv not in old_set]
            
            for new_val in valid_candidates:
                if time.time() > deadline - 2:
                    break
                current[idx] = new_val
                new_cost = _objective_fast(current)
                if new_cost < best_cost_for_idx:
                    best_cost_for_idx = new_cost
                    best_for_idx = new_val
                current[idx] = old
            
            # Apply best improvement for this index
            if best_for_idx != old:
                current[idx] = best_for_idx
                old_set.discard(old)
                old_set.add(best_for_idx)
                current_cost = best_cost_for_idx
                improved = True
                if new_cost < best_cost:
                    best_cost = current_cost
                    best = list(current)
    
    return best, best_cost

def _anneal(A, start_temp, end_temp, steps, deadline):
    """Simulated annealing with nudge and swap moves."""
    current = list(A)
    current_cost = _objective_fast(current)
    best = list(current)
    best_cost = current_cost
    temp = start_temp
    
    for step in range(steps):
        if time.time() > deadline:
            break
        r = random.random()
        if r < 0.6:
            # Nudge
            idx = random.randrange(20)
            delta = random.choice([-10, -7, -5, -3, -2, -1, 1, 2, 3, 5, 7, 10])
            new_val = current[idx] + delta
            if new_val < 1 or new_val > 10000 or new_val in current:
                continue
            new_A = list(current)
            new_A[idx] = new_val
        else:
            # Swap with random
            idx = random.randrange(20)
            old = current[idx]
            while True:
                if random.random() < 0.7:
                    new = random.randint(1, 100)
                else:
                    new = random.randint(1, 10000)
                if new != old and new not in current:
                    break
            new_A = list(current)
            new_A[idx] = new
        
        new_cost = _objective_fast(new_A)
        delta = new_cost - current_cost
        
        if delta < 0:
            accept = True
        else:
            if temp > 1e-9:
                prob = math.exp(-delta / temp)
            else:
                prob = 0.0
            accept = random.random() < prob
            
        if accept:
            current = new_A
            current_cost = new_cost
            if new_cost < best_cost:
                best = list(current)
                best_cost = new_cost
        
        temp = start_temp * (end_temp / start_temp) ** (step / max(steps - 1, 1))
    
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    random.seed(42)
    
    # Phase 1: Generate structured initial solutions via parameter enumeration
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

    # Phase 2: Simulated annealing to explore neighborhood
    remaining = deadline - time.time()
    if remaining > 15:
        num_restarts = min(5, max(1, int(remaining / 10)))
        for restart in range(num_restarts):
            if time.time() > deadline - 8:
                break
            if restart == 0:
                start_A = best_set
            else:
                # Perturb best solution
                start_A = list(best_set)
                for _ in range(random.randint(3, 6)):
                    idx = random.randrange(20)
                    old = start_A[idx]
                    while True:
                        new_val = random.randint(1, 10000)
                        if new_val != old and new_val not in start_A:
                            break
                    start_A[idx] = new_val
                start_A = sorted(start_A)
            
            start_temp = 12.0
            end_temp = 0.01
            steps = 2000
            result, cost = _anneal(start_A, start_temp, end_temp, steps, deadline)
            if cost < best_cost:
                best_cost = cost
                best_set = list(result)
        
        # Phase 3: Exhaustive deterministic local refinement (key improvement)
        if time.time() < deadline - 5:
            result, cost = _exhaustive_local_refinement(best_set, deadline)
            if cost < best_cost:
                best_cost = cost
                best_set = result
        
        # Phase 4: Another round of SA + refinement if time permits
        if time.time() < deadline - 10:
            # Small SA to escape
            start_A = list(best_set)
            for _ in range(2):
                idx = random.randrange(20)
                old = start_A[idx]
                while True:
                    new_val = random.randint(1, 10000)
                    if new_val != old and new_val not in start_A:
                        break
                start_A[idx] = new_val
            start_A = sorted(start_A)
            
            result, cost = _anneal(start_A, 8.0, 0.01, 1000, deadline)
            if cost < best_cost:
                best_cost = cost
                best_set = list(result)
            
            # Final exhaustive refinement
            if time.time() < deadline - 3:
                result, cost = _exhaustive_local_refinement(best_set, deadline)
                if cost < best_cost:
                    best_cost = cost
                    best_set = result
    
    return best_set

# EVOLVE-BLOCK-END
