import time
import random
import math

def _objective(A):
    """Compute max(|A+A|, |A*A|)."""
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
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _generate_5_smooth(limit):
    """Generate all 5-smooth numbers up to limit."""
    smooths = set([1])
    powers2 = [1]
    powers3 = [1]
    powers5 = [1]
    while powers2[-1] * 2 <= limit:
        powers2.append(powers2[-1] * 2)
    while powers3[-1] * 3 <= limit:
        powers3.append(powers3[-1] * 3)
    while powers5[-1] * 5 <= limit:
        powers5.append(powers5[-1] * 5)
    for p2 in powers2:
        for p3 in powers3:
            val = p2 * p3
            if val > limit:
                break
            for p5 in powers5:
                prod = val * p5
                if prod > limit:
                    break
                smooths.add(prod)
    return sorted(smooths)

def _generate_structured_candidates():
    """
    Generate diverse structured candidates:
    - Windows of 5-smooth numbers
    - Arithmetic progressions
    - Geometric progressions
    - Union of arithmetic and geometric progressions (key strategy)
    - Mixed structures
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
    
    # 5-smooth windows
    smooths = _generate_5_smooth(10000)
    for i in range(len(smooths) - 20 + 1):
        window = smooths[i:i+20]
        add_candidate(window)
    
    # Pure arithmetic progressions
    for start in range(1, 20):
        for step in range(1, 25):
            ap = [start + i * step for i in range(20)]
            if max(ap) <= 10000:
                add_candidate(ap)
    
    # Pure geometric progressions
    for a in range(1, 15):
        for r in range(2, 20):
            geo = []
            val = a
            for _ in range(20):
                if val > 10000:
                    break
                geo.append(val)
                val *= r
            if len(geo) == 20:
                add_candidate(geo)
    
    # KEY STRATEGY: Union of arithmetic and geometric progressions
    # This maximizes collisions in both A+A and A*A
    for n_arith in range(3, 18):
        n_geo = 20 - n_arith
        for a in range(1, 12):
            for d in range(1, 12):
                arith = [a + i * d for i in range(n_arith)]
                if max(arith) > 10000:
                    continue
                for b in range(1, 12):
                    for r in range(2, 15):
                        geo = []
                        val = b
                        for _ in range(n_geo):
                            if val > 10000:
                                break
                            geo.append(val)
                            val *= r
                        if len(geo) != n_geo:
                            continue
                        # Check no overlap
                        combined = arith + geo
                        if len(set(combined)) == 20:
                            add_candidate(combined)
    
    # Mixed: combine small arithmetic progression with larger geometric
    for small_n in [4, 5, 6, 8, 10]:
        for start in range(1, 8):
            for step in range(1, 6):
                ap_part = [start + i * step for i in range(small_n)]
                if max(ap_part) > 10000:
                    continue
                remaining = 20 - small_n
                for geo_start in range(1, 15):
                    for ratio in range(2, 10):
                        geo_part = []
                        val = geo_start
                        for _ in range(remaining):
                            if val > 10000:
                                break
                            geo_part.append(val)
                            val *= ratio
                        if len(geo_part) == remaining:
                            combined = ap_part + geo_part
                            if len(set(combined)) == 20:
                                add_candidate(combined)
    
    # Quadratic sequences
    for a in range(1, 8):
        for b in range(1, 8):
            for c in range(1, 8):
                seq = [a*i*i + b*i + c for i in range(20)]
                if max(seq) <= 10000 and min(seq) >= 1 and len(set(seq)) == 20:
                    add_candidate(seq)
    
    # Powers of small bases (mixed)
    for base1 in [2, 3, 4, 5]:
        for base2 in [2, 3, 4, 5, 6, 7]:
            if base1 == base2:
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
                for start in range(0, max(1, len(union) - 20), 3):
                    add_candidate(union[start:start+20])
    
    return candidates

def _local_search(A, smooths, deadline):
    """
    Local search restricting moves to 5-smooth numbers.
    Supports single-element replacement and swap moves.
    """
    current = list(A)
    current_cost = _objective(current)
    best = list(current)
    best_cost = current_cost
    
    valid_smooths = [x for x in smooths if x <= 10000]
    valid_smooth_set = set(valid_smooths)
    
    improved = True
    max_passes = 80
    passes = 0
    while improved and passes < max_passes and time.time() < deadline - 3:
        improved = False
        passes += 1
        indices = list(range(20))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 3:
                break
            old = current[idx]
            
            # Try replacing with other 5-smooth numbers
            candidates = random.sample(valid_smooths, min(80, len(valid_smooths)))
            found_improvement = False
            for new_val in candidates:
                if new_val == old or new_val in current:
                    continue
                current[idx] = new_val
                new_cost = _objective(current)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best = list(current)
                    found_improvement = True
                    break
                else:
                    current[idx] = old
            if found_improvement:
                continue
            
            # Try swap moves
            for idx2 in range(idx + 1, 20):
                if time.time() > deadline - 3:
                    break
                old2 = current[idx2]
                if random.random() < 0.4:
                    c1 = random.choice(valid_smooths)
                    c2 = random.choice(valid_smooths)
                    if c1 in current or c2 in current or c1 == c2 or c1 == old or c1 == old2 or c2 == old or c2 == old2:
                        continue
                    current[idx] = c1
                    current[idx2] = c2
                    new_cost = _objective(current)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best = list(current)
                    else:
                        current[idx] = old
                        current[idx2] = old2
    
    return best, best_cost

def _anneal(A, start_temp, end_temp, steps, deadline):
    """Simulated annealing with nudge and swap moves."""
    current = list(A)
    current_cost = _objective(current)
    best = list(current)
    best_cost = current_cost
    temp = start_temp
    
    for step in range(steps):
        if time.time() > deadline:
            break
        r = random.random()
        if r < 0.6:
            # Nudge move
            idx = random.randrange(20)
            delta = random.choice([-5, -3, -2, -1, 1, 2, 3, 5, 10, -10])
            new_val = current[idx] + delta
            if new_val < 1 or new_val > 10000 or new_val in current:
                continue
            current[idx] = new_val
            new_A = sorted(current)
        else:
            # Swap move
            idx = random.randrange(20)
            old = current[idx]
            while True:
                if random.random() < 0.7:
                    new = random.randint(1, 50)
                else:
                    new = random.randint(1, 10000)
                if new != old and new not in current:
                    break
            current[idx] = new
            new_A = sorted(current)
        
        new_cost = _objective(new_A)
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
    
    smooths = _generate_5_smooth(10000)
    
    best_set = None
    best_cost = float('inf')
    
    # Phase 1: Generate and evaluate structured candidates
    candidates = _generate_structured_candidates()
    
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        if not _is_valid(cand):
            continue
        cost = _objective(cand)
        if cost < best_cost:
            best_cost = cost
            best_set = list(cand)
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_cost = _objective(best_set)
    
    # Phase 2: Local search from best structured candidate
    remaining = deadline - time.time()
    if remaining > 15:
        result, cost = _local_search(best_set, smooths, deadline)
        if cost < best_cost:
            best_cost = cost
            best_set = result
    
    # Phase 3: Simulated annealing with multiple restarts
    remaining = deadline - time.time()
    if remaining > 10:
        num_restarts = min(4, int(remaining / 8))
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            if restart == 0:
                start_A = list(best_set)
            else:
                # Perturb best solution
                start_A = list(best_set)
                num_swaps = random.randint(2, 5)
                for _ in range(num_swaps):
                    idx = random.randrange(20)
                    new_val = random.choice(smooths)
                    if new_val not in start_A:
                        start_A[idx] = new_val
                start_A = sorted(start_A)
                if not _is_valid(start_A):
                    continue
            
            result, cost = _anneal(start_A, 10.0, 0.01, 200, deadline - 2)
            if cost < best_cost:
                best_cost = cost
                best_set = result
    
    # Phase 4: Final local search polish
    remaining = deadline - time.time()
    if remaining > 5:
        result, cost = _local_search(best_set, smooths, deadline)
        if cost < best_cost:
            best_cost = cost
            best_set = result
    
    return best_set
