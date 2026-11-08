import time
from math import log, floor, exp, max as py_max, min as py_min
import random
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def generate_mersenne_like(m, offset=0):
    """Generate U = {0} U {2^k - 1 + offset for k=1..m}"""
    U = [0]
    seen = {0}
    for k in range(1, m + 1):
        val = (1 << k) - 1 + offset
        if val not in seen:
            U.append(val)
            seen.add(val)
    return sorted(U)

def generate_fibonacci_like(m):
    """Generate U based on Fibonacci numbers"""
    U = [0, 1]
    a, b = 0, 1
    for _ in range(m - 1):
        a, b = b, a + b
        if b not in U:
            U.append(b)
    return sorted(U)

def generate_gapped_arithmetic(n, gap, start=0):
    """Generate U = {0, start+gap, start+2*gap, ...}"""
    U = [0]
    for i in range(1, n):
        U.append(start + i * gap)
    return sorted(U)

def local_search_sa(U, deadline, max_val_limit=2000, max_n_limit=40, start_temp=1.0, cool_rate=0.999):
    """
    Simulated Annealing local search.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    temp = start_temp
    no_improvement_count = 0
    max_no_improvement = 100
    
    U_set = set(U)
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        temp *= cool_rate
        if temp < 0.01:
            break
            
        improved = False
        move_type = random.random()
        
        # Candidate generation is implicit in the moves
        
        if move_type < 0.4:
            # Try to ADD an element
            if len(U) < max_n_limit:
                # Pick a random candidate in a reasonable range
                # Range based on current max
                current_max = max(U)
                # Try adding near existing elements or in gaps
                # Simple strategy: random value in [1, max_val_limit]
                # Or better: random sum/diff of existing elements
                candidates = []
                for a in U:
                    for b in U:
                        s = a + b
                        d = abs(a - b)
                        if 0 < s <= max_val_limit and s not in U_set:
                            candidates.append(s)
                        if 0 < d <= max_val_limit and d not in U_set:
                            candidates.append(d)
                
                if candidates:
                    c = random.choice(candidates)
                    new_U = U + [c]
                    new_U.sort()
                    new_score = score_set(new_U)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < exp(delta / temp):
                        U = new_U
                        current_score = new_score
                        U_set = set(U)
                        improved = True
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_set = list(U)
                            
        elif move_type < 0.7:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < exp(delta / temp):
                        U = new_U
                        current_score = new_score
                        U_set = set(U)
                        improved = True
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_set = list(U)
        else:
            # Try to REPLACE an element
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Generate a new value
                # Perturb old_val or pick random
                if random.random() < 0.5:
                    new_val = old_val + random.randint(-10, 10)
                else:
                    new_val = random.randint(1, max_val_limit)
                    
                if new_val <= 0 or new_val in U_set:
                    continue
                    
                new_U = U.copy()
                new_U[idx] = new_val
                new_U.sort()
                # Check for duplicates
                if len(set(new_U)) != len(new_U):
                    continue
                    
                new_score = score_set(new_U)
                delta = new_score - current_score
                if delta > 0 or random.random() < exp(delta / temp):
                    U = new_U
                    current_score = new_score
                    U_set = set(U)
                    improved = True
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local_set = list(U)
            
        if improved:
            no_improvement_count = 0
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    best_set = [0, 1, 3]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 2, 3, 5, 8, 13, 21], # Fibonacci
        [0, 1, 3, 6, 10, 15, 21, 28], # Triangular (shifted)
        [0, 1, 2, 5, 10, 17, 26, 37], # Quadratic
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Exhaustive Search for Small n
    from itertools import combinations
    
    def exhaustive_search(n, max_val):
        """Find best set of size n (including 0) with elements <= max_val."""
        best_s = -1
        best_c = None
        range_vals = range(1, max_val + 1)
        for comb in combinations(range_vals, n - 1):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_s:
                best_s = s
                best_c = U
        return best_s, best_c

    if time.time() < deadline - 50:
        # n=3
        s, c = exhaustive_search(3, 50)
        if s > best_score:
            best_score = s
            best_set = c
        
        # n=4
        s, c = exhaustive_search(4, 40)
        if s > best_score:
            best_score = s
            best_set = c
            
        # n=5
        s, c = exhaustive_search(5, 30)
        if s > best_score:
            best_score = s
            best_set = c

    # 3. Mersenne-like Search
    # Try different lengths and offsets
    for m in range(3, 15):
        for offset in range(0, 5):
            U = generate_mersenne_like(m, offset)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    # 4. Simulated Annealing Local Search
    seeds = [best_set]
    for cand in structured_candidates[:5]:
        if cand != best_set:
            seeds.append(cand)
    
    max_val_limit = 500
    max_n_limit = 30
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Cycle through seeds
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        # Run local search
        local_score, local_set = local_search_sa(U, deadline, max_val_limit, max_n_limit)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            # Add the new best to seeds for future restarts
            if len(seeds) < 10:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 5. Final Polish: Fine-grained local search on the best set
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 500:
            improved = False
            iteration += 1
            n = len(current)
            # Try replacing each element with nearby values (small delta)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                # Try deltas in a wider range but smaller steps
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            # Try adding an element near the max
            if not improved and n < 25 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 5, max_val + 6):
                    if new_val <= 0:
                        continue
                    if new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            # Try removing an element
            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    # Quick test
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
