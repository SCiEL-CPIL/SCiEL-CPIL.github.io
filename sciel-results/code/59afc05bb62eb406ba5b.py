import time
import random
import math

def _score(s):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d are in s."""
    diffs = set()
    elements = sorted(s)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s:
                diffs.add(d)
    return len(diffs)

def _score_with_diffs(s):
    """Returns (score, diffs_set) for faster delta computation."""
    diffs = set()
    elements = sorted(s)
    n = len(elements)
    for i in range(n):
        a = elements[i]
        for j in range(i + 1, n):
            b = elements[j]
            d = b - a
            c = b + d
            if c in s:
                diffs.add(d)
    return len(diffs), diffs

def _find_missing_diffs(s, max_d=200):
    """Find which small differences are missing from the set's progression differences."""
    _, diffs = _score_with_diffs(s)
    missing = []
    for d in range(1, max_d + 1):
        if d not in diffs:
            missing.append(d)
    return missing

def _targeted_move(current_set, missing_diffs):
    """Try to add an element that creates a new progression for a missing difference."""
    s_list = sorted(current_set)
    best_in = None
    best_gain = 0
    
    # For each missing difference, try to complete a progression
    for d in missing_diffs[:20]:  # Limit to top 20 missing diffs
        # Find pairs (a, a+d) in set where a+2d is not in set
        for a in s_list:
            if a + d in current_set and a + 2*d not in current_set and a + 2*d <= 512:
                # Adding a+2d would create this progression
                candidate = a + 2*d
                if candidate not in current_set:
                    # Check if this addition also helps other diffs
                    gain = 1  # At least one new diff
                    # Quick check: does it create other progressions?
                    for other_d in missing_diffs[:10]:
                        if other_d != d:
                            # Check if candidate forms a progression with existing elements
                            for b in s_list:
                                if b + other_d in current_set and b + 2*other_d == candidate:
                                    gain += 1
                                    break
                                if b - other_d in current_set and b - 2*other_d == candidate:
                                    gain += 1
                                    break
                    if gain > best_gain:
                        best_gain = gain
                        best_in = candidate
                        break
        if best_in is not None:
            break
    
    if best_in is not None:
        return best_in
    
    # Fallback: random
    return random.randint(1, 512)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_set = None
    best_score = 0
    
    # Generate diverse initial sets
    initial_sets = []
    
    # 1. Contiguous (Seed)
    initial_sets.append(set(range(1, 33)))
    
    # 2. Random
    for _ in range(10):
        initial_sets.append(set(random.sample(range(1, 513), 32)))
        
    # 3. Arithmetic Progressions
    for d in [2, 3, 4, 5, 8, 10, 16]:
        max_start = 512 - 31 * d
        if max_start >= 1:
            a = random.randint(1, max_start)
            initial_sets.append(set(a + i * d for i in range(32)))
            
    # 4. Two Blocks
    for _ in range(5):
        s1_start = random.randint(1, 200)
        s2_start = random.randint(300, 480)
        s = set()
        try:
            block1 = set(random.sample(range(s1_start, s1_start + 50), 16))
            block2 = set(random.sample(range(s2_start, s2_start + 50), 16))
            s = block1 | block2
            if len(s) == 32:
                initial_sets.append(s)
        except:
            pass
    
    # 5. Structured sets: mix of small and large values
    for _ in range(5):
        s = set()
        # 16 small values (1-100)
        s.update(random.sample(range(1, 101), 16))
        # 16 large values (400-512)
        s.update(random.sample(range(400, 513), 16))
        if len(s) == 32:
            initial_sets.append(s)
    
    # 6. Geometric-like spacing
    for base in [1, 2, 3, 5]:
        s = set()
        for i in range(32):
            val = base + i * (i + 1) // 2  # Triangular numbers offset
            if val <= 512:
                s.add(val)
        if len(s) >= 32:
            s = set(list(s)[:32])
            if len(s) == 32:
                initial_sets.append(s)
    
    # Evaluate initial sets
    for s in initial_sets:
        if len(s) != 32:
            continue
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_set = s
            
    if best_set is None:
        best_set = set(range(1, 33))
        best_score = _score(best_set)
        
    # Simulated Annealing with targeted moves
    current_set = set(best_set)
    current_score = best_score
    current_diffs = _score_with_diffs(current_set)[1]
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.1
    stuck_count = 0
    max_stuck = 3000
    use_targeted = True
    
    while time.time() < deadline:
        # Generate move: mix of random and targeted
        if use_targeted and stuck_count < max_stuck // 2:
            missing = _find_missing_diffs(current_set, max_d=150)
            if missing:
                elem_in = _targeted_move(current_set, missing)
            else:
                elem_in = random.randint(1, 512)
        else:
            elem_in = random.randint(1, 512)
        
        if elem_in in current_set:
            continue
            
        # Choose element to remove
        # Weight toward removing elements that are "less useful"
        elem_out = random.choice(list(current_set))
        
        new_set = set(current_set)
        new_set.remove(elem_out)
        new_set.add(elem_in)
        
        new_score = _score(new_set)
        delta = new_score - current_score
        
        if delta > 0:
            current_set = new_set
            current_score = new_score
            current_diffs = _score_with_diffs(current_set)[1]
            stuck_count = 0
            if new_score > best_score:
                best_score = new_score
                best_set = new_set.copy()
        else:
            # Accept with probability exp(delta/T)
            if delta < -10:
                stuck_count += 1
            elif random.random() < math.exp(delta / T):
                current_set = new_set
                current_score = new_score
                current_diffs = _score_with_diffs(current_set)[1]
            else:
                stuck_count += 1
                
        T *= cooling
        if T < min_T:
            T = min_T
            
        if stuck_count > max_stuck:
            # Stronger perturbation: replace 8-12 elements
            num_replace = random.randint(8, 12)
            for _ in range(num_replace):
                eo = random.choice(list(current_set))
                ei = random.randint(1, 512)
                if ei not in current_set:
                    current_set.remove(eo)
                    current_set.add(ei)
            current_score = _score(current_set)
            current_diffs = _score_with_diffs(current_set)[1]
            stuck_count = 0
            T = max(T, 8.0)
            
    return list(best_set)
