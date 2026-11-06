import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 82  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_arr = np.array(A, dtype=np.int64)
    sums = A_arr[:, None] + A_arr[None, :]
    num_sums = len(np.unique(sums))
    products = A_arr[:, None] * A_arr[None, :]
    num_products = len(np.unique(products))
    return max(num_sums, num_products)

def _objective_fast(A_arr):
    """Fast objective assuming A_arr is already a numpy int64 array."""
    sums = A_arr[:, None] + A_arr[None, :]
    num_sums = len(np.unique(sums))
    products = A_arr[:, None] * A_arr[None, :]
    num_products = len(np.unique(products))
    return max(num_sums, num_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _union_of_aps_seed(deadline):
    """
    Seed with unions of short arithmetic progressions with coprime differences.
    Two APs with coprime differences d1, d2 create additive collisions when
    a1 + i*d1 = a2 + j*d2 for multiple (i,j) pairs.
    """
    best_obj = float('inf')
    best_A = None
    rng = random.Random(31415)
    
    # Try pairs of APs with coprime differences
    for d1 in range(1, 30):
        if time.time() > deadline:
            return best_A, best_obj
        for d2 in range(d1 + 1, 30):
            if time.time() > deadline:
                return best_A, best_obj
            # Check coprime
            if _gcd(d1, d2) != 1:
                continue
            
            # Split 20 elements: k in first AP, 20-k in second
            for k in range(5, 16):
                k2 = 20 - k
                # First AP: a1, a1+d1, ..., a1+(k-1)*d1
                # Second AP: a2, a2+d2, ..., a2+(k2-1)*d2
                # Need all values in [1, 10000] and all distinct
                
                max_a1 = MAX_VAL - (k - 1) * d1
                max_a2 = MAX_VAL - (k2 - 1) * d2
                if max_a1 < 1 or max_a2 < 1:
                    continue
                
                # Try several starting points
                for a1 in [1, 2, 3, max_a1 // 2, max_a1 - 1, max_a1]:
                    if a1 < 1:
                        continue
                    ap1 = [a1 + i * d1 for i in range(k)]
                    
                    for a2 in [1, 2, 3, max_a2 // 2, max_a2 - 1, max_a2]:
                        if a2 < 1:
                            continue
                        ap2 = [a2 + i * d2 for i in range(k2)]
                        
                        A = ap1 + ap2
                        if len(set(A)) != 20:
                            continue
                        if any(x < 1 or x > MAX_VAL for x in A):
                            continue
                        
                        obj = _objective(A)
                        if obj < best_obj:
                            best_obj = obj
                            best_A = list(A)
    
    # Also try single AP with perturbations at strategic positions
    for d in [1, 2, 3, 4, 5, 7, 10, 13]:
        if time.time() > deadline:
            return best_A, best_obj
        max_a = MAX_VAL - 19 * d
        if max_a < 1:
            continue
        for a in [1, 2, 3, max_a // 2]:
            A = [a + i * d for i in range(20)]
            if any(x > MAX_VAL for x in A):
                continue
            obj = _objective(A)
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)
    
    return best_A, best_obj

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _rational_geometric_search(deadline):
    """
    Search for sets of the form { round(a * (p/q)^i) } for i=0..19.
    We iterate over small p, q and optimize a.
    """
    best_obj = float('inf')
    best_A = None
    rng = random.Random(42)
    
    max_pq = 20
    
    for p in range(2, max_pq + 1):
        for q in range(1, p):
            if time.time() > deadline:
                return best_A, best_obj
            
            ratio = p / q
            max_a = int(MAX_VAL / (ratio ** 19))
            if max_a < 1:
                continue
            
            candidates_a = set()
            candidates_a.add(max_a)
            
            for k in range(1, 10):
                if k <= max_a:
                    candidates_a.add(k)
            
            base = q ** 19
            max_k = max_a // base
            if max_k > 0:
                for k in range(1, min(max_k, 20) + 1):
                    candidates_a.add(k * base)
            
            for _ in range(5):
                candidates_a.add(rng.randint(1, max_a))
            
            for a in candidates_a:
                if a < 1:
                    continue
                
                A = []
                valid = True
                current = a
                for i in range(20):
                    if current > MAX_VAL:
                        valid = False
                        break
                    if current in A:
                        valid = False
                        break
                    A.append(current)
                    
                    if i < 19:
                        next_val = (current * p) // q
                        current = next_val
                
                if not valid or len(A) != 20:
                    continue
                
                if len(set(A)) != 20:
                    continue
                
                obj = _objective(A)
                if obj < best_obj:
                    best_obj = obj
                    best_A = list(A)
                
                if time.time() > deadline:
                    return best_A, best_obj

    return best_A, best_obj

def _arithmetic_progression_seed(rng, deadline):
    """
    Seed with arithmetic progressions to exploit additive collisions.
    For an AP {a, a+d, ..., a+(n-1)d}, |A+A| = 2n-1 = 39 (exact).
    We try various (a, d) pairs and small perturbations.
    """
    best_obj = float('inf')
    best_A = None
    
    # Try various common differences
    for d in range(1, 500):
        if time.time() > deadline:
            break
        
        # a must satisfy a + 19*d <= 10000
        max_a = MAX_VAL - 19 * d
        if max_a < 1:
            continue
        
        # Try a few values of a
        candidate_as = set()
        candidate_as.add(1)
        candidate_as.add(max_a)
        for _ in range(5):
            candidate_as.add(rng.randint(1, max_a))
        
        for a in candidate_as:
            A = [a + i * d for i in range(20)]
            if len(set(A)) != 20:
                continue
            if any(x < 1 or x > MAX_VAL for x in A):
                continue
            
            obj = _objective(A)
            if obj < best_obj:
                best_obj = obj
                best_A = list(A)
            
            # Also try slight perturbations of the AP
            if obj < 80:  # Only perturb good candidates
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    A_pert = list(A)
                    idx = rng.randint(0, 19)
                    delta = rng.randint(-2*d, 2*d)
                    if delta == 0:
                        continue
                    new_val = A_pert[idx] + delta
                    if new_val < 1 or new_val > MAX_VAL or new_val in A_pert:
                        continue
                    A_pert[idx] = new_val
                    obj_pert = _objective(A_pert)
                    if obj_pert < best_obj:
                        best_obj = obj_pert
                        best_A = list(A_pert)
    
    return best_A, best_obj

def _swap_local_search(A, time_limit, rng):
    """
    Local search that swaps elements to preserve multiplicative structure.
    Uses collision-aware move generation to target sumset/productset reductions.
    """
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 50.0
    cooling_rate = 0.997
    min_temperature = 0.01
    
    # Precompute current sum and product sets for collision targeting
    A_arr = np.array(A, dtype=np.int64)
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        # Try a swap
        i = rng.randint(0, 19)
        j = rng.randint(0, 19)
        if i == j:
            continue
        
        old_i, old_j = A[i], A[j]
        A_set = set(A)
        
        # Generate candidate swap values with collision awareness
        found = False
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Choose perturbation scale based on element magnitude
            scale_i = max(1, old_i // 10)
            scale_j = max(1, old_j // 10)
            di = rng.randint(-scale_i, scale_i)
            dj = rng.randint(-scale_j, scale_j)
            new_i = old_i + di
            new_j = old_j + dj
            
            if new_i < 1 or new_i > MAX_VAL or new_j < 1 or new_j > MAX_VAL:
                continue
            if new_i == new_j:
                continue
            if new_i in A_set and new_i != old_i:
                continue
            if new_j in A_set and new_j != old_j:
                continue
            
            A[i] = new_i
            A[j] = new_j
            new_obj = _objective(A)
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                found = True
                break
            else:
                A[i] = old_i
                A[j] = old_j
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing and collision-aware moves."""
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 100.0
    cooling_rate = 0.995
    min_temperature = 0.01
    
    while temperature > min_temperature:
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.randint(0, 19)
        old_val = A[idx]
        A_set = set(A)
        
        found_improvement = False
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Adaptive step size based on element magnitude
            scale = max(1, old_val // 5)
            if rng.random() < 0.3:
                new_val = old_val + rng.randint(-scale, scale)
            elif rng.random() < 0.6:
                new_val = old_val + rng.randint(-3 * scale, 3 * scale)
            elif rng.random() < 0.85:
                new_val = old_val + rng.randint(-10 * scale, 10 * scale)
            else:
                # Try to create sum collision: pick another element and adjust
                other_idx = rng.randint(0, 19)
                if other_idx == idx:
                    other_idx = (idx + 1) % 20
                target = A[other_idx] + rng.choice([-2 * scale, -scale, scale, 2 * scale])
                new_val = target
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A_set:
                continue
            
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                found_improvement = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _polish_pass(A, time_limit, rng):
    """Final polish with very small steps and pairwise swap polish."""
    A = list(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Single-element polish
        for idx in range(20):
            if time.time() - start_time > time_limit:
                break
            
            old_val = A[idx]
            A_set = set(A)
            
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                if new_val < 1 or new_val > MAX_VAL:
                    continue
                if new_val in A_set:
                    continue
                
                A[idx] = new_val
                new_obj = _objective(A)
                A[idx] = old_val
                
                if new_obj < current_obj:
                    A[idx] = new_val
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    break
        
        # Pairwise swap polish
        if time.time() - start_time > time_limit:
            break
        for i in range(20):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, 20):
                if time.time() - start_time > time_limit:
                    break
                
                old_i, old_j = A[i], A[j]
                A_set = set(A)
                
                # Try small adjustments to both
                for di in [-2, -1, 1, 2]:
                    for dj in [-2, -1, 1, 2]:
                        new_i = old_i + di
                        new_j = old_j + dj
                        if new_i < 1 or new_i > MAX_VAL or new_j < 1 or new_j > MAX_VAL:
                            continue
                        if new_i == new_j:
                            continue
                        if new_i in A_set and new_i != old_i:
                            continue
                        if new_j in A_set and new_j != old_j:
                            continue
                        
                        A[i] = new_i
                        A[j] = new_j
                        new_obj = _objective(A)
                        A[i] = old_i
                        A[j] = old_j
                        
                        if new_obj < current_obj:
                            A[i] = new_i
                            A[j] = new_j
                            current_obj = new_obj
                            if new_obj < best_obj:
                                best_obj = new_obj
                                best_A = list(A)
                            break
                    else:
                        continue
                    break
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    deadline = start_time + time_budget
    
    best_A = None
    best_obj = float('inf')
    
    # Phase 1: Deterministic Rational Geometric Search
    try:
        A, obj = _rational_geometric_search(deadline)
        if A:
            best_A = A
            best_obj = obj
    except Exception:
        pass
    
    # Phase 1b: Arithmetic Progression Seeding
    if time.time() < deadline:
        try:
            A, obj = _arithmetic_progression_seed(random.Random(789), deadline)
            if A and obj < best_obj:
                best_A = A
                best_obj = obj
        except Exception:
            pass
    
    # Phase 1c: Union of APs Seeding
    if time.time() < deadline:
        try:
            A, obj = _union_of_aps_seed(deadline)
            if A and obj < best_obj:
                best_A = A
                best_obj = obj
        except Exception:
            pass
    
    # Fallback
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    
    # Phase 2: Swap-based local search (preserves multiplicative structure)
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 2.0:
            try:
                refined_A, refined_obj = _swap_local_search(best_A, remaining * 0.5, random.Random(123))
                if refined_obj < best_obj:
                    best_A = refined_A
                    best_obj = refined_obj
            except Exception:
                pass
    
    # Phase 3: Standard local search
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 1.0:
            try:
                refined_A, refined_obj = _local_search(best_A, remaining * 0.6, random.Random(456))
                if refined_obj < best_obj:
                    best_A = refined_A
                    best_obj = refined_obj
            except Exception:
                pass
    
    # Phase 4: Polish
    if time.time() < deadline:
        remaining = deadline - time.time()
        if remaining > 0.5:
            try:
                polished_A, polished_obj = _polish_pass(best_A, remaining, random.Random(789))
                if polished_obj < best_obj:
                    best_A = polished_A
                    best_obj = polished_obj
            except Exception:
                pass
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return sorted(best_A)
