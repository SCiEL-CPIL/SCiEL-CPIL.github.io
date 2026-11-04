"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set(U):
    """Compute the lower bound score for a set U using numpy for speed."""
    if not U or 0 not in U:
        return 0.0
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n < 2:
        return 0.0
    max_U = int(U_arr[-1])
    if max_U == 0:
        return 0.0
    
    # Compute sumset U+U using numpy broadcasting
    sumset = np.unique(U_arr[:, None] + U_arr[None, :])
    
    # Compute difference set U-U using numpy broadcasting
    diffset = np.unique(U_arr[:, None] - U_arr[None, :])
    
    sumset_size = len(sumset)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    log_base = math.log(2 * max_U + 1)
    if log_base <= 0:
        return 0.0
    
    score = 1 + math.log(ratio) / log_base
    return score

def _generate_candidates(U, max_val_limit):
    """Generate diverse candidates from arithmetic neighborhoods of U."""
    candidates = set()
    U_list = list(U)
    
    # Pairs from U
    for a in U_list:
        for b in U_list:
            s = a + b
            d = abs(a - b)
            if 0 < s <= max_val_limit:
                candidates.add(s)
            if 0 < d <= max_val_limit:
                candidates.add(d)
    
    # Multiples and offsets
    for a in U_list:
        if a > 0:
            for mult in [2, 3, 4, 5]:
                val = a * mult
                if val <= max_val_limit:
                    candidates.add(val)
            for offset in [1, 2, 3, 5, 8, 13]:
                if a + offset <= max_val_limit:
                    candidates.add(a + offset)
                if a - offset > 0:
                    candidates.add(a - offset)
    
    # Powers and structured values
    for base in [2, 3, 4, 5]:
        val = base
        while val <= max_val_limit:
            candidates.add(val)
            val *= base
    
    return candidates

def _local_search_dynamic(U, num_iterations=200, max_val_limit=80000):
    """Local search with dynamic candidate generation."""
    U = set(U)
    best_score = _score_set(list(U))
    
    for _ in range(num_iterations):
        improved = False
        
        # Generate dynamic candidates
        candidates = _generate_candidates(U, max_val_limit)
        valid_candidates = [c for c in candidates if c not in U and 0 < c <= max_val_limit]
        
        if not valid_candidates:
            break
        
        # Sample candidates for efficiency
        sample_size = min(len(valid_candidates), 150)
        random.shuffle(valid_candidates)
        sampled = valid_candidates[:sample_size]
        
        # Try adding
        for c in sampled:
            new_U = U | {c}
            new_score = _score_set(list(new_U))
            if new_score > best_score:
                U = new_U
                best_score = new_score
                improved = True
                break
        
        # Try removing
        if not improved and len(U) > 3:
            elements = list(U)
            random.shuffle(elements)
            for c in elements[:20]:
                if c == 0:
                    continue
                new_U = U - {c}
                new_score = _score_set(list(new_U))
                if new_score > best_score:
                    U = new_U
                    best_score = new_score
                    improved = True
                    break
        
        # Try swapping
        if not improved and len(U) > 4:
            elements = list(U)
            random.shuffle(elements)
            for rem in elements[:8]:
                if rem == 0:
                    continue
                U_without = U - {rem}
                swap_candidates = _generate_candidates(U_without, max_val_limit)
                swap_valid = [c for c in swap_candidates if c not in U_without and 0 < c <= max_val_limit]
                if swap_valid:
                    random.shuffle(swap_valid)
                    for add in swap_valid[:20]:
                        new_U = U_without | {add}
                        new_score = _score_set(list(new_U))
                        if new_score > best_score:
                            U = new_U
                            best_score = new_score
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return sorted(U)

def _continuous_geometric_search(deadline):
    """Search for optimal geometric progression parameters."""
    best_set = [0, 1]
    best_score = _score_set(best_set)
    
    start_time = time.time()
    
    for base in np.linspace(1.5, 5.0, 50):
        if time.time() > deadline - 10:
            break
        for num_terms in range(3, 20):
            if time.time() > deadline - 10:
                break
            U = [0]
            current = 1.0
            for _ in range(num_terms):
                U.append(int(round(current)))
                current *= base
            U = list(set(U))
            if max(U) > 100000 or len(U) < 2:
                continue
            score = _score_set(U)
            if score > best_score:
                best_score = score
                best_set = U
    
    return best_set, best_score

def _build_geometric_set(base, num_digits, start=0):
    """Build a set using a geometric/digit representation."""
    elements = set()
    elements.add(0)
    
    # Try to build a set with good sum/difference ratio
    # Using a base-b representation with limited digits
    max_val = base ** num_digits
    if max_val > 100000:
        return list(elements)
    
    # Simple geometric progression
    elements = set()
    elements.add(0)
    current = 1
    for _ in range(num_digits):
        elements.add(current)
        current *= base
    
    return sorted(elements)

def _greedy_construct(target_size, max_val_limit=100000):
    """Greedy construction: add elements that maximize local ratio improvement."""
    U = {0}
    U_list = [0]
    
    # Precompute some candidates
    candidates = set()
    for i in range(1, min(max_val_limit + 1, 5000)):
        candidates.add(i)
    
    # Add some structured elements
    for power in [2, 3, 4, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]:
        if power <= max_val_limit:
            candidates.add(power)
    
    # Fibonacci-like
    a, b = 1, 2
    while a <= max_val_limit:
        candidates.add(a)
        a, b = b, a + b
    
    best_score = _score_set(list(U))
    
    while len(U) < target_size:
        best_candidate = None
        best_new_score = best_score
        
        # Sample from candidates
        candidate_list = list(candidates - U)
        if not candidate_list:
            break
        
        # Try a subset for speed
        sample_size = min(len(candidate_list), 100)
        sampled = random.sample(candidate_list, sample_size)
        
        for c in sampled:
            if c > max_val_limit:
                continue
            new_U = U | {c}
            new_score = _score_set(list(new_U))
            if new_score > best_new_score:
                best_new_score = new_score
                best_candidate = c
        
        if best_candidate is not None:
            U.add(best_candidate)
            U_list.append(best_candidate)
            best_score = best_new_score
            # Update candidates with new sums/differences
            for c in list(U):
                if best_candidate + c <= max_val_limit:
                    candidates.add(best_candidate + c)
                if abs(best_candidate - c) <= max_val_limit:
                    candidates.add(abs(best_candidate - c))
        else:
            break
    
    return sorted(U)

def _local_search(U, num_iterations=500, max_val_limit=100000):
    """Local search: try to improve the set by swapping or adding elements."""
    U = set(U)
    best_score = _score_set(list(U))
    
    for _ in range(num_iterations):
        improved = False
        
        # Try adding a new element
        candidates = set()
        for a in U:
            for b in U:
                s = a + b
                d = abs(a - b)
                if s <= max_val_limit:
                    candidates.add(s)
                if d <= max_val_limit:
                    candidates.add(d)
        
        # Filter valid candidates
        valid_candidates = [c for c in candidates if c not in U and 0 < c <= max_val_limit]
        
        if valid_candidates:
            # Sample more candidates for better exploration
            sample_size = min(len(valid_candidates), 100)
            sampled = random.sample(valid_candidates, sample_size)
            
            for c in sampled:
                new_U = U | {c}
                new_score = _score_set(list(new_U))
                if new_score > best_score:
                    U = new_U
                    best_score = new_score
                    improved = True
                    break
        
        # Try removing an element
        if len(U) > 2:
            elements = list(U)
            random.shuffle(elements)
            for c in elements[:30]:
                if c == 0:
                    continue
                new_U = U - {c}
                new_score = _score_set(list(new_U))
                if new_score > best_score:
                    U = new_U
                    best_score = new_score
                    improved = True
                    break
        
        # Try swapping: remove one, add another
        if len(U) > 3 and not improved:
            elements = list(U)
            random.shuffle(elements)
            for rem in elements[:10]:
                if rem == 0:
                    continue
                U_without = U - {rem}
                # Generate candidates from U_without
                swap_candidates = set()
                for a in U_without:
                    for b in U_without:
                        s = a + b
                        d = abs(a - b)
                        if s <= max_val_limit:
                            swap_candidates.add(s)
                        if d <= max_val_limit:
                            swap_candidates.add(d)
                swap_candidates = [c for c in swap_candidates if c not in U_without and 0 < c <= max_val_limit]
                if swap_candidates:
                    sample_size = min(len(swap_candidates), 30)
                    sampled = random.sample(swap_candidates, sample_size)
                    for add in sampled:
                        new_U = U_without | {add}
                        new_score = _score_set(list(new_U))
                        if new_score > best_score:
                            U = new_U
                            best_score = new_score
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return sorted(U)

def _construct_paper_style():
    """Try to construct a set similar to the paper's constructions."""
    # The paper mentions sizes 2003 and 54265
    # Let's try a structured approach
    
    # Method 1: Use a base-b number system with restricted digits
    best_set = None
    best_score = 0
    
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]:
        for num_digits in range(3, 12):
            max_val = base ** num_digits
            if max_val > 200000:
                break
            
            # Generate all numbers with digits in {0, 1, ..., base-1}
            # But restrict to keep the set manageable
            elements = set()
            elements.add(0)
            
            # Add powers of base
            for d in range(num_digits):
                elements.add(base ** d)
            
            # Add some combinations
            if len(elements) < 500:
                for d1 in range(num_digits):
                    for d2 in range(d1 + 1, num_digits):
                        val = base ** d1 + base ** d2
                        if val <= max_val:
                            elements.add(val)
            
            if elements:
                score = _score_set(list(elements))
                if score > best_score:
                    best_score = score
                    best_set = list(elements)
    
    return best_set if best_set else [0, 1, 2]

def _random_search(deadline):
    """Random search with deadline."""
    best_set = [0, 1, 2]
    best_score = _score_set(best_set)
    
    start_time = time.time()
    
    for attempt in range(10000):
        if time.time() > deadline - 5:
            break
        
        # Random strategy
        strategy = random.random()
        
        if strategy < 0.3:
            # Geometric progression
            base = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
            num_terms = random.randint(3, 15)
            U = [0]
            current = 1
            for _ in range(num_terms):
                U.append(current)
                current *= base
            if max(U) > 100000:
                continue
            U = list(set(U))
            
        elif strategy < 0.6:
            # Random subset of [0, N]
            N = random.randint(10, 500)
            size = random.randint(5, min(50, N))
            U = [0] + random.sample(range(1, N + 1), size - 1)
            
        else:
            # Fibonacci-like
            U = [0, 1]
            while len(U) < random.randint(5, 20):
                U.append(U[-1] + U[-2])
            if max(U) > 100000:
                U = U[:10]
        
        score = _score_set(U)
        if score > best_score:
            best_score = score
            best_set = U
    
    return best_set, best_score

def construct():
    """Main construction function."""
    start_time = time.time()
    deadline = start_time + 145  # Use more of the 150s budget
    
    random.seed(42)
    
    # Start with a reasonable initial set
    best_set = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_score = _score_set(best_set)
    
    # Phase 1: Try structured constructions
    if time.time() < deadline - 15:
        paper_set = _construct_paper_style()
        if paper_set:
            score = _score_set(paper_set)
            if score > best_score:
                best_score = score
                best_set = paper_set
    
    # Phase 2: Greedy construction with larger target
    if time.time() < deadline - 30:
        greedy_set = _greedy_construct(target_size=40, max_val_limit=80000)
        if greedy_set:
            score = _score_set(greedy_set)
            if score > best_score:
                best_score = score
                best_set = greedy_set
    
    # Phase 3: Continuous geometric search for better seeds
    if time.time() < deadline - 20:
        geom_set, geom_score = _continuous_geometric_search(deadline)
        if geom_score > best_score:
            best_score = geom_score
            best_set = geom_set
    
    # Phase 4: Dynamic local search from best
    if time.time() < deadline - 30:
        improved_set = _local_search_dynamic(best_set, num_iterations=300, max_val_limit=80000)
        if improved_set:
            score = _score_set(improved_set)
            if score > best_score:
                best_score = score
                best_set = improved_set
    
    # Phase 5: Random search
    if time.time() < deadline - 15:
        random_set, random_score = _random_search(deadline)
        if random_score > best_score:
            best_score = random_score
            best_set = random_set
    
    # Final local search polish with dynamic candidates
    if time.time() < deadline - 5 and len(best_set) > 2:
        polished = _local_search_dynamic(best_set, num_iterations=200, max_val_limit=max(best_set) * 2 + 10000)
        if polished:
            score = _score_set(polished)
            if score > best_score:
                best_score = score
                best_set = polished
    
    return best_set
# EVOLVE-BLOCK-END
