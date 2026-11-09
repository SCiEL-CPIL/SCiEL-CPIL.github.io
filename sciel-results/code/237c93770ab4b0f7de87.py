"""Optimized ternary covering code constructor using Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math


def _hamming_dist(a, b):
    return sum(x != y for x, y in zip(a, b))


def _is_covered(word, codewords, q, n, radius):
    for cw in codewords:
        if _hamming_dist(word, cw) <= radius:
            return True
    return False


def _deltas_within_radius(q, n, radius):
    """Generate all delta vectors with Hamming weight <= radius."""
    deltas = []
    deltas.append([0] * n)
    for weight in range(1, radius + 1):
        for positions in itertools.combinations(range(n), weight):
            for values in itertools.product(range(1, q), repeat=weight):
                delta = [0] * n
                for pos, val in zip(positions, values):
                    delta[pos] = val
                deltas.append(delta)
    return deltas


def _compute_coverage_counts(codewords, q, n, radius, deltas):
    """
    For each word in the space, count how many codewords cover it.
    Returns a dict mapping word (tuple) -> count.
    Only words with count >= 1 are included.
    """
    coverage = {}
    for cw in codewords:
        cw_tuple = tuple(cw)
        for delta in deltas:
            word = tuple((cw_tuple[i] + delta[i]) % q for i in range(n))
            coverage[word] = coverage.get(word, 0) + 1
    return coverage


def _coverage_deficit(coverage, total_words):
    """Compute the number of uncovered words (deficit from full coverage)."""
    uncovered = 0
    for word in range(total_words):
        # Convert integer to tuple
        word_tuple = []
        w = word
        for i in range(6):
            word_tuple.append(w % 3)
            w //= 3
        word_tuple = tuple(word_tuple)
        if word_tuple not in coverage:
            uncovered += 1
    return uncovered


def _sa_optimize(codewords, q, n, radius, time_limit, start_time):
    """
    Simulated Annealing optimization starting from initial code.
    """
    deltas = _deltas_within_radius(q, n, radius)
    total_words = q ** n
    
    # Initial coverage
    coverage = _compute_coverage_counts(codewords, q, n, radius, deltas)
    
    # Track best solution
    best_codewords = list(codewords)
    best_coverage = dict(coverage)
    best_size = len(codewords)
    
    # SA parameters
    T = 1.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    
    current_size = len(codewords)
    current_energy = 0  # 0 if fully covered, positive if not
    
    # Check if fully covered
    for word_tuple, count in coverage.items():
        if count <= 0:
            current_energy = 1000  # Large penalty
            break
    
    iteration = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Choose move type:
        # 1. Remove a random codeword (if coverage allows)
        # 2. Move a random codeword to a nearby position
        # 3. Add a new codeword (if size < some limit)
        
        move_type = random.random()
        
        if move_type < 0.3 and current_size > 1:
            # Try removing a codeword
            idx = random.randint(0, current_size - 1)
            cw = codewords[idx]
            cw_tuple = tuple(cw)
            
            # Check if removal would leave any word uncovered
            removable = True
            for delta in deltas:
                word = tuple((cw_tuple[i] + delta[i]) % q for i in range(n))
                if coverage.get(word, 0) <= 1:
                    removable = False
                    break
            
            if removable:
                # Perform removal
                codewords.pop(idx)
                current_size -= 1
                
                for delta in deltas:
                    word = tuple((cw_tuple[i] + delta[i]) % q for i in range(n))
                    coverage[word] -= 1
                    if coverage[word] == 0:
                        del coverage[word]
                
                # This is always an improvement (smaller code, still covered)
                if current_size < best_size:
                    best_size = current_size
                    best_codewords = list(codewords)
                    best_coverage = dict(coverage)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        elif move_type < 0.9:
            # Move a random codeword to a nearby position
            idx = random.randint(0, current_size - 1)
            old_cw = codewords[idx]
            old_tuple = tuple(old_cw)
            
            # Generate new position: change 1-2 random positions
            new_cw = list(old_cw)
            num_changes = random.randint(1, 2)
            positions = random.sample(range(n), num_changes)
            for pos in positions:
                new_cw[pos] = (new_cw[pos] + random.randint(1, q - 1)) % q
            new_tuple = tuple(new_cw)
            
            # Check if new position is already occupied
            if new_tuple in [tuple(c) for c in codewords]:
                no_improve_count += 1
                continue
            
            # Calculate energy change
            # Remove old, add new, check if any word becomes uncovered
            old_sphere = set()
            for delta in deltas:
                word = tuple((old_tuple[i] + delta[i]) % q for i in range(n))
                old_sphere.add(word)
            
            new_sphere = set()
            for delta in deltas:
                word = tuple((new_tuple[i] + delta[i]) % q for i in range(n))
                new_sphere.add(word)
            
            # Words that would become uncovered: in old_sphere but not in new_sphere, and count == 1
            would_uncover = 0
            for word in old_sphere:
                if word not in new_sphere:
                    if coverage.get(word, 0) <= 1:
                        would_uncover += 1
            
            if would_uncover > 0:
                # This move would break coverage - accept with probability exp(-delta/T)
                delta_energy = would_uncover * 10  # Penalty per uncovered word
                if random.random() < math.exp(-delta_energy / T):
                    # Accept
                    codewords[idx] = new_cw
                    
                    # Update coverage
                    for word in old_sphere:
                        coverage[word] -= 1
                        if coverage[word] == 0:
                            del coverage[word]
                    for word in new_sphere:
                        coverage[word] = coverage.get(word, 0) + 1
                    
                    current_energy = would_uncover
                else:
                    no_improve_count += 1
            else:
                # Move is safe - always accept
                codewords[idx] = new_cw
                
                for word in old_sphere:
                    coverage[word] -= 1
                    if coverage[word] == 0:
                        del coverage[word]
                for word in new_sphere:
                    coverage[word] = coverage.get(word, 0) + 1
                
                current_energy = 0
                no_improve_count = 0
        
        else:
            # Try adding a new codeword (only if we've been stuck)
            if current_size < 100 and no_improve_count > 100:
                # Find a word with low coverage and place a codeword nearby
                # Sample some words
                candidates = []
                for _ in range(50):
                    word = tuple(random.randint(0, q - 1) for _ in range(n))
                    cov = coverage.get(word, 0)
                    if cov <= 1:
                        # Place codeword at this word or nearby
                        new_cw = list(word)
                        # Maybe perturb slightly
                        if random.random() < 0.5:
                            pos = random.randint(0, n - 1)
                            new_cw[pos] = (new_cw[pos] + random.randint(1, q - 1)) % q
                        candidates.append(tuple(new_cw))
                
                if candidates:
                    new_cw = random.choice(candidates)
                    if new_cw not in [tuple(c) for c in codewords]:
                        codewords.append(list(new_cw))
                        current_size += 1
                        
                        new_tuple = new_cw
                        for delta in deltas:
                            word = tuple((new_tuple[i] + delta[i]) % q for i in range(n))
                            coverage[word] = coverage.get(word, 0) + 1
                        
                        # Now try to remove redundant codewords
                        for _ in range(5):
                            removable_idx = -1
                            for i in range(current_size):
                                cw = codewords[i]
                                cw_t = tuple(cw)
                                can_remove = True
                                for delta in deltas:
                                    word = tuple((cw_t[j] + delta[j]) % q for j in range(n))
                                    if coverage.get(word, 0) <= 1:
                                        can_remove = False
                                        break
                                if can_remove:
                                    removable_idx = i
                                    break
                            
                            if removable_idx != -1:
                                cw = codewords.pop(removable_idx)
                                current_size -= 1
                                cw_t = tuple(cw)
                                for delta in deltas:
                                    word = tuple((cw_t[j] + delta[j]) % q for j in range(n))
                                    coverage[word] -= 1
                                    if coverage[word] == 0:
                                        del coverage[word]
                                if current_size < best_size:
                                    best_size = current_size
                                    best_codewords = list(codewords)
                                    best_coverage = dict(coverage)
                            else:
                                break
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # If stuck for too long, restart with best solution
        if no_improve_count > 500:
            codewords = list(best_codewords)
            coverage = dict(best_coverage)
            current_size = best_size
            current_energy = 0
            T = 0.5  # Reset temperature
            no_improve_count = 0
    
    return best_codewords


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet of size q
    with covering radius radius.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s margin for the 120s limit
    
    # Handle edge cases
    if radius >= n:
        return [[0] * n]
    
    if n == 0:
        return [[]]
    
    if q == 1:
        return [[0] * n]
    
    # For small cases, use exact or near-exact methods
    if n <= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 1:
            return [[i] for i in range(q)]
        elif n == 2:
            return [list(p) for p in itertools.product(range(q), repeat=2)]
        elif n == 3:
            return [[a, b, (a + b) % 3] for a in range(3) for b in range(3)]
        else:
            return tetracode[:9] if len(tetracode) >= 9 else tetracode
    
    # Build initial code using direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    random.shuffle(initial_code)
    
    # Run simulated annealing
    result = _sa_optimize(initial_code, q, n, radius, time_limit, start_time)
    
    # Safety check: verify coverage if time permits
    if time.time() - start_time < time_limit - 5:
        deltas = _deltas_within_radius(q, n, radius)
        sample_size = min(500, 3 ** n // 10)
        for _ in range(sample_size):
            word = [random.randint(0, q - 1) for _ in range(n)]
            if not _is_covered(word, result, q, n, radius):
                # Coverage failed, fall back to direct sum
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    
    return result
# EVOLVE-BLOCK-END
