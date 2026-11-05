"""Rewritten solver for K_3(7,1) ternary covering code (football pool)."""
import time
import random
import numpy as np

# Fixed deadline guard: evaluator kills at 150s; we return with a safety margin.
_DEADLINE = 145.0  # seconds


def _hamming_dist(a, b):
    return np.count_nonzero(a != b)


def _is_covering(code, q, n, radius):
    """Check if every word is within `radius` of some codeword."""
    code_arr = np.array(code, dtype=np.int8)
    m = len(code_arr)
    # Precompute the union of all radius-r spheres.
    # For q=3, n=7, radius=1: each sphere has 1 + 7*2 = 15 words.
    # Total space = 3^7 = 2187.
    covered = np.zeros(q ** n, dtype=bool)
    # Map a word (tuple of ints) to an index: sum(w[i] * q^(n-1-i))
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    # For each codeword, mark all words within radius
    for cw in code_arr:
        # distance 0
        idx = sum(cw[i] * powers[i] for i in range(n))
        covered[idx] = True
        if radius >= 1:
            # distance 1: change one position
            for pos in range(n):
                orig = cw[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    cw[pos] = val
                    idx1 = sum(cw[i] * powers[i] for i in range(n))
                    covered[idx1] = True
                    cw[pos] = orig
    return covered.all()


def _base_solution(q, n, radius):
    """Direct sum construction: tetracode (9 words len 4) x all words on remaining coords."""
    if n <= 4:
        # For small n, use greedy or exhaustive
        # For n=4, tetracode is perfect
        if n == 4:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Fallback: greedy
        return _greedy_cover(q, n, radius)
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    result = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            result.append(h + list(tail))
    return result


def _greedy_cover(q, n, radius, time_limit=None):
    """Greedy set cover: repeatedly pick the word covering the most uncovered words."""
    start_time = time.time()
    total_words = q ** n
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    # Precompute spheres for all possible words
    spheres = []
    for w in range(total_words):
        # Decode
        digits = []
        tmp = w
        for i in range(n - 1, -1, -1):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        
        sphere = set()
        sphere.add(w)
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    digits[pos] = val
                    idx = sum(digits[i] * powers[i] for i in range(n))
                    sphere.add(idx)
                    digits[pos] = orig
        spheres.append(sphere)
    
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        if time_limit and (time.time() - start_time) > time_limit:
            break
        best_word = -1
        best_count = 0
        for w in range(total_words):
            count = len(spheres[w] & uncovered)
            if count > best_count:
                best_count = count
                best_word = w
        if best_word == -1:
            break
        code.append(best_word)
        uncovered -= spheres[best_word]
    
    # Decode to list of lists
    result = []
    for w in code:
        digits = []
        tmp = w
        for i in range(n - 1, -1, -1):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        result.append(digits)
    return result


def _simulated_annealing(code, q, n, radius, time_limit=140.0):
    """Simulated annealing to minimize number of codewords while maintaining coverage."""
    start_time = time.time()
    total_words = q ** n
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    # Precompute spheres for all possible words
    spheres = []
    for w in range(total_words):
        digits = []
        tmp = w
        for i in range(n - 1, -1, -1):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        
        sphere = set()
        sphere.add(w)
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    digits[pos] = val
                    idx = sum(digits[i] * powers[i] for i in range(n))
                    sphere.add(idx)
                    digits[pos] = orig
        spheres.append(sphere)
    
    # Convert code to set of word indices
    code_set = set()
    for cw in code:
        idx = sum(cw[i] * powers[i] for i in range(n))
        code_set.add(idx)
    
    # Compute coverage
    covered = set()
    for idx in code_set:
        covered |= spheres[idx]
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.01
    max_iter = 500000
    
    current_code = set(code_set)
    current_covered = set(covered)
    best_code = set(current_code)
    best_covered = set(current_covered)
    best_size = len(best_code)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Cooling
        T = T_start * (T_end / T_start) ** (it / max_iter)
        
        # Move: try to remove a random codeword
        if len(current_code) > 0:
            candidate = random.choice(list(current_code))
            new_code = current_code - {candidate}
            new_covered = set()
            for idx in new_code:
                new_covered |= spheres[idx]
            
            if new_covered == covered:  # Still covers everything
                current_code = new_code
                current_covered = new_covered
                if len(current_code) < best_size:
                    best_code = set(current_code)
                    best_covered = set(current_covered)
                    best_size = len(best_code)
            else:
                # Accept with probability exp(-delta/T) where delta is number of newly uncovered
                delta = len(covered - new_covered)
                if delta > 0 and random.random() < np.exp(-delta / T):
                    current_code = new_code
                    current_covered = new_covered
        
        # Move: try to replace a codeword with another
        if len(current_code) > 0:
            old = random.choice(list(current_code))
            # Pick a random new word
            new_word = random.randint(0, total_words - 1)
            if new_word not in current_code:
                new_code = current_code - {old} | {new_word}
                new_covered = set()
                for idx in new_code:
                    new_covered |= spheres[idx]
                
                if new_covered == covered:
                    current_code = new_code
                    current_covered = new_covered
                    if len(current_code) < best_size:
                        best_code = set(current_code)
                        best_covered = set(current_covered)
                        best_size = len(best_code)
                else:
                    delta = len(covered - new_covered)
                    if delta > 0 and random.random() < np.exp(-delta / T):
                        current_code = new_code
                        current_covered = new_covered
    
    # Decode best code
    result = []
    for idx in best_code:
        digits = []
        tmp = idx
        for i in range(n - 1, -1, -1):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        result.append(digits)
    return result


def _local_search(code, q, n, radius, time_limit=140.0):
    """Iterated local search: try to remove codewords that are redundant."""
    start_time = time.time()
    total_words = q ** n
    powers = [q ** (n - 1 - i) for i in range(n)]
    
    # Precompute spheres
    spheres = []
    for w in range(total_words):
        digits = []
        tmp = w
        for i in range(n - 1, -1, -1):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        
        sphere = set()
        sphere.add(w)
        if radius >= 1:
            for pos in range(n):
                orig = digits[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    digits[pos] = val
                    idx = sum(digits[i] * powers[i] for i in range(n))
                    sphere.add(idx)
                    digits[pos] = orig
        spheres.append(sphere)
    
    # Convert to set of indices
    code_set = set()
    for cw in code:
        idx = sum(cw[i] * powers[i] for i in range(n))
        code_set.add(idx)
    
    # Try to remove redundant codewords
    changed = True
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        for idx in list(code_set):
            if len(code_set) <= 1:
                break
            new_code = code_set - {idx}
            new_covered = set()
            for i in new_code:
                new_covered |= spheres[i]
            if len(new_covered) == total_words:
                code_set = new_code
                changed = True
    
    # Decode
    result = []
    for idx in code_set:
        digits = []
        tmp = idx
        for i in range(n - 1, -1, -1):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        result.append(digits)
    return result


def construct(q, n, radius):
    """Main entry point: return a covering code for K_q(n, radius)."""
    start_time = time.time()
    
    # Get base solution
    code = _base_solution(q, n, radius)
    base_size = len(code)
    
    # For the specific case q=3, n=7, radius=1
    if q == 3 and n == 7 and radius == 1:
        # Use simulated annealing to try to improve
        remaining_time = _DEADLINE - (time.time() - start_time) - 5.0  # 5s safety margin
        if remaining_time > 10.0:
            sa_code = _simulated_annealing(code, q, n, radius, time_limit=remaining_time)
            if len(sa_code) < base_size:
                code = sa_code
                base_size = len(code)
        
        # Local search to remove redundant codewords
        remaining_time = _DEADLINE - (time.time() - start_time) - 3.0
        if remaining_time > 5.0:
            ls_code = _local_search(code, q, n, radius, time_limit=remaining_time)
            if len(ls_code) < base_size:
                code = ls_code
                base_size = len(code)
    
    # Verify coverage
    if not _is_covering(code, q, n, radius):
        # Fallback to base
        code = _base_solution(q, n, radius)
    
    return code
