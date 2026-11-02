import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline, start_time)

    # Fallbacks
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _solve_ternary_8_1(deadline, start_time):
    n = 8
    q = 3
    total = q ** n  # 6561
    num_neighbors = 1 + n * (q - 1)  # 17

    # Precompute neighbors (balls) for all 6561 words
    weights = [3 ** (n - 1 - i) for i in range(n)]

    balls = np.zeros((total, num_neighbors), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1

    # Precompute ball membership: ball_membership[c, v] = True if v is in balls[c]
    ball_membership = np.zeros((total, total), dtype=bool)
    for w in range(total):
        ball_membership[w, balls[w]] = True

    # Tetracode-based structured initialization
    def make_tetracode_init(seed_val):
        """Create a diverse initial covering set based on tetracode structure."""
        random.seed(seed_val)
        # Tetracode: 9 codewords of length 4
        tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
        
        # Build initial set: for each tetracode word, create variants in positions 4-7
        # This gives a structured starting point with good coverage properties
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        
        # Phase 1: Add tetracode-based words
        for tc in tetracode:
            for tail in itertools.product(range(3), repeat=4):
                word = list(tc) + list(tail)
                # Convert to integer
                w_int = 0
                for d in word:
                    w_int = w_int * 3 + d
                if uncovered[w_int]:
                    selected.append(w_int)
                    cover_count[balls[w_int]] += 1
                    uncovered[balls[w_int]] = False
        
        # Phase 2: Greedy completion
        while np.any(uncovered):
            if time.time() > deadline - 15.0:
                break
            counts = uncovered[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered = (cover_count == 0)
        
        return selected, cover_count

    def make_random_greedy_init(seed_val):
        """Standard random greedy initialization."""
        random.seed(seed_val)
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        
        while np.any(uncovered):
            if time.time() > deadline - 15.0:
                break
            counts = uncovered[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered = (cover_count == 0)
        
        return selected, cover_count

    def remove_redundant(selected, cover_count):
        """Remove redundant codewords."""
        if not selected:
            return selected, cover_count
        
        current_set = set(selected)
        cover_count = cover_count.copy()
        
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            min_count = np.min(cover_count)
            if min_count >= 2:
                c_to_remove = next(iter(current_set))
                current_set.remove(c_to_remove)
                cover_count[balls[c_to_remove]] -= 1
                changed = True
                continue
            
            unique_words_mask = (cover_count == 1)
            candidates = list(current_set)
            random.shuffle(candidates)
            
            for c in candidates:
                num_unique_covered = np.sum(cover_count[balls[c]] == 1)
                if num_unique_covered == 0:
                    current_set.remove(c)
                    cover_count[balls[c]] -= 1
                    changed = True
                    break
        
        return list(current_set), cover_count

    def local_search(selected, cover_count, time_limit=None):
        """Local search with swap optimization. Returns improved set and counts."""
        if time_limit is None:
            time_limit = deadline - 5.0
        
        current_set = set(selected)
        cover_count = cover_count.copy()
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            candidates = list(current_set)
            random.shuffle(candidates)
            
            for c in candidates:
                if time.time() > time_limit:
                    break
                
                c_ball = balls[c]
                U_c = c_ball[cover_count[c_ball] == 1]
                
                if len(U_c) == 0:
                    current_set.remove(c)
                    cover_count[c_ball] -= 1
                    improved = True
                    break
                
                # Find swap candidates: c' that covers all critical words U_c
                # Use ball_membership: c' is valid if for all w in U_c, ball_membership[c', w] = True
                candidate_mask = np.ones(total, dtype=bool)
                for w in U_c:
                    candidate_mask &= ball_membership[:, w]
                
                valid_candidates = np.where(candidate_mask)[0]
                valid_candidates = valid_candidates[valid_candidates != c]
                
                if len(valid_candidates) > 0:
                    # Try multiple candidates to find the best swap
                    # Pick randomly but try a few if the first doesn't help
                    tried = set()
                    for attempt in range(min(3, len(valid_candidates))):
                        c_prime = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                        if c_prime in tried:
                            continue
                        tried.add(c_prime)
                        
                        # Check if this swap is beneficial:
                        # After removing c and adding c', can we remove more codewords?
                        # Simple heuristic: check if c_prime's ball has more overlap
                        new_cover = cover_count.copy()
                        new_cover[c_ball] -= 1
                        new_cover[balls[c_prime]] += 1
                        
                        # Check if any codeword becomes redundant after this swap
                        can_remove_more = False
                        for other in current_set:
                            if other == c:
                                continue
                            if np.all(new_cover[balls[other]] > 1):
                                can_remove_more = True
                                break
                        
                        if can_remove_more:
                            # Perform the swap
                            current_set.remove(c)
                            cover_count[c_ball] -= 1
                            current_set.add(c_prime)
                            cover_count[balls[c_prime]] += 1
                            improved = True
                            break
                        # If not beneficial, try another candidate
                    # If no beneficial swap found, continue to next codeword
                
                if improved:
                    break
        
        return list(current_set), cover_count

    def polish(selected, cover_count, time_limit):
        """High-precision polish pass with tighter local search."""
        if time.time() > time_limit:
            return selected, cover_count
        
        current_set = set(selected)
        cover_count = cover_count.copy()
        
        # Multiple passes of removal + swap
        for _ in range(10):
            if time.time() > time_limit:
                break
            
            # Removal pass
            changed = True
            while changed and time.time() < time_limit:
                changed = False
                min_count = np.min(cover_count)
                if min_count >= 2:
                    c_to_remove = next(iter(current_set))
                    current_set.remove(c_to_remove)
                    cover_count[balls[c_to_remove]] -= 1
                    changed = True
                    continue
                
                candidates = list(current_set)
                random.shuffle(candidates)
                for c in candidates:
                    if np.sum(cover_count[balls[c]] == 1) == 0:
                        current_set.remove(c)
                        cover_count[balls[c]] -= 1
                        changed = True
                        break
            
            # Swap pass - try harder
            improved = True
            while improved and time.time() < time_limit:
                improved = False
                candidates = list(current_set)
                random.shuffle(candidates)
                
                for c in candidates:
                    if time.time() > time_limit:
                        break
                    
                    c_ball = balls[c]
                    U_c = c_ball[cover_count[c_ball] == 1]
                    
                    if len(U_c) == 0:
                        current_set.remove(c)
                        cover_count[c_ball] -= 1
                        improved = True
                        break
                    
                    candidate_mask = np.ones(total, dtype=bool)
                    for w in U_c:
                        candidate_mask &= ball_membership[:, w]
                    
                    valid_candidates = np.where(candidate_mask)[0]
                    valid_candidates = valid_candidates[valid_candidates != c]
                    
                    if len(valid_candidates) > 0:
                        # Try all valid candidates, pick the one that maximizes removals
                        best_c_prime = None
                        best_removals = 0
                        
                        for c_prime in valid_candidates[:20]:  # Limit to 20 candidates
                            new_cover = cover_count.copy()
                            new_cover[c_ball] -= 1
                            new_cover[balls[c_prime]] += 1
                            
                            # Count how many codewords become removable
                            removals = 0
                            for other in current_set:
                                if other != c and np.all(new_cover[balls[other]] > 1):
                                    removals += 1
                            
                            if removals > best_removals:
                                best_removals = removals
                                best_c_prime = c_prime
                        
                        if best_c_prime is not None and best_removals > 0:
                            current_set.remove(c)
                            cover_count[c_ball] -= 1
                            current_set.add(best_c_prime)
                            cover_count[balls[best_c_prime]] += 1
                            improved = True
                            break
            
            if not improved and not changed:
                break
        
        return list(current_set), cover_count

    best_size = total
    best_set = None
    best_cover = None

    # Run multiple restarts with different initialization strategies
    num_restarts = 60
    for i in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Alternate between structured and random initialization
        if i % 3 == 0:
            selected, cover_count = make_tetracode_init(i * 7 + 1)
        elif i % 3 == 1:
            selected, cover_count = make_random_greedy_init(i * 13 + 2)
        else:
            # Hybrid: start with tetracode, then random greedy
            selected, cover_count = make_tetracode_init(i * 19 + 3)
            if time.time() < deadline - 10.0:
                # Add random greedy completion
                uncovered = (cover_count == 0)
                while np.any(uncovered) and time.time() < deadline - 10.0:
                    counts = uncovered[balls].sum(axis=1)
                    max_count = np.max(counts)
                    if max_count == 0:
                        break
                    candidates = np.where(counts == max_count)[0]
                    best_idx = candidates[random.randint(0, len(candidates) - 1)]
                    selected.append(best_idx)
                    cover_count[balls[best_idx]] += 1
                    uncovered = (cover_count == 0)
        
        if len(selected) >= best_size:
            continue
        
        # Redundancy removal
        selected, cover_count = remove_redundant(selected, cover_count)
        
        if len(selected) >= best_size:
            continue
        
        # Local search
        selected, cover_count = local_search(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected
            best_cover = cover_count.copy()
            # Polish the best solution found so far
            if time.time() < deadline - 10.0:
                best_set, best_cover = polish(best_set, best_cover, deadline - 5.0)
                best_size = len(best_set)

    # Final polish on the best solution
    if best_set is not None and time.time() < deadline - 3.0:
        best_set, best_cover = polish(best_set, best_cover, deadline - 2.0)
        best_size = len(best_set)

    # Fallback
    if best_set is None:
        return [list(np.base_repr(w, 3).zfill(n)) for w in range(total)]

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)

    return result
