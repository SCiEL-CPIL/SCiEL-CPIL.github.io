import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
    
    # Generic fallback: return all words
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1_optimized(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls for all words
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    balls = np.zeros((total, ball_size), dtype=np.int32)
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
    
    # Convert to list of sets for faster membership testing in local search
    ball_sets = [set(balls[i].tolist()) for i in range(total)]
    
    def greedy_cover(randomize=False, seed=0):
        """Greedy set cover with optional randomization."""
        if randomize:
            rng = random.Random(seed)
        else:
            rng = None
        
        cover_count = np.zeros(total, dtype=np.int32)
        selected = []
        current_uncovered = np.ones(total, dtype=bool)
        
        while np.any(current_uncovered):
            if time.time() > deadline - 35.0:
                break
            
            counts = np.sum(current_uncovered[balls], axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            
            candidates = np.where(counts == max_count)[0]
            
            if randomize and rng and len(candidates) > 1:
                # Add small random perturbation
                perturb = np.array([rng.random() for _ in range(len(candidates))])
                best_idx = candidates[np.argmax(perturb)]
            else:
                best_idx = candidates[0]
            
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
        
        return selected, cover_count
    
    def prune(selected, cover_count):
        """Iteratively remove codewords that are redundant."""
        changed = True
        while changed and time.time() < deadline - 20.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
        return selected, cover_count
    
    def local_search(selected, cover_count):
        """Try 1-exchange moves: remove one codeword, add another if it reduces size."""
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 15.0:
            improved = False
            iteration += 1
            
            new_selected = list(selected)
            new_cover = cover_count.copy()
            
            for i in range(len(new_selected)):
                if time.time() > deadline - 15.0:
                    break
                
                c = new_selected[i]
                # Check if c can be removed
                if np.min(new_cover[balls[c]]) > 1:
                    new_cover[balls[c]] -= 1
                    new_selected.pop(i)
                    improved = True
                    # After removal, try to add back a better codeword
                    i -= 1
                    continue
                
                # Try swapping c with another codeword
                # Look for a codeword w not in new_selected that covers more uncovered words
                best_swap = None
                best_gain = 0
                
                # Sample some candidates from the ball of c
                candidates = balls[c].tolist()
                random.shuffle(candidates)
                
                for w in candidates[:30]:  # Limit to 30 candidates for speed
                    if w in new_selected:
                        continue
                    
                    # Check if adding w and removing c is beneficial
                    # This is expensive, so use heuristic:
                    # Count how many words w covers that c doesn't
                    w_ball = set(balls[w].tolist())
                    c_ball = set(balls[c].tolist())
                    
                    # Words covered by w but not by c
                    new_cover_by_w = w_ball - c_ball
                    # Words that would become uncovered if we remove c
                    only_c = c_ball - w_ball
                    
                    # Check if only_c words are covered by others
                    can_remove_c = True
                    for word in only_c:
                        if new_cover[word] <= 1:
                            can_remove_c = False
                            break
                    
                    if can_remove_c:
                        # Count gain: words newly covered by w
                        gain = sum(1 for word in new_cover_by_w if new_cover[word] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = w
                
                if best_swap is not None and best_gain > 0:
                    # Perform swap
                    new_cover[balls[c]] -= 1
                    new_cover[balls[best_swap]] += 1
                    new_selected[i] = best_swap
                    improved = True
            
            selected = new_selected
            cover_count = new_cover
    
    # Multi-start local search
    best_selected = None
    best_size = float('inf')
    best_cover = None
    
    num_restarts = 0
    while time.time() < deadline - 40.0:
        num_restarts += 1
        seed = num_restarts * 12345 + 67890
        
        # Greedy with randomization
        selected, cover_count = greedy_cover(randomize=True, seed=seed)
        
        # Prune
        selected, cover_count = prune(selected, cover_count)
        
        # Local search
        selected, cover_count = local_search(selected, cover_count)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected
            best_cover = cover_count
            
            # If we've found a good solution, try more aggressive local search
            if best_size <= 500:
                # Extra pruning passes
                for _ in range(3):
                    selected, cover_count = prune(selected, cover_count)
                    if len(selected) < best_size:
                        best_size = len(selected)
                        best_selected = selected
                        best_cover = cover_count
    
    # Final greedy without randomization for consistency
    if best_selected is None:
        selected, cover_count = greedy_cover(randomize=False)
        selected, cover_count = prune(selected, cover_count)
        best_selected = selected
        best_cover = cover_count
    
    # Final verification and conversion
    final_cover = np.zeros(total, dtype=np.int32)
    for c in best_selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            best_selected.append(m)
            final_cover[balls[m]] += 1
    
    # Convert to digit lists
    result = []
    for w in best_selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
