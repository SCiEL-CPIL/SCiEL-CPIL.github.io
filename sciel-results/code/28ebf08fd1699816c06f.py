import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)

    # Fallback for other cases (not expected in this task)
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n  # 17

    weights = [3 ** (n - 1 - i) for i in range(n)]

    # Precompute balls: balls[w] is an array of 17 indices
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

    best_size = total
    best_set = None

    def greedy_construct_random():
        """Randomized greedy construction."""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        remaining_uncovered = total

        while remaining_uncovered > 0:
            # Find the word that covers the most uncovered neighbors
            # counts[i] = number of uncovered neighbors of i
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break # Should not happen
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[random.randint(0, len(candidates) - 1)]

            selected.append(int(best_idx))
            cover_count[balls[best_idx]] += 1
            newly_covered = (cover_count[balls[best_idx]] == 1)
            if np.any(newly_covered):
                remaining_uncovered -= np.sum(newly_covered)
            uncovered_mask = (cover_count == 0)

        return selected, cover_count

    def remove_redundant(selected, cover_count):
        """Remove redundant codewords."""
        changed = True
        while changed:
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

    def local_search(selected, cover_count, deadline):
        """
        Local search combining 1-swaps and 2-exchanges.
        1-swap: Replace c with c' if c' covers all critical neighbors of c.
        2-exchange: Remove c1, c2 and add c3 if c3 covers all critical neighbors of c1 and c2.
        """
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # Phase 1: Try to remove redundant codewords
            selected, cover_count = remove_redundant(selected, cover_count)
            
            # Phase 2: Try 1-swaps to rearrange
            # We iterate through codewords. If a codeword is not redundant, we look for a swap.
            # To be efficient, we only try swaps for a random subset or in a random order.
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                c = selected[i]
                # Check if redundant (might have become redundant due to previous swaps)
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    selected.pop(i)
                    indices.pop(i)
                    improved = True
                    continue

                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    # Should have been caught by redundancy check
                    cover_count[balls[c]] -= 1
                    selected.pop(i)
                    indices.pop(i)
                    improved = True
                    continue

                # Find candidates that cover all critical words
                # candidates = intersection of balls[cw] for cw in critical_words
                candidates = set(balls[critical_words[0]].tolist())
                for cw in critical_words[1:]:
                    candidates.intersection_update(balls[cw].tolist())
                    if not candidates:
                        break

                candidates.discard(c)
                
                if candidates:
                    c_prime = random.choice(list(candidates))
                    cover_count[balls[c]] -= 1
                    cover_count[balls[c_prime]] += 1
                    selected[i] = c_prime
                    improved = True
            
            # Phase 3: Try 2-exchanges (remove 2, add 1)
            if len(selected) > 403:
                num_trials = 200
                for _ in range(num_trials):
                    if time.time() > deadline - 2.0:
                        break
                    
                    idx1 = random.randint(0, len(selected) - 1)
                    idx2 = random.randint(0, len(selected) - 1)
                    if idx1 == idx2:
                        continue
                    
                    c1 = selected[idx1]
                    c2 = selected[idx2]
                    
                    # Temporarily remove
                    cover_count[balls[c1]] -= 1
                    cover_count[balls[c2]] -= 1
                    
                    uncovered_after = (cover_count == 0)
                    
                    if np.any(uncovered_after):
                        uncovered_words = np.where(uncovered_after)[0]
                        
                        # Find candidate that covers all uncovered words
                        candidates = set(balls[uncovered_words[0]].tolist())
                        for uw in uncovered_words[1:]:
                            candidates.intersection_update(balls[uw].tolist())
                            if not candidates:
                                break
                        
                        candidates.discard(c1)
                        candidates.discard(c2)
                        
                        if candidates:
                            c3 = random.choice(list(candidates))
                            cover_count[balls[c3]] += 1
                            
                            # Update selected list
                            # Remove c1 and c2, add c3
                            # Note: c1 and c2 are distinct
                            if c1 == c2:
                                # Should not happen
                                cover_count[balls[c1]] += 1
                                continue
                            
                            new_selected = []
                            for c in selected:
                                if c != c1 and c != c2:
                                    new_selected.append(c)
                            new_selected.append(c3)
                            
                            selected = new_selected
                            improved = True
                        else:
                            # Restore
                            cover_count[balls[c1]] += 1
                            cover_count[balls[c2]] += 1
                    else:
                        # No uncovered words, so we can just remove them
                        new_selected = [c for c in selected if c != c1 and c != c2]
                        selected = new_selected
                        improved = True

        return selected, cover_count

    # Main loop
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        selected, cover_count = greedy_construct_random()
        
        if time.time() > deadline - 5.0:
            continue
        
        selected, cover_count = local_search(selected, cover_count, deadline)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break

    if best_set is None:
        selected, cover_count = greedy_construct_random()
        best_set = selected

    # Convert to output format
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
