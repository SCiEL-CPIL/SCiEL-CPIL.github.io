import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def compute_score(U):
        """
        Computes the score: 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
        """
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        n = len(U)
        sumset = set()
        diffset = set()
        
        # Efficient O(N^2) calculation
        for i in range(n):
            a = U[i]
            for j in range(n):
                b = U[j]
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        base = 2 * max_u + 1
        if base <= 1:
            return 1.0
            
        return 1.0 + math.log(ratio) / math.log(base)

    def greedy_ratio_construct(n_marks, max_val=200):
        """
        Greedy construction: add elements one at a time to maximize |U-U|/|U+U|
        while keeping max(U) bounded.
        """
        U = [0]
        for _ in range(n_marks - 1):
            best_candidate = None
            best_ratio = 0.0
            # Try candidates from current max+1 up to max_val
            for v in range(1, max_val + 1):
                if v in U:
                    continue
                candidate = sorted(U + [v])
                s = compute_score(candidate)
                if s > best_ratio:
                    best_ratio = s
                    best_candidate = v
            if best_candidate is None:
                break
            U.append(best_candidate)
            U.sort()
        return sorted(U)

    # Known optimal Golomb Rulers
    # These are the shortest rulers for N marks
    known_golomb = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 18, 23, 25],
        8: [0, 1, 4, 10, 18, 23, 29, 34],
        9: [0, 1, 4, 10, 18, 23, 29, 34, 41],
        10: [0, 1, 4, 10, 18, 23, 29, 34, 41, 52],
        11: [0, 1, 4, 10, 18, 23, 29, 34, 41, 52, 59],
        12: [0, 1, 4, 10, 18, 23, 29, 34, 41, 52, 59, 69],
    }

    best_score = 0.0
    best_set = [0, 1, 3]

    # Evaluate known optimal Golomb rulers
    for n, ruler in known_golomb.items():
        score = compute_score(ruler)
        if score > best_score:
            best_score = score
            best_set = list(ruler)

    # Add greedy ratio-based constructions as additional initial candidates
    if time.time() < deadline:
        for n_marks in [5, 6, 7, 8, 9, 10, 11, 12]:
            greedy_set = greedy_ratio_construct(n_marks, max_val=150)
            score = compute_score(greedy_set)
            if score > best_score:
                best_score = score
                best_set = list(greedy_set)

    # Local search to potentially find a better set
    # We try to improve the best set found so far
    current = list(best_set)
    current_score = best_score
    
    # Local search with wider range and 2-swap moves
    # Focus on small sets (N < 15) as they have higher potential scores
    if len(current) < 15 and time.time() < deadline:
        no_improve = 0
        max_no_improve = 200
        while no_improve < max_no_improve and time.time() < deadline - 5:
            improved = False
            # Try single-element moves with wider range
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                i = random.randint(1, len(current) - 1)
                old_val = current[i]
                # Try nearby values with wider range
                for delta in range(-20, 21):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val < 0 or new_val == 0:
                        continue
                    # Check if new_val is already in set
                    if new_val in current:
                        continue
                    
                    candidate = list(current)
                    candidate[i] = new_val
                    candidate = sorted(candidate)
                    
                    score = compute_score(candidate)
                    if score > current_score + 1e-9:
                        current = candidate
                        current_score = score
                        if score > best_score:
                            best_score = score
                            best_set = list(candidate)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            
            # Try 2-swap moves: replace two elements with new values
            if not improved and len(current) >= 2 and time.time() < deadline - 5:
                for _ in range(15):
                    if time.time() > deadline - 5:
                        break
                    i, j = random.sample(range(1, len(current)), 2)
                    old_i, old_j = current[i], current[j]
                    # Try replacing both with nearby values
                    for delta_i in range(-10, 11):
                        for delta_j in range(-10, 11):
                            if delta_i == 0 and delta_j == 0:
                                continue
                            new_i = old_i + delta_i
                            new_j = old_j + delta_j
                            if new_i < 0 or new_i == 0 or new_j < 0 or new_j == 0:
                                continue
                            if new_i in current or new_j in current:
                                continue
                            if new_i == new_j:
                                continue
                            
                            candidate = list(current)
                            candidate[i] = new_i
                            candidate[j] = new_j
                            candidate = sorted(candidate)
                            
                            score = compute_score(candidate)
                            if score > current_score + 1e-9:
                                current = candidate
                                current_score = score
                                if score > best_score:
                                    best_score = score
                                    best_set = list(candidate)
                                improved = True
                                no_improve = 0
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                no_improve += 1
                # Random restart from a known Golomb ruler or greedy construction if stuck
                if no_improve % 8 == 0:
                    if random.random() < 0.5:
                        n = random.randint(3, 12)
                        if n in known_golomb:
                            current = list(known_golomb[n])
                            current_score = compute_score(current)
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current)
                    else:
                        n_marks = random.randint(5, 12)
                        greedy_set = greedy_ratio_construct(n_marks, max_val=150)
                        current = list(greedy_set)
                        current_score = compute_score(current)
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current)

    return best_set
