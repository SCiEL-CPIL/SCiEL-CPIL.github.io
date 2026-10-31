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

    # Local search to potentially find a better set
    # We try to improve the best set found so far
    current = list(best_set)
    current_score = best_score
    
    # Simple local search: try moving one element
    # Focus on small sets (N < 15) as they have higher potential scores
    if len(current) < 15 and time.time() < deadline:
        no_improve = 0
        while no_improve < 100 and time.time() < deadline - 5:
            improved = False
            # Try to modify a random non-zero element
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                i = random.randint(1, len(current) - 1)
                old_val = current[i]
                # Try nearby values
                for delta in range(-5, 6):
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
            
            if not improved:
                no_improve += 1
                # Random restart from a known Golomb ruler if stuck
                if no_improve % 10 == 0:
                    n = random.randint(3, 12)
                    if n in known_golomb:
                        current = list(known_golomb[n])
                        current_score = compute_score(current)
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current)

    return best_set
