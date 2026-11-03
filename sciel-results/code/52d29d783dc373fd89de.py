"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    sums = set()
    diffs = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(U[i] + U[j])
        for j in range(i + 1, n):
            d = U[j] - U[i]
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def try_random_search(deadline):
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Known good sets - expanded with more diverse candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 8, 13],
        [0, 2, 3, 7, 12],
        [0, 1, 4, 11, 13],
        [0, 1, 4, 9, 13],
        [0, 1, 5, 10, 15],
        [0, 1, 6, 12, 18],
        [0, 1, 4, 7, 10],
        [0, 1, 5, 9, 13],
        [0, 2, 5, 9, 13],
        [0, 1, 4, 8, 12],
        [0, 1, 3, 6, 9, 12],
        [0, 1, 4, 7, 10, 13],
        [0, 1, 5, 10, 15, 20],
        [0, 1, 4, 9, 14, 19],
        [0, 1, 3, 8, 13, 18],
        [0, 1, 4, 10, 16, 22],
        [0, 1, 5, 11, 17, 23],
        [0, 2, 4, 6, 8, 10],
        [0, 1, 3, 5, 7, 9],
        [0, 1, 4, 8, 12, 16],
        [0, 1, 6, 11, 16, 21],
        [0, 1, 4, 9, 14, 19, 24],
        [0, 1, 5, 10, 15, 20, 25],
        [0, 1, 3, 7, 11, 15, 19],
        [0, 1, 4, 8, 12, 16, 20],
        [0, 2, 5, 8, 11, 14],
        [0, 1, 4, 7, 10, 13, 16],
        [0, 1, 5, 9, 13, 17, 21],
        [0, 1, 3, 6, 9, 12, 15],
        [0, 1, 4, 8, 12, 16, 20, 24],
    ]
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
    
    # Random local search with improved neighborhood
    random.seed(42)
    start_time = time.time()
    
    while time.time() < deadline - 5:
        # Start from a random valid set with varying sizes and ranges
        n = random.randint(4, 9)
        max_val = random.randint(15, 80)
        
        # Generate a random set with 0 and n-1 other elements
        other_elements = random.sample(range(1, max_val), n - 1)
        current = sorted([0] + other_elements)
        
        current_score = score_set(current)
        if current_score < 0:
            continue
        
        # Local search: try moving one element with wider neighborhood
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(current)):
                if i == 0:
                    continue  # Keep 0
                old_val = current[i]
                # Try changing to a nearby value with wider range
                for delta in [-3, -2, -1, 1, 2, 3]:
                    new_val = old_val + delta
                    if new_val <= 0 or new_val >= max_val:
                        continue
                    new_set = sorted(set([0] + [current[j] if j != i else new_val for j in range(len(current))]))
                    if len(new_set) != len(current):
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        break
        
        # Also try adding/removing elements
        if time.time() < deadline - 5:
            # Try adding an element
            if len(current) < 9:
                for new_val in range(1, max_val + 10):
                    if new_val not in current:
                        new_set = sorted(current + [new_val])
                        new_score = score_set(new_set)
                        if new_score > current_score:
                            current = new_set
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_set = current[:]
                            break
            
            # Try removing an element (not 0)
            if len(current) > 4:
                for i in range(1, len(current)):
                    new_set = [current[j] for j in range(len(current)) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        break
    
    # Final high-precision polish pass on the best set found
    if best_score > 0 and time.time() < deadline - 2:
        polish_set = best_set[:]
        polish_score = best_score
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try moving each element with wider neighborhood
            for i in range(len(polish_set)):
                if i == 0:
                    continue
                old_val = polish_set[i]
                for delta in range(-20, 21):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = sorted(set([0] + [polish_set[j] if j != i else new_val for j in range(len(polish_set))]))
                    if len(new_set) != len(polish_set):
                        continue
                    new_score = score_set(new_set)
                    if new_score > polish_score:
                        polish_set = new_set
                        polish_score = new_score
                        improved = True
                        break
            # Try pairwise swaps
            for i in range(1, len(polish_set)):
                for j in range(i + 1, len(polish_set)):
                    new_set = polish_set[:]
                    new_set[i], new_set[j] = new_set[j], new_set[i]
                    new_score = score_set(sorted(set(new_set)))
                    if new_score > polish_score:
                        polish_set = sorted(set(new_set))
                        polish_score = new_score
                        improved = True
                        break
                if improved:
                    break
            # Try adding an element with wider range
            if len(polish_set) < 10:
                max_current = max(polish_set)
                for new_val in range(1, max_current + 30):
                    if new_val not in polish_set:
                        new_set = sorted(polish_set + [new_val])
                        new_score = score_set(new_set)
                        if new_score > polish_score:
                            polish_set = new_set
                            polish_score = new_score
                            improved = True
                            break
            # Try removing an element
            if len(polish_set) > 4:
                for i in range(1, len(polish_set)):
                    new_set = [polish_set[j] for j in range(len(polish_set)) if j != i]
                    new_score = score_set(new_set)
                    if new_score > polish_score:
                        polish_set = new_set
                        polish_score = new_score
                        improved = True
                        break
        
        if polish_score > best_score:
            best_score = polish_score
            best_set = polish_set[:]
    
    return best_set

def construct():
    # Use a safety margin of 100s to stay well within the 120s limit
    deadline = time.time() + 100
    try:
        best_set = try_random_search(deadline)
    except Exception:
        # Fallback to a known valid set if search fails
        best_set = [0, 1, 4, 9, 11]
    return best_set

# EVOLVE-BLOCK-END
