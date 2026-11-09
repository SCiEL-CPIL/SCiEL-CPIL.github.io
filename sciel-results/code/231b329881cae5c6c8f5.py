"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import log

# Known optimal Golomb rulers (Sidon sets) for small n
# These are minimal-length Golomb rulers
KNOWN_GOLOMB = {
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 10, 12],
    6: [0, 1, 4, 10, 18, 23],
    7: [0, 1, 7, 12, 18, 23, 31],
    8: [0, 1, 7, 12, 18, 25, 34, 41],
    9: [0, 1, 7, 12, 18, 25, 34, 44, 51],
    10: [0, 1, 7, 12, 18, 25, 34, 44, 52, 62],
    11: [0, 1, 7, 12, 18, 25, 34, 44, 52, 62, 70],
    12: [0, 1, 7, 12, 18, 25, 34, 44, 52, 62, 70, 79],
}

def compute_score(U):
    """Compute the raw score for a given set U."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def is_sidon(U):
    """Check if U is a Sidon set (all pairwise sums are distinct or symmetric)."""
    sums = {}
    for i in range(len(U)):
        for j in range(i, len(U)):
            s = U[i] + U[j]
            if s in sums:
                # Check if it's the same pair
                if (sums[s][0], sums[s][1]) != (i, j) and (sums[s][1], sums[s][0]) != (i, j):
                    return False
            sums[s] = (i, j)
    return True

def local_search(U, time_limit=10):
    """Perform local search to improve the Golomb ruler."""
    start_time = time.time()
    U = sorted(set(U))
    best_U = list(U)
    best_score = compute_score(U)
    
    n = len(U)
    if n < 2:
        return best_U
    
    # Try moving each element
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(1, n):  # Keep 0 fixed
            # Try small perturbations
            for delta in range(-3, 4):
                if delta == 0:
                    continue
                new_U = list(U)
                new_val = new_U[i] + delta
                if new_val < 0 or new_val in new_U:
                    continue
                new_U[i] = new_val
                new_U.sort()
                if new_U[0] != 0:
                    continue
                # Check Sidon property
                if not is_sidon(new_U):
                    continue
                score = compute_score(new_U)
                if score > best_score:
                    best_score = score
                    best_U = new_U
                    U = new_U
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_U

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave safety margin
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    # Try known Golomb rulers
    for n in range(2, 13):
        if time.time() > deadline:
            break
        if n in KNOWN_GOLOMB:
            U = list(KNOWN_GOLOMB[n])
            score = compute_score(U)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Local search on the best candidate
    if time.time() < deadline - 5:
        improved_U = local_search(best_U, time_limit=10)
        score = compute_score(improved_U)
        if score > best_score:
            best_score = score
            best_U = improved_U
    
    # Try random Sidon sets
    if time.time() < deadline - 10:
        random.seed(42)
        for n in range(4, 10):
            if time.time() > deadline:
                break
            for _ in range(200):
                if time.time() > deadline:
                    break
                # Generate random Sidon set
                U = [0]
                for _ in range(n - 1):
                    # Try to add a random element
                    for _ in range(50):
                        val = random.randint(1, 100)
                        if val not in U:
                            new_U = U + [val]
                            new_U.sort()
                            if is_sidon(new_U):
                                U = new_U
                                break
                if len(U) == n:
                    score = compute_score(U)
                    if score > best_score:
                        best_score = score
                        best_U = U
                        # Local search on this
                        if time.time() < deadline - 5:
                            improved_U = local_search(best_U, time_limit=5)
                            score = compute_score(improved_U)
                            if score > best_score:
                                best_score = score
                                best_U = improved_U
    
    return best_U
# EVOLVE-BLOCK-END
