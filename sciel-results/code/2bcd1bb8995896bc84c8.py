"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from math import log
import numpy as np

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    arr = np.array(U, dtype=np.int64)
    # Compute all pair sums (i <= j) using broadcasting
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    unique_sums = np.unique(sums)
    # Compute all differences (i < j), then add negatives and 0
    i_idx2, j_idx2 = np.triu_indices(n, k=1)
    diffs = arr[j_idx2] - arr[i_idx2]
    # Include negatives and 0
    all_diffs = np.concatenate([diffs, -diffs, np.array([0], dtype=np.int64)])
    unique_diffs = np.unique(all_diffs)
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1.0
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
    return 1.0 + log(ratio) / log(D)

def generate_structural_candidates():
    """Generate candidates using arithmetic, geometric, and binary-structure patterns."""
    candidates = []
    seen = set()
    
    def add_cand(c):
        c = sorted(set(c))
        if 0 in c and len(c) >= 3 and max(c) < 200:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                candidates.append(c)
    
    # Arithmetic progressions with various starts and steps
    for start in range(1, 8):
        for step in range(1, 12):
            for length in range(3, 12):
                cand = [0] + [start + i * step for i in range(length)]
                if max(cand) < 200:
                    add_cand(cand)
    
    # Geometric-like sequences
    for base in [2, 3, 4, 5]:
        for length in range(3, 9):
            cand = [0]
            val = 1
            for i in range(length):
                cand.append(val)
                val *= base
                if val > 200:
                    break
            if len(cand) >= 3:
                add_cand(cand)
    
    # Binary-structure candidates: subsets of {1, 2, 4, 8, 16, 32, ...}
    # These have high difference/sum ratios due to unique binary representations
    for bits in range(3, 8):
        base_vals = [2**k for k in range(bits)]
        # Try all subsets of size 3 to bits
        for size in range(3, bits + 1):
            for subset in itertools.combinations(base_vals, size):
                cand = [0] + list(subset)
                if max(cand) < 200:
                    add_cand(cand)
    
    # Mixed binary + arithmetic: perturb binary structures
    random.seed(123)
    for _ in range(80):
        bits = random.randint(3, 6)
        base_vals = [2**k for k in range(bits)]
        # Pick a random subset
        size = random.randint(3, min(bits, 7))
        subset = random.sample(base_vals, min(size, len(base_vals)))
        # Perturb some elements
        cand = [0]
        for v in subset:
            perturbed = v + random.randint(-2, 2)
            if perturbed > 0:
                cand.append(perturbed)
        add_cand(cand)
    
    # Mixed: arithmetic with perturbations
    for _ in range(60):
        n = random.randint(4, 10)
        base = random.randint(2, 8)
        step = random.randint(1, 5)
        cand = [0]
        val = base
        for i in range(n - 1):
            val += step + random.randint(-2, 2)
            if val > 150:
                break
            if val > 0:
                cand.append(val)
        add_cand(cand)
    
    # Known good sets
    known = [
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 8, 13],
        [0, 1, 4, 11, 13],
        [0, 1, 5, 10, 15],
        [0, 1, 6, 12, 18],
        [0, 1, 4, 7, 10],
        [0, 1, 5, 9, 13],
        [0, 1, 4, 8, 12],
        [0, 1, 3, 6, 9, 12],
        [0, 1, 4, 7, 10, 13],
        [0, 1, 5, 10, 15, 20],
        [0, 1, 4, 9, 14, 19],
        [0, 1, 3, 8, 13, 18],
        [0, 1, 4, 10, 16, 22],
        [0, 1, 5, 11, 17, 23],
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
        [0, 1, 2, 4, 8, 16],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 3, 4, 8, 12, 16],
        [0, 1, 2, 5, 8, 13, 21],
    ]
    for k in known:
        add_cand(k)
    
    # Generate Golomb Rulers (unique differences)
    # A Golomb ruler maximizes |U-U|.
    # We will generate some known small Golomb rulers and also search for near-Golomb rulers.
    golomb_rulers = [
        [0, 1, 4],
        [0, 1, 4, 9],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10, 18, 27],
        [0, 1, 4, 10, 18, 27, 37],
        [0, 1, 4, 10, 18, 27, 37, 49],
        [0, 1, 4, 10, 18, 27, 37, 49, 62],
        [0, 1, 4, 10, 18, 27, 37, 49, 62, 76],
        [0, 1, 4, 10, 18, 27, 37, 49, 62, 76, 91],
    ]
    for gr in golomb_rulers:
        if max(gr) < 200:
            add_cand(gr)
            
    # Generate "Sparse" sets that are not strictly Golomb but have high difference ratio
    # e.g. Powers of 3 (Ternary)
    for base in [3, 4]:
        for length in range(3, 8):
            cand = [0]
            val = 1
            for i in range(length):
                cand.append(val)
                val *= base
                if val > 200:
                    break
            if len(cand) >= 3:
                add_cand(cand)

    return candidates

def try_random_search(deadline):
    best_score = -1.0
    best_set = [0, 1, 4, 9, 11]
    
    # Generate structural candidates
    candidates = generate_structural_candidates()
    
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
    
    # Random local search with improved neighborhood
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Start from a random valid set with varying sizes and ranges
        n = random.randint(4, 10)
        max_val = random.randint(20, 120)
        
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
                # Use a mix of small and medium steps
                deltas = [-15, -10, -5, -3, -2, -1, 1, 2, 3, 5, 10, 15]
                random.shuffle(deltas)
                for delta in deltas:
                    new_val = old_val + delta
                    if new_val <= 0 or new_val > 150:
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
            if len(current) < 11:
                max_current = max(current)
                for new_val in range(1, min(max_current + 30, 150)):
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
    if best_score > 0 and time.time() < deadline - 3:
        polish_set = best_set[:]
        polish_score = best_score
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try moving each element with wider neighborhood
            for i in range(len(polish_set)):
                if i == 0:
                    continue
                old_val = polish_set[i]
                for delta in range(-40, 41):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0 or new_val > 180:
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
            if len(polish_set) < 12:
                max_current = max(polish_set)
                for new_val in range(1, min(max_current + 50, 180)):
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
    # Use a safety margin of 140s to stay within the 150s limit
    deadline = time.time() + 140
    try:
        best_set = try_random_search(deadline)
    except Exception:
        # Fallback to a known valid set if search fails
        best_set = [0, 1, 4, 9, 11]
    return best_set

# EVOLVE-BLOCK-END
