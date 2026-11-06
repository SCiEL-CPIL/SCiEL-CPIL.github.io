import itertools
import random
import time
from itertools import combinations

def is_sidon_fast(subset):
    """Check if a subset has all unordered pair sums unique (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size_fast(S):
    """Compute the size of the largest Sidon subset of S using bitmask enumeration."""
    n = len(S)
    max_size = 0
    # Precompute for speed
    for size in range(n, max_size, -1):
        if size <= max_size:
            break
        found = False
        # Enumerate all subsets of this size
        for mask in range(1, 1 << n):
            if bin(mask).count('1') != size:
                continue
            subset = [S[i] for i in range(n) if (mask >> i) & 1]
            if is_sidon_fast(subset):
                max_size = size
                found = True
                break
        if found:
            break
    return max_size

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    # Generate diverse candidate sets
    candidates = []
    
    # 1. Contiguous low
    candidates.append(list(range(12)))
    # 2. Contiguous high
    candidates.append(list(range(117, 129)))
    # 3. Arithmetic progression
    candidates.append(list(range(0, 128, 11)))
    # 4. Pairs summing to same value (creates many collisions)
    for target in [64, 100, 128]:
        pairs = [(i, target-i) for i in range(6) if 0 <= target-i <= 128 and target-i != i]
        cand = []
        for p in pairs:
            cand.extend(p)
        if len(cand) == 12:
            candidates.append(cand)
    # 5. Quadratic residues mod small prime
    for p in [13, 17, 19]:
        quad_residues = sorted(set((i*i) % p for i in range(p)))
        # Scale to fit in [0,128]
        scale = 128 // (p-1) if p > 1 else 1
        cand = [min(128, (r * scale)) for r in quad_residues[:12]]
        # Ensure distinct
        cand = list(set(cand))
        if len(cand) < 12:
            # Fill with random
            remaining = [x for x in range(129) if x not in cand]
            random.seed(p)
            cand.extend(random.sample(remaining, 12 - len(cand)))
        candidates.append(cand)
    
    # 6. Random sets with various seeds
    for seed in range(200):
        random.seed(seed)
        cand = random.sample(range(129), 12)
        candidates.append(cand)
    
    best_set = None
    best_score = float('inf')
    
    for cand in candidates:
        if time.time() > deadline:
            break
        score = max_sidon_size_fast(cand)
        if score < best_score:
            best_score = score
            best_set = list(cand)
            if best_score <= 4:
                # Can't do much better than 4 for 12 elements
                break
    
    if best_set is None:
        return list(range(12))
    
    # Local refinement: try replacing one element with another
    # Use a faster heuristic: only evaluate if the change could plausibly help
    if time.time() < deadline - 10:
        current_best = list(best_set)
        current_score = best_score
        best_set = list(current_best)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                old_val = current_best[i]
                # Try a subset of replacement values for speed
                for val in range(0, 129, 4):  # Step by 4 to reduce evaluations
                    if val in current_best:
                        continue
                    current_best[i] = val
                    new_set = current_best[:]
                    new_score = max_sidon_size_fast(new_set)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        best_set = list(current_best)
                    else:
                        current_best[i] = old_val
                # Also try exact neighbors
                for val in [old_val - 1, old_val + 1, old_val - 2, old_val + 2]:
                    if val < 0 or val > 128 or val in current_best:
                        continue
                    current_best[i] = val
                    new_set = current_best[:]
                    new_score = max_sidon_size_fast(new_set)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        best_set = list(current_best)
                    else:
                        current_best[i] = old_val
        
        # Final polish: try all values for each position
        if time.time() < deadline - 2:
            current_best = list(best_set)
            current_score = best_score
            for i in range(12):
                if time.time() > deadline - 2:
                    break
                old_val = current_best[i]
                for val in range(129):
                    if val in current_best:
                        continue
                    current_best[i] = val
                    new_set = current_best[:]
                    new_score = max_sidon_size_fast(new_set)
                    if new_score < current_score:
                        current_score = new_score
                        best_set = list(current_best)
                    else:
                        current_best[i] = old_val
    
    return best_set
