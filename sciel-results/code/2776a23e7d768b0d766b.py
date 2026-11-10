import time
import random
from itertools import combinations

def is_sidon_subset(subset):
    """Check if a subset is Sidon (all pair sums including diagonals are unique)."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    """Find the largest Sidon subset size of A by exhaustive search."""
    n = len(A)
    # Try from largest possible down
    for size in range(n, 0, -1):
        for subset in combinations(A, size):
            if is_sidon_subset(subset):
                return size
    return 0

def count_sidon_k_subsets(A, k):
    """Count the number of k-element subsets of A that are Sidon."""
    count = 0
    for subset in combinations(A, k):
        if is_sidon_subset(subset):
            count += 1
    return count

def greedy_collision_build(deadline, seed=0):
    """Build ambient set greedily: each new element maximizes pair-sum collisions."""
    rng = random.Random(seed)
    chosen = []
    all_sums = set()  # sums of pairs among chosen elements
    
    for step in range(12):
        if time.time() > deadline:
            break
        
        best_val = None
        best_collision_score = -1
        best_sum_count = float('inf')
        
        # Try candidate values
        candidates = list(range(0, 129))
        rng.shuffle(candidates)
        
        for val in candidates:
            if val in chosen:
                continue
            
            # Compute new sums if we add val
            new_sums = set()
            collision_count = 0
            for prev in chosen:
                s = val + prev
                if s in all_sums:
                    collision_count += 1
                new_sums.add(s)
            # Diagonal
            diag = val + val
            if diag in all_sums:
                collision_count += 1
            new_sums.add(diag)
            
            # Score: maximize collisions, minimize total distinct sums
            # We want many collisions (bad for Sidon) and few new distinct sums
            score = collision_count * 10 - len(new_sums)
            
            if score > best_collision_score or (score == best_collision_score and len(new_sums) < best_sum_count):
                best_collision_score = score
                best_sum_count = len(new_sums)
                best_val = val
        
        if best_val is not None:
            chosen.append(best_val)
            # Update all_sums
            for prev in chosen[:-1]:
                all_sums.add(best_val + prev)
            all_sums.add(best_val + best_val)
    
    if len(chosen) == 12:
        return chosen
    else:
        return list(range(12))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s limit to be safe (90s wall clock)
    
    # Start with greedy collision build
    best_A = greedy_collision_build(deadline - 10, seed=42)
    best_ms = max_sidon_size(best_A)
    
    # Also try the interval as a baseline
    interval_A = list(range(12))
    interval_ms = max_sidon_size(interval_A)
    if interval_ms < best_ms:
        best_ms = interval_ms
        best_A = interval_A
    
    # Try multiple greedy builds with different seeds
    for seed in range(1, 50):
        if time.time() > deadline - 15:
            break
        A = greedy_collision_build(deadline - 10, seed=seed)
        ms = max_sidon_size(A)
        if ms < best_ms:
            best_ms = ms
            best_A = A
            if best_ms <= 3:
                break
    
    # Local search: try to reduce max Sidon size further
    if best_ms > 3 and time.time() < deadline - 5:
        A = best_A[:]
        current_ms = best_ms
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            # Try random swaps
            for _ in range(50):
                idx = random.randint(0, 11)
                old_val = A[idx]
                new_val = random.randint(0, 128)
                if new_val in A:
                    continue
                A[idx] = new_val
                new_ms = max_sidon_size(A)
                if new_ms < current_ms:
                    current_ms = new_ms
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_A = A[:]
                        if best_ms <= 3:
                            return best_A
                    improved = True
                else:
                    A[idx] = old_val
    
    return best_A
