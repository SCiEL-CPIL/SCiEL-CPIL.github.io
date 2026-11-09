"""Rewritten program with a stronger algorithm for integer sum-product extremal sets."""
import time
import random
import sys
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s kill
    
    def generate_smooth_numbers(limit=10000):
        """Generate all 7-smooth numbers <= limit."""
        nums = set([1])
        # 2^a * 3^b * 5^c * 7^d <= 10000
        for a in range(14):
            p2 = 2 ** a
            if p2 > limit: break
            for b in range(9):
                p3 = 3 ** b
                if p2 * p3 > limit: break
                for c in range(6):
                    p5 = 5 ** c
                    if p2 * p3 * p5 > limit: break
                    for d in range(5):
                        p7 = 7 ** d
                        val = p2 * p3 * p5 * p7
                        if val <= limit:
                            nums.add(val)
        return sorted(nums)

    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using numpy for speed."""
        A = np.array(sorted(A), dtype=np.int64)
        n = len(A)
        i_idx, j_idx = np.triu_indices(n)
        sums = A[i_idx] + A[j_idx]
        sumset_size = len(np.unique(sums))
        prods = A[i_idx] * A[j_idx]
        prodset_size = len(np.unique(prods))
        return max(sumset_size, prodset_size)

    smooth_numbers = generate_smooth_numbers(10000)
    # Filter out 1 if we want positive integers > 1? Task says positive integers. 1 is allowed.
    # However, including 1 might increase sumset size relative to productset?
    # Let's keep 1 in the pool but be careful.
    
    # Initial candidates
    candidates = []
    
    # 1. Smallest 20 smooth numbers
    if len(smooth_numbers) >= 20:
        candidates.append(smooth_numbers[:20])
        
    # 2. Random subsets of smooth numbers
    random.seed(42)
    for _ in range(10):
        subset = random.sample(smooth_numbers, 20)
        candidates.append(sorted(subset))
        
    # 3. Arithmetic progression (often good for sumset, bad for productset, but maybe mixed?)
    # Try small APs
    for start in [1, 2, 3]:
        for step in [1, 2, 3]:
            cand = [start + i*step for i in range(20)]
            if max(cand) <= 10000:
                candidates.append(cand)
                
    # Evaluate initial candidates and pick the best
    best_set = None
    best_score = float('inf')
    
    for cand in candidates:
        if len(set(cand)) != 20: continue
        if min(cand) < 1 or max(cand) > 10000: continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
        
    # Local Search with Restarts
    # We will perform local search on the best candidate found so far.
    # Strategy:
    # 1. Take current best set.
    # 2. Try swapping one element with a random smooth number.
    # 3. If score improves, accept.
    # 4. If no improvement for X steps, or time runs out, restart from a random candidate.
    
    current = best_set[:]
    current_score = best_score
    
    # Pre-compute indices for speed in evaluate? No, np.triu_indices is fast.
    # To speed up evaluate, we can avoid sorting if we maintain sorted order?
    # Swapping breaks order. We must sort before evaluate.
    # Sorting 20 elements is very fast.
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart
            # Pick a random smooth number subset as new starting point
            if random.random() < 0.5:
                new_start = random.sample(smooth_numbers, 20)
            else:
                # Perturb current best
                new_start = current[:]
                for _ in range(5):
                    idx = random.randint(0, 19)
                    new_start[idx] = random.choice(smooth_numbers)
                    # Ensure distinct
                    # Simple check
                    if len(set(new_start)) < 20:
                        # Fix duplicates
                        seen = set()
                        for i in range(20):
                            if new_start[i] in seen:
                                # Replace with a random smooth number not in seen
                                while True:
                                    r = random.choice(smooth_numbers)
                                    if r not in seen:
                                        new_start[i] = r
                                        break
                            seen.add(new_start[i])
            
            new_start = sorted(new_start)
            if len(set(new_start)) != 20:
                continue
            if min(new_start) < 1 or max(new_start) > 10000:
                continue
            
            new_score = evaluate(new_start)
            if new_score < current_score:
                current = new_start[:]
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]
            no_improve_count = 0
            continue
            
        # Try a random swap
        idx = random.randint(0, 19)
        old_val = current[idx]
        new_val = random.choice(smooth_numbers)
        
        if new_val == old_val:
            continue
            
        # Check if new_val is already in current (excluding idx)
        # To make this O(1), we can maintain a set of current values
        # But for n=20, linear search is fine.
        is_dup = False
        for j in range(20):
            if j != idx and current[j] == new_val:
                is_dup = True
                break
        if is_dup:
            continue
            
        # Perform swap
        current[idx] = new_val
        # Sort is required for correct sumset/productset calculation?
        # Actually, the set of sums and products is independent of order.
        # But np.triu_indices relies on the array structure.
        # The set of values {A[i] + A[j] | i<=j} is the same regardless of order.
        # So we don't strictly need to sort for the *set* calculation, 
        # but we do need distinct elements.
        # However, if we don't sort, the indices i,j are just positions.
        # The set of pairs (val_i, val_j) is the same.
        # So we can skip sorting in evaluate if we just use the values.
        # Let's optimize evaluate to not sort.
        
        # Re-evaluate
        # We can write a faster evaluate that assumes distinct positive ints.
        def evaluate_fast(A):
            A = np.array(A, dtype=np.int64)
            n = len(A)
            i_idx, j_idx = np.triu_indices(n)
            sums = A[i_idx] + A[j_idx]
            sumset_size = len(np.unique(sums))
            prods = A[i_idx] * A[j_idx]
            prodset_size = len(np.unique(prods))
            return max(sumset_size, prodset_size)
            
        new_score = evaluate_fast(current)
        
        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_set = current[:]
        else:
            # Revert
            current[idx] = old_val
            no_improve_count += 1
            
    return best_set
# EVOLVE-BLOCK-END
