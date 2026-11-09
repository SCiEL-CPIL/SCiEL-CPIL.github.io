"""Rewritten program with a stronger algorithm for integer sum-product extremal sets."""
import time
import random
import sys
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using pure Python for speed."""
        A = sorted(A)
        n = len(A)
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(i, n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    def generate_smooth_numbers(limit=10000):
        """Generate all 7-smooth numbers <= limit."""
        nums = set([1])
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

    smooth_numbers = generate_smooth_numbers(10000)
    random.seed(42)
    
    candidates = []
    
    # Strategy 1: Arithmetic progression
    for start in range(1, 30):
        for step in range(1, 15):
            ap = [start + i * step for i in range(20)]
            if max(ap) <= 10000 and len(set(ap)) == 20:
                candidates.append(ap)
    
    # Strategy 2: Geometric progression
    for base in range(2, 10):
        gp = [base ** i for i in range(20)]
        if max(gp) <= 10000 and len(set(gp)) == 20:
            candidates.append(gp)
    
    # Strategy 3: Mixed AP + GP with overlap
    for start in range(1, 20):
        for step in range(1, 10):
            ap_part = [start + i * step for i in range(10)]
            if max(ap_part) > 5000:
                continue
            for base in range(2, 6):
                gp_part = [base ** i for i in range(10)]
                if max(gp_part) > 10000:
                    continue
                combined = list(set(ap_part + gp_part))
                if len(combined) >= 20:
                    combined = sorted(combined[:20])
                    if len(set(combined)) == 20 and max(combined) <= 10000:
                        candidates.append(combined)
    
    # Strategy 4: Smooth number subsets
    if len(smooth_numbers) >= 20:
        candidates.append(smooth_numbers[:20])
        indices = np.linspace(0, len(smooth_numbers)-1, 20).astype(int)
        candidates.append([smooth_numbers[i] for i in indices])
        for _ in range(15):
            subset = random.sample(smooth_numbers, 20)
            candidates.append(sorted(subset))
    
    # Strategy 5: Small integers with structure
    candidates.append(list(range(1, 21)))
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36])
    
    # Evaluate initial candidates
    for cand in candidates:
        if len(cand) != 20 or len(set(cand)) != 20 or min(cand) < 1 or max(cand) > 10000:
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
        
    # Local Search with Incremental Updates
    def build_counters_dict(A):
        A = sorted(A)
        n = len(A)
        sum_counts = {}
        prod_counts = {}
        for i in range(n):
            for j in range(i, n):
                s = A[i] + A[j]
                p = A[i] * A[j]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
        return sum_counts, prod_counts
        
    def get_score_dict(sum_counts, prod_counts):
        return max(len(sum_counts), len(prod_counts))
        
    def update_counters_remove(sum_counts, prod_counts, A, idx):
        val = A[idx]
        n = len(A)
        for j in range(n):
            if j == idx:
                continue
            s = val + A[j]
            p = val * A[j]
            sum_counts[s] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]
            prod_counts[p] -= 1
            if prod_counts[p] == 0:
                del prod_counts[p]
        s = val + val
        p = val * val
        sum_counts[s] -= 1
        if sum_counts[s] == 0:
            del sum_counts[s]
        prod_counts[p] -= 1
        if prod_counts[p] == 0:
            del prod_counts[p]
            
    def update_counters_add(sum_counts, prod_counts, A, idx, new_val):
        n = len(A)
        for j in range(n):
            if j == idx:
                continue
            s = new_val + A[j]
            p = new_val * A[j]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        s = new_val + new_val
        p = new_val * new_val
        sum_counts[s] = sum_counts.get(s, 0) + 1
        prod_counts[p] = prod_counts.get(p, 0) + 1

    # Use a pool of candidate values: smooth numbers + some structured values
    pool = smooth_numbers.copy()
    for start in range(1, 50):
        for step in range(1, 20):
            for i in range(20):
                val = start + i * step
                if val <= 10000:
                    pool.append(val)
    pool = sorted(set(pool))
    
    temp = 10.0
    cooling = 0.9998
    
    num_iterations = 500000
    for iteration in range(num_iterations):
        if time.time() > deadline:
            break
        
        if temp < 0.1:
            temp = 5.0
            current = best_set[:]
            current.sort()
            sum_counts, prod_counts = build_counters_dict(current)
            current_score = get_score_dict(sum_counts, prod_counts)
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
                
        i = random.randint(0, 19)
        old_val = current[i]
        
        new_val = random.choice(pool)
        
        if new_val == old_val:
            continue
            
        is_dup = False
        for j in range(20):
            if j != i and current[j] == new_val:
                is_dup = True
                break
        if is_dup:
            continue
            
        update_counters_remove(sum_counts, prod_counts, current, i)
        current[i] = new_val
        update_counters_add(sum_counts, prod_counts, current, i, new_val)
        
        new_score = get_score_dict(sum_counts, prod_counts)
        delta_score = new_score - current_score
        
        if delta_score < 0 or (delta_score == 0 and random.random() < 0.5) or \
           (delta_score > 0 and random.random() < np.exp(-delta_score / temp)):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_set = current[:]
        else:
            update_counters_remove(sum_counts, prod_counts, current, i)
            current[i] = old_val
            update_counters_add(sum_counts, prod_counts, current, i, old_val)
            
        temp *= cooling
        
    return best_set
# EVOLVE-BLOCK-END

