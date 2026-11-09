"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from itertools import combinations

P = {'n': 256}
_DEADLINE = None

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _valid_set(S):
    """Check if set S is valid: for all distinct a,b,c in S, a does not divide b*c."""
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            for k in range(j+1, m):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 85  # 85s budget, 5s safety margin
    
    n = P['n']
    nums = list(range(2, n + 1))
    N = len(nums)
    
    # Precompute forbidden pairs for each x
    # forbidden[x_idx] = set of (i, j) indices such that nums[i] and nums[j] are distinct 
    # and {x, nums[i], nums[j]} is invalid.
    
    forbidden = [set() for _ in range(N)]
    
    for x_idx, x in enumerate(nums):
        for i in range(N):
            if i == x_idx: continue
            a = nums[i]
            for j in range(i+1, N):
                if j == x_idx: continue
                b = nums[j]
                # Check if {x, a, b} is invalid
                # x | a*b OR a | x*b OR b | x*a
                if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                    forbidden[x_idx].add((i, j))
    
    # Precompute forbidden pair counts for sorting
    forbidden_counts = [len(forbidden[i]) for i in range(N)]
    
    def can_add(S_indices, x_idx):
        """Check if x_idx can be added to set S_indices (set of indices)."""
        for (i, j) in forbidden[x_idx]:
            if i in S_indices and j in S_indices:
                return False
        return True
    
    def greedy_search(order):
        S = set()
        for idx in order:
            if can_add(S, idx):
                S.add(idx)
        return S
    
    # Generate multiple greedy orders
    primes_idx = [i for i, v in enumerate(nums) if _is_prime(v)]
    others_idx = [i for i in range(N) if i not in primes_idx]
    
    others_sorted_safe = sorted(others_idx, key=lambda i: forbidden_counts[i])
    others_sorted_danger = sorted(others_idx, key=lambda i: -forbidden_counts[i])
    
    orders = [
        primes_idx + others_sorted_safe,
        primes_idx + others_sorted_danger,
        sorted(range(N), key=lambda i: forbidden_counts[i]),
        sorted(range(N), key=lambda i: -forbidden_counts[i]),
        list(range(N)),
    ]
    
    best_set = set()
    best_size = 0
    
    for order in orders:
        S = greedy_search(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    def local_search(S, max_iter=50):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < _DEADLINE:
            improved = False
            iters += 1
            S_list = list(S)
            for rem in S_list:
                if time.time() > _DEADLINE:
                    return S
                S2 = S - {rem}
                best_add = None
                for add_idx in range(N):
                    if add_idx in S2:
                        continue
                    if can_add(S2, add_idx):
                        if best_add is None or forbidden_counts[add_idx] < forbidden_counts[best_add]:
                            best_add = add_idx
                
                if best_add is not None:
                    S2.add(best_add)
                    if len(S2) > len(S):
                        S = S2
                        improved = True
                        break
        
        return S
    
    def local_search_1_for_2(S, max_iter=20):
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < _DEADLINE:
            improved = False
            iters += 1
            S_list = list(S)
            for rem in S_list:
                if time.time() > _DEADLINE:
                    return S
                S2 = S - {rem}
                found = False
                cands = [i for i in range(N) if i not in S2]
                cands.sort(key=lambda i: forbidden_counts[i])
                
                for i1 in cands:
                    if time.time() > _DEADLINE:
                        return S
                    if not can_add(S2, i1):
                        continue
                    S3 = S2 | {i1}
                    for i2 in cands:
                        if i2 == i1:
                            continue
                        if i2 in S3:
                            continue
                        if can_add(S3, i2):
                            S = S3 | {i2}
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        return S
    
    def local_search_2_for_3(S, max_iter=10):
        """Remove 2 elements, try to add 3."""
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < _DEADLINE:
            improved = False
            iters += 1
            S_list = list(S)
            for r1_idx in range(len(S_list)):
                if time.time() > _DEADLINE:
                    return S
                r1 = S_list[r1_idx]
                for r2_idx in range(r1_idx + 1, len(S_list)):
                    r2 = S_list[r2_idx]
                    S2 = S - {r1, r2}
                    found = False
                    cands = [i for i in range(N) if i not in S2]
                    cands.sort(key=lambda i: forbidden_counts[i])
                    
                    # Try to add 3 elements
                    for i1 in cands:
                        if time.time() > _DEADLINE:
                            return S
                        if not can_add(S2, i1):
                            continue
                        S3 = S2 | {i1}
                        for i2 in cands:
                            if i2 == i1 or i2 in S3:
                                continue
                            if not can_add(S3, i2):
                                continue
                            S4 = S3 | {i2}
                            for i3 in cands:
                                if i3 == i1 or i3 == i2 or i3 in S4:
                                    continue
                                if can_add(S4, i3):
                                    S = S4 | {i3}
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
        
        return S
    
    def random_perturbation_search(S, k_remove=2, n_trials=10):
        """Remove k random elements, then greedily refill with multiple orders."""
        S = set(S)
        best = S
        best_size = len(S)
        
        for trial in range(n_trials):
            if time.time() > _DEADLINE:
                return best
            
            # Remove k random elements
            S_trial = set(S)
            if len(S_trial) > k_remove:
                to_remove = random.sample(list(S_trial), k_remove)
                for r in to_remove:
                    S_trial.discard(r)
            
            # Try multiple refill orders
            cands = [i for i in range(N) if i not in S_trial]
            cands_safe = sorted(cands, key=lambda i: forbidden_counts[i])
            cands_danger = sorted(cands, key=lambda i: -forbidden_counts[i])
            cands_random = cands[:]
            random.shuffle(cands_random)
            
            for order in [cands_safe, cands_danger, cands_random]:
                S_refill = set(S_trial)
                for idx in order:
                    if can_add(S_refill, idx):
                        S_refill.add(idx)
                
                if len(S_refill) > best_size:
                    best_size = len(S_refill)
                    best = S_refill
        
        return best
    
    # Apply local searches
    best_set = local_search(best_set)
    best_size = len(best_set)
    
    best_set = local_search_1_for_2(best_set)
    best_size = len(best_set)
    
    # Use remaining time for random perturbation searches
    while time.time() < _DEADLINE - 5:
        candidate = random_perturbation_search(best_set, k_remove=2, n_trials=5)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate
            # Continue improving the new best
            best_set = local_search(best_set, max_iter=20)
            best_size = len(best_set)
            best_set = local_search_1_for_2(best_set, max_iter=10)
            best_size = len(best_set)
        elif time.time() > _DEADLINE - 10:
            break
    
    # Try 2-for-3 if we have time
    if time.time() < _DEADLINE - 10:
        candidate = local_search_2_for_3(best_set, max_iter=5)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = candidate
            best_set = local_search(best_set, max_iter=20)
            best_size = len(best_set)
    
    # Final verification
    S_list = sorted([nums[i] for i in best_set])
    if not _valid_set(S_list):
        # Fallback to primes
        S_list = sorted([nums[i] for i in primes_idx])
    
    return S_list

# EVOLVE-BLOCK-END
