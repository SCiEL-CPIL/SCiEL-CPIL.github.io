import time
import random
import itertools
from math import log
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    # Use NumPy for faster computation
    U_arr = np.array(U, dtype=np.int64)
    
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 1:
        return -1
        
    max_u = U_arr[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def binary_structure_score(U):
    """Score how well U's elements have non-overlapping bit patterns (fewer carry collisions)"""
    if len(U) < 2:
        return 0
    # Count how many pairs have overlapping 1-bits (potential carry sources)
    overlap_count = 0
    for i in range(len(U)):
        for j in range(i+1, len(U)):
            overlap_count += bin(U[i] & U[j]).count('1')
    # Normalize: fewer overlaps is better
    return -overlap_count / (len(U) * (len(U)-1) / 2)

def score_geometric(b, m):
    """Score a geometric progression {floor(b^k) for k=0..m}"""
    if m < 1:
        return -1
    U = [int(b**k) for k in range(m + 1)]
    # Remove duplicates
    U = sorted(set(U))
    if len(U) < 2 or 0 not in U:
        return -1
    return score_set(U)

def ternary_search_geometric(m, b_lo, b_hi, deadline, iters=50):
    """Ternary search over base b for fixed m"""
    for _ in range(iters):
        if time.time() > deadline:
            break
        b1 = b_lo + (b_hi - b_lo) / 3.0
        b2 = b_hi - (b_hi - b_lo) / 3.0
        s1 = score_geometric(b1, m)
        s2 = score_geometric(b2, m)
        if s1 < s2:
            b_lo = b1
        else:
            b_hi = b2
    b_opt = (b_lo + b_hi) / 2.0
    s_opt = score_geometric(b_opt, m)
    return b_opt, s_opt

def continuous_geometric_search(deadline):
    """Sweep over m and ternary search over b for each m"""
    best_score = -1
    best_set = None
    best_params = None
    
    for m in range(2, 20):
        if time.time() > deadline - 5:
            break
        # Search over base b in a reasonable range
        b_lo, b_hi = 1.01, 3.0
        b_opt, s_opt = ternary_search_geometric(m, b_lo, b_hi, deadline, iters=30)
        if s_opt > best_score:
            best_score = s_opt
            U = sorted(set([int(b_opt**k) for k in range(m + 1)]))
            best_set = U
            best_params = (b_opt, m)
    
    return best_set, best_score

def generate_parametric_candidates(deadline):
    """Generate candidate sets from parametric families"""
    candidates = []
    
    # Geometric progressions with various bases
    for b in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        for m in range(3, 12):
            if time.time() > deadline - 10:
                return candidates
            U = sorted(set([int(b**k) for k in range(m + 1)]))
            if len(U) >= 2 and 0 in U:
                candidates.append(U)
    
    # Arithmetic progressions
    for d in [1, 2, 3, 4, 5]:
        for n in range(3, 12):
            if time.time() > deadline - 10:
                return candidates
            U = list(range(0, d * n, d))
            if len(U) >= 2:
                candidates.append(U)
    
    # Mixed: geometric with small perturbations
    for b in [2.0, 2.5, 3.0]:
        for m in range(4, 10):
            if time.time() > deadline - 10:
                return candidates
            base = [int(b**k) for k in range(m + 1)]
            # Add small offsets
            U = sorted(set([base[i] + (i % 3) for i in range(m + 1)]))
            if 0 not in U:
                U = sorted(set(U + [0]))
            if len(U) >= 2:
                candidates.append(U)
    
    # Binary-structure-based candidates: select elements with sparse, non-overlapping bit patterns
    # Strategy: pick elements that use distinct "bit blocks" to minimize carry collisions in sums
    for n_bits in range(3, 8):
        if time.time() > deadline - 10:
            return candidates
        # Generate candidates using subsets of bit positions
        for n_elems in range(3, min(n_bits + 2, 11)):
            if time.time() > deadline - 10:
                return candidates
            # Choose which bit positions to use (ensuring spacing)
            # Use a greedy approach: pick elements that minimize pairwise bit overlap
            best_U = [0]
            for _ in range(n_elems - 1):
                best_val = None
                best_overlap = float('inf')
                # Try candidates in a range
                for trial in range(1, 1 << n_bits):
                    # Check overlap with existing elements
                    overlap = sum(bin(best_U[i] & trial).count('1') for i in range(len(best_U)))
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_val = trial
                if best_val is not None and best_val not in best_U:
                    best_U.append(best_val)
            best_U = sorted(set(best_U))
            if len(best_U) >= 2 and 0 in best_U:
                candidates.append(best_U)
    
    # Fibonacci-like sequences (known to have good sum/diff properties)
    for start in [1, 2]:
        for n_fib in range(4, 12):
            if time.time() > deadline - 10:
                return candidates
            fib = [start, start]
            for _ in range(n_fib - 2):
                fib.append(fib[-1] + fib[-2])
            U = sorted(set([0] + fib[:n_fib]))
            if len(U) >= 2:
                candidates.append(U)
    
    return candidates

def local_search(start_set, deadline, max_iter=5000):
    current = sorted(set(start_set))
    if 0 not in current:
        current = sorted(set(current + [0]))
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    # Simulated annealing with adaptive temperature
    T = 0.08
    cooling = 0.995
    T_min = 0.001
    
    # Track recent scores for adaptive cooling
    recent_scores = []
    
    while time.time() < deadline - 2 and max_iter > 0:
        max_iter -= 1
        if len(current) <= 1:
            break
            
        # Randomly choose an operation with weighted probabilities
        op = random.random()
        
        if op < 0.6:
            # Move an element - try different strategies
            i = random.randint(1, len(current) - 1)
            old_val = current[i]
            
            strategy = random.random()
            if strategy < 0.25:
                # Small local perturbation
                delta = random.randint(-5, 5)
                new_val = old_val + delta
            elif strategy < 0.45:
                # Scale relative to current max
                new_val = random.randint(1, max(2, int(current[-1] * 1.5)))
            elif strategy < 0.65:
                # Try to create geometric-like spacing
                if i > 0:
                    prev = current[i-1]
                    next_val = current[i+1] if i + 1 < len(current) else current[-1] + 1
                    mid = (prev + next_val) // 2
                    new_val = max(1, min(next_val - 1, mid + random.randint(-3, 3)))
                else:
                    new_val = random.randint(1, current[-1] + 10)
            elif strategy < 0.85:
                # Binary-structure-aware move: try values with low bit overlap with neighbors
                # Find a value that minimizes bit overlap with existing elements
                best_candidate = None
                best_overlap = float('inf')
                max_bit = max(8, current[-1].bit_length() + 1)
                for _ in range(20):
                    trial = random.randint(1, (1 << max_bit) - 1)
                    if trial in current:
                        continue
                    overlap = sum(bin(current[j] & trial).count('1') for j in range(len(current)) if j != i)
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_candidate = trial
                if best_candidate is not None:
                    new_val = best_candidate
                else:
                    new_val = random.randint(1, current[-1] + 30)
            else:
                # Large jump
                new_val = random.randint(1, current[-1] + 30)
                
            if new_val <= 0:
                continue
                
            new_set = current[:]
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
            if len(new_set) != len(current):
                continue
                
        elif op < 0.8:
            # Add an element
            if len(current) >= 12: # Cap size
                continue
            # Try to add in a structured way
            if random.random() < 0.4:
                # Add near existing elements
                i = random.randint(1, len(current) - 1)
                new_val = current[i] + random.randint(1, 5)
            elif random.random() < 0.7:
                # Binary-structure-aware addition: find value with low bit overlap
                best_candidate = None
                best_overlap = float('inf')
                max_bit = max(8, current[-1].bit_length() + 1)
                for _ in range(20):
                    trial = random.randint(1, (1 << max_bit) - 1)
                    if trial in current:
                        continue
                    overlap = sum(bin(current[j] & trial).count('1') for j in range(len(current)))
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_candidate = trial
                if best_candidate is not None:
                    new_val = best_candidate
                else:
                    new_val = random.randint(1, current[-1] + 15)
            else:
                # Add at end or random position
                new_val = random.randint(1, current[-1] + 15)
            if new_val in current:
                continue
            new_set = sorted(current + [new_val])
            
        else:
            # Remove an element (not 0)
            if len(current) <= 3:
                continue
            # Prefer removing elements that might be causing issues
            # (e.g., very close to neighbors or high bit overlap)
            i = random.randint(1, len(current) - 1)
            new_set = current[:i] + current[i+1:]
            
        new_score = score_set(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best_set = new_set[:]
            recent_scores.append(new_score)
            if len(recent_scores) > 20:
                recent_scores.pop(0)
            
        # Accept if better or with probability
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
            else:
                prob = 1.0
                
            if random.random() < prob:
                current = new_set
                current_score = new_score
        
        # Adaptive cooling: slow down if making progress, speed up if stuck
        if recent_scores and len(recent_scores) >= 5:
            if new_score > max(recent_scores[-5:]):
                cooling = max(0.99, cooling * 0.99)  # Slow cooling
            else:
                cooling = min(0.999, cooling * 1.01)  # Fast cooling
        
        T = max(T_min, T * cooling)
        
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    best_set = [0, 1, 4]
    best_score = score_set(best_set)
    
    random.seed(42)
    
    # Phase 1: Parametric search - geometric, arithmetic, and binary-structured
    if time.time() < deadline - 30:
        param_candidates = generate_parametric_candidates(deadline)
        for cand in param_candidates:
            if time.time() > deadline - 25:
                break
            s = score_set(cand)
            if s > best_score:
                best_score = s
                best_set = cand[:]
        
        # Continuous geometric search
        geo_set, geo_score = continuous_geometric_search(deadline - 15)
        if geo_score > best_score:
            best_score = geo_score
            best_set = geo_set[:]
    
    # Phase 2: Local search from diverse seeds
    seeds = [
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
        [0, 1, 4],
        [0, 1, 4, 16],
        [0, 1, 3, 9, 27],
        [0, 1, 2, 4, 7, 11],
        [0, 1, 3, 6, 10, 15],
        [0, 2, 5, 9, 14],
        [0, 1, 4, 9, 16, 25],
    ]
    
    # Add binary-structure-based seeds
    for n_bits in [4, 5, 6]:
        # Greedy binary-structure construction
        U = [0]
        for _ in range(min(5, n_bits + 1)):
            best_val = None
            best_overlap = float('inf')
            for trial in range(1, 1 << n_bits):
                if trial in U:
                    continue
                overlap = sum(bin(U[j] & trial).count('1') for j in range(len(U)))
                if overlap < best_overlap:
                    best_overlap = overlap
                    best_val = trial
            if best_val is not None:
                U.append(best_val)
        U = sorted(set(U))
        if len(U) >= 3:
            seeds.append(U)
    
    # Add random seeds
    for _ in range(15):
        n = random.randint(4, 10)
        max_val = random.randint(20, 100)
        U = [0]
        other_elements = random.sample(range(1, max_val), n - 1)
        U.extend(other_elements)
        seeds.append(sorted(U))
    
    # Add best parametric result as a seed
    if best_set != [0, 1, 4]:
        seeds.insert(0, best_set[:])
    
    # Local search from each seed
    for seed in seeds:
        if time.time() > deadline - 8:
            break
        local_best, local_score = local_search(seed, deadline, max_iter=4000)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
    
    return best_set
