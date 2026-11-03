import time
from math import log, floor, sqrt
import random
import numpy as np
from itertools import combinations

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def popcount(x):
    """Count number of 1-bits in x."""
    return bin(x).count('1')

def construct_bit_sparse_set(max_bits, max_weight, target_size, deadline):
    """
    Construct a set from numbers with sparse binary representations.
    Select numbers with at most max_weight 1-bits, up to max_bits total bits.
    Use greedy selection to maximize |U-U|/|U+U|.
    """
    # Generate all numbers with at most max_weight bits set in [0, 2^max_bits)
    candidates = [0]
    for bits in range(1, max_bits + 1):
        for combo in combinations(range(bits), min(max_weight, bits)):
            val = 0
            for b in combo:
                val |= (1 << b)
            if val not in candidates:
                candidates.append(val)
    
    candidates = sorted(set(candidates))
    
    if len(candidates) < target_size:
        target_size = len(candidates)
    
    # Greedy: start with 0 and 1, then greedily add elements that improve score
    U = [0, 1]
    current_score = score_set(U)
    
    remaining = [c for c in candidates if c not in U]
    
    while len(U) < target_size and time.time() < deadline - 5:
        best_cand = None
        best_new_score = current_score
        
        # Sample a subset of remaining candidates for speed
        sample_size = min(len(remaining), 50)
        sampled = random.sample(remaining, sample_size) if len(remaining) > sample_size else remaining
        
        for c in sampled:
            new_U = sorted(U + [c])
            new_score = score_set(new_U)
            if new_score > best_new_score:
                best_new_score = new_score
                best_cand = c
        
        if best_cand is not None:
            U.append(best_cand)
            U.sort()
            current_score = best_new_score
            remaining = [c for c in remaining if c != best_cand]
        else:
            break
    
    return U, current_score

def construct_hamming_style(bits, deadline):
    """
    Construct set using Hamming code structure: select numbers where 
    the positions of 1-bits correspond to codeword positions.
    This ensures controlled distances between elements.
    """
    # Generate candidate numbers with specific bit patterns
    # Use a stride-based approach: numbers at positions that are powers of 2 times a base
    U = [0]
    
    # Try different stride patterns
    for base in [1, 2, 3, 5, 7]:
        for stride in [1, 2, 3, 5]:
            candidate_set = [0]
            val = base
            while val < (1 << bits) and len(candidate_set) < 25:
                candidate_set.append(val)
                val += base * stride
                if val % (base * stride) != 0:
                    val = (val // (base * stride) + 1) * base * stride
            
            if len(candidate_set) >= 3:
                s = score_set(candidate_set)
                if s > 0:
                    return candidate_set, s
    
    return U, 0

def local_search_bit_aware(U, deadline, max_val_limit=2048, max_n_limit=30):
    """
    Local search that respects binary structure: prefer moves that maintain
    sparsity of binary representations.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_local_score = current_score
    best_local_set = list(U)
    
    no_improvement_count = 0
    max_no_improvement = 20
    
    while time.time() < deadline - 2:
        if no_improvement_count > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to ADD an element with sparse bits
            if len(U) < max_n_limit:
                # Generate candidates with sparse bits
                max_u = max(U)
                max_bits = max_u.bit_length() + 2
                
                candidates = []
                for weight in [1, 2, 3]:
                    for bits in range(1, max_bits + 1):
                        for combo in combinations(range(bits), weight):
                            val = 0
                            for b in combo:
                                val |= (1 << b)
                            if 0 < val <= max_val_limit and val not in U:
                                candidates.append(val)
                
                if candidates:
                    random.shuffle(candidates)
                    for c in candidates[:20]:
                        new_U = sorted(U + [c])
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
                            
        elif move_type < 0.6:
            # Try to REMOVE an element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    c = random.choice(removable)
                    new_U = [x for x in U if x != c]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
        else:
            # Try to REPLACE with a bit-flip variant
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                
                # Generate variants by flipping one bit
                variants = []
                for bit in range(old_val.bit_length() + 2):
                    new_val = old_val ^ (1 << bit)
                    if 0 < new_val <= max_val_limit and new_val not in U:
                        variants.append(new_val)
                
                if variants:
                    random.shuffle(variants)
                    for new_val in variants[:10]:
                        new_U = U.copy()
                        new_U[idx] = new_val
                        new_U = sorted(set(new_U))
                        if len(new_U) != len(U):
                            continue
                        new_score = score_set(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
            
        if improved:
            no_improvement_count = 0
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_set = list(U)
        else:
            no_improvement_count += 1
                
    return best_local_score, best_local_set

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    best_set = [0, 1, 4, 9, 11]
    best_score = score_set(best_set)

    # 1. Structured Candidates
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
        [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
        [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
        [0, 1, 3, 9, 27, 81, 243], # Powers of 3
    ]
    
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # 2. Bit-sparse construction (NEW STRUCTURAL APPROACH)
    # Try various parameter combinations
    for max_bits in [6, 8, 10, 12]:
        for max_weight in [1, 2, 3]:
            for target_size in [8, 12, 16, 20]:
                if time.time() > deadline - 30:
                    break
                U, s = construct_bit_sparse_set(max_bits, max_weight, target_size, deadline)
                if s > best_score:
                    best_score = s
                    best_set = U

    # 3. Hamming-style construction
    for bits in [6, 8, 10]:
        if time.time() > deadline - 20:
            break
        U, s = construct_hamming_style(bits, deadline)
        if s > best_score:
            best_score = s
            best_set = U

    # 4. Bit-aware local search
    seeds = [best_set]
    for cand in structured_candidates[:3]:
        if cand != best_set:
            seeds.append(cand)
    
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        base = seeds[restarts % len(seeds)]
        U = list(base)
        
        local_score, local_set = local_search_bit_aware(U, deadline, max_val_limit=2048, max_n_limit=25)
        
        if local_score > best_score:
            best_score = local_score
            best_set = local_set
            if len(seeds) < 5:
                seeds.append(local_set)
            else:
                seeds[-1] = local_set

    # 5. Final polish
    if time.time() < deadline - 2:
        current = list(best_set)
        current_score = best_score
        improved = True
        iteration = 0
        while improved and time.time() < deadline - 1 and iteration < 200:
            improved = False
            iteration += 1
            n = len(current)
            for i in range(1, n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    new_set = [current[j] if j != i else new_val for j in range(n)]
                    new_set = sorted(set(new_set))
                    if len(new_set) != n:
                        continue
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved and n < 20 and time.time() < deadline - 1:
                max_val = max(current)
                for new_val in range(max_val - 5, max_val + 6):
                    if new_val <= 0 or new_val in current:
                        continue
                    new_set = sorted(current + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

            if not improved and n > 3 and time.time() < deadline - 1:
                for i in range(1, n):
                    new_set = [current[j] for j in range(n) if j != i]
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        improved = True
                        break

        if current_score > best_score:
            best_score = current_score
            best_set = current

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
