"""
Combined approach: Focus on small sets with binary-like structures (high difference/sum ratio)
+ Efficient local search with structural initialization.
"""
import time
import random
import numpy as np
from math import log

# EVOLVE-BLOCK-START

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    # Use sets for small n to avoid numpy overhead for tiny sets
    # For n <= 10, python sets are fast enough and simpler
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
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def generate_initial_candidates():
    """
    Generate candidates biased towards small, binary-like structures.
    """
    candidates = []
    
    # 1. Small binary-like sets (powers of 2, or 2^k - 1, etc.)
    # Size 3
    for a in range(1, 20):
        for b in range(a+1, 50):
            candidates.append([0, a, b])
            
    # Size 4
    for a in range(1, 10):
        for b in range(a+1, 20):
            for c in range(b+1, 50):
                # Bias towards geometric growth
                if c > 2*b or c < b: 
                    candidates.append([0, a, b, c])
                else:
                    # Include some arithmetic
                    if random.random() < 0.2:
                        candidates.append([0, a, b, c])

    # Specific known good structures
    known_good = [
        [0, 1, 3, 7],
        [0, 1, 3, 8],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 5, 10],
        [0, 1, 2, 5],
        [0, 1, 2, 6],
        [0, 1, 3, 6],
        [0, 1, 4, 7],
        [0, 1, 5, 9],
        [0, 1, 2, 7],
        [0, 1, 3, 9],
        [0, 1, 4, 11],
        [0, 1, 5, 12],
        [0, 1, 2, 8],
        [0, 1, 3, 10],
        [0, 1, 4, 12],
        [0, 1, 5, 13],
        [0, 1, 2, 9],
        [0, 1, 3, 11],
        [0, 1, 4, 13],
        [0, 1, 5, 14],
        [0, 1, 2, 10],
        [0, 1, 3, 12],
        [0, 1, 4, 14],
        [0, 1, 5, 15],
        [0, 1, 2, 11],
        [0, 1, 3, 13],
        [0, 1, 4, 15],
        [0, 1, 5, 16],
    ]
    candidates.extend(known_good)
    
    # 2. Random small sets
    random.seed(42)
    for _ in range(100):
        n = random.randint(3, 6)
        max_val = random.randint(10, 50)
        other_elements = random.sample(range(1, max_val), n - 1)
        cand = sorted([0] + other_elements)
        candidates.append(cand)
        
    return candidates

def local_search(U, deadline):
    """
    Local search with a focus on small sets and structural moves.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = U[:]
    
    no_improvement = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2:
        if no_improvement > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if move_type < 0.6:
            # Move 1: Perturb an element
            # Focus on small deltas for small sets
            i = random.randint(1, len(U) - 1)
            old_val = U[i]
            # Try small deltas
            deltas = list(range(-5, 6))
            random.shuffle(deltas)
            for delta in deltas:
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                # Keep max value bounded to maintain high score potential
                if new_val > 100:
                    continue
                new_set = sorted(set(U[:i] + [new_val] + U[i+1:]))
                if len(new_set) != len(U):
                    continue
                new_score = score_set(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = U[:]
                    break
        elif move_type < 0.8:
            # Move 2: Replace an element with a "binary-like" neighbor
            i = random.randint(1, len(U) - 1)
            old_val = U[i]
            # Find closest power of 2
            p2 = 1
            while p2 < old_val:
                p2 *= 2
            candidates = [p2, p2//2, p2*2, old_val//2, old_val*2]
            random.shuffle(candidates)
            for new_val in candidates:
                if new_val <= 0 or new_val > 100:
                    continue
                new_set = sorted(set(U[:i] + [new_val] + U[i+1:]))
                if len(new_set) != len(U):
                    continue
                new_score = score_set(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = U[:]
                    break
        else:
            # Move 3: Add or Remove element (only for small sets)
            if len(U) < 6 and random.random() < 0.5:
                # Add
                # Try adding a value that is sum or diff of existing
                max_current = max(U)
                add_candidates = set()
                for a in U:
                    for b in U:
                        s = a + b
                        d = abs(a - b)
                        if 0 < s <= 100:
                            add_candidates.add(s)
                        if 0 < d <= 100:
                            add_candidates.add(d)
                add_candidates = list(add_candidates)
                random.shuffle(add_candidates)
                for c in add_candidates[:10]:
                    if c in U:
                        continue
                    new_set = sorted(U + [c])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        U = new_set
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = U[:]
                        break
            elif len(U) > 3:
                # Remove
                i = random.randint(1, len(U) - 1)
                new_set = U[:i] + U[i+1:]
                new_score = score_set(new_set)
                if new_score > current_score:
                    U = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = U[:]
                    break
        
        if improved:
            no_improvement = 0
        else:
            no_improvement += 1
            
    return best_set

def construct():
    # Safety margin: 120s limit, use 115s
    deadline = time.time() + 115
    
    # 1. Generate and score initial candidates
    candidates = generate_initial_candidates()
    best_score = -1
    best_set = [0, 1, 3, 7] # Fallback
    
    for c in candidates:
        if time.time() > deadline - 5:
            break
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
            
    # 2. Local search from the best initial candidate
    if time.time() < deadline - 5:
        refined_set = local_search(best_set, deadline)
        refined_score = score_set(refined_set)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
            
    # 3. Try a few other good candidates with local search if time permits
    # Sort candidates by score to prioritize good ones
    scored_candidates = []
    for c in candidates:
        s = score_set(c)
        scored_candidates.append((s, c))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    for s, c in scored_candidates[:10]:
        if time.time() > deadline - 5:
            break
        if s <= best_score:
            break
        refined_set = local_search(c, deadline)
        refined_score = score_set(refined_set)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
            
    return best_set

# EVOLVE-BLOCK-END
