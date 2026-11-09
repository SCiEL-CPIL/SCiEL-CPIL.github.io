"""Optimized construction for 32 integers in 1..512 maximizing distinct 3-AP differences."""
import time
import random
import itertools

def count_progression_diffs(S):
    """Count distinct d>0 such that a, a+d, a+2d all in S."""
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            if b > a:
                d = b - a
                if a + 2 * d in S_set:
                    diffs.add(d)
    return len(diffs)

def count_progression_diffs_fast(S):
    """Faster count using set membership."""
    S_set = set(S)
    diffs = set()
    # For each pair (a, b) with a < b, check if a+2(b-a) = 2b-a is in S
    # d = b - a, need 2b - a in S
    for a in S:
        for b in S:
            if b > a:
                c = 2 * b - a
                if c in S_set and c > b:
                    diffs.add(b - a)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    best_score = 0
    best_set = None
    
    # Strategy 1: Start with structured initial solutions
    # Use arithmetic progressions and other structured sets
    
    # Initial candidate: try some structured sets
    candidates = []
    
    # Candidate 1: First 32 integers
    candidates.append(list(range(1, 33)))
    
    # Candidate 2: Geometric-like spread
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 2, 3, 5, 9, 17, 33, 65, 129,
                       257, 10, 20, 40, 80, 160, 320, 6, 12, 24, 48, 96, 192, 384, 7, 14])
    # Deduplicate and keep only 32 distinct in 1..512
    seen = set()
    c2 = []
    for x in candidates[-1]:
        if 1 <= x <= 512 and x not in seen:
            seen.add(x)
            c2.append(x)
    if len(c2) == 32:
        candidates.append(c2)
    
    # Candidate 3: Mix of small and large values
    c3 = list(range(1, 17)) + [i * 16 + 8 for i in range(16)]
    c3 = [x for x in c3 if 1 <= x <= 512]
    if len(c3) == 32:
        candidates.append(c3)
    
    # Candidate 4: Powers and multiples
    c4 = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 32, 40,
          48, 60, 64, 80, 96, 120, 128, 160, 192, 240, 256, 320, 384, 480, 12, 14]
    seen = set()
    c4d = []
    for x in c4:
        if 1 <= x <= 512 and x not in seen:
            seen.add(x)
            c4d.append(x)
    if len(c4d) == 32:
        candidates.append(c4d)
    
    # Evaluate initial candidates
    for cand in candidates:
        if len(cand) == 32 and all(1 <= x <= 512 for x in cand) and len(set(cand)) == 32:
            score = count_progression_diffs(cand)
            if score > best_score:
                best_score = score
                best_set = list(cand)
    
    # Strategy 2: Local search with targeted moves
    if best_set is None:
        best_set = list(range(1, 33))
        best_score = count_progression_diffs(best_set)
    
    random.seed(42)
    
    # Multi-start local search
    num_starts = 20
    for start_i in range(num_starts):
        if time.time() > deadline:
            break
        
        # Create initial solution
        if start_i == 0:
            current = list(best_set)
        elif start_i == 1:
            # Structured start
            current = list(range(1, 33))
        elif start_i == 2:
            # Random but biased towards structured
            current = random.sample(range(1, 513), 32)
        elif start_i == 3:
            # Start from best with some randomization
            current = list(best_set)
            for _ in range(5):
                idx = random.randint(0, 31)
                current[idx] = random.randint(1, 512)
            # Ensure uniqueness
            while len(set(current)) < 32:
                idx = random.randint(0, 31)
                current[idx] = random.randint(1, 512)
        else:
            # Random start
            current = random.sample(range(1, 513), 32)
        
        current_score = count_progression_diffs(current)
        
        # Local search: try swapping elements
        max_iter = 5000
        no_improve = 0
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current)
            
            # Try a random swap
            i, j = random.sample(range(32), 2)
            old_i, old_j = current[i], current[j]
            
            # Try swapping with a random new value
            new_val = random.randint(1, 512)
            if new_val in current:
                continue
            
            # Try replacing element i
            current[i] = new_val
            new_score = count_progression_diffs(current)
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
            else:
                current[i] = old_i
                no_improve += 1
            
            # Occasionally try swapping two elements
            if random.random() < 0.3:
                current[i], current[j] = current[j], current[i]
                new_score = count_progression_diffs(current)
                if new_score > current_score:
                    current_score = new_score
                    no_improve = 0
                else:
                    current[i], current[j] = current[j], current[i]
                    no_improve += 1
            
            if no_improve > 200:
                break
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_set = list(current)
    
    # Strategy 3: Targeted polish - try to add elements that complete 3-APs
    if time.time() < deadline - 5:
        current = list(best_set)
        current_score = count_progression_diffs(current)
        S_set = set(current)
        
        # Try removing one element and adding another to complete more 3-APs
        for trial in range(2000):
            if time.time() > deadline - 2:
                break
            
            # Pick an element to remove
            rem_idx = random.randint(0, 31)
            rem_val = current[rem_idx]
            
            # Find candidate additions that would create new 3-APs
            best_add = None
            best_add_score = current_score
            
            # For each pair in S\{rem_val}, check if adding a third completes a 3-AP
            # and if that third is not already in S
            candidates_add = set()
            for a in current:
                for b in current:
                    if a == b or a == rem_val or b == rem_val:
                        continue
                    # Check if a, b, c form 3-AP with c = 2b-a
                    c = 2 * b - a
                    if 1 <= c <= 512 and c not in S_set and c != rem_val:
                        candidates_add.add(c)
                    # Check if a, c, b form 3-AP with c = (a+b)/2
                    if (a + b) % 2 == 0:
                        c = (a + b) // 2
                        if 1 <= c <= 512 and c not in S_set and c != rem_val:
                            candidates_add.add(c)
                    # Check if c, a, b form 3-AP with c = 2a-b
                    c = 2 * a - b
                    if 1 <= c <= 512 and c not in S_set and c != rem_val:
                        candidates_add.add(c)
            
            # Try a few candidates
            for add_val in list(candidates_add)[:10]:
                test_set = current.copy()
                test_set[rem_idx] = add_val
                test_set.sort()
                if len(set(test_set)) != 32:
                    continue
                s = count_progression_diffs(test_set)
                if s > best_add_score:
                    best_add_score = s
                    best_add = add_val
            
            if best_add is not None:
                current[rem_idx] = best_add
                current_score = best_add_score
                S_set = set(current)
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current)
            else:
                # Random try
                new_val = random.randint(1, 512)
                if new_val not in current:
                    test_set = current.copy()
                    test_set[rem_idx] = new_val
                    if len(set(test_set)) == 32:
                        s = count_progression_diffs(test_set)
                        if s > current_score:
                            current[rem_idx] = new_val
                            current_score = s
                            S_set = set(current)
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(current)
    
    # Final verification
    best_set = sorted(set(best_set))
    if len(best_set) != 32:
        # Fallback
        best_set = list(range(1, 33))
    
    return best_set
