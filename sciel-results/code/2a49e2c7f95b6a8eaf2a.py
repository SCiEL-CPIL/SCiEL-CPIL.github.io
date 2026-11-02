import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(U)
        if len(U) < 2:
            return 0.0
        
        max_u = U[-1]
        if max_u == 0:
            return 0.0
            
        sum_set = set()
        diff_set = set()
        
        n = len(U)
        for i in range(n):
            for j in range(n):
                sum_set.add(U[i] + U[j])
                diff_set.add(U[i] - U[j])
        
        size_sum = len(sum_set)
        size_diff = len(diff_set)
        
        if size_sum == 0:
            return 0.0
            
        ratio = size_diff / size_sum
        if ratio <= 1.0:
            return 1.0
            
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
            
        return 1.0 + math.log(ratio) / denom

    def is_golomb_ruler(U):
        """Check if all pairwise differences are distinct."""
        diffs = set()
        n = len(U)
        for i in range(n):
            for j in range(i + 1, n):
                d = U[j] - U[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def generate_golomb_ruler(n, max_val=200):
        """Generate a Golomb ruler with n marks starting at 0."""
        if n <= 1:
            return [0]
        if n == 2:
            return [0, 1]
        
        # Backtracking with pruning
        def backtrack(marks, remaining, max_allowed):
            if len(marks) == n:
                return marks[:]
            
            for pos in range(marks[-1] + 1, max_allowed + 1):
                # Check if adding pos creates duplicate differences
                valid = True
                for m in marks:
                    d = pos - m
                    for m2 in marks:
                        if m2 != m and d == pos - m2:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    marks.append(pos)
                    result = backtrack(marks, remaining - 1, max_allowed)
                    if result is not None:
                        return result
                    marks.pop()
            return None
        
        return backtrack([0], n - 1, max_val)

    def mian_chowla(n):
        """Generate first n terms of Mian-Chowla sequence (starting with 0,1)."""
        seq = [0, 1]
        while len(seq) < n:
            candidates = set()
            # All pairwise sums of existing elements
            for i in range(len(seq)):
                for j in range(len(seq)):
                    candidates.add(seq[i] + seq[j])
            # Find smallest non-negative integer not in candidates
            x = 0
            while x in candidates:
                x += 1
            seq.append(x)
        return seq

    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Evaluate seed
    best_score = calc_score(best_set)
    
    # Strategy 1: Try structured candidates
    structured_candidates = []
    
    # Mian-Chowla sequences
    for n in range(3, 10):
        mc = mian_chowla(n)
        structured_candidates.append(mc)
    
    # Golomb rulers
    for n in range(3, 8):
        gr = generate_golomb_ruler(n, max_val=50)
        if gr is not None:
            structured_candidates.append(gr)
    
    # Power-of-2 based
    for n in range(3, 8):
        p2 = [0] + [2**(i) - 1 for i in range(1, n)]
        structured_candidates.append(p2)
    
    # Fibonacci-like
    for n in range(3, 8):
        fib = [0, 1]
        while len(fib) < n:
            fib.append(fib[-1] + fib[-2])
        structured_candidates.append(fib)
    
    # Mixed structured sets
    mixed_sets = [
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 5, 11],
        [0, 1, 2, 5, 11, 23],
        [0, 1, 4, 10, 22],
        [0, 2, 3, 7, 15],
        [0, 1, 3, 4, 10, 21],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 9, 27],
        [0, 1, 2, 5, 9, 18],
        [0, 1, 3, 5, 9, 17],
        [0, 2, 3, 5, 8, 13],
    ]
    structured_candidates.extend(mixed_sets)
    
    for U in structured_candidates:
        if time.time() > deadline:
            break
        s = calc_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Strategy 2: Local search from best structured set
    def local_search(U, max_iters=5000):
        nonlocal best_score, best_set
        current = U[:]
        current_score = calc_score(current)
        if current_score > best_score:
            best_score = current_score
            best_set = current[:]
        
        no_improve = 0
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try different moves
            move_type = random.choice(['swap', 'add', 'remove', 'shift'])
            
            if move_type == 'swap' and len(current) > 2:
                # Swap one element with a new value
                idx = random.randint(1, len(current) - 1)
                max_u = current[-1]
                new_val = random.randint(1, max_u + 5)
                if new_val not in current:
                    U_new = current[:]
                    U_new[idx] = new_val
                    U_new.sort()
                    s = calc_score(U_new)
                    if s > current_score:
                        current = U_new
                        current_score = s
                        no_improve = 0
                        if s > best_score:
                            best_score = s
                            best_set = current[:]
                    else:
                        no_improve += 1
            elif move_type == 'add' and len(current) < 10:
                # Add a new element
                max_u = current[-1]
                new_val = random.randint(1, max_u + 10)
                if new_val not in current:
                    U_new = current + [new_val]
                    U_new.sort()
                    s = calc_score(U_new)
                    if s > current_score:
                        current = U_new
                        current_score = s
                        no_improve = 0
                        if s > best_score:
                            best_score = s
                            best_set = current[:]
                    else:
                        no_improve += 1
            elif move_type == 'remove' and len(current) > 3:
                # Remove a random element (not 0)
                idx = random.randint(1, len(current) - 1)
                U_new = current[:idx] + current[idx+1:]
                s = calc_score(U_new)
                if s > current_score:
                    current = U_new
                    current_score = s
                    no_improve = 0
                    if s > best_score:
                        best_score = s
                        best_set = current[:]
                else:
                    no_improve += 1
            elif move_type == 'shift':
                # Shift one element by a small amount
                idx = random.randint(1, len(current) - 1)
                delta = random.choice([-2, -1, 1, 2, 3])
                new_val = current[idx] + delta
                if new_val >= 1 and new_val not in current:
                    U_new = current[:]
                    U_new[idx] = new_val
                    U_new.sort()
                    s = calc_score(U_new)
                    if s > current_score:
                        current = U_new
                        current_score = s
                        no_improve = 0
                        if s > best_score:
                            best_score = s
                            best_set = current[:]
                    else:
                        no_improve += 1
            else:
                no_improve += 1
            
            if no_improve > 200:
                # Restart with perturbation
                current = best_set[:]
                current_score = best_score
                no_improve = 0
                # Perturb
                idx = random.randint(1, len(current) - 1)
                new_val = random.randint(1, current[-1] + 10)
                if new_val not in current:
                    current[idx] = new_val
                    current.sort()
        
        return current

    # Run local search from best set
    if time.time() < deadline - 10:
        local_search(best_set[:], max_iters=3000)
    
    # Strategy 3: Random restarts with local search
    random.seed(42)
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline - 5:
            break
        
        # Generate a random starting set
        k = random.randint(4, 8)
        m = random.randint(10, 60)
        others = random.sample(range(1, m + 1), k - 1)
        U_start = [0] + sorted(others)
        
        local_search(U_start, max_iters=1000)
    
    return best_set
# EVOLVE-BLOCK-END
