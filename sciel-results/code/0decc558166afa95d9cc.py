import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    prescribed = [1, 2, 5, 11]
    N = 284
    target_sizes = [20, 19, 18, 17, 16]
    
    best_set = list(prescribed)
    
    # Helper to check validity and get sums
    def get_sums(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        return sums
    
    def is_valid_add(A, sums, x):
        # Check if x can be added to A
        if x in A:
            return False
        new_sums = set()
        for y in A:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        if new_sums & sums:
            return False
        return True

    def try_find_size(target_size):
        nonlocal best_set
        A = list(prescribed)
        sums = get_sums(A)
        
        # Backtracking search
        # We need to add (target_size - len(A)) elements
        needed = target_size - len(A)
        if needed <= 0:
            return True
        
        # Candidate pool
        candidates = [x for x in range(1, N + 1) if x not in A]
        random.shuffle(candidates)
        
        def backtrack(current_A, current_sums, needed):
            nonlocal best_set
            if time.time() > deadline:
                return False
            
            if needed == 0:
                if len(current_A) > len(best_set):
                    best_set = list(current_A)
                return True
            
            # Heuristic: Try candidates that are "far" from existing elements? 
            # Or just iterate through shuffled candidates.
            for x in candidates:
                if x in current_A:
                    continue
                
                # Check if x can be added
                valid = True
                new_sums = set()
                for y in current_A:
                    s = x + y
                    if s in current_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s = 2 * x
                    if s in current_sums:
                        valid = False
                    else:
                        new_sums.add(s)
                
                if not valid:
                    continue
                
                # Add x
                current_A.append(x)
                current_sums.update(new_sums)
                
                if backtrack(current_A, current_sums, needed - 1):
                    return True
                
                # Remove x
                current_A.pop()
                current_sums -= new_sums
                
            return False
        
        # To avoid deep recursion stack issues and to allow randomization in deeper levels,
        # we can implement this iteratively or with limited depth first.
        # But for size 20, depth is small.
        
        # We might get stuck in a bad branch quickly.
        # Let's add a random restart mechanism within the search if it's too slow.
        # For now, just try once with shuffled candidates.
        
        if backtrack(A, sums, needed):
            return True
        
        return False

    # Try to find a set of size 20
    # We do multiple restarts with different random seeds/shuffles
    for target in target_sizes:
        if time.time() > deadline:
            break
            
        # Try multiple times for each target size
        attempts = 0
        max_attempts = 10000 # Adjust based on speed
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            if try_find_size(target):
                return best_set
            # If we found a larger set in best_set, we can break early if target is met?
            # try_find_size updates best_set only if it finds a full valid set of that size.
            # Actually, my backtrack only updates best_set if needed==0.
            # We should also update best_set with partial results?
            # No, we want max size. If we fail to find size 20, we try 19.
        
    # If we didn't find any of the target sizes, return the best found (prescribed)
    # But we should probably run a greedy to get a baseline > 4
    # The current seed program does greedy. Let's include a quick greedy to ensure we don't return just 4.
    
    if len(best_set) == 4:
        # Run a simple greedy
        A = list(prescribed)
        sums = get_sums(A)
        for x in range(12, N + 1):
            if is_valid_add(A, sums, x):
                A.append(x)
                for y in A[:-1]:
                    sums.add(x + y)
                sums.add(2 * x)
        best_set = A

    return best_set
