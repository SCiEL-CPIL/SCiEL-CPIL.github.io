import time
import math

def compute_score(U):
    if not U:
        return 0.0
    U_set = set(U)
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    sum_set = set()
    diff_set = set()
    
    # Optimization: use list for U
    U_list = list(U_set)
    n = len(U_list)
    
    for i in range(n):
        for j in range(n):
            sum_set.add(U_list[i] + U_list[j])
            diff_set.add(U_list[i] - U_list[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / size_sum
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def find_golomb_ruler(k, max_time=10):
    """
    Finds a Golomb ruler of order k with minimal length.
    Returns the set of marks or None if not found in time.
    """
    start_time = time.time()
    best_ruler = None
    best_length = float('inf')
    
    # Initial guess for length
    # Lower bound: k(k-1)/2
    current_length = k * (k - 1) // 2
    
    def backtrack(marks, length):
        nonlocal best_ruler, best_length, current_length
        
        if time.time() - start_time > max_time:
            return False
            
        if len(marks) == k:
            if length < best_length:
                best_length = length
                best_ruler = tuple(marks)
            return True
            
        last = marks[-1]
        # Try adding a new mark 'new_mark'
        # new_mark must be > last
        # To prune, we can limit the search space
        # A simple bound: new_mark < best_length
        # Also, differences must be unique
        
        # Heuristic: try smaller gaps first? Or larger?
        # Usually, trying to fill the ruler densely is good.
        
        # We iterate possible positions for the next mark
        # We can bound the max position by best_length
        # If best_length is inf, we need a heuristic bound.
        # For k=5, length is 11.
        
        # Let's just iterate up to a reasonable limit or best_length
        max_possible = best_length - 1 if best_length < float('inf') else current_length + k
        
        for new_mark in range(last + 1, max_possible + 1):
            # Check differences
            valid = True
            for m in marks:
                diff = new_mark - m
                # Check if this diff already exists
                # We can maintain a set of diffs for speed, but for small k, O(k^2) is fine
                # To speed up, we can precompute diffs?
                # For now, simple check
                pass
            
            # Efficient check:
            # We need to ensure all differences |new_mark - m| for m in marks are unique
            # AND not equal to any existing differences between pairs in marks
            
            # Let's maintain a set of all pairwise differences
            # This requires passing the diff set around or computing it.
            # For simplicity and speed in Python, we'll compute on the fly if k is small.
            
            diffs = []
            for m in marks:
                d = new_mark - m
                if d in diffs:
                    valid = False
                    break
                diffs.append(d)
            
            if not valid:
                continue
                
            # Check against existing diffs in marks
            # We need a set of all diffs in 'marks'
            # To avoid recomputing, we can pass it, but let's just recompute for clarity in the prototype
            # Actually, for k up to 12, O(k^3) is fine.
            
            existing_diffs = set()
            for i in range(len(marks)):
                for j in range(i+1, len(marks)):
                    existing_diffs.add(abs(marks[i] - marks[j]))
                    
            if any(d in existing_diffs for d in diffs):
                continue
                
            marks.append(new_mark)
            if backtrack(marks, new_mark):
                marks.pop()
                return True
            marks.pop()
            
        return False

    # Start with 0
    backtrack([0], 0)
    return best_ruler

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    # Search for Golomb rulers of increasing size
    # We know k=3 is [0,1,3]
    # Try k=4 to k=15
    
    for k in range(4, 16):
        if time.time() > deadline - 5:
            break
            
        ruler = find_golomb_ruler(k, max_time=5)
        if ruler:
            # Convert to list
            U = list(ruler)
            score = compute_score(U)
            if score > best_score:
                best_score = score
                best_set = U
                
    return best_set
