import time
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for 120s limit
    
    # Target marks relative to 0
    # Prescribed: 1, 2, 5, 11 -> 0, 1, 4, 10
    base_marks = [0, 1, 4, 10]
    max_val = 283 # Length of ruler (max mark - min mark)
    
    # We want to find the largest K such that a Golomb ruler of size K 
    # containing base_marks exists with length <= max_val.
    
    # Start with K=20 (known max for length 283) and go down
    for K in range(20, 3, -1):
        if time.time() - start_time > time_limit:
            break
            
        # DFS to find a ruler of size K
        solution = dfs_search(base_marks, K, max_val, start_time, time_limit)
        if solution is not None:
            # Convert back to 1..284
            return [x + 1 for x in solution]
            
    # If no solution found for K > 4, return the base case (invalid if K<4? No, base is 4)
    # The base case [0, 1, 4, 10] is valid.
    return [x + 1 for x in base_marks]

def dfs_search(initial_marks, target_size, max_length, start_time, time_limit):
    # initial_marks is sorted
    marks = list(initial_marks)
    sums = set()
    for i in range(len(marks)):
        for j in range(i, len(marks)):
            sums.add(marks[i] + marks[j])
            
    # We need to add target_size - len(marks) more marks
    remaining_slots = target_size - len(marks)
    if remaining_slots < 0:
        return None
    if remaining_slots == 0:
        # Check if length is valid (it is, since max mark <= max_length is enforced during addition)
        return marks
        
    # Candidates are integers from max(marks) + 1 to max_length
    # To improve search, we might want to try larger gaps first? 
    # Or just iterate?
    # A common heuristic is to try to place marks that are "far" from existing ones.
    
    # We will use a recursive helper
    def backtrack(current_marks, current_sums, slots_left):
        if time.time() - start_time > time_limit:
            return None
            
        if slots_left == 0:
            return current_marks
            
        last_mark = current_marks[-1]
        # The next mark must be > last_mark
        # And last_mark + next_mark <= max_length + last_mark? 
        # Actually, the constraint is max_mark <= max_length.
        # So next_mark <= max_length.
        
        # Heuristic: Try candidates in a specific order.
        # Randomization or specific step?
        # Let's iterate from max_length down to last_mark + 1.
        # Why down? Placing large marks first might help?
        # Actually, for Golomb rulers, the distribution is roughly uniform.
        # Let's just iterate. To break symmetry or find solutions faster, 
        # we can try a few random candidates if the search space is large.
        
        # For efficiency, let's iterate all candidates.
        # Pruning: 
        # If the minimum possible increase in span for 'slots_left' marks is 
        # greater than remaining span, prune.
        # Min span for m marks is m*(m-1)/2 (if packed tightly).
        # But we don't pack tightly.
        
        # Simple check: 
        # If (max_length - last_mark) < slots_left:
        #     return None # Not enough room for distinct integers
        
        # Better check:
        # The smallest possible new mark is last_mark + 1.
        # The largest is max_length.
        
        # Let's try candidates in a random order to avoid getting stuck in deep dead-ends?
        # Or sorted?
        # Given the time limit, a simple DFS with backtracking is standard.
        
        # Optimization: Check if adding a candidate creates a sum conflict.
        # new_sums = {c + m for m in current_marks} | {2*c}
        
        # We will iterate c from last_mark + 1 to max_length
        # To speed up, we can jump?
        
        # Let's use a helper to check validity
        def is_valid(c, marks, sums):
            new_sums = set()
            for m in marks:
                s = c + m
                if s in sums or s in new_sums:
                    return False
                new_sums.add(s)
            s2 = 2 * c
            if s2 in sums or s2 in new_sums:
                return False
            new_sums.add(s2)
            return True
            
        # To make the search faster, we can use a "jump" heuristic or just linear scan.
        # Linear scan from last_mark+1 to max_length.
        
        # To prevent exponential blowup, we rely on the time limit.
        
        # Heuristic: Try candidates that are "far" from the last mark first?
        # Let's just iterate.
        
        # One important optimization: 
        # If we are close to the limit, we might want to try larger steps.
        
        # Let's try iterating from last_mark + 1 to max_length.
        # If this is too slow, we can randomize the order.
        
        # For now, linear scan.
        # To improve, we can check if the remaining capacity is sufficient.
        # Max number of marks we can fit in [last_mark+1, max_length] is max_length - last_mark.
        # If slots_left > max_length - last_mark: return None
        
        if slots_left > max_length - last_mark:
            return None
            
        # Try candidates
        # To diversify, we can try a few random candidates first?
        # Let's stick to deterministic first, it's safer for reproducibility if needed, 
        # but here we just need *a* solution.
        
        # Actually, iterating in increasing order is standard.
        for c in range(last_mark + 1, max_length + 1):
            # Check time
            if time.time() - start_time > time_limit:
                return None
                
            if is_valid(c, current_marks, current_sums):
                new_marks = current_marks + [c]
                new_sums = set(current_sums)
                for m in current_marks:
                    new_sums.add(c + m)
                new_sums.add(2 * c)
                
                res = backtrack(new_marks, new_sums, slots_left - 1)
                if res is not None:
                    return res
        return None
        
    return backtrack(marks, sums, remaining_slots)
