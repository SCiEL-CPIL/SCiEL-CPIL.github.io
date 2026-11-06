import itertools
import time

def construct():
    N = 192
    # Sort in descending order to prioritize larger numbers which are "safer"
    candidates = list(range(N, 0, -1))
    
    best_size = 0
    best_set = set()
    start_time = time.time()
    timeout = 50  # Leave 10s buffer
    
    def is_valid(candidate, current_list):
        """
        Check if candidate can be added to current_list.
        current_list is a list of selected integers.
        """
        # Check 1: candidate does not divide the sum of any 3 distinct elements in current_list
        if len(current_list) >= 3:
            # Optimization: If candidate is large, only specific sums matter.
            # But we just check all.
            for a, b, c in itertools.combinations(current_list, 3):
                if (a + b + c) % candidate == 0:
                    return False
        
        # Check 2: For each y in current_list, y does not divide (candidate + a + b)
        # where a, b are distinct elements in current_list \ {y}
        for y in current_list:
            # We need to check pairs (a, b) from current_list excluding y
            # Sum = candidate + a + b
            # Condition: (candidate + a + b) % y == 0
            
            # Optimization: 
            # Let S_ab = a + b.
            # We need S_ab == k * y - candidate.
            # Min a+b is sum of two smallest in current_list\{y}.
            # Max a+b is sum of two largest in current_list\{y}.
            
            # Simple loop
            for i in range(len(current_list)):
                a = current_list[i]
                if a == y:
                    continue
                for j in range(i + 1, len(current_list)):
                    b = current_list[j]
                    if b == y:
                        continue
                    if (candidate + a + b) % y == 0:
                        return False
        return True

    def backtrack(start_idx, current_list):
        nonlocal best_size, best_set, start_time
        
        # Check timeout
        if time.time() - start_time > timeout:
            return
            
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_set = set(current_list)
            
        # Pruning: If we can't possibly beat best_size, return
        # Max elements we can still add is len(candidates) - start_idx
        if len(current_list) + (len(candidates) - start_idx) <= best_size:
            return
            
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            
            # Heuristic: If x is too small to be useful? No, we want max cardinality.
            
            if is_valid(x, current_list):
                current_list.append(x)
                backtrack(i + 1, current_list)
                current_list.pop()
                
                # If we found a solution of size 48 (the max allowed), we can stop early?
                # The problem says "at most 48 elements".
                # If we reach 48, we can't add more.
                if len(current_list) == 48:
                    return

    backtrack(0, [])
    
    # If no solution found (shouldn't happen), return empty or greedy
    if not best_set:
        return []
        
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(len(res))
