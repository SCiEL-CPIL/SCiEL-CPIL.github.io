import time
import random

def construct():
    start_time = time.time()
    # 120s limit, use 115s to be safe
    deadline = start_time + 115.0
    
    fixed = [1, 2, 5, 11]
    fixed_set = set(fixed)
    
    # Helper to check if a list of ints is a valid Sidon set
    # Returns True if valid, False otherwise
    def is_sidon(A):
        sums = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                s = ai + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def find_sidon_set(N):
        """
        Attempts to find a Sidon set of size N containing the fixed elements.
        Returns the sorted list if found, else None.
        """
        num_vars = N - 4
        if num_vars < 0:
            return None
        
        # Values available for variables (1..284 excluding fixed)
        available = set(range(1, 285))
        for f in fixed:
            available.remove(f)
        avail_list = list(available)
        
        # Local search with random restarts
        while time.time() < deadline:
            # Initialize random variables
            current_vars = random.sample(avail_list, num_vars)
            
            # Limit steps per restart to avoid getting stuck in a bad local minimum
            max_steps = 2000
            for _ in range(max_steps):
                if time.time() >= deadline:
                    return None
                
                # Construct full set
                full_set = fixed + current_vars
                
                # Check for distinct integers
                if len(set(full_set)) != len(full_set):
                    # Find a duplicate and resolve
                    seen = {}
                    resolved = False
                    for i, v in enumerate(full_set):
                        if v in seen:
                            # v is duplicate of seen[v]
                            # If i >= 4, it's a variable
                            if i >= 4:
                                var_idx = i - 4
                                new_val = random.choice(avail_list)
                                while new_val in full_set:
                                    new_val = random.choice(avail_list)
                                current_vars[var_idx] = new_val
                                resolved = True
                                break
                            else:
                                # Duplicate involves a fixed element and a later element
                                # The later element must be a variable if fixed are distinct
                                # Find the variable
                                for j in range(i+1, len(full_set)):
                                    if full_set[j] == v:
                                        if j >= 4:
                                            var_idx = j - 4
                                            new_val = random.choice(avail_list)
                                            while new_val in full_set:
                                                new_val = random.choice(avail_list)
                                            current_vars[var_idx] = new_val
                                            resolved = True
                                            break
                                break
                        else:
                            seen[v] = i
                    if not resolved:
                        # Should not happen if logic is correct, but restart if stuck
                        break
                    continue
                
                # Check Sidon property and find conflicts
                sums = {}
                conflict_found = False
                n = len(full_set)
                
                for i in range(n):
                    ai = full_set[i]
                    for j in range(i, n):
                        s = ai + full_set[j]
                        if s in sums:
                            # Conflict found
                            # Previous pair: sums[s]
                            # Current pair: (i, j)
                            
                            # Identify variable indices involved
                            # We want to change a variable to resolve the conflict
                            candidates = []
                            
                            # Check current pair (i, j)
                            if i >= 4:
                                candidates.append(i - 4)
                            if j >= 4:
                                candidates.append(j - 4)
                                
                            # Check previous pair
                            pi, pj = sums[s]
                            if pi >= 4:
                                candidates.append(pi - 4)
                            if pj >= 4:
                                candidates.append(pj - 4)
                                
                            if candidates:
                                # Pick a random candidate to change
                                target_var_idx = random.choice(candidates)
                                old_val = current_vars[target_var_idx]
                                
                                # Generate new value
                                # Ensure it doesn't create immediate duplicate integer
                                new_val = random.choice(avail_list)
                                # Simple retry loop for distinctness
                                attempts = 0
                                while new_val in full_set:
                                    new_val = random.choice(avail_list)
                                    attempts += 1
                                    if attempts > 10:
                                        # Fallback: pick any distinct
                                        break
                                        
                                current_vars[target_var_idx] = new_val
                                conflict_found = True
                                break
                        else:
                            sums[s] = (i, j)
                    
                    if conflict_found:
                        break
                
                if not conflict_found:
                    # Valid Sidon set found
                    return sorted(full_set)
            
            # Restart if local search exhausted or timed out inside
            if time.time() >= deadline:
                return None

        return None

    # Try to find the largest possible set
    # Max possible is 20
    for N in range(20, 11, -1):
        result = find_sidon_set(N)
        if result:
            return result
        
        # If we've used too much time, break and return what we have?
        # The task asks to maximize size. If we fail to find 20, we should try 19, etc.
        # But if we run out of time, we should return the best found so far.
        # However, find_sidon_set returns None if it fails.
        # We can keep track of the best result found in any attempt.
        
        # Actually, the loop above returns immediately on success.
        # If it returns None, we try the next N.
        
    # Fallback: return the fixed set
    return fixed
