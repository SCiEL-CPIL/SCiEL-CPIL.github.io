import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    N = 4096
    MAX_SUM = 3 * N
    
    # used_sums[s] is True if sum s is already formed by a triple in S
    # We use a list of booleans for speed. Size is 3*4096 + 1.
    used_sums = [False] * (MAX_SUM + 1)
    
    def try_add(x, S):
        """
        Attempts to add x to S.
        Returns True if successful, False otherwise.
        Modifies S and used_sums if successful.
        """
        # Check for conflicts with existing triples involving x
        # Triples are (x, a, b) where a, b are in S (a <= b to handle unordered)
        # Since S is sorted, we can just iterate all pairs.
        # Actually, we need to check if x+a+b is in used_sums for any a,b in S.
        # Also, we must ensure x itself doesn't form a conflict with itself?
        # The problem says "unordered triple, allowing repetitions".
        # So (x, x, x) is a triple. (x, x, a) is a triple. (x, a, b) is a triple.
        
        # 1. Check (x, x, x)
        if used_sums[3 * x]:
            return False
        
        # 2. Check (x, x, a) for all a in S
        for a in S:
            if used_sums[2 * x + a]:
                return False
        
        # 3. Check (x, a, b) for all a, b in S with a < b
        # Note: (x, a, a) is covered by case 2 if we iterate all a? 
        # Yes, if we iterate all a in S, we cover (x, x, a) and (x, a, a) is not a standard form 
        # unless we consider indices. 
        # Let's be precise. 
        # Triples involving x are of the form {x, u, v} where u, v are from S U {x}.
        # Since x is new, u, v can be from S.
        # If u=x, v=x: sum 3x.
        # If u=x, v=a (a in S): sum 2x+a.
        # If u=a, v=b (a,b in S): sum x+a+b.
        
        # We already checked 3x and 2x+a.
        # Now check x+a+b for all a,b in S.
        # To avoid duplicate checks for (a,b) and (b,a), we can just check all pairs.
        # Since S is small (<=80), k^2 is fine.
        k = len(S)
        for i in range(k):
            a = S[i]
            for j in range(i, k):  # a <= b
                b = S[j]
                s = x + a + b
                if s > MAX_SUM:
                    continue
                if used_sums[s]:
                    return False
        
        # If we get here, x is valid.
        # Update used_sums
        used_sums[3 * x] = True
        for a in S:
            used_sums[2 * x + a] = True
            # Wait, (x, a, a) is covered by x+a+b where i=j.
            # My loop above covers x+a+b for all a,b in S.
            # So I don't need to separately mark 2x+a here?
            # Yes, the loop `for i... for j...` covers all pairs (a,b) from S.
            # So 2x+a is NOT covered by x+a+b because a,b are from S, not x.
            # So I DO need to mark 2x+a separately.
            # And 3x separately.
            pass
        
        # Correct update logic:
        # 1. 3x
        used_sums[3 * x] = True
        # 2. 2x + a for all a in S
        for a in S:
            used_sums[2 * x + a] = True
        # 3. x + a + b for all a, b in S
        for i in range(k):
            a = S[i]
            for j in range(i, k):
                b = S[j]
                used_sums[x + a + b] = True
                
        S.append(x)
        S.sort()
        return True

    def remove(x, S):
        """
        Removes x from S and updates used_sums.
        This is tricky because we don't know which sums were *only* formed by x.
        However, we can rebuild used_sums from scratch if needed, but that's O(k^3).
        Alternatively, we can use a counter array instead of boolean.
        Let's switch to a counter array for robustness in swapping.
        """
        pass # We will implement a counter-based version for the main loop.

    # Let's restart with a counter-based approach for safety and ease of swapping.
    sum_counts = [0] * (MAX_SUM + 1)
    S = []
    
    def add(x):
        """Adds x to S, updating sum_counts. Returns True if valid."""
        k = len(S)
        
        # Check 3x
        if sum_counts[3 * x] > 0:
            return False
        
        # Check 2x + a
        for a in S:
            if sum_counts[2 * x + a] > 0:
                return False
                
        # Check x + a + b
        for i in range(k):
            a = S[i]
            for j in range(i, k):
                b = S[j]
                s = x + a + b
                if sum_counts[s] > 0:
                    return False
        
        # Valid, update counts
        sum_counts[3 * x] += 1
        for a in S:
            sum_counts[2 * x + a] += 1
        for i in range(k):
            a = S[i]
            for j in range(i, k):
                b = S[j]
                sum_counts[x + a + b] += 1
                
        S.append(x)
        S.sort()
        return True

    def remove(x):
        """Removes x from S, updating sum_counts."""
        # Find index
        idx = S.index(x)
        S.pop(idx)
        k = len(S)
        
        sum_counts[3 * x] -= 1
        for a in S:
            sum_counts[2 * x + a] -= 1
        for i in range(k):
            a = S[i]
            for j in range(i, k):
                b = S[j]
                sum_counts[x + a + b] -= 1

    # Phase 1: Greedy Initialization
    # Try to add numbers from 1 to N
    for x in range(1, N + 1):
        if add(x):
            pass # Added
        if time.time() > deadline:
            break

    best_S = list(S)
    best_size = len(S)
    best_sum = sum(S)

    # Phase 2: Local Search
    # We will try to insert unused numbers or swap existing ones with smaller unused ones.
    # To make this efficient, we maintain a set of unused numbers.
    used_set = set(S)
    unused = set(range(1, N + 1)) - used_set
    
    # List for faster iteration, but set for O(1) check
    # We'll iterate over sorted unused numbers for insertion attempts.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        
        # 1. Try to insert any unused number
        # Sort unused to try smaller ones first? Or random?
        # Smaller ones are often easier to fit in gaps.
        # Let's try a few random unused numbers to escape local optima.
        if unused:
            # Try up to 10 random insertions
            candidates = random.sample(list(unused), min(10, len(unused)))
            for x in candidates:
                if add(x):
                    used_set.add(x)
                    unused.remove(x)
                    improved = True
                    best_S = list(S)
                    best_size = len(S)
                    best_sum = sum(S)
                    no_improve_count = 0
                    break # Restart loop to re-evaluate state
            if improved:
                continue
                
        # 2. Try to swap an existing element with a smaller unused one
        # This reduces the sum, potentially allowing larger elements later.
        # We look for (y in S, x in unused) such that x < y and swap is valid.
        # Valid swap: Remove y, Add x.
        
        # Heuristic: Try to replace the largest elements with smaller unused ones.
        # Or just random pairs.
        
        # To be fast, we can pick a random y from S and a random x from unused where x < y.
        # If S is large, random sampling is good.
        
        if len(S) > 0 and unused:
            # Pick random y from S
            y = random.choice(S)
            # Find unused x < y
            # This is slow if we scan. 
            # We can maintain a sorted list of unused? 
            # Or just check a few random unused numbers.
            
            # Let's try a few random swaps
            for _ in range(10):
                if not unused:
                    break
                x = random.choice(list(unused))
                if x >= y:
                    continue
                
                # Check if swap is valid
                # 1. Remove y
                remove(y)
                # 2. Try to add x
                if add(x):
                    # Swap successful
                    used_set.remove(y)
                    used_set.add(x)
                    unused.remove(x)
                    unused.add(y)
                    improved = True
                    # Update best if size is same but sum is lower?
                    # Or just keep the current state as it's valid.
                    # Since size didn't change, but sum decreased, it's a "better" local state.
                    # We track best_S by size primarily.
                    # If size is same, we don't necessarily update best_S unless we want to minimize sum for final output?
                    # The task says "Maximize cardinality". Sum doesn't matter for score, but lower sum might help future insertions.
                    # So we keep the state.
                    no_improve_count = 0
                    break
                else:
                    # Revert: Remove x, Add y
                    remove(x)
                    add(y)
            
            if improved:
                continue

        no_improve_count += 1

    # Return the best set found (largest size)
    return best_S

# Final check: The function construct() is defined.
# Ensure no syntax errors.
# The `remove` function relies on `S.index(x)` which is O(k). This is fine for k<=80.
# The `add` function is O(k^2).
# The local search loop runs until deadline.

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Sample: {res[:10]}")
