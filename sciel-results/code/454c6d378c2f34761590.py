import time
import random

def construct():
    t0 = time.time()
    deadline = 55.0
    N = 128
    
    def elapsed():
        return time.time() - t0
    
    # --- Phase 1: Greedy removal from full set ---
    S = set(range(1, N + 1))
    
    def has_violation(S):
        """Check if S contains any violating triple."""
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                ab = a + b
                if ab > N:
                    break
                if ab not in S:
                    continue
                for k in range(j + 1, n):
                    c = S_list[k]
                    ac = a + c
                    if ac not in S:
                        continue
                    bc = b + c
                    if bc in S:
                        return True
        return False
    
    def count_violations_for_elem(S, x):
        """Count how many violating triples involve x."""
        count = 0
        S_list = [e for e in S if e != x]
        n = len(S_list)
        
        # Case 1: x is one of a, b, c
        for i in range(n):
            u = S_list[i]
            xu = x + u
            if xu not in S:
                continue
            for j in range(i + 1, n):
                v = S_list[j]
                xv = x + v
                if xv not in S:
                    continue
                uv = u + v
                if uv in S:
                    count += 1
        
        # Case 2: x is one of the sums
        # Subcase: x = a + b, need c in S\{a,b} with a+c in S and b+c in S
        for a in S_list:
            b = x - a
            if b in S and b != a and b != x:
                # Check for c
                for c in S_list:
                    if c == a or c == b:
                        continue
                    if (a + c) in S and (b + c) in S:
                        count += 1
        
        return count
    
    # Greedy removal
    while has_violation(S) and elapsed() < deadline - 10.0:
        # Find element with highest violation count
        best_elem = None
        best_count = -1
        for x in S:
            if elapsed() > deadline - 10.0:
                break
            cnt = count_violations_for_elem(S, x)
            if cnt > best_count:
                best_count = cnt
                best_elem = x
        
        if best_elem is None:
            break
        S.remove(best_elem)
    
    # --- Phase 2: Try to add back removed elements ---
    removed = [x for x in range(1, N + 1) if x not in S]
    random.shuffle(removed)
    
    def can_add(S, x):
        """Check if adding x to S keeps it valid."""
        S_list = sorted(S)
        n = len(S_list)
        
        # Case 1: x is one of a, b, c
        for i in range(n):
            u = S_list[i]
            xu = x + u
            if xu not in S and xu != x:
                continue
            if xu in S:
                for j in range(i + 1, n):
                    v = S_list[j]
                    xv = x + v
                    if xv not in S:
                        continue
                    uv = u + v
                    if uv in S or uv == x:
                        return False
        
        # Case 2: x is one of the sums
        for a in S_list:
            b = x - a
            if b in S and b != a and b != x:
                for c in S_list:
                    if c == a or c == b or c == x:
                        continue
                    if (a + c) in S and (b + c) in S:
                        return False
        
        return True
    
    for x in removed:
        if elapsed() > deadline - 5.0:
            break
        if can_add(S, x):
            S.add(x)
    
    # --- Phase 3: Local search with swaps ---
    best_size = len(S)
    no_improve = 0
    
    while elapsed() < deadline - 2.0:
        improved = False
        
        # Try adding
        for x in range(1, N + 1):
            if x in S or elapsed() > deadline - 2.0:
                continue
            if can_add(S, x):
                S.add(x)
                improved = True
                break
        
        if improved:
            best_size = len(S)
            no_improve = 0
            continue
        
        # Try swaps
        S_list = list(S)
        random.shuffle(S_list)
        for y in S_list:
            if elapsed() > deadline - 2.0:
                break
            S.remove(y)
            # Try to add some x
            found = False
            for x in range(1, N + 1):
                if x == y or x in S:
                    continue
                if can_add(S, x):
                    S.add(x)
                    found = True
                    improved = True
                    break
            
            if not found:
                S.add(y)
        
        if improved:
            best_size = len(S)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 20:
                break
    
    # Convert to list of Python ints
    result = sorted([int(x) for x in S])
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    S = set(res)
    valid = True
    res_sorted = sorted(res)
    for i, a in enumerate(res_sorted):
        for j, b in enumerate(res_sorted):
            if i >= j:
                continue
            for k, c in enumerate(res_sorted):
                if k <= j:
                    continue
                if (a + b) in S and (a + c) in S and (b + c) in S:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print("Valid:", valid)
