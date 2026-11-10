# EVOLVE-BLOCK-START
import sys
import time

def construct():
    n = 192
    max_size = 48
    best_S = []
    best_size = 0
    
    def is_valid_addition(a, S):
        """
        Check if adding a to S maintains the constraint.
        S is a list of selected integers.
        """
        len_S = len(S)
        
        # 1. Check constraint for a:
        # No 3 distinct elements in S should sum to a multiple of a.
        if a == 1:
            if len_S >= 3:
                return False
        else:
            # Iterate through all pairs in S
            # Optimization: Use a set of residues? 
            # For a fixed a, we can count residues of elements in S.
            # But we need to ensure distinctness.
            # Given small N, direct iteration is fine.
            for i in range(len_S):
                x = S[i]
                for j in range(i + 1, len_S):
                    y = S[j]
                    target_res = (-x - y) % a
                    # Check if any z in S, z != x, z != y, satisfies z % a == target_res
                    # We can break early
                    for k in range(len_S):
                        z = S[k]
                        if z == x or z == y:
                            continue
                        if z % a == target_res:
                            return False
        
        # 2. Check constraint for existing elements b in S:
        # Adding a creates new triples (a, x, y) where x, y in S.
        # For each b in S, we must ensure b does NOT divide (a + x + y) for any distinct x, y in S\{b}.
        # Note: The triple is {a, x, y}. b is NOT in the triple sum, but b is the divisor.
        # The elements x, y must be distinct from b.
        # The elements x, y are from S.
        # So we iterate b in S.
        for b in S:
            if b == 1:
                # 1 divides everything.
                # We need to ensure there are NO distinct x, y in S\{1} such that the triple (a, x, y) exists.
                # The triple elements are a, x, y. They must be distinct from b=1.
                # a != 1 (since 1 is in S and a is new, or if 1 not in S, a could be 1? No, a is candidate).
                # If 1 is in S, and we add a, the triple is {a, x, y}.
                # If b=1, the condition is 1 | (a+x+y). This is always true.
                # So we must ensure that NO such triple {a, x, y} exists where x,y \in S\{1}.
                # This means we cannot have 2 distinct elements x, y in S\{1}.
                count_non_one = len_S - (1 if 1 in S else 0)
                if count_non_one >= 2:
                    return False
            else:
                # Check if there exist distinct x, y in S\{b} such that b | (a + x + y)
                # (a + x + y) % b == 0  =>  (x + y) % b == (-a) % b
                target_res = (-a) % b
                # Iterate through pairs in S\{b}
                # Optimization: Precompute pair sums mod b? 
                # For now, iterate.
                found = False
                # Create list of S without b for cleaner logic, or just skip in loop
                # S is small, list comprehension is fast
                S_wo_b = [v for v in S if v != b]
                len_wo_b = len(S_wo_b)
                for i in range(len_wo_b):
                    x = S_wo_b[i]
                    for j in range(i + 1, len_wo_b):
                        y = S_wo_b[j]
                        if (x + y) % b == target_res:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return False
        
        return True

    # Greedy initialization
    S_greedy = []
    for a in range(n, 0, -1):
        if is_valid_addition(a, S_greedy):
            S_greedy.append(a)
    best_S = S_greedy[:]
    best_size = len(S_greedy)
    
    if best_size == max_size:
        return best_S

    candidates = list(range(n, 0, -1))
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    
    def dfs(idx, current_S, current_size):
        nonlocal best_S, best_size
        
        if time.time() - start_time > 55:
            return
        
        # Pruning
        if current_size + (n - idx) <= best_size:
            return
        
        if current_size == max_size:
            if current_size > best_size:
                best_S = current_S[:]
                best_size = current_size
            return
        
        if idx == n:
            if current_size > best_size:
                best_S = current_S[:]
                best_size = current_size
            return
        
        a = candidates[idx]
        
        # Try to include a
        if is_valid_addition(a, current_S):
            current_S.append(a)
            dfs(idx + 1, current_S, current_size + 1)
            current_S.pop()
            
        # Try to skip a
        dfs(idx + 1, current_S, current_size)

    dfs(0, [], 0)
    
    return best_S
# EVOLVE-BLOCK-END
