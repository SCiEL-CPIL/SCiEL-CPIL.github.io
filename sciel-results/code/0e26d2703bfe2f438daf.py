import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    N = 256
    # Precompute reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
        
    best_score = 0.0
    best_S = set()
    
    def is_valid_addition(S, x):
        # Check if adding x to S creates a 3-AP
        # 1. x as middle: a + c = 2x => c = 2x - a
        # 2. x as endpoint: a + x = 2c => c = (a+x)/2
        for a in S:
            c = 2 * x - a
            if c in S:
                return False
            s = a + x
            if s % 2 == 0:
                c = s // 2
                if c in S:
                    return False
        return True

    def get_conflicts(S):
        # Return set of numbers that cannot be added to S
        conflicts = set()
        # x as middle: y = (a+c)/2
        for a in S:
            for c in S:
                if a != c:
                    s = a + c
                    if s % 2 == 0:
                        y = s // 2
                        conflicts.add(y)
        # x as endpoint: y = 2a - b
        for a in S:
            for b in S:
                y = 2 * a - b
                conflicts.add(y)
        return conflicts

    def heuristic_solve():
        # Greedy + Local Search to get a good initial solution
        S = set()
        for x in range(1, N + 1):
            if is_valid_addition(S, x):
                S.add(x)
        
        # Simple 1-swap hill climb
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            x_outs = sorted([x for x in S], reverse=True)
            x_ins = sorted([x for x in range(1, N + 1) if x not in S])
            
            for x_out in x_outs:
                if time.time() > deadline - 5.0:
                    break
                S_temp = S.copy()
                S_temp.remove(x_out)
                for x_in in x_ins:
                    if x_in >= x_out:
                        break
                    if is_valid_addition(S_temp, x_in):
                        S = S_temp
                        S.add(x_in)
                        improved = True
                        break
                if improved:
                    break
        return S

    # Initial Heuristic
    S_init = heuristic_solve()
    best_score = sum(recip[x] for x in S_init)
    best_S = S_init.copy()
    
    # B&B
    # State: S (set), C (set), current_sum
    # We use a stack for DFS
    
    # Initial C is all numbers not in S_init? 
    # Actually, B&B should start from scratch to be thorough, 
    # but using S_init as a "forced" start might bias it.
    # Better: Start B&B from S=empty, C=all.
    # The incumbent (best_score) will prune branches that can't beat the heuristic.
    
    stack = []
    # Root: S=empty, C={1..256}, sum=0
    # To prioritize small numbers, we process min(C)
    
    # We'll use a recursive function for clarity, but Python recursion limit might be an issue if depth is large.
    # Depth is max 256. Default limit is 1000. So recursion is safe.
    
    def dfs(S, C, current_sum):
        nonlocal best_score, best_S
        
        if time.time() > deadline:
            return
            
        # Upper Bound
        # Sum of reciprocals of C
        # Optimization: If C is large, summing every time is slow.
        # We can pass remaining_sum as a parameter? 
        # If we remove x from C, new_remaining = remaining - recip[x].
        # But we also remove conflicts. So we need to subtract their reciprocals too.
        # So we need to know which elements are removed.
        
        # Let's calculate upper bound strictly.
        # upper_bound = current_sum + sum(recip[y] for y in C)
        # This is O(|C|). |C| <= 256.
        # If we have 10k nodes, 10k * 256 = 2.5M ops. Acceptable.
        
        ub = current_sum
        for y in C:
            ub += recip[y]
            
        if ub <= best_score:
            return
            
        if not C:
            if current_sum > best_score:
                best_score = current_sum
                best_S = S.copy()
            return
            
        # Pick smallest x
        x = min(C)
        
        # Branch 1: Include x
        if is_valid_addition(S, x):
            new_S = S.copy()
            new_S.add(x)
            
            # Calculate conflicts with new_S
            # We need to remove from C any y that conflicts with new_S
            # Note: x itself is removed from C.
            # Also, any y that formed a 3-AP with S and x is now forbidden.
            # Actually, the condition "y conflicts with new_S" covers all cases.
            
            # Optimization: Instead of recalculating all conflicts for new_S,
            # we can just check each y in C (excluding x) against new_S.
            new_C = set()
            for y in C:
                if y == x:
                    continue
                if is_valid_addition(new_S, y):
                    new_C.add(y)
                    
            # Wait, is_valid_addition(new_S, y) checks if y can be added to new_S.
            # If y can be added, it is a valid candidate.
            # If y cannot be added, it conflicts.
            # So new_C should contain y if is_valid_addition(new_S, y) is True.
            # This is correct.
            
            # However, is_valid_addition is O(|S|).
            # So this loop is O(|C| * |S|).
            
            dfs(new_S, new_C, current_sum + recip[x])
            
        # Branch 2: Exclude x
        new_C_excl = C.copy()
        new_C_excl.remove(x)
        dfs(S, new_C_excl, current_sum)

    # Start B&B
    # S = empty set
    # C = {1, ..., 256}
    # current_sum = 0
    dfs(set(), set(range(1, N + 1)), 0.0)
    
    return list(best_S)
