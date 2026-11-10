# EVOLVE-BLOCK-START
import sys

def construct():
    n = 192
    best_S = []
    best_size = 0
    
    def is_valid_addition(a, S):
        """
        Check if adding a to S maintains the constraint.
        S is a list of selected integers.
        """
        # S is small (<= 50), so O(|S|^2) or O(|S|^3) is acceptable per check
        # if the number of checks is not too high.
        
        # 1. Check constraint for a:
        # No 3 distinct elements in S should sum to a multiple of a.
        if a == 1:
            if len(S) >= 3:
                return False
        else:
            # Iterate through all pairs in S
            len_S = len(S)
            for i in range(len_S):
                x = S[i]
                for j in range(i + 1, len_S):
                    y = S[j]
                    # We need z in S, z != x, z != y, such that (x + y + z) % a == 0
                    # z % a == (-x - y) % a
                    target_res = (-x - y) % a
                    # Check if any z in S satisfies this
                    # We can break early if found
                    for k in range(len_S):
                        z = S[k]
                        if z == x or z == y:
                            continue
                        if z % a == target_res:
                            return False
                    # If no z found for this pair, continue to next pair
        
        # 2. Check constraint for existing elements b in S:
        # Adding a creates new triples (a, x, y) where x, y in S.
        # For each b in S, we must ensure b does NOT divide (a + x + y) for any distinct x, y in S\{b}.
        for b in S:
            if b == 1:
                # 1 divides everything.
                # We need to ensure there are NO distinct x, y in S\{1} such that the triple (a, x, y) exists.
                # The triple elements are a, x, y. They must be distinct from b=1.
                # a != 1 (since 1 is in S and a is new).
                # So we need x, y != 1.
                # If there are at least 2 elements in S that are not 1, then such a triple exists.
                count_non_one = len(S) - (1 if 1 in S else 0)
                if count_non_one >= 2:
                    return False
            else:
                # Check if there exist distinct x, y in S\{b} such that b | (a + x + y)
                # (a + x + y) % b == 0  =>  (x + y) % b == (-a) % b
                target_res = (-a) % b
                # Iterate through pairs in S\{b}
                # To optimize, we can just iterate S and skip b
                found = False
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

    # Greedy initialization to get a strong lower bound
    S_greedy = []
    for a in range(n, 0, -1):
        if is_valid_addition(a, S_greedy):
            S_greedy.append(a)
    best_S = S_greedy[:]
    best_size = len(S_greedy)
    
    # Backtracking search to try to find a larger set
    # We use a DFS that tries to include elements first.
    # Pruning: If current_size + remaining_candidates <= best_size, stop.
    
    candidates = list(range(n, 0, -1))
    
    sys.setrecursionlimit(10000)
    
    def dfs(idx, current_S, current_size):
        nonlocal best_S, best_size
        
        # Pruning
        if current_size + (n - idx) <= best_size:
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

    # Start DFS
    # To ensure we finish within 60s, we might need to limit the depth or use a timer.
    # However, the pruning is very strong if best_size is close to optimal.
    # If greedy finds 48, and optimal is 48, DFS prunes almost everything.
    # If optimal is 49, DFS will search for it.
    
    # We add a simple time limit check if needed, but standard pruning should suffice.
    import time
    start_time = time.time()
    
    def dfs_with_timeout(idx, current_S, current_size):
        nonlocal best_S, best_size
        
        if time.time() - start_time > 55: # Leave 5s buffer
            return False
        
        if current_size + (n - idx) <= best_size:
            return True
        
        if idx == n:
            if current_size > best_size:
                best_S = current_S[:]
                best_size = current_size
            return True
        
        a = candidates[idx]
        
        # Try to include a
        if is_valid_addition(a, current_S):
            current_S.append(a)
            if not dfs_with_timeout(idx + 1, current_S, current_size + 1):
                current_S.pop()
                return False
            current_S.pop()
        
        # Try to skip a
        if not dfs_with_timeout(idx + 1, current_S, current_size):
            return False
        
        return True

    dfs_with_timeout(0, [], 0)
    
    return best_S
# EVOLVE-BLOCK-END
