import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Helper to check validity and find conflict
    def check_and_find_conflict(S):
        # Returns (is_valid, conflict_subset)
        # S is a set of ints
        reachable = [False] * 201
        reachable[0] = True
        # To reconstruct, we can store the last element that made a sum reachable
        # Or just re-run DP with backtracking if needed
        # For speed, just check validity first.
        for x in S:
            # Shift reachable by x
            # new_reachable[i] = reachable[i] or reachable[i-x]
            # We only care up to 200
            # Optimization: iterate backwards
            # Actually, just a simple loop is fine for N=100, T=200
            new_reachable = reachable[:]
            for i in range(200 - x, -1, -1):
                if reachable[i]:
                    new_reachable[i+x] = True
            reachable = new_reachable
            if reachable[200]:
                # Find conflict
                # Backtrack to find which elements sum to 200
                # This is slow if done every time.
                # Alternative: Just return False, None.
                # For the swap logic, we need the conflict.
                # We can find it by re-running DP storing parents.
                pass
        return not reachable[200], None

    # Faster check
    def is_valid(S):
        reachable = 1  # Bit 0 set
        mask = (1 << 201) - 1
        for x in S:
            reachable |= (reachable << x)
            reachable &= mask
            if (reachable >> 200) & 1:
                return False
        return True

    # Find a subset summing to 200
    def find_conflict(S):
        # DP to find one subset
        # dp[i] = set of elements? No, too big.
        # dp[i] = (prev_sum, element)
        parent = [-1] * 201
        prev_sum = [-1] * 201
        reachable = [False] * 201
        reachable[0] = True
        
        for x in S:
            # Iterate backwards to avoid using x multiple times
            for i in range(200, x - 1, -1):
                if reachable[i - x] and not reachable[i]:
                    reachable[i] = True
                    parent[i] = x
                    prev_sum[i] = i - x
        
        if not reachable[200]:
            return None
        
        # Reconstruct
        conflict = []
        cur = 200
        while cur > 0:
            x = parent[cur]
            conflict.append(x)
            cur = prev_sum[cur]
        return conflict

    # Initialize with Seed Strategy
    S = set()
    for x in range(100, 0, -1):
        if is_valid(S | {x}):
            S.add(x)
    
    best_S = set(S)
    
    # Randomized Local Search
    while time.time() < deadline:
        # Try to add a random element
        if len(S) < 100:
            candidates = [x for x in range(1, 101) if x not in S]
            if candidates:
                u = random.choice(candidates)
                new_S = S | {u}
                if is_valid(new_S):
                    S = new_S
                    if len(S) > len(best_S):
                        best_S = set(S)
                else:
                    # Try swap
                    conflict = find_conflict(new_S)
                    if conflict:
                        # Remove u from conflict consideration for removal
                        # We must remove one from conflict \ {u}
                        removable = [v for v in conflict if v != u]
                        if removable:
                            v = random.choice(removable)
                            swap_S = (S - {v}) | {u}
                            if is_valid(swap_S):
                                S = swap_S
                                if len(S) > len(best_S):
                                    best_S = set(S)
        
        # Occasionally try to remove an element and add a different one?
        # The swap above handles this.
        
        # Random restart? No, keep improving current best.
        
    return list(best_S)
