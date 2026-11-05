import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 4096
    best_set = []
    best_size = 0
    
    # Helper to compute all triple sums of a set S
    def compute_all_sums(S):
        sums = set()
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                for m in range(j, k):
                    sums.add(S[i] + S[j] + S[m])
        return sums

    # Helper to get new sums introduced by adding x to S
    def get_new_sums(x, S):
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                new_sums.add(x + S[i] + S[j])
        return new_sums

    # Helper to remove sums involving x from a sum set
    def remove_sums_involving_x(x, S, sums):
        # We need to be careful: only remove sums that are exclusively formed by x and other elements
        # If a sum can be formed in multiple ways, we can't just remove it.
        # However, since we are maintaining a valid B_3[1] set, each sum is unique.
        # So if we remove x, we remove all sums that include x.
        # But wait, if we just remove x from the set, the sums involving x are no longer present.
        # The sums NOT involving x remain.
        # So we just need to subtract the set of sums that included x.
        
        sums_to_remove = set()
        sums_to_remove.add(3 * x)
        for y in S:
            if y != x:
                sums_to_remove.add(2 * x + y)
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                if S[i] != x and S[j] != x:
                    sums_to_remove.add(x + S[i] + S[j])
                elif S[i] == x and S[j] == x:
                    # 3x already handled
                    pass
                elif S[i] == x:
                    # 2x + S[j] already handled
                    pass
                elif S[j] == x:
                    # x + S[i] + x = 2x + S[i] already handled
                    pass
        
        # Actually, simpler:
        # The sums involving x are exactly:
        # 1. 3x
        # 2. 2x + y for y in S, y != x
        # 3. x + y + z for y, z in S, y <= z, and not (y==x and z==x) and not (y==x) and not (z==x)
        # Wait, the set S contains x.
        # Let S' = S \ {x}.
        # Sums involving x:
        # - 3x
        # - 2x + y for y in S'
        # - x + y + z for y, z in S', y <= z
        
        S_prime = [y for y in S if y != x]
        sums_to_remove = set()
        sums_to_remove.add(3 * x)
        for y in S_prime:
            sums_to_remove.add(2 * x + y)
        for i in range(len(S_prime)):
            for j in range(i, len(S_prime)):
                sums_to_remove.add(x + S_prime[i] + S_prime[j])
                
        for s in sums_to_remove:
            sums.discard(s)

    def can_add(x, S, sums):
        if x in S:
            return False, set()
        new_sums = get_new_sums(x, S)
        # Check for internal collisions in new_sums
        # The number of unique sums should match the number of generated triples
        k = len(S)
        expected = 1 + k + k * (k + 1) // 2
        if len(new_sums) != expected:
            return False, set()
        # Check for collisions with existing sums
        if new_sums & sums:
            return False, set()
        return True, new_sums

    # Initial solution: Randomized Greedy
    def initial_solution():
        S = []
        sums = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            valid, new_sums = can_add(x, S, sums)
            if valid:
                S.append(x)
                sums.update(new_sums)
        return S, sums

    S, sums = initial_solution()
    best_set = S.copy()
    best_size = len(S)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    iterations = 0
    
    while time.time() < deadline and T > min_T:
        iterations += 1
        current_size = len(S)
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random element
            candidates = [x for x in range(1, n + 1) if x not in S]
            if not candidates:
                continue
            x = random.choice(candidates)
            valid, new_sums = can_add(x, S, sums)
            if valid:
                # Accept
                S.append(x)
                sums.update(new_sums)
                new_size = len(S)
                if new_size > best_size:
                    best_size = new_size
                    best_set = S.copy()
            # If invalid, reject (size would not change, or invalid)
            # In SA, we usually compare fitness. Here fitness is size.
            # Adding invalid element is not a valid state.
            # So we only accept if valid.
            
        elif move_type < 0.7:
            # Try to remove a random element
            if not S:
                continue
            idx = random.randint(0, len(S) - 1)
            x = S[idx]
            
            # Remove x
            S.pop(idx)
            remove_sums_involving_x(x, S, sums)
            # Note: S is now S \ {x}. The sums set is updated.
            # This move decreases size. We accept it with probability exp(-delta/T)
            # delta = new_size - old_size = -1
            # prob = exp(1/T) which is > 1? No, we usually define delta as cost increase.
            # Cost = -size. Delta cost = -new_size - (-old_size) = old_size - new_size = 1.
            # prob = exp(-1/T)
            if random.random() < math.exp(-1.0 / T):
                # Accept
                pass
            else:
                # Reject: restore
                S.append(x)
                # Recompute sums for x to add back
                # Since we removed sums involving x from the previous S (which included x),
                # we need to be careful.
                # Let's just recompute sums for the restored S.
                # Actually, remove_sums_involving_x modified the global `sums` set.
                # To restore, we need to add back the sums involving x with respect to the CURRENT S (which is S \ {x}).
                # The sums to add back are exactly the ones removed.
                # But we didn't save them.
                # Easier: Just recompute all sums for S.
                # This is expensive O(k^3).
                # Alternative: Keep a stack of (x, sums_to_remove)
                # Or, simply, if we reject, we revert S and recompute sums.
                # Given k is small (< 30), recomputing is fast.
                sums = compute_all_sums(S)
                
        else:
            # Try to swap
            if not S:
                continue
            candidates = [x for x in range(1, n + 1) if x not in S]
            if not candidates:
                continue
            
            idx = random.randint(0, len(S) - 1)
            x_out = S[idx]
            x_in = random.choice(candidates)
            
            # Try to replace x_out with x_in
            # 1. Remove x_out
            S_temp = S.copy()
            S_temp.pop(idx)
            # 2. Check if x_in can be added to S_temp
            valid, new_sums = can_add(x_in, S_temp, sums)
            
            # Wait, `sums` currently contains sums for S (including x_out).
            # We need sums for S_temp.
            # Let's compute sums_temp = sums - sums_involving_x_out
            sums_temp = set(sums)
            remove_sums_involving_x(x_out, S, sums_temp)
            
            valid, new_sums = can_add(x_in, S_temp, sums_temp)
            
            if valid:
                # New size is same as old size.
                # We accept with probability 1 (or based on some heuristic)
                # Since size is same, we accept to explore.
                S = S_temp + [x_in]
                sums = sums_temp
                sums.update(new_sums)
            # If invalid, reject
            
        T *= cooling
        
    return best_set
