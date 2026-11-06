import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin for 60s limit
    max_val = 4096
    
    # Helper to check if adding x is valid
    def can_add(S, used_sums, pair_sums, x):
        k = len(S)
        # 1. x + x + x
        s1 = 3 * x
        if s1 in used_sums:
            return False
        
        # Pre-check internal collisions with existing pair sums
        # 4. 2x must not be a pair sum in S
        if 2 * x in pair_sums:
            return False
            
        # 2. x + x + s
        for s in S:
            s2 = 2 * x + s
            if s2 in used_sums:
                return False
            # 5. x + s must not be a pair sum in S
            # Collision if 2x + s = x + s_i + s_j  => x + s = s_i + s_j
            if x + s in pair_sums:
                return False
                
        # 3. x + s_i + s_j
        # Iterate over pair sums
        for p in pair_sums:
            s3 = x + p
            if s3 in used_sums:
                return False
                
        return True

    def get_sums_for(x, S_old):
        """Returns list of sums contributed by x given S_old (which does not contain x)."""
        sums = [3 * x]
        for s in S_old:
            sums.append(2 * x + s)
        for i in range(len(S_old)):
            for j in range(i, len(S_old)):
                sums.append(x + S_old[i] + S_old[j])
        return sums

    def get_pair_sums_for(x, S_old):
        """Returns set of pair sums involving x, given S_old."""
        ps = {2 * x}
        for s in S_old:
            ps.add(x + s)
        return ps

    # Initial Greedy to get a baseline
    S = []
    S_set = set()
    used_sums = set()
    pair_sums = set()
    
    for x in range(1, max_val + 1):
        if x not in S_set:
            if can_add(S, used_sums, pair_sums, x):
                S_old = S
                S.append(x)
                S_set.add(x)
                
                for s_val in get_sums_for(x, S_old):
                    used_sums.add(s_val)
                    
                for p_val in get_pair_sums_for(x, S_old):
                    pair_sums.add(p_val)
                
    best_set = list(S)
    best_len = len(S)
    
    def rollback_add(S, S_set, used_sums, pair_sums, x, S_old):
        S.pop()
        S_set.remove(x)
        for s_val in get_sums_for(x, S_old):
            used_sums.remove(s_val)
        for p_val in get_pair_sums_for(x, S_old):
            pair_sums.remove(p_val)
            
    def remove_element(S, S_set, used_sums, pair_sums, idx):
        y = S[idx]
        S_rest = S[:idx] + S[idx+1:]
        
        S.pop(idx)
        S_set.remove(y)
        
        for s_val in get_sums_for(y, S_rest):
            used_sums.remove(s_val)
            
        for p_val in get_pair_sums_for(y, S_rest):
            pair_sums.remove(p_val)
            
        return y, S_rest

    def add_element(S, S_set, used_sums, pair_sums, x):
        S.append(x)
        S_set.add(x)
        S_old = S[:-1]
        for s_val in get_sums_for(x, S_old):
            used_sums.add(s_val)
        for p_val in get_pair_sums_for(x, S_old):
            pair_sums.add(p_val)

    # Local Search Loop
    # We will alternate between trying to add elements and swapping elements to escape local optima.
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        # Phase 1: Greedy Add
        # Scan all x from 1 to max_val
        for x in range(1, max_val + 1):
            if x not in S_set:
                if can_add(S, used_sums, pair_sums, x):
                    S_old = S
                    add_element(S, S_set, used_sums, pair_sums, x)
                    improved = True
                    if len(S) > best_len:
                        best_set = list(S)
                        best_len = len(S)
                    
        if not improved:
            # Phase 2: Swap (1-for-1)
            # Try to swap a random element with a better one?
            # Or just any one to escape local optimum?
            # Let's try to swap to allow growth.
            # Pick random y
            if len(S) > 0:
                idx = random.randint(0, len(S) - 1)
                y, S_rest = remove_element(S, S_set, used_sums, pair_sums, idx)
                
                swapped = False
                # Try to find x to replace y
                # Scan 1..max_val
                for x in range(1, max_val + 1):
                    if x == y:
                        continue
                    if x not in S_set:
                        if can_add(S, used_sums, pair_sums, x):
                            # Swap successful
                            add_element(S, S_set, used_sums, pair_sums, x)
                            swapped = True
                            break
                
                if not swapped:
                    # Restore y
                    add_element(S, S_set, used_sums, pair_sums, y)
                else:
                    improved = True # State changed, might allow growth next iter
                    # Note: Size didn't increase, but structure changed.
                    # We don't update best_set here unless size increases, which it doesn't.
                    
            # Phase 3: 2-for-1
            if not improved and len(S) > 2:
                # Pick two random elements to remove
                idx1 = random.randint(0, len(S) - 1)
                idx2 = random.randint(0, len(S) - 2)
                if idx2 >= idx1:
                    idx2 += 1
                    
                y1 = S[idx1]
                y2 = S[idx2]
                
                # Remove y1
                y1_rest = remove_element(S, S_set, used_sums, pair_sums, idx1)
                # Adjust index for y2
                if idx2 > idx1:
                    idx2 -= 1
                y2_rest = remove_element(S, S_set, used_sums, pair_sums, idx2)
                
                found = False
                # Try to find x1, x2 to replace
                for x1 in range(1, max_val + 1):
                    if x1 == y1 or x1 == y2 or x1 in S_set:
                        continue
                    if can_add(S, used_sums, pair_sums, x1):
                        # Tentatively add x1
                        S_old_temp = S
                        add_element(S, S_set, used_sums, pair_sums, x1)
                        
                        for x2 in range(x1 + 1, max_val + 1):
                            if x2 == y1 or x2 == y2 or x2 in S_set:
                                continue
                            if can_add(S, used_sums, pair_sums, x2):
                                # Add x2
                                add_element(S, S_set, used_sums, pair_sums, x2)
                                found = True
                                break
                        
                        if found:
                            break
                        else:
                            # Rollback x1
                            rollback_add(S, S_set, used_sums, pair_sums, x1, S_old_temp)
                            
                if not found:
                    # Restore y1, y2
                    add_element(S, S_set, used_sums, pair_sums, y2)
                    add_element(S, S_set, used_sums, pair_sums, y1)
                else:
                    improved = True
                    if len(S) > best_len:
                        best_set = list(S)
                        best_len = len(S)

        # If no improvement in a full cycle, we might be stuck.
        # We could implement a more sophisticated move like 2-for-2, but 2-for-1 is a good start.
        # To avoid infinite loops on the same state, we rely on randomization in swaps.
        
    return best_set

# Quick sanity check / test
if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
    print(res[:10])
