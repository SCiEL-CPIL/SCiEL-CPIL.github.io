import time
import random

N = 256
FULL_MASK = (1 << (2 * N + 1)) - 1  # sums go up to 512

def construct():
    start_time = time.time()
    deadline = start_time + 27.0  # safety margin
    
    best_size = N + 1
    best_set = []
    
    # Precompute: for each element x, the sum bits it would create
    # and check validity using bitmasks
    
    def is_valid_addition(x, S_mask, sums_mask):
        """Check if x can be added to S without violating Sidon property."""
        # 2x must not be in sums_mask
        if (sums_mask >> (2 * x)) & 1:
            return False
        # x + a must not be in sums_mask for any a in S
        # sums_mask has bits at positions a+b for a,b in S (a<=b)
        # We need: for all a in S, (x+a) not in sums_mask
        # Equivalent: (sums_mask >> x) & S_mask == 0
        if (sums_mask >> x) & S_mask:
            return False
        return True
    
    def get_new_sums_bits(x, S_mask, sums_mask):
        """Get the bitmask of new sums created by adding x."""
        bits = 1 << (2 * x)
        temp = S_mask
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            bits |= (1 << (x + a))
            temp ^= lsb
        return bits
    
    def is_maximal(S_mask, sums_mask):
        """Check if S is inclusion-maximal: no element can be added."""
        # For each x in [1, N] not in S, check if it's blocked
        for x in range(1, N + 1):
            if (S_mask >> x) & 1:
                continue
            # Check if x can be added
            if is_valid_addition(x, S_mask, sums_mask):
                return False
        return True
    
    # Backtracking search
    # State: S_mask, sums_mask, current list
    # Goal: find a maximal Sidon set with minimum size
    
    # Use iterative deepening: try to find maximal set of size <= k
    # But first, let's do a smart greedy to get an initial upper bound
    
    # Greedy initialization
    S_mask = 0
    sums_mask = 0
    greedy_list = []
    
    for _ in range(N):
        best_x = -1
        best_block = -1
        for x in range(1, N + 1):
            if (S_mask >> x) & 1:
                continue
            if not is_valid_addition(x, S_mask, sums_mask):
                continue
            # Count how many elements this would block
            new_sums = sums_mask | get_new_sums_bits(x, S_mask, sums_mask)
            new_S = S_mask | (1 << x)
            block_count = 0
            for w in range(1, N + 1):
                if (new_S >> w) & 1:
                    continue
                if not is_valid_addition(w, new_S, new_sums):
                    block_count += 1
            if block_count > best_block:
                best_block = block_count
                best_x = x
        if best_x == -1:
            break
        S_mask |= (1 << best_x)
        sums_mask |= get_new_sums_bits(best_x, S_mask ^ (1 << best_x), sums_mask)
        greedy_list.append(best_x)
    
    if is_maximal(S_mask, sums_mask):
        best_size = len(greedy_list)
        best_set = greedy_list[:]
    else:
        # Greedy didn't produce maximal, extend it
        while True:
            best_x = -1
            best_block = -1
            for x in range(1, N + 1):
                if (S_mask >> x) & 1:
                    continue
                if not is_valid_addition(x, S_mask, sums_mask):
                    continue
                new_sums = sums_mask | get_new_sums_bits(x, S_mask, sums_mask)
                new_S = S_mask | (1 << x)
                block_count = 0
                for w in range(1, N + 1):
                    if (new_S >> w) & 1:
                        continue
                    if not is_valid_addition(w, new_S, new_sums):
                        block_count += 1
                if block_count > best_block:
                    best_block = block_count
                    best_x = x
            if best_x == -1:
                break
            S_mask |= (1 << best_x)
            sums_mask |= get_new_sums_bits(best_x, S_mask ^ (1 << best_x), sums_mask)
            greedy_list.append(best_x)
        if is_maximal(S_mask, sums_mask):
            best_size = len(greedy_list)
            best_set = greedy_list[:]
    
    # Now do local search: try to remove elements and see if still maximal
    # If removing an element keeps it maximal, we have a smaller set
    def try_shrink(S_list):
        """Try to remove elements while maintaining maximality."""
        S = set(S_list)
        changed = True
        while changed:
            changed = False
            for x in sorted(S):
                S_try = S - {x}
                if not S_try:
                    continue
                # Check if S_try is Sidon
                sums = set()
                valid = True
                S_list_try = sorted(S_try)
                for i, a in enumerate(S_list_try):
                    for b in S_list_try[i:]:
                        s = a + b
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if not valid:
                    continue
                # Check if maximal
                is_max = True
                for w in range(1, N + 1):
                    if w in S_try:
                        continue
                    # Check if w can be added
                    can_add = True
                    if 2 * w in sums:
                        can_add = False
                    else:
                        for a in S_try:
                            if w + a in sums:
                                can_add = False
                                break
                    if can_add:
                        is_max = False
                        break
                if is_max:
                    S = S_try
                    changed = True
                    break
        return list(S)
    
    # Try shrinking
    shrunk = try_shrink(best_set)
    if len(shrunk) < best_size:
        best_size = len(shrunk)
        best_set = shrunk
    
    # Now do randomized local search / hill climbing
    # Try random restarts with different greedy strategies
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        random.seed(restart * 7919 + 13)
        
        S_mask = 0
        sums_mask = 0
        S_list = []
        
        # Randomized greedy: pick from top-k candidates
        for step in range(N):
            if time.time() > deadline:
                break
            
            candidates = []
            for x in range(1, N + 1):
                if (S_mask >> x) & 1:
                    continue
                if not is_valid_addition(x, S_mask, sums_mask):
                    continue
                new_sums = sums_mask | get_new_sums_bits(x, S_mask, sums_mask)
                new_S = S_mask | (1 << x)
                block_count = 0
                for w in range(1, N + 1):
                    if (new_S >> w) & 1:
                        continue
                    if not is_valid_addition(w, new_S, new_sums):
                        block_count += 1
                candidates.append((block_count, x))
            
            if not candidates:
                break
            
            # Sort by block count descending, pick from top few
            candidates.sort(reverse=True)
            top_k = min(5, len(candidates))
            if random.random() < 0.7:
                # Pick from top 3
                top_k = min(3, len(candidates))
            chosen = random.choice(candidates[:top_k])
            x = chosen[1]
            
            S_mask |= (1 << x)
            sums_mask |= get_new_sums_bits(x, S_mask ^ (1 << x), sums_mask)
            S_list.append(x)
        
        # Check if maximal, if not extend
        while True:
            if time.time() > deadline:
                break
            candidates = []
            for x in range(1, N + 1):
                if (S_mask >> x) & 1:
                    continue
                if not is_valid_addition(x, S_mask, sums_mask):
                    continue
                new_sums = sums_mask | get_new_sums_bits(x, S_mask, sums_mask)
                new_S = S_mask | (1 << x)
                block_count = 0
                for w in range(1, N + 1):
                    if (new_S >> w) & 1:
                        continue
                    if not is_valid_addition(w, new_S, new_sums):
                        block_count += 1
                candidates.append((block_count, x))
            if not candidates:
                break
            candidates.sort(reverse=True)
            x = candidates[0][1]
            S_mask |= (1 << x)
            sums_mask |= get_new_sums_bits(x, S_mask ^ (1 << x), sums_mask)
            S_list.append(x)
        
        # Try to shrink
        shrunk = try_shrink(S_list)
        if len(shrunk) < best_size:
            best_size = len(shrunk)
            best_set = shrunk
    
    return best_set
