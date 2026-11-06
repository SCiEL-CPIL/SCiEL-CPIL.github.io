import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 1000
    MAX_SUM = 2 * N

    def is_valid_add(S_bits, counts, x):
        # Check if x can be added to current set represented by S_bits and counts
        # x is not in S
        # New sums: 2x and x+y for y in S
        if counts[2 * x] >= 2:
            return False
        # Check x+y for y in S
        # We need to iterate y in S. 
        # To make this fast, we can check if there exists y in S such that counts[x+y] >= 2.
        # But we need to check ALL y.
        # This is O(|S|).
        # Alternative: Precompute? No, S changes.
        # We'll rely on the local search doing the heavy lifting and use this for greedy init.
        # For greedy init, |S| is small initially.
        y = 1
        while y <= N:
            if (S_bits >> y) & 1:
                if counts[x + y] >= 2:
                    return False
            y += 1
        return True

    def add_element(S, S_bits, counts, x):
        # Add x to S
        counts[2 * x] += 1
        y = 1
        while y <= N:
            if (S_bits >> y) & 1:
                counts[x + y] += 1
            y += 1
        S.add(x)
        S_bits |= (1 << x)
        return S, S_bits, counts

    def remove_element(S, S_bits, counts, x):
        # Remove x from S
        counts[2 * x] -= 1
        y = 1
        while y <= N:
            if (S_bits >> y) & 1 and y != x:
                counts[x + y] -= 1
            y += 1
        S.discard(x)
        S_bits &= ~(1 << x)
        return S, S_bits, counts

    def greedy_init(order):
        S = set()
        S_bits = 0
        counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
        for x in order:
            if is_valid_add(S_bits, counts, x):
                S, S_bits, counts = add_element(S, S_bits, counts, x)
        return S, S_bits, counts

    def local_search(S, S_bits, counts, max_iters, rng):
        best_S = S.copy()
        best_S_bits = S_bits
        best_counts = counts.copy()
        best_size = len(S)
        
        S_list = list(S)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random candidate x not in S
            # To avoid O(N) scan, we can pick random x in 1..N and check membership
            # Or maintain a list of non-members. Maintaining list is O(1) add/remove.
            # Let's just pick random x.
            
            # Try up to 10 random candidates per iteration?
            # Or just one.
            
            x = rng.randint(1, N)
            if x in S:
                continue
                
            # Calculate blocked sums
            blocked = []
            if counts[2 * x] >= 2:
                blocked.append(2 * x)
            
            y = 1
            while y <= N:
                if (S_bits >> y) & 1:
                    s = x + y
                    if counts[s] >= 2:
                        blocked.append(s)
                y += 1
            
            if not blocked:
                # Can add directly
                S, S_bits, counts = add_element(S, S_bits, counts, x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S.copy()
                    best_S_bits = S_bits
                    best_counts = counts.copy()
                continue
            
            # Try to find a y in S to remove
            # Shuffle S_list to randomize order of trying removals
            # But we only try a few?
            # Let's try up to 5 random y's from S
            # Or iterate all? |S| is ~300. 300 checks per candidate.
            # Candidates per iter: 1.
            # 300 * 3000 iters = 900k checks. Should be fine.
            
            # To speed up, we can pre-shuffle S_list once per iteration?
            # No, just pick random y's.
            
            found_swap = False
            # Try up to 10 random y's
            for _ in range(10):
                if not S_list:
                    break
                y = S_list[rng.randrange(len(S_list))]
                
                # Check if removing y fixes all blocked sums
                valid_swap = True
                for s in blocked:
                    if s == x + y:
                        # The pair (x,y) is not added, so no increase.
                        # But we must ensure counts[s] <= 2 after removal.
                        # counts[s] was >= 2.
                        # Removing y decreases counts[s] by 1 if y contributes to s.
                        # y contributes to s=x+y? Only if there is z in S such that y+z=x+y => z=x. x not in S.
                        # So y does NOT contribute to x+y in the current S.
                        # So counts[s] does NOT change due to removal of y.
                        # And it does NOT increase due to addition of x (pair x,y gone).
                        # So counts[s] remains the same.
                        # Since it was >= 2, and max allowed is 2, it must be exactly 2.
                        # So it is valid.
                        continue
                    else:
                        # s != x + y
                        # The pair (x, s-x) is added (if s-x in S and s-x != y).
                        # Wait, if s-x == y, then s = x+y, which is handled above.
                        # So s-x != y.
                        # So the pair (x, s-x) IS added, increasing counts[s] by 1.
                        # We need counts[s] - N_y(s) + 1 <= 2.
                        # N_y(s) is 1 if s-y in S, else 0.
                        # So we need: if s-y in S, then counts[s] <= 2. (Since 1-1+1=1? No. counts[s] - 1 + 1 = counts[s].)
                        # If s-y NOT in S, then counts[s] - 0 + 1 = counts[s] + 1 <= 2 => counts[s] <= 1.
                        
                        # Let's re-verify the delta.
                        # Old count: C
                        # Remove y: C - N_y(s)
                        # Add x: (C - N_y(s)) + 1 (because x + (s-x) is a new pair, and s-x != y)
                        # New count: C - N_y(s) + 1
                        # We need C - N_y(s) + 1 <= 2 => C - N_y(s) <= 1.
                        
                        if (S_bits >> (s - y)) & 1:
                            # s-y is in S, so N_y(s) = 1
                            # Need C <= 2
                            if counts[s] > 2:
                                valid_swap = False
                                break
                        else:
                            # s-y not in S, so N_y(s) = 0
                            # Need C <= 1
                            if counts[s] > 1:
                                valid_swap = False
                                break
                
                if valid_swap:
                    # Perform swap
                    S, S_bits, counts = remove_element(S, S_bits, counts, y)
                    S_list.remove(y)
                    S, S_bits, counts = add_element(S, S_bits, counts, x)
                    S_list.append(x)
                    found_swap = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                        best_S_bits = S_bits
                        best_counts = counts.copy()
                    break
            
            if found_swap:
                continue
            
            # If no swap found, we might try to just add x if we missed something?
            # No, logic says it's blocked.
            
        return best_S, best_S_bits, best_counts

    def final_polish(S, S_bits, counts):
        added = True
        while added and time.time() < deadline - 2:
            added = False
            for x in range(1, N + 1):
                if x not in S and is_valid_add(S_bits, counts, x):
                    S, S_bits, counts = add_element(S, S_bits, counts, x)
                    added = True
                    break
        return S, S_bits, counts

    rng = random.Random(42)
    best_size = 0
    best_S = set()
    best_S_bits = 0
    best_counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    
    # Phase 1: Greedy Init
    order = list(range(1, N + 1))
    S, S_bits, counts = greedy_init(order)
    best_size = len(S)
    best_S = S.copy()
    best_S_bits = S_bits
    best_counts = counts.copy()
    
    # Phase 2: Local Search
    S, S_bits, counts = best_S, best_S_bits, best_counts.copy()
    best_S, best_S_bits, best_counts = local_search(S, S_bits, counts, 5000, rng)
    
    # Phase 3: Polish
    best_S, best_S_bits, best_counts = final_polish(best_S, best_S_bits, best_counts.copy())
    
    return list(best_S)
