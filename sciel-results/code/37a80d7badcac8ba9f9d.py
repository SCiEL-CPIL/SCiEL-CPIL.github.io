import time
import random

N = 512
DEADLINE = 85.0  # Safety margin before 90s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute all 4-APs
    # aps_by_element[x] = list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP.
    aps_by_element = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps_by_element[a].append((b, c, e))
            aps_by_element[b].append((a, c, e))
            aps_by_element[c].append((a, b, e))
            aps_by_element[e].append((a, b, c))

    def valid_set(S):
        """Check if S is 4-AP free."""
        in_S = [False] * (N + 1)
        for x in S:
            in_S[x] = True
        for a in range(1, N + 1):
            for d in range(1, (N - a) // 3 + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                if in_S[a] and in_S[b] and in_S[c] and in_S[e]:
                    return False
        return True

    def fast_greedy(order):
        """Greedy construction using a specific order."""
        in_set = [False] * (N + 1)
        bad_count = [0] * (N + 1)
        S_size = 0
        
        for x in order:
            if bad_count[x] == 0:
                in_set[x] = True
                S_size += 1
                # Update bad_count for others
                for (y, z, w) in aps_by_element[x]:
                    if in_set[z] and in_set[w]:
                        bad_count[y] += 1
                    if in_set[y] and in_set[w]:
                        bad_count[z] += 1
                    if in_set[y] and in_set[z]:
                        bad_count[w] += 1
                        
        return [i for i in range(1, N + 1) if in_set[i]]

    def local_search(S, time_limit):
        """WalkSAT-style local search.
        Start with S (valid).
        Goal: Maximize size.
        Moves:
        1. Add x if valid.
        2. Swap: Remove r, Add a. Valid if resulting set is 4-AP free.
        3. Multi-swap: Remove set R, Add set A. (Complex, skip for now or use simple 2-swap)
        
        We will focus on 1-swap and maybe 2-swap if time permits.
        To speed up validity check, we can use the bad_count structure, 
        but for simplicity and robustness in a "from scratch" rewrite with time pressure,
        let's use a hybrid: 
        - Maintain bad_count for O(1) add check.
        - For swap, we need to check if removing r allows adding a.
          This is equivalent to: In S \ {r}, is a addable?
          We can simulate this quickly.
        """
        in_set = [False] * (N + 1)
        for x in S:
            in_set[x] = True
        
        bad_count = [0] * (N + 1)
        # Initialize bad_count
        # This is O(N * avg_aps). 
        # We can compute it by iterating all APs.
        # Or, simpler: for each x in S, for each AP (x, y, z, w), 
        # if y, z, w in S, then bad_count[x]++, bad_count[y]++, etc? No.
        # bad_count[x] = # APs containing x where OTHER 3 are in S.
        # So for each AP (a, b, c, d):
        #   count = sum(in_set[v] for v in [a,b,c,d])
        #   if count == 3:
        #       the missing one has bad_count += 1
        #   if count == 4:
        #       # This shouldn't happen if S is valid. 
        #       # But if S is invalid, we have a problem. 
        #       # The input S is guaranteed valid by fast_greedy.
        #       pass
        
        # Efficient init:
        for a in range(1, N + 1):
            for d in range(1, (N - a) // 3 + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                cnt = 0
                if in_set[a]: cnt += 1
                if in_set[b]: cnt += 1
                if in_set[c]: cnt += 1
                if in_set[e]: cnt += 1
                if cnt == 3:
                    if not in_set[a]: bad_count[a] += 1
                    if not in_set[b]: bad_count[b] += 1
                    if not in_set[c]: bad_count[c] += 1
                    if not in_set[e]: bad_count[e] += 1
                # If cnt == 4, S is invalid. We assume S is valid.

        S_size = len(S)
        best_size = S_size
        best_set = list(S)
        
        rng = random.Random(12345)
        
        def do_add(x):
            nonlocal S_size
            in_set[x] = True
            S_size += 1
            for (y, z, w) in aps_by_element[x]:
                if in_set[z] and in_set[w]:
                    bad_count[y] += 1
                if in_set[y] and in_set[w]:
                    bad_count[z] += 1
                if in_set[y] and in_set[z]:
                    bad_count[w] += 1

        def do_remove(x):
            nonlocal S_size
            in_set[x] = False
            S_size -= 1
            for (y, z, w) in aps_by_element[x]:
                if in_set[z] and in_set[w]:
                    bad_count[y] -= 1
                if in_set[y] and in_set[w]:
                    bad_count[z] -= 1
                if in_set[y] and in_set[z]:
                    bad_count[w] -= 1

        def check_and_update_best():
            nonlocal best_size, best_set
            if S_size > best_size:
                best_size = S_size
                best_set = [i for i in range(1, N + 1) if in_set[i]]

        # Local Search Loop
        while time.time() < time_limit:
            # 1. Try to add
            added = False
            # Find x with bad_count 0
            # Optimization: Keep a list of candidates? 
            # N=512 is small, scanning is fine.
            for x in range(1, N + 1):
                if not in_set[x] and bad_count[x] == 0:
                    do_add(x)
                    check_and_update_best()
                    added = True
                    break
            
            if added:
                continue

            # 2. Try to swap
            # Find a with bad_count[a] > 0.
            # We want to remove r such that bad_count[a] becomes 0.
            # This happens if r is in all blocking APs of a.
            # If bad_count[a] == 1, r is in the unique blocking AP.
            # If bad_count[a] > 1, we might need to remove multiple, or find r in intersection.
            
            # Strategy: Pick random a with bad_count[a] == 1.
            # Find its blocker r.
            # Remove r, Add a.
            
            swap_done = False
            # Collect candidates for a
            candidates_a = [x for x in range(1, N + 1) if not in_set[x] and bad_count[x] == 1]
            
            if candidates_a:
                a = rng.choice(candidates_a)
                # Find blocker r
                # Find the AP (a, y, z, w) where y,z,w in S.
                r = -1
                for (y, z, w) in aps_by_element[a]:
                    if in_set[y] and in_set[z] and in_set[w]:
                        # This is the blocking AP.
                        # We can remove any of y, z, w.
                        # Which one to remove?
                        # Heuristic: Remove the one that is "least central" or has high bad_count?
                        # Or just random.
                        # Let's try removing the one that is in the most APs with other elements?
                        # For now, random.
                        r = rng.choice([y, z, w])
                        break
                
                if r != -1:
                    # Perform swap
                    do_remove(r)
                    # Now a should be addable?
                    # bad_count[a] should be 0 now.
                    if bad_count[a] == 0:
                        do_add(a)
                        check_and_update_best()
                        swap_done = True
                    else:
                        # Something went wrong? Revert?
                        # This shouldn't happen if logic is correct.
                        # If bad_count[a] > 0, we created an invalid state or didn't clear the block.
                        # Revert
                        do_remove(a) if in_set[a] else None # a is not in set yet
                        do_add(r)
            
            if not swap_done:
                # Try 2-swap? Or random restart?
                # Let's try a random restart of the local search state from best_set?
                # Or just break and rely on outer loop restarts.
                # If we are stuck, we might try to pick a random r and a random a and see if it works?
                # That's O(N^2) checks.
                
                # Let's try a simple "kick": 
                # Remove a random element from S, then greedily add as many as possible.
                # This is a form of restart.
                
                if S_size > 0:
                    r = rng.choice([i for i in range(1, N + 1) if in_set[i]])
                    do_remove(r)
                    # Now greedily add
                    for x in range(1, N + 1):
                        if not in_set[x] and bad_count[x] == 0:
                            do_add(x)
                    check_and_update_best()
                    # If we didn't improve, maybe revert?
                    # The "kick" is destructive. 
                    # Let's just continue the loop. If we don't improve, we'll try again.
                    # To avoid getting stuck in a bad local minimum, this helps.

        return best_set

    # Main Logic
    best_overall = []
    best_size = 0
    
    # 1. Initial Greedy with natural order
    S1 = fast_greedy(list(range(1, N + 1)))
    if len(S1) > best_size:
        best_size = len(S1)
        best_overall = S1
    
    # 2. Local Search from S1
    remaining_time = deadline - time.time()
    if remaining_time > 1.0:
        S2 = local_search(S1, time.time() + remaining_time * 0.4)
        if len(S2) > best_size:
            best_size = len(S2)
            best_overall = S2

    # 3. Try Random Greedy restarts
    rng_main = random.Random(999)
    while time.time() < deadline - 2.0:
        # Generate a random order
        order = list(range(1, N + 1))
        rng_main.shuffle(order)
        
        S_rand = fast_greedy(order)
        
        if len(S_rand) > best_size:
            best_size = len(S_rand)
            best_overall = S_rand
            
            # If we found a new best, run local search on it
            remaining = deadline - time.time()
            if remaining > 1.0:
                S_ls = local_search(S_rand, time.time() + remaining * 0.3)
                if len(S_ls) > best_size:
                    best_size = len(S_ls)
                    best_overall = S_ls

    # Final check
    if not best_overall:
        # Fallback: simple set
        best_overall = list(range(1, N + 1, 4))

    return best_overall

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")
    # Verify
    in_res = [False] * (N + 1)
    for x in res:
        in_res[x] = True
    valid = True
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if in_res[a] and in_res[b] and in_res[c] and in_res[e]:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
