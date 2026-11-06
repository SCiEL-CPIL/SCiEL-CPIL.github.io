import time
import random

def construct():
    start_time = time.time()
    # 90s limit, use 55s for safety
    deadline = start_time + 55.0
    N = 4096
    
    def check_and_get_new_sums(S, triple_sums, pair_sums, x):
        """
        Checks if x can be added to S without violating unique triple sums.
        S: list of current elements
        triple_sums: set of all i<=j<=k sums
        pair_sums: set of all i<=j sums
        Returns (True, new_triple_sums, new_pair_sums) if valid, (False, None, None) otherwise.
        """
        n = len(S)
        new_triple = set()
        new_pair = set()
        
        # 1. (x, x, x)
        s3 = 3 * x
        if s3 in triple_sums:
            return False, None, None
        new_triple.add(s3)
        
        # 2. (x, x, s_i) -> sum is 2x + s_i
        # Also generates new pairs: (x, x) -> 2x, (x, s_i) -> x + s_i
        new_pair.add(2 * x)
        for s_i in S:
            s_val = 2 * x + s_i
            if s_val in triple_sums or s_val in new_triple:
                return False, None, None
            new_triple.add(s_val)
            
            p_val = x + s_i
            new_pair.add(p_val)
            
        # 3. (x, s_i, s_j) with i <= j
        # Sum is x + (s_i + s_j)
        # We iterate over existing pair_sums
        for p in pair_sums:
            s_val = x + p
            if s_val in triple_sums or s_val in new_triple:
                return False, None, None
            new_triple.add(s_val)
            
        return True, new_triple, new_pair

    def greedy_search(order, time_limit):
        """
        Performs a greedy search given a specific order of candidates.
        Returns the selected set.
        """
        S = []
        triple_sums = set()
        pair_sums = set()
        
        for x in order:
            if time.time() > time_limit:
                break
            valid, new_triple, new_pair = check_and_get_new_sums(S, triple_sums, pair_sums, x)
            if valid:
                S.append(x)
                triple_sums.update(new_triple)
                pair_sums.update(new_pair)
                
        return S

    def compute_all_sums(S):
        """Recomputes the full set of triple and pair sums for a given set S."""
        triple_sums = set()
        pair_sums = set()
        n = len(S)
        for i in range(n):
            si = S[i]
            pair_sums.add(2 * si) # (si, si)
            for j in range(i + 1, n):
                sj = S[j]
                pair_sums.add(si + sj)
                
        # Now compute triple sums using pair sums?
        # Triple sums are (a,b,c) with a<=b<=c.
        # We can just do triple loop for correctness in this helper, it's only called for kicks.
        for i in range(n):
            si = S[i]
            for j in range(i, n):
                sj = S[j]
                for k in range(j, n):
                    sk = S[k]
                    triple_sums.add(si + sj + sk)
        return triple_sums, pair_sums

    best_set = []
    best_len = 0
    
    # Strategy 1: Natural Order (Smallest to Largest)
    # This is a standard heuristic for B_h sets as it keeps sums small.
    order_natural = list(range(1, N + 1))
    S_nat = greedy_search(order_natural, deadline)
    if len(S_nat) > best_len:
        best_set = S_nat
        best_len = len(S_nat)
        
    if time.time() > deadline - 5.0:
        return best_set

    # Strategy 2: Randomized Greedy Restarts
    random.seed(42)
    
    # We can afford roughly 50-100 restarts depending on speed.
    # Each restart is O(N * K^2).
    # Let's loop until time is up.
    
    while time.time() < deadline - 15.0: # Leave 15s for local improvement
        # Create a random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S_try = greedy_search(order, deadline - 15.0)
        
        if len(S_try) > best_len:
            best_set = S_try
            best_len = len(S_try)
            
        # Optional: Occasionally try reverse order or biased order
        if random.random() < 0.1:
             order_rev = list(range(N, 0, -1))
             random.shuffle(order_rev)
             S_try_rev = greedy_search(order_rev, deadline - 15.0)
             if len(S_try_rev) > best_len:
                 best_set = S_try_rev
                 best_len = len(S_try_rev)

    # Strategy 3: Local Improvement (Kicks)
    # Try to improve the best set found by removing one element and adding one or more.
    if best_len > 0 and time.time() < deadline - 2.0:
        # First, try simple additions (in case greedy stopped early or order was bad)
        S_cur = list(best_set)
        triple_cur, pair_cur = compute_all_sums(S_cur)
        S_cur_set = set(S_cur)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for x in range(1, N + 1):
                if x in S_cur_set:
                    continue
                valid, new_triple, new_pair = check_and_get_new_sums(S_cur, triple_cur, pair_cur, x)
                if valid:
                    S_cur.append(x)
                    S_cur_set.add(x)
                    triple_cur.update(new_triple)
                    pair_cur.update(new_pair)
                    best_len = len(S_cur)
                    best_set = S_cur
                    improved = True
                    break # Restart the loop as the set changed
                
        # Now try Kicks: Remove one, add one.
        if time.time() < deadline - 2.0:
            S_kick = list(best_set)
            triple_kick, pair_kick = compute_all_sums(S_kick)
            S_kick_set = set(S_kick)
            
            # Try a few random kicks
            kick_attempts = 0
            while time.time() < deadline - 1.0 and kick_attempts < 20:
                kick_attempts += 1
                if not S_kick:
                    break
                # Remove a random element
                idx_to_remove = random.randint(0, len(S_kick) - 1)
                removed_elem = S_kick.pop(idx_to_remove)
                S_kick_set.discard(removed_elem)
                
                # Recompute sums for the reduced set? 
                # It's safer to recompute for Kicks as we are modifying the set structure significantly.
                # Or we can try to subtract sums, but that's complex.
                # Given K is small, recompute is fast.
                triple_kick, pair_kick = compute_all_sums(S_kick)
                
                # Try to add any element
                found_addition = False
                for x in range(1, N + 1):
                    if x in S_kick_set:
                        continue
                    valid, new_triple, new_pair = check_and_get_new_sums(S_kick, triple_kick, pair_kick, x)
                    if valid:
                        S_kick.append(x)
                        S_kick_set.add(x)
                        triple_kick.update(new_triple)
                        pair_kick.update(new_pair)
                        found_addition = True
                        break
                        
                if found_addition:
                    # We have a new set of the same size.
                    # Let's see if we can add MORE to this new set (greedy expansion)
                    improved2 = True
                    while improved2 and time.time() < deadline - 1.0:
                        improved2 = False
                        for x in range(1, N + 1):
                            if x in S_kick_set:
                                continue
                            valid, new_triple, new_pair = check_and_get_new_sums(S_kick, triple_kick, pair_kick, x)
                            if valid:
                                S_kick.append(x)
                                S_kick_set.add(x)
                                triple_kick.update(new_triple)
                                pair_kick.update(new_pair)
                                improved2 = True
                                break
                                
                    if len(S_kick) > best_len:
                        best_set = S_kick
                        best_len = len(S_kick)
                else:
                    # Put the element back if we didn't find an addition? 
                    # Actually, we popped it. If we don't find an addition, we should revert 
                    # to try a different kick, or accept the smaller set? 
                    # Usually we want to keep the size. So revert.
                    S_kick.append(removed_elem)
                    S_kick_set.add(removed_elem)
                    triple_kick, pair_kick = compute_all_sums(S_kick)

    return best_set
