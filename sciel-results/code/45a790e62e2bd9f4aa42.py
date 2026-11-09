import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    N = 1000
    # Elements are 1..1000. Sums are 2..2000.
    # Map sum s to index s-2 in a vector of size 1999.
    max_sum = 2 * N
    sum_size = max_sum - 1 # indices 0..1998 for sums 2..2000
    
    def get_counts(S):
        """
        Compute sum counts for a set S (represented as a boolean numpy array of length N+1, 1-indexed).
        Returns a numpy array of length sum_size where index i holds count of sum i+2.
        """
        counts = np.zeros(sum_size, dtype=np.int32)
        # Vectorized approach is hard for arbitrary sets without loops due to varying S size.
        # But we can do it semi-vectorized or just use a fast loop if S is small.
        # Given N=1000, S size ~ 400-500. Pairs ~ 100k-125k.
        # A pure Python loop over pairs is ~0.05s.
        # We can speed it up by extracting indices.
        if not isinstance(S, np.ndarray):
            S = np.array(S)
        elems = np.nonzero(S)[0]
        if len(elems) == 0:
            return counts
        
        # Using numpy for pair sums is memory intensive (500x500 = 250k ints, ok).
        # But we need a<=b.
        # Let's stick to a fast Python loop for counts generation as it's called only at init and occasionally.
        # For SA moves, we update incrementally.
        
        # Actually, let's write a fast C-like loop in Python for init.
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i, len(elems)):
                b = elems[j]
                s = a + b
                counts[s - 2] += 1
        return counts

    # Initial state: Random set
    # A good initial density is around 3/2 * sqrt(N) ? No, B2[2] allows ~ c * N. 
    # Wait, the prompt says "asymptotic constant is at least 3/2". 
    # Usually B2 sets (Sidon) have size ~ sqrt(N). 
    # B2[2] sets (weak Sidon) have size ~ c * N? 
    # Let's check the prompt: "Large subsets of {1, ..., 1000} in which every sum has at most two representations".
    # Standard Sidon (B2) is 1 rep. B2[2] is 2 reps.
    # It is known that B2[2] sets can have linear size? 
    # Actually, for B_h sets, the size is N^(1/h). For h=2, size is N^(1/2) for B2.
    # For B2[2] (allowing 2 reps), the size is still O(N^(1/2))? 
    # No, the "asymptotic constant is at least 3/2" likely refers to the coefficient in the bound or a specific construction.
    # Wait, if the constant is 3/2, and N=1000, sqrt(1000) ~ 31. 3/2 * 31 ~ 46?
    # Or is it 3/2 * N? That would be 1500, which is impossible as max size is 1000.
    # Let's re-read carefully: "asymptotic constant is at least 3/2".
    # In additive combinatorics, the maximum size of a B2 set in [N] is ~ sqrt(N).
    # For B2[2], the maximum size is also ~ sqrt(N) but with a larger constant.
    # If the constant is 3/2, and the base is sqrt(N), then size ~ 1.5 * sqrt(1000) ~ 47.
    # Let's assume the size is around 40-60.
    
    # Let's start with a random set of size ~ 50.
    initial_size = 50
    S = np.zeros(N + 1, dtype=bool)
    random.seed(42)
    candidates = random.sample(range(1, N + 1), initial_size)
    for x in candidates:
        S[x] = True
        
    counts = get_counts(S)
    
    # Calculate initial penalty: number of sums with count > 2
    # We want to minimize penalty, then maximize size.
    # Objective: Minimize (Penalty * LargeWeight - Size)
    # Or simply: Maximize Size - Penalty * 1000
    
    def get_penalty(counts):
        return np.sum(counts > 2)
    
    def get_score(S, counts):
        size = np.sum(S)
        pen = get_penalty(counts)
        return size * 1000 - pen # Prioritize validity, then size
    
    best_score = get_score(S, counts)
    best_S = S.copy()
    best_counts = counts.copy()
    
    # Simulated Annealing
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    
    # Precompute neighbors? No, random moves.
    # Move: Flip a random bit.
    
    while time.time() < deadline:
        # Pick a random element to flip
        x = random.randint(1, N)
        
        old_val = S[x]
        old_score = get_score(S, counts)
        
        # Tentatively flip
        S[x] = not old_val
        
        # Update counts incrementally
        # If adding x:
        #   For all y in S (including x if y>=x? No, a<=b. If we add x, new pairs are (x, y) where y in S and y>=x? 
        #   Wait, the set S changes. 
        #   Pairs involving x: (x, y) for all y in S such that y >= x? 
        #   Actually, the representation is a+b with a<=b.
        #   If we add x to S, the new pairs are (x, y) where y is in the NEW S and y >= x?
        #   No, y is in the OLD S (before adding x, x wasn't there).
        #   The pairs are (x, y) where y is in S_old.
        #   Condition: min(x, y) <= max(x, y). This is always true.
        #   But we count each pair once.
        #   If we maintain counts for all pairs (a,b) with a<=b in the current set:
        #   When adding x:
        #     New pairs are (x, y) for all y in S_old.
        #     We must ensure we don't double count.
        #     Since x is new, it wasn't in S_old.
        #     The pairs are (x, y) for all y in S_old.
        #     Do we need to check order? No, the sum x+y is the same.
        #     The count of sum s increases by the number of y in S_old such that x+y=s.
        #     i.e., y = s-x.
        #     So for each y in S_old, sum s = x+y gets +1.
        #     This includes y=x (if x is in S_old? No, x was not in S_old).
        #     So we iterate over all y in S_old.
        
        # When removing x:
        #     Remove pairs (x, y) for all y in S_new (which is S_old minus x).
        #     Wait, if we remove x, the pairs that disappear are those involving x.
        #     These are (x, y) for all y in S_after_removal.
        #     Note: if y=x, the pair is (x,x).
        
        if not old_val: # Adding x
            # Iterate over current S (which does NOT contain x)
            # We need the list of elements in S.
            # For speed, we can maintain a list of active elements?
            # Or just iterate 1..N and check S[y]. N=1000 is small.
            for y in range(1, N + 1):
                if S[y]:
                    s = x + y
                    counts[s - 2] += 1
        else: # Removing x
            # S[x] is now False.
            # Iterate over current S (which does NOT contain x)
            for y in range(1, N + 1):
                if S[y]:
                    s = x + y
                    counts[s - 2] -= 1
            
            # Don't forget the pair (x, x) if it existed?
            # When we added x, we added (x, y) for all y in S_old.
            # If x was not in S_old, (x,x) was NOT added in the loop above?
            # Wait. If we add x, and x is not in S_old, then the pair (x,x) is a NEW pair.
            # Did my loop add it?
            # Loop: for y in S_old. If x not in S_old, y never equals x.
            # So (x,x) was NOT added!
            # We need to explicitly handle (x,x).
            
            # Correction:
            # When adding x:
            #   Add (x, y) for all y in S_old.
            #   Add (x, x) ? Yes, (x,x) is a valid pair if x is in the set.
            #   So we must add 1 to count[2x-2].
            
            # Let's re-verify the "Adding" logic.
            # New pairs: (x, y) for y in S_new.
            # S_new = S_old U {x}.
            # Pairs: (x, y) for y in S_old, and (x, x).
            # So we should add 1 for each y in S_old, AND 1 for (x,x).
            
            # My code above: `if S[y]: counts[x+y-2] += 1`.
            # Since S[x] is now True, if y=x, it would add 1.
            # But wait, in the "Adding" branch, I set `S[x] = True` BEFORE the loop.
            # So `S[x]` is True.
            # So when y=x, `S[x]` is True, so it adds 1 to count[2x-2].
            # This correctly accounts for (x,x).
            
            # Now for "Removing":
            # I set `S[x] = False` BEFORE the loop.
            # So `S[x]` is False.
            # The loop iterates y in S (which excludes x).
            # So it removes (x, y) for all y in S_old \ {x}.
            # But it DOES NOT remove (x, x).
            # Because y=x is not in S (since S[x] is False).
            # So I must explicitly remove (x, x).
            counts[2 * x - 2] -= 1
            
        new_score = get_score(S, counts)
        
        delta = new_score - old_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            # Accept
            if new_score > best_score:
                best_score = new_score
                best_S = S.copy()
                best_counts = counts.copy()
        else:
            # Reject: revert
            S[x] = old_val
            if old_val: # Was True, so we removed it. Now we add it back.
                for y in range(1, N + 1):
                    if S[y]:
                        s = x + y
                        counts[s - 2] += 1
                # Wait, if we revert an addition (old_val=False, new_val=True, rejected -> back to False)
                # We need to undo the addition.
                # Addition logic:
                #   S[x] = True.
                #   Loop y in S (with x=True): adds (x,y) for all y in S.
                #   This includes (x,x).
                # So to undo, we subtract those.
                
            else: # Was False, so we added it. Now we remove it.
                # Removal logic:
                #   S[x] = False.
                #   Loop y in S (with x=False): subtracts (x,y) for y in S.
                #   Explicitly subtracts (x,x).
                # To undo, we add them back.
                for y in range(1, N + 1):
                    if S[y]:
                        s = x + y
                        counts[s - 2] += 1
                counts[2 * x - 2] += 1
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    # Final verification and cleanup
    # Ensure best_S is valid
    # The score function penalizes invalid sets, so best_S should be valid or close.
    # Let's do a final greedy repair on best_S to ensure validity.
    
    S_list = np.nonzero(best_S)[0].tolist()
    S_set = set(S_list)
    counts_dict = {}
    for i, x in enumerate(S_list):
        for y in S_list[i:]:
            s = x + y
            counts_dict[s] = counts_dict.get(s, 0) + 1
            
    # Repair: Remove elements that cause violations
    # This is a simple heuristic repair
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Find sums with count > 2
        bad_sums = [s for s, c in counts_dict.items() if c > 2]
        if not bad_sums:
            break
            
        # For each bad sum, remove one of the elements involved?
        # This is complex. Simpler: just return the set if it's valid, else trim.
        # Given the SA minimizes penalty, it's likely valid.
        # Let's just check validity.
        
    # Check validity of best_S
    is_valid = True
    counts_check = {}
    for i, x in enumerate(S_list):
        for y in S_list[i:]:
            s = x + y
            counts_check[s] = counts_check.get(s, 0) + 1
            if counts_check[s] > 2:
                is_valid = False
                break
        if not is_valid:
            break
            
    if not is_valid:
        # Fallback: Use a greedy construction which is guaranteed valid
        S_list = []
        S_set = set()
        counts = {}
        for x in range(1, N + 1):
            can_add = True
            for y in S_list:
                s = x + y
                if counts.get(s, 0) >= 2:
                    can_add = False
                    break
            if can_add:
                s = 2 * x
                if counts.get(s, 0) >= 2:
                    can_add = False
            
            if can_add:
                S_list.append(x)
                S_set.add(x)
                for y in S_list[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1

    return S_list
