import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Greedy initialization
    S = []
    counts = {}
    
    def try_add(x):
        if x in S:
            return False
        # Check if adding x violates B2[2]
        # New sums: x + y for y in S, and 2*x
        # We need to ensure count[s] + 1 <= 2 for all new sums
        # i.e., count[s] < 2 for all new sums
        valid = True
        for y in S:
            s = x + y
            if counts.get(s, 0) >= 2:
                valid = False
                break
        if valid:
            if counts.get(2*x, 0) >= 2:
                valid = False
        
        if valid:
            S.append(x)
            S.sort() # Keep sorted for determinism or efficiency if needed, but set is better for lookup
            # Actually, let's use a set for S to make 'in' check O(1)
            # But we need to iterate over S. A list is fine for iteration, set for lookup.
            # Let's stick to a list for S and a set for fast membership if needed, or just list.
            # Since N=1000, list iteration is fast.
            for y in S:
                if y != x:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
            counts[2*x] = counts.get(2*x, 0) + 1
            return True
        return False

    # Initial greedy pass
    for x in range(1, 1001):
        try_add(x)
    
    S_set = set(S)
    
    # Local search
    # We will try to insert elements not in S, and swap elements in S with those not in S.
    # To maximize size, insertion is primary. Swap is secondary to break plateaus.
    
    candidates = [x for x in range(1, 1001) if x not in S_set]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try insertions
        # We can't just iterate candidates once, because adding one might make another invalid?
        # No, adding elements only increases counts, so if x was valid before, it might become invalid later.
        # But if x was invalid before, it will remain invalid (counts only go up).
        # So we should check candidates in some order.
        # Let's try to insert any candidate that is currently valid.
        
        # To be efficient, we might not want to check all 1000 candidates every time.
        # But 1000 checks * (size of S) operations is ~1000*300 = 300k ops. Very fast.
        
        # Re-shuffle or iterate through a pool of candidates?
        # Let's iterate through all possible x not in S.
        # This is O(1000 * |S|).
        
        new_S = S[:]
        new_S_set = S_set.copy()
        
        # We will perform a single pass of potential insertions.
        # If we insert one, we should re-evaluate others?
        # Actually, inserting one makes it harder to insert others.
        # So greedy insertion in a fixed order is fine.
        
        # Let's try to insert from the candidate list.
        # If we find one, insert it, mark improved, and restart the scan (or continue).
        # Restarting is safer to ensure we don't miss a "best" insertion, but costly.
        # Given the speed, we can just try to insert the first valid one found, then restart.
        
        # Optimization: Pre-compute which candidates are currently "blocked"?
        # A candidate x is blocked if there exists y in S such that count[x+y] == 2 or count[2x] == 2.
        
        for x in range(1, 1001):
            if x in S_set:
                continue
            
            # Check validity of adding x
            valid = True
            for y in S:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                if counts.get(2*x, 0) >= 2:
                    valid = False
            
            if valid:
                # Perform insertion
                S.append(x)
                S_set.add(x)
                for y in S:
                    if y != x:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                counts[2*x] = counts.get(2*x, 0) + 1
                improved = True
                break # Restart the loop to re-evaluate other candidates
        
        if improved:
            continue
            
        # If no insertion was possible, try swaps
        # Swap u (in S) with v (not in S)
        # This is more expensive. O(|S| * (1000-|S|) * |S|)
        # We can limit this to a few iterations or random sampling.
        
        # Let's try a limited number of random swaps
        # If |S| is large, say 300, then 300 * 700 * 300 is ~63 million ops. Too slow for Python loops.
        # We need to be smarter or limit swaps.
        
        # Maybe just try to swap out the "most constrained" element?
        # Or just give up on swaps if insertions stall?
        # The seed score was 1.0. Insertions alone might get us to ~1.02-1.05 based on knowledge.
        # Let's see if we can afford some swaps.
        
        # A cheaper swap check:
        # For a swap u->v to be valid, the new set must satisfy B2[2].
        # This is complex to check incrementally without full rebuild.
        # Given the time limit, let's focus on maximizing insertions first.
        
        # If we stall, we can try random restarts?
        # Or just break.
        
        # Let's add a simple random restart mechanism if we stall for a while?
        # No, we don't have a timer for "stall".
        
        # Let's just break if no insertion is found.
        break
        
    return S

# To ensure we don't miss easy gains, let's refine the insertion logic.
# The above loop breaks after ONE insertion. This is effectively greedy insertion.
# Is greedy insertion optimal? No.
# But it's a good baseline.

# Can we do better?
# What if we try all possible insertions in a random order?
# The current loop `for x in range(1, 1001)` is deterministic.
# Randomizing might help escape local minima where a specific order prevents a larger set.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_S = []
    best_size = 0
    
    # We will run multiple random restarts of the greedy+insertion process
    # Since N=1000 is small, we can afford a few restarts.
    
    for restart in range(5): # Max 5 restarts
        S = []
        S_set = set()
        counts = {}
        
        def try_add(x):
            if x in S_set:
                return False
            valid = True
            for y in S:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                if counts.get(2*x, 0) >= 2:
                    valid = False
            if valid:
                S.append(x)
                S_set.add(x)
                for y in S:
                    if y != x:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                counts[2*x] = counts.get(2*x, 0) + 1
                return True
            return False

        # Phase 1: Greedy initialization with random perturbation?
        # Or just standard greedy?
        # Standard greedy is deterministic.
        # Let's try a randomized greedy: at each step, pick a random valid x?
        # That's expensive to find valid x.
        
        # Let's stick to standard greedy for init, then local search.
        for x in range(1, 1001):
            try_add(x)
            
        # Phase 2: Local Search (Insertions)
        # We will try to insert elements not in S.
        # We will iterate through candidates in random order to avoid bias.
        
        candidates = [x for x in range(1, 1001) if x not in S_set]
        random.shuffle(candidates)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to insert one candidate
            # We pick the first valid one from the shuffled list?
            # Or we scan the whole list and pick the first valid one?
            # Scanning the whole list is O(1000 * |S|).
            
            for x in candidates:
                if x in S_set:
                    continue
                valid = True
                for y in S:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
                if valid:
                    if counts.get(2*x, 0) >= 2:
                        valid = False
                
                if valid:
                    S.append(x)
                    S_set.add(x)
                    for y in S:
                        if y != x:
                            s = x + y
                            counts[s] = counts.get(s, 0) + 1
                    counts[2*x] = counts.get(2*x, 0) + 1
                    improved = True
                    # Re-shuffle candidates to avoid bias?
                    # Or just remove x from candidates?
                    candidates.remove(x)
                    random.shuffle(candidates)
                    break # Restart scan with new state
            
            if not improved:
                # No more insertions possible with current set
                # Try swaps?
                # Swaps are expensive. Let's skip for now and rely on restarts.
                break
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
            
        if time.time() > deadline - 10.0:
            break
            
    return best_S
