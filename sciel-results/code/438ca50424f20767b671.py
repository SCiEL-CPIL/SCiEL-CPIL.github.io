import time
import itertools

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # Standard construction fallback
    def standard_superperm(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # Try to find a shorter solution using a greedy/beam search heuristic
    # State: last n-1 symbols
    # Goal: Visit all n! permutations as windows of size n
    
    if n <= 4:
        return standard_superperm(n)
        
    # For n=7, the standard is 5913.
    # We try a greedy approach.
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    target_count = len(all_perms)
    
    # Initial state: 1, 2, ..., n-1
    # We need to build the string.
    # Let's maintain a list of the current string.
    # Start with n-1 symbols.
    curr_str = list(range(1, n))
    # The first window of size n is not yet formed.
    # We need to append symbols.
    
    visited_perms = set()
    
    # Check if the current suffix of length n (if length >= n) is a new perm
    # Initially length is n-1.
    
    # Heuristic: 
    # At each step, choose the next symbol that:
    # 1. Creates a new permutation window (if possible)
    # 2. Maximizes the "potential" (number of unvisited perms that can be reached soon)
    
    # Simplified Heuristic:
    # Prefer symbols that create a NEW permutation.
    # If multiple, prefer the one that creates a permutation which has the "most unique" prefix/suffix?
    # Actually, just picking any new permutation is a good start.
    
    # To avoid getting stuck, we can use a randomized greedy with restarts.
    # But time is tight.
    
    # Let's try a deterministic greedy first.
    # If it fails, fall back.
    
    # Optimization: 
    # The state is the last n-1 symbols.
    # There are n^(n-1) states.
    # We can do a BFS for the shortest path to visit all perms? 
    # No, state space of visited perms is 2^(n!). Too big.
    
    # So we stick to a heuristic.
    
    # Greedy: 
    # 1. Start with 1..n-1
    # 2. Next symbol must be n to form the first perm (1..n).
    #    Actually, any symbol works, but only n forms a perm with 1..n-1.
    #    If we pick a symbol already in 1..n-1, the window has duplicates.
    #    So, to form a perm, we MUST pick the missing symbol.
    #    This implies that IF we want the current window to be a permutation, the next symbol is FORCED.
    #    Wait, this logic from before was: "If we want the window to be a perm, the next symbol is the missing one."
    #    So, if we are in a state where the last n-1 are distinct, there is exactly ONE symbol that creates a new permutation.
    #    Any other symbol creates a non-permutation window.
    
    # So, the strategy is:
    # - Sometimes we "waste" a step by creating a non-perm window to reposition for a better path of perms.
    # - Sometimes we take the "forced" step to collect a perm.
    
    # The standard construction essentially always takes the forced step when possible?
    # No, it inserts the new symbol in a specific way.
    
    # Let's try to find a path that collects all perms with minimal "waste" steps.
    
    # Since the forced step is unique, the "choices" are only when we decide to NOT take the forced step.
    # Or rather, we are always in a state of last n-1 distinct symbols?
    # If we append a duplicate, the last n-1 might have duplicates.
    # If the last n-1 have duplicates, the next window CANNOT be a permutation.
    # So we must "wash out" the duplicate before we can form a perm again.
    
    # This suggests a graph where nodes are "distinct suffixes".
    # If we are at a distinct suffix, we have 1 "perm" option and n-1 "non-perm" options.
    # If we take a non-perm option, we enter a state with duplicates.
    # We must then move to a state with distinct suffixes again to collect more perms.
    
    # Let's run a simple search for n=7 with a time limit.
    
    best_word = standard_superperm(n)
    best_len = len(best_word)
    
    # Heuristic Search
    # We will try to build a word.
    # We track the set of visited permutations.
    # We use a greedy approach with a "temperature" or randomization.
    
    import random
    random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
            
        curr = list(range(1, n)) # Start with 1..n-1
        visited = set()
        
        # First window is not formed yet.
        # We need to append n to form 1..n.
        # Or we can append something else.
        # Let's start by appending the missing symbol to form the first perm.
        missing = next(x for x in range(1, n+1) if x not in curr)
        curr.append(missing)
        first_window = tuple(curr[-n:])
        if len(set(first_window)) == n:
            visited.add(first_window)
        
        # Now we have a state of last n-1 distinct symbols.
        # We iterate.
        # Max steps: best_len (we don't want to be worse)
        for _ in range(best_len - n + 1):
            if len(visited) == target_count:
                break
            if time.time() > deadline - 5:
                break
                
            suffix = tuple(curr[-(n-1):])
            
            # Find the next symbol that forms a new permutation
            # The missing symbol from the suffix
            missing = next((x for x in range(1, n+1) if x not in suffix), None)
            
            if missing is not None:
                candidate_window = tuple(list(suffix) + [missing])
                if len(set(candidate_window)) == n:
                    if candidate_window not in visited:
                        # Greedy: always take the new perm?
                        # This might lead to a dead end.
                        # Let's just take it.
                        curr.append(missing)
                        visited.add(candidate_window)
                        continue
            
            # If no new perm is possible (or we choose to skip), pick a symbol to reposition.
            # Pick a symbol that is NOT in the suffix? 
            # If we pick a symbol in the suffix, we create a duplicate, making it harder to form a perm next step.
            # If we pick a symbol NOT in the suffix, it's the same as the "missing" one, which we just skipped or didn't exist?
            # If missing is not None, it's the only one not in suffix.
            # So if we didn't take it, we MUST take a symbol IN the suffix.
            # This creates a duplicate in the window, so the next window will not be a perm.
            
            # So, we are forced to "break" the chain of perms.
            # We pick a symbol from the suffix.
            # Which one?
            # Heuristic: Pick the one that appears most frequently in unvisited perms' prefixes?
            # Simple heuristic: Randomly pick one from the suffix.
            
            # To be more effective, let's look ahead.
            # If we pick x from suffix, the new suffix will have x duplicated?
            # No, the new suffix is (s_2, ..., s_{n-1}, x).
            # If x was s_1, the new suffix is (s_2, ..., s_{n-1}, s_1).
            # This is a rotation. It is still distinct!
            # Wait.
            # Suffix S = (s_1, ..., s_{n-1}). Distinct.
            # If we append s_1, new suffix S' = (s_2, ..., s_{n-1}, s_1). Distinct.
            # The window formed is (s_1, s_2, ..., s_{n-1}, s_1). Duplicate s_1. Not a perm.
            # But the new STATE S' is distinct.
            # So we can form a perm in the NEXT step.
            
            # So, if we append s_1, we waste 1 step (no new perm) but stay in a "good" state.
            # If we append s_i (i>1), new suffix is (s_2, ..., s_{i-1}, s_i, s_{i+1}, ..., s_{n-1}, s_i).
            # Here s_i is duplicated in the suffix?
            # S = (s_1, s_2, s_3). Append s_2. S' = (s_2, s_3, s_2). Duplicate.
            # So S' has duplicates. We cannot form a perm in the next step.
            # We must wait until the duplicate is washed out.
            
            # So, appending s_1 (the first element of the suffix) is the "cheapest" waste.
            # It rotates the suffix, keeping it distinct.
            
            # So the strategy:
            # If the forced perm is visited, or we want to explore, we can rotate the suffix.
            # Rotating k times takes k steps and visits 0 new perms (if the perms formed by rotations are already visited).
            
            # Let's just try the "forced" step always. If it fails to cover all, we fail.
            # But the forced step always visits a new perm if the state is distinct and the perm hasn't been visited.
            # Since we start with a distinct state, and rotating keeps it distinct, we will keep visiting new perms.
            # Until we run out of new perms for the current "cycle"?
            
            # Actually, if we always take the forced step, we are just cycling through perms.
            # How many distinct perms can we visit with forced steps?
            # It depends on the starting point.
            
            # For now, this heuristic is too weak to guarantee < 5913.
            # I will return the standard construction.
            pass
        
        # Check if this attempt produced a valid shorter string
        if len(curr) < best_len:
            # Verify
            windows = set()
            for i in range(len(curr) - n + 1):
                w = tuple(curr[i:i+n])
                if len(set(w)) == n:
                    windows.add(w)
            if len(windows) == target_count:
                best_word = "".join(map(str, curr))
                best_len = len(curr)

    return best_word
