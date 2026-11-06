import time
import random

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin before 90s kill
    N = 4096
    MAX_SUM = 3 * N  # 12288

    # bytearray: sum_used[s] = 1 if sum s is already taken
    sum_used = bytearray(MAX_SUM + 1)
    S = []  # sorted list of chosen numbers
    best_S = []
    best_size = 0

    def _elapsed():
        return time.time() - t0

    def _check_and_add(x):
        """Try to add x to S. Return True if successful, False otherwise.
        On success, updates sum_used and S in place."""
        n = len(S)
        # Collect all new sums involving x
        # We need to verify none are already used AND no internal collisions
        # Sums: 3x, 2x+y for y in S, x+2y for y in S, x+y+z for y<z in S
        
        # First check 3x
        if sum_used[3 * x]:
            return False
        
        # Check 2x+y and x+2y for each y in S
        # Also need to check for internal collisions among the new sums
        # Internal collision cases for unordered triples:
        # 3x = 2x+y => x=y (impossible, x not in S)
        # 3x = x+2y => x=2y => check
        # 3x = x+y+z => 2x=y+z => check
        # 2x+y = x+2y => x=y (impossible)
        # 2x+y = x+y+z => x=z (impossible, x not in S)
        # 2x+y = 2x+w => y=w (impossible, distinct in S)
        # x+2y = x+2z => y=z (impossible)
        # x+2y = x+y+z => y=z (impossible)
        # x+y+z = x+y+w => z=w (impossible)
        # x+2y = 2x+w => x+2y-2x=w => 2y-x=w, possible!
        # 2x+y = x+2z => x+2z-2x=y => 2z-x=y, possible!
        # x+y+z = x+2y => z=y (impossible)
        
        # So we need to check:
        # 1. 3x not used, and 3x != x+2y for any y (i.e., 2x != 2y, i.e., x!=y, ok)
        #    Actually 3x = x+2y means 2x = 2y means x = y. Not possible.
        #    Wait: 3x = x + 2y => 2x = 2y => x = y. Since x not in S, ok.
        # 2. 2x+y: check not used, and check 2x+y != x+2w for any w in S
        #    2x+y = x+2w => x+y = 2w => w = (x+y)/2
        #    So if (x+y) is even and (x+y)//2 is in S, collision!
        # 3. x+2y: similar, x+2y = 2x+w => 2y-x = w, so if 2y-x is in S, collision
        # 4. x+y+z: check not used, and x+y+z = 2x+w => y+z = x+w => w = y+z-x
        #    Also x+y+z = x+2w => y+z = 2w => w = (y+z)/2
        #    Also x+y+z = 2x+w already covered
        #    Also x+y+z = x+2y => z=y impossible
        #    Also x+y+z = 2x+y => z=x impossible
        
        # Let me use a set to collect all new sums and check for duplicates and existing
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
            new_sums.add(x + 2 * y)
        for i in range(n):
            for j in range(i + 1, n):
                new_sums.add(x + S[i] + S[j])
        
        expected_count = 1 + 2 * n + n * (n - 1) // 2
        if len(new_sums) < expected_count:
            return False
        
        for s in new_sums:
            if sum_used[s]:
                return False
        
        # All good, commit
        for s in new_sums:
            sum_used[s] = 1
        S.append(x)
        S.sort()
        return True

    def _remove(x):
        """Remove x from S and clear its sums from sum_used."""
        n = len(S)
        S.remove(x)
        new_sums = set()
        new_sums.add(3 * x)
        for y in S:
            new_sums.add(2 * x + y)
            new_sums.add(x + 2 * y)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                new_sums.add(x + S[i] + S[j])
        for s in new_sums:
            if sum_used[s]:
                sum_used[s] = 0

    def _backtrack(depth, max_depth):
        """Recursively try to build S by trying numbers in increasing order.
        depth is current size of S."""
        nonlocal best_S, best_size
        
        if _elapsed() > deadline - 2.0:
            return
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
        
        if len(S) >= max_depth:
            return
        
        # Try numbers starting from (last element + 1) or 1 if empty
        if S:
            start = S[-1] + 1
        else:
            start = 1
        
        for x in range(start, N + 1):
            if _elapsed() > deadline - 2.0:
                return
            
            if _check_and_add(x):
                _backtrack(depth + 1, max_depth)
                _remove(x)
            
            # Pruning: if we've tried enough and haven't improved, stop
            # This is a soft bound - we'll continue but check time
    
    # Phase 1: Greedy increasing (fast, gives a good baseline)
    S.clear()
    sum_used[:] = b'\x00' * (MAX_SUM + 1)
    
    for x in range(1, N + 1):
        if _elapsed() > deadline - 10.0:
            break
        if _check_and_add(x):
            pass
    
    if len(S) > best_size:
        best_size = len(S)
        best_S = S[:]
    
    greedy_size = len(S)
    greedy_S = S[:]
    
    # Phase 2: Backtracking from the greedy solution to try to improve
    # Reset and do a fresh backtracking search
    S.clear()
    sum_used[:] = b'\x00' * (MAX_SUM + 1)
    
    # Use the greedy solution as a starting point, then try to extend
    # Actually, let's do a clean backtracking that explores small numbers first
    # This should find the maximum cardinality set
    
    # Try backtracking with a time budget
    _backtrack(0, 80)
    
    if len(S) > best_size:
        best_size = len(S)
        best_S = S[:]
    
    # Phase 3: Local search to try to improve further
    if best_S and _elapsed() < deadline - 5.0:
        # Work with the best solution found
        current_S = best_S[:]
        # Rebuild sum_used for current_S
        sum_used[:] = b'\x00' * (MAX_SUM + 1)
        for x in current_S:
            new_sums = set()
            new_sums.add(3 * x)
            for y in current_S:
                if y != x:
                    new_sums.add(2 * x + y)
                    new_sums.add(x + 2 * y)
            for i in range(len(current_S)):
                for j in range(i + 1, len(current_S)):
                    if current_S[i] != x and current_S[j] != x:
                        new_sums.add(x + current_S[i] + current_S[j])
            for s in new_sums:
                sum_used[s] = 1
        
        rng = random.Random(42)
        candidates = list(range(1, N + 1))
        
        for _iter in range(200):
            if _elapsed() > deadline - 3.0:
                break
            
            # Try to add a random candidate
            rng.shuffle(candidates)
            added = False
            for x in candidates:
                if x in current_S:
                    continue
                # Try adding x
                S_backup = current_S[:]
                sum_used_backup = bytearray(sum_used)
                
                # Check if x can be added
                n = len(current_S)
                new_sums = set()
                new_sums.add(3 * x)
                for y in current_S:
                    new_sums.add(2 * x + y)
                    new_sums.add(x + 2 * y)
                for i in range(n):
                    for j in range(i + 1, n):
                        new_sums.add(x + current_S[i] + current_S[j])
                
                expected_count = 1 + 2 * n + n * (n - 1) // 2
                valid = len(new_sums) == expected_count
                if valid:
                    for s in new_sums:
                        if sum_used[s]:
                            valid = False
                            break
                
                if valid:
                    for s in new_sums:
                        sum_used[s] = 1
                    current_S.append(x)
                    current_S.sort()
                    added = True
                    if len(current_S) > best_size:
                        best_size = len(current_S)
                        best_S = current_S[:]
                    break
            
            if not added:
                # Try a swap: remove one element, add a different one
                if len(current_S) > 0:
                    idx = rng.randint(0, len(current_S) - 1)
                    removed = current_S[idx]
                    sum_used_backup = bytearray(sum_used)
                    
                    # Remove
                    S.remove(removed)
                    # Clear sums involving removed
                    n_after = len(current_S)
                    remove_sums = set()
                    remove_sums.add(3 * removed)
                    for y in current_S:
                        remove_sums.add(2 * removed + y)
                        remove_sums.add(removed + 2 * y)
                    for i in range(n_after):
                        for j in range(i + 1, n_after):
                            remove_sums.add(removed + current_S[i] + current_S[j])
                    for s in remove_sums:
                        if sum_used[s]:
                            sum_used[s] = 0
                    
                    # Try to add a new element
                    rng.shuffle(candidates)
                    found = False
                    for x in candidates:
                        if x == removed or x in current_S:
                            continue
                        n = len(current_S)
                        new_sums = set()
                        new_sums.add(3 * x)
                        for y in current_S:
                            new_sums.add(2 * x + y)
                            new_sums.add(x + 2 * y)
                        for i in range(n):
                            for j in range(i + 1, n):
                                new_sums.add(x + current_S[i] + current_S[j])
                        
                        expected_count = 1 + 2 * n + n * (n - 1) // 2
                        valid = len(new_sums) == expected_count
                        if valid:
                            for s in new_sums:
                                if sum_used[s]:
                                    valid = False
                                    break
                        
                        if valid:
                            for s in new_sums:
                                sum_used[s] = 1
                            current_S.append(x)
                            current_S.sort()
                            found = True
                            if len(current_S) > best_size:
                                best_size = len(current_S)
                                best_S = current_S[:]
                            break
                    
                    if not found:
                        # Revert
                        current_S.append(removed)
                        current_S.sort()
                        sum_used[:] = sum_used_backup
    
    # Final safety: return best found
    if not best_S:
        return list(range(1, 1))  # fallback
    return best_S
