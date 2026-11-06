import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 4096
    max_sum = 3 * n
    best_set = []
    
    used_sums = bytearray(max_sum + 1)
    
    def reset_sums():
        for i in range(len(used_sums)):
            used_sums[i] = 0
            
    def get_new_sums_indices(x, S):
        """Generate indices of new sums involving x and elements of S.
        Returns None if collision, else list of new sum indices."""
        new_sums = []
        s = 3 * x
        if used_sums[s]:
            return None
        new_sums.append(s)
        
        for y in S:
            s = 2 * x + y
            if used_sums[s]:
                return None
            new_sums.append(s)
            
        k = len(S)
        for i in range(k):
            for j in range(i, k):
                s = x + S[i] + S[j]
                if used_sums[s]:
                    return None
                new_sums.append(s)
                
        return new_sums

    def mark_sums(sums):
        for s in sums:
            used_sums[s] = 1

    def unmark_sums(sums):
        for s in sums:
            used_sums[s] = 0

    # Strategy 1: Fast Greedy to get a good initial set
    def greedy_build(order):
        reset_sums()
        S = []
        for x in order:
            new_sums = get_new_sums_indices(x, S)
            if new_sums is not None:
                S.append(x)
                mark_sums(new_sums)
        return S

    # Initialize with a sorted greedy pass
    S1 = greedy_build(range(1, n + 1))
    if len(S1) > len(best_set):
        best_set = S1

    def try_local_improvement(S):
        """
        Try to increase the size of S by swapping out one element for a new one.
        Returns a new set if an improvement is found, else None.
        """
        if time.time() > deadline:
            return None
        
        # Rebuild used_sums for S
        reset_sums()
        # We need to mark all sums for S
        # To do this efficiently, we can just re-run the marking logic or use a helper.
        # Since S is small (~20-30), O(k^2) is fine.
        for i in range(len(S)):
            for j in range(i, len(S)):
                for k in range(j, len(S)):
                    s = S[i] + S[j] + S[k]
                    used_sums[s] = 1
        
        # Create a set of elements in S for quick lookup
        S_set = set(S)
        
        # Try swapping each element x in S with a candidate y not in S
        # To make this fast, we only try a subset of candidates or check conditions quickly.
        # A swap is valid if:
        # 1. All sums involving y are unique (not in used_sums after removing x's sums)
        # 2. y is not in S
        
        # Optimization: Iterate through S elements. For each, remove its sums, 
        # then try to add a new element.
        # Removing sums is O(k^2). Adding is O(k^2).
        # Total O(k^4) per improvement attempt? No, O(k^3) per swap check.
        # With k=25, k^3 = 15625. 25 swaps * 15625 = ~400k ops. Very fast.
        
        for idx, x in enumerate(S):
            # 1. Remove x from S and unmark its sums
            # We need to be careful: unmarking sums that might be shared?
            # No, sums are unique. So we can just unmark all sums involving x.
            # But wait, if we unmark sums involving x, we might unmark sums that 
            # are also formed by other triples? No, B_3[1] means ALL triples are unique.
            # So each sum belongs to exactly one triple.
            
            # List of sums to unmark
            sums_to_unmark = []
            # Triple (x,x,x)
            sums_to_unmark.append(3*x)
            # Triples (x,x,y) for y in S, y != x
            for y in S:
                if y != x:
                    sums_to_unmark.append(2*x + y)
            # Triples (x,y,z) for y,z in S, y<z, y!=x, z!=x
            # We can iterate over S
            for i in range(len(S)):
                if S[i] == x: continue
                for j in range(i+1, len(S)):
                    if S[j] == x: continue
                    # Ensure y < z to avoid duplicates? 
                    # The sum x+y+z is the same regardless of order.
                    # We just need to unmark the sum value.
                    sums_to_unmark.append(x + S[i] + S[j])
            
            # Unmark
            for s in sums_to_unmark:
                if used_sums[s]:
                    used_sums[s] = 0
            
            # Now try to find a candidate y to add
            # We can try candidates from 1..n not in S
            # To be fast, we can try a random subset or just check if there's *any* valid y.
            # Let's try a few random candidates from the full range [1, n]
            
            # Optimization: We can pre-compute a list of "available" numbers?
            # Or just pick randoms.
            
            # Let's try 100 random candidates
            found = False
            for _ in range(50):
                if time.time() > deadline:
                    break
                y = random.randint(1, n)
                if y in S_set:
                    continue
                
                # Check if y can be added
                # 1. 3y not in used_sums
                if used_sums[3*y]:
                    continue
                # 2. 2y + z not in used_sums for z in S
                ok = True
                for z in S:
                    if used_sums[2*y + z]:
                        ok = False
                        break
                if not ok:
                    continue
                # 3. y + z1 + z2 not in used_sums
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        if used_sums[y + S[i] + S[j]]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                
                # Valid!
                # Mark sums for y
                used_sums[3*y] = 1
                for z in S:
                    used_sums[2*y + z] = 1
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_sums[y + S[i] + S[j]] = 1
                
                # Add y to S
                new_S = S[:]
                new_S.append(y)
                new_S.sort()
                
                # We found an improvement. 
                # Note: We modified used_sums. We should return the new set.
                # The caller will handle resetting used_sums if needed.
                return new_S

            # If no y found, restore x's sums? 
            # Actually, we just modified used_sums by unmarking x.
            # If we don't add y, we must re-mark x's sums to restore state.
            # But since we return None if no improvement, and the caller 
            # might want to try other swaps, we should restore state.
            # However, for simplicity in this "try once" function, 
            # let's just re-mark x if we didn't find a swap.
            
            for s in sums_to_unmark:
                used_sums[s] = 1

        return None

    # Main search loop
    # 1. Run many randomized greedy passes
    # 2. Try to improve the best set found via local swaps
    
    num_greedy_attempts = 5000
    current_best_size = len(best_set)
    
    for seed in range(num_greedy_attempts):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = greedy_build(candidates)
        
        if len(S) > current_best_size:
            current_best_size = len(S)
            best_set = S
            
            # Try to improve this new best
            improved = try_local_improvement(S)
            if improved and len(improved) > current_best_size:
                current_best_size = len(improved)
                best_set = improved
                # Try again on the improved set?
                improved2 = try_local_improvement(best_set)
                if improved2 and len(improved2) > current_best_size:
                    current_best_size = len(improved2)
                    best_set = improved2

        # Also periodically try to improve the global best
        if seed % 100 == 0 and len(best_set) > 0:
            improved = try_local_improvement(best_set)
            if improved and len(improved) > current_best_size:
                current_best_size = len(improved)
                best_set = improved

    return best_set
