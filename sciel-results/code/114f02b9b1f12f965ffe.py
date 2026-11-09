import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    def get_initial_set():
        """
        Generate a valid B2[2] set using a fast greedy method.
        We iterate 1..1000 and add if possible.
        """
        A = []
        counts = {}
        for x in range(1, 1001):
            # Check if x can be added
            ok = True
            # Check sums x+y for y in A
            # We need to be careful: if we add x, sums x+y (y in A) and 2x are formed.
            # A sum s is invalid if count[s] == 2 already, because adding x would make it 3.
            # Wait, the constraint is "at most two representations".
            # So if count[s] == 2, we cannot add any new representation of s.
            # If count[s] < 2, we can add.
            
            # Sums involving x:
            # 1. 2*x
            # 2. x + y for y in A
            
            # Check 2*x
            if counts.get(2*x, 0) == 2:
                ok = False
            else:
                for y in A:
                    s = x + y
                    if counts.get(s, 0) == 2:
                        ok = False
                        break
            
            if ok:
                A.append(x)
                # Update counts
                counts[2*x] = counts.get(2*x, 0) + 1
                for y in A[:-1]: # A[:-1] is the set before adding x? No, A includes x now.
                    # Actually, we just appended x. So y in A[:-1] are the old elements.
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
        return A, counts

    # Initialize with greedy
    current_set, current_counts = get_initial_set()
    current_size = len(current_set)
    best_set = current_set[:]
    best_size = current_size
    
    # To speed up checks, we maintain a sorted list or just use the list.
    # Since N=1000, list size is ~700. O(N) check is ~700 ops.
    # Swap check: remove a, add b.
    # Cost of check: O(N) for add b, O(N) for remove a?
    # Actually, we can compute the delta in counts.
    
    # For efficiency, let's keep current_set as a list.
    # We will perform Simulated Annealing.
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.999
    
    T = T_start
    
    # Precompute all possible pairs? No, too many.
    
    # Helper to check validity of a set (for debugging or final verification)
    def is_valid(A):
        counts = {}
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                counts[s] = counts.get(s, 0) + 1
                if counts[s] > 2:
                    return False
        return True

    # We will use a "lazy" update strategy for counts.
    # When swapping a (remove) and b (add):
    # 1. Decrement counts for sums involving a (with a and others)
    # 2. Increment counts for sums involving b (with b and others)
    # 3. Check if any count > 2 during increment.
    
    # To make this faster, we can just re-compute validity of the "affected" sums.
    # But checking "can add b" to A\{a} is O(|A|).
    
    def try_swap(a, b, A, counts):
        """
        Attempt to swap a for b in A.
        Returns (success, new_A, new_counts)
        """
        if a == b:
            return False, A, counts
        
        # Remove a from A temporarily? 
        # Better: Check if b can be added to A \ {a}.
        
        # Sums involving a that will be removed:
        # 2a, a+y for y in A, y != a
        
        # Sums involving b that will be added:
        # 2b, b+y for y in A, y != b (and y != a, since a is removed)
        
        # Let's simulate the count changes.
        
        # 1. Decrement counts for a
        # We need to know which sums to decrement.
        # sums_a = {2a} U {a+y | y in A, y != a}
        
        # 2. Increment counts for b
        # sums_b = {2b} U {b+y | y in A, y != b, y != a}
        
        # We can just check if the resulting counts are valid.
        # Instead of updating the dict in place and rolling back, 
        # we can check the "delta".
        
        # A sum s is invalid if count[s] + delta[s] > 2.
        # delta[s] = (1 if s in sums_b else 0) - (1 if s in sums_a else 0)
        
        # This is complex. Let's just do a direct check on the candidate set A' = (A \ {a}) U {b}.
        # Cost: O(|A|^2) is too slow.
        # Cost: O(|A|) for checking b against A\{a}.
        
        # Check if b can be added to A' = A - {a}
        # We need counts for A'.
        # counts[A'] = counts[A] - counts_a + counts_b
        # But we don't have counts[A'] yet.
        
        # Let's just reconstruct the check for b against A without a.
        
        # Optimization: 
        # If we precompute the counts, we can just check:
        # For each sum s that b participates in, what is the count in A \ {a}?
        
        # Let's stick to a simpler approach for now: 
        # 1. Remove a from counts and A.
        # 2. Try to add b.
        # 3. If fail, revert.
        
        # This is O(N) for remove, O(N) for add check. Total O(N).
        
        # Remove a
        A.remove(a)
        # Decrement counts
        # 2a
        counts[2*a] -= 1
        if counts[2*a] == 0: del counts[2*a]
        
        for y in A:
            s = a + y
            counts[s] -= 1
            if counts[s] == 0: del counts[s]
            
        # Now try to add b to A (which is now A_orig \ {a})
        # Check if b can be added
        can_add = True
        if counts.get(2*b, 0) == 2:
            can_add = False
        else:
            for y in A:
                s = b + y
                if counts.get(s, 0) == 2:
                    can_add = False
                    break
        
        if can_add:
            # Add b
            A.append(b)
            counts[2*b] = counts.get(2*b, 0) + 1
            for y in A[:-1]:
                s = b + y
                counts[s] = counts.get(s, 0) + 1
            return True, A, counts
        else:
            # Revert removal of a
            # This is tricky because we modified counts.
            # We should have copied or done it more carefully.
            # Let's just return failure and let the caller handle state?
            # No, we modified A and counts. We must revert.
            
            # Re-add a to counts
            counts[2*a] = counts.get(2*a, 0) + 1
            for y in A:
                s = a + y
                counts[s] = counts.get(s, 0) + 1
            
            # Re-insert a into A. 
            # A is currently A_orig \ {a}.
            # We don't care about order for the set logic, but for reproducibility maybe.
            A.append(a)
            return False, A, counts

    # The above try_swap is O(N) but has overhead.
    # Let's optimize by not using .remove() which is O(N) search + O(N) shift.
    # We can use a set for A? No, we need to iterate.
    # Let's keep A as a list.
    
    # Alternative: 
    # Maintain A as a sorted list? 
    # Or just use a set for membership and a list for iteration?
    # Since N is small (1000), O(N) is fine.
    
    # Let's refine the swap logic to be faster.
    # We will maintain:
    # A: list of elements
    # counts: dict of sum counts
    
    # To swap a for b:
    # 1. Check if b is valid in A \ {a}.
    #    We need to know counts in A \ {a}.
    #    counts_A_minus_a[s] = counts[s] - (1 if s in Sums(a, A) else 0)
    #    Sums(a, A) = {2a} U {a+y | y in A, y!=a}
    
    # Let's precompute the "footprint" of each element?
    # Too much memory/time.
    
    # Let's just use the "remove then add" logic but optimize the revert.
    # Actually, if we fail, we just re-add a.
    
    # One issue: A.remove(a) is O(N).
    # We can find index of a.
    
    def execute_swap(a, b):
        # Returns True if swap was successful and state updated
        nonlocal current_set, current_counts
        
        if a == b:
            return False
        
        # Find index of a
        try:
            idx_a = current_set.index(a)
        except ValueError:
            return False
            
        # Remove a
        # We pop idx_a
        current_set.pop(idx_a)
        
        # Decrement counts for a
        # 2a
        c_2a = current_counts.get(2*a, 0)
        if c_2a == 1:
            del current_counts[2*a]
        else:
            current_counts[2*a] = c_2a - 1
            
        for y in current_set:
            s = a + y
            c_s = current_counts.get(s, 0)
            if c_s == 1:
                del current_counts[s]
            else:
                current_counts[s] = c_s - 1
                
        # Now current_set is A \ {a}
        # Check if b can be added
        ok = True
        if current_counts.get(2*b, 0) == 2:
            ok = False
        else:
            for y in current_set:
                s = b + y
                if current_counts.get(s, 0) == 2:
                    ok = False
                    break
        
        if ok:
            # Add b
            current_set.append(b)
            current_counts[2*b] = current_counts.get(2*b, 0) + 1
            for y in current_set[:-1]:
                s = b + y
                current_counts[s] = current_counts.get(s, 0) + 1
            return True
        else:
            # Revert: re-add a
            # We need to re-insert a into current_set. 
            # Order doesn't matter for correctness, but let's just append.
            current_set.append(a)
            
            # Re-increment counts for a
            current_counts[2*a] = current_counts.get(2*a, 0) + 1
            for y in current_set[:-1]: # current_set now has a at end, so y in others
                s = a + y
                current_counts[s] = current_counts.get(s, 0) + 1
            return False

    # Main SA Loop
    # We also try to ADD new elements if space allows?
    # The greedy already added as many as it could.
    # Swaps allow rearranging to potentially fit MORE elements?
    # No, a swap keeps size constant.
    # To increase size, we need to find a "hole" or rearrange to free up a sum count.
    # But if greedy failed to add x, it's because x+y had count 2.
    # If we swap some elements, maybe the count of x+y drops?
    # Yes.
    
    # So the strategy is:
    # 1. Start with Greedy Set.
    # 2. Try to add any excluded element. If fail, perform swaps to "make room".
    # 3. Actually, just do random swaps and check if size can be increased afterwards?
    #    Or, do swaps that preserve size, and occasionally try to add.
    
    # Let's try:
    # While time:
    #   1. Pick a random element a in A.
    #   2. Pick a random element b in [1, 1000] \ A.
    #   3. Try swap a->b.
    #   4. If successful, check if we can now add some c in [1, 1000] \ A_new.
    #      If yes, add c.
    #   5. Update best if size increased.
    
    # This is a "Swap and Insert" strategy.
    
    # To make it faster, we should prioritize swaps that might free up constraints.
    # But random is fine for now.
    
    # Let's optimize the "Try Add" part.
    # We maintain a list of "excluded" elements.
    # Initially, excluded = [1..1000] \ A.
    
    excluded = [x for x in range(1, 1001) if x not in current_set]
    
    # To check if we can add ANY excluded element, we scan excluded.
    # This is O(N * M) where M is size of A.
    
    def try_add_any():
        nonlocal current_set, current_counts, excluded, best_set, best_size
        added = False
        new_excluded = []
        for x in excluded:
            # Check if x can be added
            ok = True
            if current_counts.get(2*x, 0) == 2:
                ok = False
            else:
                for y in current_set:
                    s = x + y
                    if current_counts.get(s, 0) == 2:
                        ok = False
                        break
            if ok:
                # Add x
                current_set.append(x)
                current_counts[2*x] = current_counts.get(2*x, 0) + 1
                for y in current_set[:-1]:
                    s = x + y
                    current_counts[s] = current_counts.get(s, 0) + 1
                added = True
                # x is no longer excluded
            else:
                new_excluded.append(x)
        if added:
            excluded = new_excluded
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
        return added

    # Initial try_add_any (should be empty since greedy was maximal, but just in case)
    try_add_any()
    
    T = T_start
    iterations = 0
    
    while time.time() < deadline:
        # Pick random a from current_set
        if not current_set:
            break
        a = random.choice(current_set)
        
        # Pick random b from excluded? Or from 1..1000?
        # If b is in current_set, it's a swap within set (reordering? no, set is unique).
        # So b must be in excluded.
        if not excluded:
            # No elements to swap in. We are stuck?
            # If excluded is empty, we have all 1000 elements. Max size.
            break
        
        b = random.choice(excluded)
        
        # Try swap a -> b
        # We need to update excluded list if swap succeeds.
        # If a->b succeeds:
        #   a becomes excluded.
        #   b becomes in set.
        
        # Let's do the swap logic carefully with excluded list.
        
        # 1. Remove a from current_set and counts.
        # 2. Check if b can be added.
        # 3. If yes, add b. Update excluded: remove b, add a.
        # 4. If no, revert.
        
        # To avoid the complex revert of counts, let's just use the execute_swap logic 
        # but we need to handle the excluded list.
        
        # Actually, let's just use a simpler "Recompute" approach for the swap check?
        # No, too slow.
        
        # Let's use the execute_swap logic but wrap it to handle excluded.
        
        # We need to know the index of a in current_set for execute_swap.
        # execute_swap modifies current_set and current_counts.
        
        # Let's modify execute_swap to return the new state or just mutate.
        # It already mutates.
        
        # Problem: execute_swap doesn't update excluded.
        # We can update excluded manually.
        
        # Before swap:
        # a in current_set, b in excluded.
        
        # Call execute_swap(a, b)
        # If True:
        #   current_set has b, not a.
        #   Update excluded: remove b, add a.
        
        # If False:
        #   current_set has a, not b.
        #   Excluded unchanged.
        
        success = execute_swap(a, b)
        
        if success:
            # Update excluded
            # Remove b from excluded
            # Add a to excluded
            # This is O(N) for list ops.
            # We can use a set for excluded for O(1) lookup, but we need to iterate?
            # We only pick random b. So a set is fine for membership, 
            # but random.choice needs a list or we can use random.sample?
            # Let's keep excluded as a list for random.choice.
            # O(N) removal is acceptable if N=1000 and iterations are ~10k.
            
            try:
                excluded.remove(b)
            except ValueError:
                pass # Should not happen
            excluded.append(a)
            
            # Try to add more elements
            if try_add_any():
                pass # Updated best_set inside
            
            # Update T
            T *= cooling_rate
            if T < T_end:
                T = T_end
        else:
            # Accept with probability exp(-delta / T)?
            # Delta is 0 for size-preserving swaps.
            # But we want to explore.
            # If swap fails, we just stay.
            # Should we accept "bad" swaps? 
            # In this problem, "bad" means invalid. We can't have invalid sets.
            # So we only accept valid swaps.
            # To escape local minima, we might need to allow temporary "worsening"?
            # But there is no cost function other than size.
            # All valid sets have cost 0 (or size).
            # So we can only move to other valid sets.
            # This is a random walk on the graph of valid sets.
            # If the graph is connected, we will eventually find the max size.
            # If not, we are stuck.
            
            # To help, we can bias the choice of b.
            # But random is okay.
            
            # Cooling
            T *= cooling_rate
            if T < T_end:
                T = T_end

        iterations += 1
        if iterations % 1000 == 0:
            # Optional: check time
            pass

    return best_set

# Quick self-test
if __name__ == "__main__":
    s = construct()
    print(len(s))
    # Verify
    counts = {}
    for i in range(len(s)):
        for j in range(i, len(s)):
            sum_val = s[i] + s[j]
            counts[sum_val] = counts.get(sum_val, 0) + 1
            if counts[sum_val] > 2:
                print("Invalid!")
                break
