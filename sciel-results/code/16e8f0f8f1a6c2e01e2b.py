import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s buffer for 90s limit
    
    # Initial best known valid set (powers of 2 up to 128, 256 doesn't fit if 128 is there? 
    # 1, 2, 4, 8, 16, 32, 64, 128 is size 8. 
    # Sums for k=0: {0}
    # k=1: {1,2,4,8,16,32,64,128}
    # k=2: {3,5,9,...} min sum 1+2=3. Max sum for k=1 is 128. 3 <= 128, so collision?
    # Wait, the condition is: "Whenever two subsets have different cardinalities their sums must differ."
    # This means the set of sums for cardinality k, denoted Sum(k), must be disjoint from Sum(j) for k != j.
    # For powers of 2:
    # Sum(0) = {0}
    # Sum(1) = {1, 2, 4, ..., 128}
    # Sum(2) = {3, 5, 6, ..., 190} (e.g., 1+2=3, 1+4=5, 2+4=6, ..., 64+128=190)
    # Is Sum(1) disjoint from Sum(2)?
    # Max of Sum(1) is 128. Min of Sum(2) is 3. They overlap?
    # 128 is in Sum(1). Is 128 in Sum(2)? 128 = 64 + 64 (not allowed, distinct elements). 
    # 128 = 128 + 0 (0 not in set). 
    # Actually, for powers of 2, every sum is unique to its binary representation.
    # The number of 1s in the binary representation determines the cardinality.
    # So Sum(k) consists of numbers with exactly k bits set.
    # Numbers with different numbers of bits set are distinct.
    # So {1, 2, 4, 8, 16, 32, 64, 128} is valid. Size 8.
    # Can we add 256?
    # Sum(1) gets 256. Sum(2) gets 256+1=257, etc.
    # Sum(1) = {1..128, 256}. Sum(2) = {3..190, 257..384}.
    # Disjoint? Yes.
    # So {1, 2, 4, 8, 16, 32, 64, 128, 256} is valid. Size 9.
    
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)
    
    def compute_sums_bitset(elements):
        """
        Returns a list of integers (bitsets) S[0..n] where S[k] has bit s set
        if sum s is achievable by a subset of size k.
        """
        n = len(elements)
        if n == 0:
            return [1]  # S[0] = 1 (bit 0 set)
        
        # Max possible sum is sum(elements)
        # We use Python big integers as bitsets.
        S = [0] * (n + 1)
        S[0] = 1  # bit 0 is set
        
        for v in elements:
            # Update in reverse order to avoid using v multiple times
            for k in range(n, 0, -1):
                S[k] |= (S[k-1] << v)
        
        return S

    def count_collisions(S):
        """
        Counts the number of cross-cardinality sum collisions.
        """
        n = len(S) - 1
        total = 0
        # Only check pairs (i, j) with i < j
        for i in range(n + 1):
            if S[i] == 0:
                continue
            for j in range(i + 1, n + 1):
                if S[j] == 0:
                    continue
                inter = S[i] & S[j]
                if inter:
                    total += bin(inter).count('1')
        return total

    def is_valid(S):
        return count_collisions(S) == 0

    # Strategy 1: Greedy Construction
    # Try to build a set by adding elements that don't cause collisions.
    # We'll try a few different orderings.
    
    def greedy_construct(order):
        current_set = []
        S = [1]  # S[0]
        
        for v in order:
            # Check if adding v creates a collision
            # New S' where S'[k] = S[k] | (S[k-1] << v)
            new_S = [0] * (len(current_set) + 2)
            new_S[0] = 1
            for k in range(1, len(current_set) + 1):
                new_S[k] = S[k] | (S[k-1] << v)
            
            # Check validity of new_S
            # We can just check collisions involving new sums? 
            # Or just run count_collisions on new_S.
            # Since n is small, count_collisions is fast.
            if is_valid(new_S):
                current_set.append(v)
                S = new_S
        
        return current_set

    # Try greedy with different orders
    random.seed(42)
    
    # Order 1: Descending
    greedy_set = greedy_construct(list(range(256, 0, -1)))
    if len(greedy_set) > best_size:
        best_set = greedy_set
        best_size = len(best_set)
        
    # Order 2: Random
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        random_order = list(range(1, 257))
        random.shuffle(random_order)
        greedy_set = greedy_construct(random_order)
        if len(greedy_set) > best_size:
            best_set = greedy_set
            best_size = len(best_set)

    # Strategy 2: Conflict-Driven Removal from a Dense Set
    # Start with a large random set, remove elements that cause the most conflicts.
    
    def conflict_removal(initial_set, time_limit):
        current_set = list(initial_set)
        if not current_set:
            return current_set, 0
            
        S = compute_sums_bitset(current_set)
        collisions = count_collisions(S)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_idx = -1
            best_collisions = collisions
            best_new_set = None
            best_new_S = None
            
            n = len(current_set)
            
            # To speed up, we can estimate which elements are "bad".
            # But calculating exact reduction requires re-computing S for each removal.
            # Optimization: Only check elements that are part of a collision?
            # Hard to determine easily without enumerating subsets.
            # For n < 40, full check is O(n^2 * bitset_ops). Bitset ops are fast.
            
            for idx in range(n):
                v = current_set[idx]
                new_set = current_set[:idx] + current_set[idx+1:]
                new_S = compute_sums_bitset(new_set)
                new_collisions = count_collisions(new_S)
                
                if new_collisions < best_collisions:
                    best_collisions = new_collisions
                    best_idx = idx
                    best_new_set = new_set
                    best_new_S = new_S
                    improved = True
                    
            if improved:
                current_set = best_new_set
                S = best_new_S
                collisions = best_collisions
            else:
                # No single removal improves collisions. 
                # Try a random swap or just break?
                # If collisions > 0, we are stuck.
                break
                
        return current_set, collisions

    # Run conflict removal on several random dense sets
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
            
        # Start with a random set of size 30-50
        size = random.randint(30, 50)
        random_set = random.sample(range(1, 257), size)
        
        result_set, final_collisions = conflict_removal(random_set, deadline - 10)
        
        if final_collisions == 0:
            if len(result_set) > best_size:
                best_set = result_set
                best_size = len(best_set)
        else:
            # Even if not valid, maybe we can repair it?
            # Or maybe it's close. Let's try to add elements to the best valid set found so far.
            pass

    # Strategy 3: Local Search / Expansion
    # Take the best valid set and try to add more elements.
    
    def try_expand(valid_set):
        current_set = list(valid_set)
        S = compute_sums_bitset(current_set)
        
        # Candidates to add
        current_elements = set(current_set)
        candidates = [v for v in range(1, 257) if v not in current_elements]
        
        # Try adding candidates one by one, in a random order
        random.shuffle(candidates)
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i, v in enumerate(candidates):
                # Check if adding v is valid
                new_S = [0] * (len(current_set) + 2)
                new_S[0] = 1
                for k in range(1, len(current_set) + 1):
                    new_S[k] = S[k] | (S[k-1] << v)
                
                if is_valid(new_S):
                    # Add it
                    current_set.append(v)
                    S = new_S
                    candidates.pop(i)
                    changed = True
                    break # Restart loop with new state
                    
        return current_set

    # Try expanding the best set found
    if time.time() < deadline - 5:
        expanded_set = try_expand(best_set)
        if len(expanded_set) > best_size:
            best_set = expanded_set
            best_size = len(best_set)

    # Final Check and Return
    # Ensure types are correct
    return [int(x) for x in best_set]

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Best size: {len(res)}")
    print(res)
