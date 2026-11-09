import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    N = 1000
    max_sum = 2 * N
    
    def is_valid(S):
        if not S:
            return True
        S_arr = np.array(S, dtype=np.int32)
        # Compute all pairwise sums a+b with a<=b
        # Using broadcasting for clarity, though O(n^2) memory
        # For n~400, 400x400 = 160k ints, fine.
        sums = S_arr[:, None] + S_arr[None, :]
        # We only care about a <= b, which is upper triangle including diagonal
        # np.triu_indices
        iu = np.triu_indices(len(S_arr))
        relevant_sums = sums[iu]
        
        # Count occurrences of each sum
        # np.unique with return_counts
        _, counts = np.unique(relevant_sums, return_counts=True)
        return np.all(counts <= 2)

    def repair(S):
        """
        Greedy repair: remove elements that cause the most conflicts.
        Returns a valid subset of S.
        """
        S = list(S)
        if not S:
            return []
        
        while not is_valid(S):
            # Find the element whose removal reduces the "excess" most, or just pick one involved in a conflict
            # To be fast, we can just pick a random element involved in a conflict
            # Or, simpler: iterate and remove the first element that is part of a >2 sum
            
            # Efficient check: find a sum with count > 2
            S_arr = np.array(S, dtype=np.int32)
            sums = S_arr[:, None] + S_arr[None, :]
            iu = np.triu_indices(len(S_arr))
            relevant_sums = sums[iu]
            vals, counts = np.unique(relevant_sums, return_counts=True)
            
            # Find sums with count > 2
            bad_sums = vals[counts > 2]
            if len(bad_sums) == 0:
                break # Should be valid now
            
            # Pick a random bad sum
            bad_sum = random.choice(bad_sums)
            
            # Find all elements involved in this sum
            # a + b = bad_sum
            # We need to remove enough elements so that remaining pairs summing to bad_sum <= 2
            
            # Identify indices i, j such that S[i] + S[j] == bad_sum and i <= j
            # We can just iterate through S
            involved = []
            S_set = set(S)
            for idx, val in enumerate(S):
                # Check if val pairs with any other element to make bad_sum
                # To avoid double counting and self-pairs, we just collect candidates
                # Actually, we just need to pick one element from the set of pairs causing the excess
                # Let's find all pairs
                pairs = []
                for j in range(idx, len(S)):
                    if val + S[j] == bad_sum:
                        pairs.append((idx, j))
                
                if pairs:
                    # This element is involved
                    involved.append(idx)
                    # We don't need all, just one to break the chain? 
                    # Removing one element from a pair reduces count by 1.
                    # If count is 3, removing one element from one pair makes it 2. Valid.
                    # So we can just remove S[idx]
                    break
            
            if involved:
                # Remove the first found involved element
                S.pop(involved[0])
            else:
                # Should not happen, but safety
                break
        
        return S

    def expand(S):
        """
        Try to add elements from 1..1000 not in S.
        """
        S_set = set(S)
        S_list = list(S)
        
        # Precompute current sum counts for fast checking?
        # Since S is valid, counts are 0, 1, or 2.
        # We can maintain a dict of counts.
        counts = {}
        for i, x in enumerate(S_list):
            for y in S_list[i:]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            # Candidates not in S
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            if not candidates:
                break
            
            # Shuffle for randomness
            random.shuffle(candidates)
            
            for x in candidates:
                # Check if x can be added
                can_add = True
                # Check sums with existing elements
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
                    # Add x
                    S_list.append(x)
                    S_set.add(x)
                    for y in S_list[:-1]:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    added_any = True
                    break # Restart loop as set changed
        
        return S_list

    def greedy_b2_base():
        """
        Construct a B2[1] set (Sidon sequence) greedily.
        This ensures no sum has >1 representation initially.
        Then we relax to B2[2].
        """
        S = []
        counts = {} # sum -> count
        
        # Sort candidates to encourage spreading out? Or random?
        # Random usually works well for Sidon
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            can_add = True
            for y in S:
                s = x + y
                if counts.get(s, 0) >= 1:
                    can_add = False
                    break
            if can_add:
                s = 2 * x
                if counts.get(s, 0) >= 1:
                    can_add = False
            
            if can_add:
                S.append(x)
                for y in S[:-1]:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
        
        return S

    best_size = 0
    best_set = []
    
    # Strategy:
    # 1. Start with a dense random valid set (greedy B2[2] or B2[1] base)
    # 2. Expand it
    # 3. If stuck, try to "shake" by removing some random elements and re-expanding
    # 4. Use repair if we allow invalid intermediate states (but we stick to valid for simplicity first)
    
    # Let's try multiple restarts with different strategies
    
    for _ in range(50):
        if time.time() > deadline - 5:
            break
            
        # Variant 1: Greedy B2[2] directly
        S1 = []
        counts1 = {}
        candidates1 = list(range(1, N + 1))
        random.shuffle(candidates1)
        for x in candidates1:
            can_add = True
            for y in S1:
                s = x + y
                if counts1.get(s, 0) >= 2:
                    can_add = False
                    break
            if can_add:
                s = 2 * x
                if counts1.get(s, 0) >= 2:
                    can_add = False
            if can_add:
                S1.append(x)
                for y in S1[:-1]:
                    s = x + y
                    counts1[s] = counts1.get(s, 0) + 1
                counts1[2 * x] = counts1.get(2 * x, 0) + 1
        
        # Expand S1
        S1 = expand(S1)
        
        if len(S1) > best_size:
            best_size = len(S1)
            best_set = S1[:]
            
        # Variant 2: Start with Sidon (B2[1]), then expand to B2[2]
        # Sidon sets are smaller but "cleaner", might allow better packing when relaxed
        S2 = greedy_b2_base()
        S2 = expand(S2)
        
        if len(S2) > best_size:
            best_size = len(S2)
            best_set = S2[:]
            
        # Variant 3: Random valid set generation via rejection sampling (slower but different topology)
        # Pick random subset, repair, expand
        if time.time() < deadline - 10:
             # Generate a random set of size ~ best_size
             target_size = max(100, best_size + 10)
             S3 = random.sample(range(1, N + 1), min(target_size, N))
             S3 = repair(S3)
             S3 = expand(S3)
             
             if len(S3) > best_size:
                 best_size = len(S3)
                 best_set = S3[:]

    # Final local improvement on best_set
    # Try to swap elements to free up space
    if time.time() < deadline - 5:
        S_best = best_set[:]
        S_set = set(S_best)
        
        # Calculate counts
        counts = {}
        for i, x in enumerate(S_best):
            for y in S_best[i:]:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Try to remove a random element and add a better one?
            # Or just try to add more?
            # We already expanded. 
            # Let's try a simple swap: remove x, add y if it allows adding z later?
            # Too complex. Just try random swaps to see if we can increase size.
            
            if not S_best:
                break
                
            x_to_remove = random.choice(S_best)
            S_best.remove(x_to_remove)
            S_set.remove(x_to_remove)
            
            # Update counts
            for y in S_best:
                s = x_to_remove + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
            s = 2 * x_to_remove
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
            
            # Try to add a new element
            candidates = [x for x in range(1, N + 1) if x not in S_set]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                can_add = True
                for y in S_best:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        can_add = False
                        break
                if can_add:
                    s = 2 * x
                    if counts.get(s, 0) >= 2:
                        can_add = False
                
                if can_add:
                    S_best.append(x)
                    S_set.add(x)
                    for y in S_best[:-1]:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    added = True
                    improved = True
                    break
            
            if not added:
                # Revert
                S_best.append(x_to_remove)
                S_set.add(x_to_remove)
                for y in S_best[:-1]:
                    s = x_to_remove + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x_to_remove] = counts.get(2 * x_to_remove, 0) + 1
                
        if len(S_best) > best_size:
            best_size = len(S_best)
            best_set = S_best[:]

    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
