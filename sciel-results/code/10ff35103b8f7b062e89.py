import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Precompute reciprocals as Fractions
    reciprocals = {i: Fraction(1, i) for i in range(1, 23)}
    universe = list(range(1, 23))
    
    def is_dissociated(S):
        """
        Check if all subset sums of reciprocals of S are distinct.
        S is a list of integers.
        """
        if not S:
            return True
        
        n = len(S)
        # If n is large, 2^n is large.
        # For n=20, 2^20 = 1,048,576.
        # Generating 1M Fractions might be slow.
        # Let's optimize.
        
        sums = [Fraction(0, 1)]
        for x in S:
            val = reciprocals[x]
            # Extend sums list
            # Optimization: use list comprehension or loop
            # new_sums = [s + val for s in sums]
            # sums.extend(new_sums)
            
            # To avoid creating a new list every time, we can do:
            current_len = len(sums)
            for i in range(current_len):
                sums.append(sums[i] + val)
                
        # Check for duplicates
        # Using a set is O(2^n).
        # If there are duplicates, len(set) < len(sums)
        if len(sums) != len(set(sums)):
            return False
        return True

    def find_collision(S):
        """
        Find a non-trivial linear combination of reciprocals of S that sums to 0.
        Returns a list of (coefficient, element) pairs, or None if dissociated.
        Coefficients are in {-1, 0, 1}.
        """
        n = len(S)
        sums = [Fraction(0, 1)]
        masks = [0]
        
        for i, x in enumerate(S):
            val = reciprocals[x]
            current_len = len(sums)
            for j in range(current_len):
                new_sum = sums[j] + val
                sums.append(new_sum)
                masks.append(masks[j] | (1 << i))
                
        # Map sum to mask
        seen = {}
        for i, s in enumerate(sums):
            if s in seen:
                # Collision found between masks[i] and seen[s]
                # The difference is the collision
                # We need to extract the coefficients
                # c_k = 1 if bit k in mask1 XOR mask2?
                # Actually, sum(mask1) - sum(mask2) = 0
                # So coefficients are 1 for bits in mask1 not in mask2, -1 for bits in mask2 not in mask1.
                mask1 = masks[i]
                mask2 = seen[s]
                if mask1 == mask2:
                    continue # Should not happen if sums are unique for masks
                # Construct the collision vector
                # We just need to know which elements are involved to potentially remove one.
                # For now, just return the elements involved.
                involved = []
                for k in range(n):
                    bit1 = (mask1 >> k) & 1
                    bit2 = (mask2 >> k) & 1
                    if bit1 != bit2:
                        involved.append(S[k])
                return involved
            else:
                seen[s] = masks[i]
        return None

    best_set = []
    best_size = 0
    
    # Strategy 1: Greedy from large to small
    # This is usually very effective for dissociated sets.
    current_S = []
    for x in sorted(universe, reverse=True):
        if time.time() > deadline:
            break
        new_S = current_S + [x]
        if is_dissociated(new_S):
            current_S = new_S
            
    if len(current_S) > best_size:
        best_set = current_S
        best_size = len(current_S)
        
    # Strategy 2: Try to extend the greedy set
    # The greedy set might have "gaps" where smaller numbers could fit if we removed some larger ones?
    # Or just add remaining numbers?
    remaining = [x for x in universe if x not in current_S]
    # Try adding remaining numbers in random order or sorted order
    random.seed(42)
    for _ in range(10): # A few attempts
        if time.time() > deadline:
            break
        temp_S = list(best_set)
        rem = [x for x in universe if x not in temp_S]
        random.shuffle(rem)
        for x in rem:
            if time.time() > deadline:
                break
            new_S = temp_S + [x]
            if is_dissociated(new_S):
                temp_S = new_S
        if len(temp_S) > best_size:
            best_set = temp_S
            best_size = len(best_S)
            
    # Strategy 3: Randomized Greedy with Restarts
    # Start with a random subset of large numbers, then greedily add others.
    for i in range(20): # More restarts
        if time.time() > deadline:
            break
        
        # Random initialization: Pick a random number of large elements
        # Or just shuffle the universe and do greedy?
        # Shuffling and doing greedy is effectively a randomized greedy.
        perm = universe.copy()
        random.shuffle(perm)
        
        current_S = []
        for x in perm:
            if time.time() > deadline:
                break
            new_S = current_S + [x]
            if is_dissociated(new_S):
                current_S = new_S
                
        if len(current_S) > best_size:
            best_set = current_S
            best_size = len(current_S)
            
        # Local search: Try to add any remaining
        rem = [x for x in universe if x not in current_S]
        rem.sort(reverse=True) # Large first is safer
        for x in rem:
            if time.time() > deadline:
                break
            new_S = current_S + [x]
            if is_dissociated(new_S):
                current_S = new_S
                
        if len(current_S) > best_size:
            best_set = current_S
            best_size = len(current_S)

    return best_set

if __name__ == "__main__":
    print(construct())
