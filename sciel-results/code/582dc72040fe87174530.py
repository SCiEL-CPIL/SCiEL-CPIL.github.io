import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill

    universe = list(range(1, 23))
    
    # Precompute scaled reciprocals using a common denominator
    # LCM of 1..22 is 232792560
    # We use this as a fixed base to convert 1/i into integer 232792560 // i
    # This is exact since i divides 232792560 for all i in 1..22
    BASE = 232792560
    scaled = {i: BASE // i for i in range(1, 23)}

    best_set = []
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_set = list(S)
            best_size = len(S)

    def is_dissociated(S):
        """
        Check if all subset sums of reciprocals of S are distinct.
        Uses integer arithmetic with scaling factor BASE.
        S is a list of integers from 1 to 22.
        """
        if not S:
            return True
        n = len(S)
        # We store all subset sums as integers
        # sums[0] = 0 (empty set)
        # sums[k] for k>0 represents sum of reciprocals of some subset
        # We use a list and check for duplicates at the end, or use a set incrementally
        # For n up to ~14, 2^14 = 16384, which is fast.
        # For n=15, 32768. n=16, 65536. n=17, 131072.
        
        sums = [0]
        for x in S:
            val = scaled[x]
            current_len = len(sums)
            for i in range(current_len):
                sums.append(sums[i] + val)
        
        # Check for duplicates
        # Converting to set is O(2^n) but with small integers it's very fast
        return len(sums) == len(set(sums))

    def find_collision(S):
        """
        Find a collision in subset sums. Returns two masks that produce the same sum.
        Returns (mask1, mask2) or None.
        """
        n = len(S)
        sums = [0]
        masks = [0]
        
        for i, x in enumerate(S):
            val = scaled[x]
            current_len = len(sums)
            for j in range(current_len):
                new_sum = sums[j] + val
                new_mask = masks[j] | (1 << i)
                sums.append(new_sum)
                masks.append(new_mask)
                
        seen = {}
        for i, s in enumerate(sums):
            if s in seen:
                return (masks[i], seen[s])
            else:
                seen[s] = masks[i]
        return None

    # Strategy 1: Greedy from large to small
    current_S = []
    for x in sorted(universe, reverse=True):
        if time.time() > deadline:
            break
        new_S = current_S + [x]
        if is_dissociated(new_S):
            current_S = new_S
    update_best(current_S)

    # Strategy 2: Randomized Greedy with Restarts
    random.seed(42)
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        perm = universe.copy()
        random.shuffle(perm)
        
        current_S = []
        for x in perm:
            if time.time() > deadline:
                break
            new_S = current_S + [x]
            if is_dissociated(new_S):
                current_S = new_S
        
        update_best(current_S)
        
        # Local Search: Try to add any remaining elements
        rem = [x for x in universe if x not in current_S]
        rem.sort(reverse=True)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in rem:
                if x in current_S:
                    continue
                new_S = current_S + [x]
                if is_dissociated(new_S):
                    current_S = new_S
                    rem.remove(x)
                    update_best(current_S)
                    improved = True
                    break # Restart the loop since rem changed
            
            if not improved:
                # Try swaps: remove one, add one
                # This can help escape local optima
                rem2 = [x for x in universe if x not in current_S]
                for a in current_S:
                    if time.time() > deadline:
                        break
                    for b in rem2:
                        new_S = [x for x in current_S if x != a] + [b]
                        if is_dissociated(new_S):
                            current_S = new_S
                            update_best(current_S)
                            improved = True
                            break
                    if improved:
                        break

    # Strategy 3: Iterative improvement on the best set found
    # Try to add elements to best_set
    current_S = list(best_set)
    rem = [x for x in universe if x not in current_S]
    rem.sort(reverse=True)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in rem:
            if x in current_S:
                continue
            new_S = current_S + [x]
            if is_dissociated(new_S):
                current_S = new_S
                rem.remove(x)
                update_best(current_S)
                improved = True
                break

    # Strategy 4: More aggressive swaps on the best set
    current_S = list(best_set)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rem = [x for x in universe if x not in current_S]
        for a in current_S:
            if time.time() > deadline:
                break
            for b in rem:
                new_S = [x for x in current_S if x != a] + [b]
                if is_dissociated(new_S):
                    current_S = new_S
                    update_best(current_S)
                    improved = True
                    # After a swap, try to add more
                    rem2 = [x for x in universe if x not in current_S]
                    for c in rem2:
                        if time.time() > deadline:
                            break
                        new_S2 = current_S + [c]
                        if is_dissociated(new_S2):
                            current_S = new_S2
                            update_best(current_S)
                    break
            if improved:
                break

    if not best_set:
        best_set = [22]

    return best_set

if __name__ == "__main__":
    print(construct())
