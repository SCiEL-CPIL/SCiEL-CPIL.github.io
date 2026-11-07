import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # We want to maximize k.
    # Theoretical max is likely around 15-20.
    # Current best is 18 (from ILP).
    # Let's try to beat 18.
    
    def is_valid(s):
        """
        Check if subset sums for different cardinalities are disjoint.
        Returns True if valid.
        Uses bitsets for efficiency.
        """
        n = len(s)
        # dp[c] is a set of sums for cardinality c
        # Since sums are <= 256*60 = 15360, we can use Python ints as bitsets
        # bit i set means sum i is possible.
        
        # dp[c] is an integer where bit i is 1 if sum i is achievable with c elements.
        dp = [0] * (n + 1)
        dp[0] = 1  # sum 0 is achievable with 0 elements
        
        for x in s:
            # Update in reverse to avoid using the same element twice
            # Note: x is at most 256.
            for c in range(n - 1, -1, -1):
                # Shift dp[c] by x bits
                # dp[c+1] |= (dp[c] << x)
                # We only care about bits up to some limit? 
                # Max sum is sum(s).
                # But we need to check disjointness at the end.
                # To save time, we can mask to a reasonable max sum?
                # No, we need exact sums.
                # However, if sums exceed 2^20 or so, Python ints handle it fine.
                dp[c + 1] |= (dp[c] << x)
        
        # Check disjointness
        # We need to ensure that no sum value appears in dp[c1] and dp[c2] for c1 != c2.
        # We can do this by checking bitwise AND of all pairs? 
        # Or simpler: maintain a 'seen' bitset.
        seen = 0
        for c in range(n + 1):
            if dp[c] & seen:
                return False
            seen |= dp[c]
        return True

    def random_valid_set(k, max_val=256):
        """
        Generate a random set of size k that is likely to be valid or close.
        Strategy: Pick numbers with large gaps to start, then refine.
        Or just pick random distinct ints and check.
        """
        # Pick k distinct integers from 1..max_val
        # To make it easier, we can bias towards larger numbers or spread out.
        # A good heuristic: pick a base step size.
        # If we pick 1, 1+d, 1+2d, ..., 1+(k-1)d, sums are roughly arithmetic.
        # But random is better for escaping local minima.
        
        # Let's try a mix: start with a structured set, then randomize.
        # Structured: powers of 2 scaled? No, max 256.
        # Let's just pick random distinct integers.
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        return sorted(candidates[:k])

    def local_search(s, max_iters=2000, time_limit=5.0):
        """
        Perform local search to fix invalidities or find a valid set.
        Move: Pick an index i, change s[i] to a random distinct value.
        Objective: Minimize number of conflicts? 
        Actually, we just want to find ANY valid set.
        So we can use a random walk with bias towards "more spread out" numbers.
        """
        current = s[:]
        current_valid = is_valid(current)
        if current_valid:
            return current
            
        best = current
        best_valid = False
        
        # We'll do a random walk. If we find a valid one, return it.
        # To guide the search, we can define a "cost" function:
        # Cost = sum of overlaps between sum sets.
        # But computing cost is same as validity check.
        # So we just do random moves and keep the last one if valid.
        # To avoid getting stuck, we can restart if no progress.
        
        # A better heuristic: 
        # If invalid, it means there's a collision.
        # We can try to resolve collisions by increasing gaps.
        
        # Let's try a simple hill climbing on "spread".
        # Actually, just random restarts of the whole set might be faster for k > 15.
        
        for _ in range(max_iters):
            if time.time() > time_limit:
                break
                
            # Pick a random index to change
            idx = random.randint(0, len(current) - 1)
            old_val = current[idx]
            
            # Pick a new value distinct from others
            new_val = random.randint(1, 256)
            attempts = 0
            while new_val in current and attempts < 10:
                new_val = random.randint(1, 256)
                attempts += 1
            
            if new_val in current:
                continue
                
            current[idx] = new_val
            current.sort()
            
            if is_valid(current):
                return current
                
            # Optional: keep track of best "almost valid" set?
            # Hard to define. Just random walk.
            
        return None

    # Try increasing k
    # We know 8 is easy (powers of 2).
    # ILP found 18.
    # Let's try to find 19, 20, 21...
    
    max_k_to_try = 25
    best_set = [1, 2, 4, 8, 16, 32, 64, 128] # Fallback
    
    # Pre-generate some good starting points?
    # For k=19, a linear sequence 1, 2, ..., 19 has sums 0..190.
    # Sums of cardinality c range from c(c+1)/2 to c(2*19-c+1)/2.
    # For c=1: 1..19
    # For c=2: 3..37
    # Overlap!
    # So linear sequences are bad for large k.
    # We need "super-increasing" like sequences, but they grow too fast.
    # 1, 2, 4, 8, 16, 32, 64, 128 (8 elements)
    # Next would be 256. 9 elements: 1..256.
    # 10 elements? 1, 2, 4, ..., 256, 257?
    # Sum of largest 9: 2+4+...+256 = 510.
    # Sum of smallest 10: 1+2+...+257? No.
    # The gap condition is strict.
    
    # Let's use the random local search approach.
    # We will try k from 25 down to 19.
    # For each k, we do multiple random restarts.
    
    for k in range(max_k_to_try, 18, -1):
        if time.time() > deadline:
            break
            
        found = False
        num_restarts = 50
        restarts_done = 0
        
        while restarts_done < num_restarts and not found:
            if time.time() > deadline:
                break
                
            # Generate a random starting set
            # Bias towards larger numbers to allow more "room" for sums?
            # Or bias towards a geometric progression?
            # Let's try a geometric progression with random noise.
            base = random.uniform(1.1, 1.5)
            s = []
            val = 1
            for i in range(k):
                # Add noise
                noise = random.randint(-2, 2)
                v = int(val) + noise
                if v < 1: v = 1
                if v > 256: v = 256
                # Ensure distinct
                while v in s:
                    v = random.randint(1, 256)
                s.append(v)
                val *= base
                
            s.sort()
            
            # Check if valid immediately
            if is_valid(s):
                best_set = s
                found = True
                break
            
            # Try local search
            res = local_search(s, max_iters=1000, time_limit=1.0)
            if res:
                best_set = res
                found = True
                break
                
            restarts_done += 1
            
        if found:
            # We found a valid set of size k.
            # Since we iterate k downwards, this is the max k we found.
            # We can stop here? 
            # The task says "Maximize set size". 
            # So yes, return the largest k found.
            return [int(x) for x in best_set]
            
    # If we didn't find anything > 18, return the known 18 or fallback.
    # The ILP found 18. We don't have that set here, but we can try to find 18 quickly.
    if time.time() < deadline - 5:
        # Try to find 18
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            s = random_valid_set(18)
            if is_valid(s):
                return [int(x) for x in s]
            res = local_search(s, max_iters=2000, time_limit=2.0)
            if res:
                return [int(x) for x in res]
                
    return [int(x) for x in best_set]

if __name__ == "__main__":
    # Quick test
    pass
