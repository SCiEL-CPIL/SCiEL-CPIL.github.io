import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    def check_validity(S):
        """
        Checks if S is a valid set.
        Returns True if valid.
        """
        n = len(S)
        if n < 3:
            return True
        
        S_set = set(S)
        
        for a in S:
            Pos = []
            Neg = []
            for t in S_set:
                if t == a:
                    continue
                d = t - a
                if d > 0:
                    Pos.append(d)
                elif d < 0:
                    Neg.append(-d)
            
            # Compute subset sums of Pos using bitset
            # sums_pos bit i is 1 if sum i is achievable
            sums_pos = 1
            for d in Pos:
                sums_pos |= sums_pos << d
            
            # Compute subset sums of Neg
            sums_neg = 1
            for d in Neg:
                sums_neg |= sums_neg << d
            
            # Check intersection
            # If any common sum > 0 exists, violation.
            # Bit 0 is always set (empty subset). We ignore it.
            intersection = sums_pos & sums_neg
            if intersection & ~1:
                return False
        
        return True

    def is_valid_addition(S, x):
        """
        Checks if adding x to S results in a valid set.
        Optimized to only check the new element x and the existing elements in S.
        """
        if x in S:
            return False
        
        # 1. Check if x is the mean of a subset of S
        # This is equivalent to: Pos_x and Neg_x (relative to S) have common positive subset sum
        Pos_x = []
        Neg_x = []
        for t in S:
            d = t - x
            if d > 0:
                Pos_x.append(d)
            elif d < 0:
                Neg_x.append(-d)
        
        if Pos_x and Neg_x:
            sums_pos = 1
            for d in Pos_x:
                sums_pos |= sums_pos << d
            sums_neg = 1
            for d in Neg_x:
                sums_neg |= sums_neg << d
            if (sums_pos & sums_neg) & ~1:
                return False
        # If one is empty, no common sum > 0 is possible (sums of empty set is 0, 
        # but we need sum > 0 for non-empty subsets on both sides? 
        # Actually, if Pos is empty, intersection is just sums_neg. 
        # sums_neg has bit 0 set. sums_pos is 1 (bit 0 set). 
        # Intersection is 1. 1 & ~1 is 0. So valid. Correct.)

        # 2. Check if any existing element a in S becomes a mean of a subset of S + {x}
        # For each a in S, the set of "other" elements is S \ {a} U {x}.
        # The differences relative to a change by adding (x-a) or (a-x).
        
        for a in S:
            # We need to check if Pos_a' and Neg_a' have common positive subset sum.
            # Pos_a' = Pos_a U {x-a} if x > a
            # Neg_a' = Neg_a U {a-x} if x < a
            
            # Reconstructing Pos_a and Neg_a from scratch is O(N).
            # Since we are in a loop over S (size N), this is O(N^2) total.
            # Given N <= 48, this is fine.
            
            Pos_a = []
            Neg_a = []
            for t in S:
                if t == a:
                    continue
                d = t - a
                if d > 0:
                    Pos_a.append(d)
                elif d < 0:
                    Neg_a.append(-d)
            
            # Add the contribution of x
            dx = x - a
            if dx > 0:
                Pos_a.append(dx)
            elif dx < 0:
                Neg_a.append(-dx)
            # if dx == 0, x==a, but x is distinct from S, so this doesn't happen.
            
            # Compute bitsets
            sums_pos = 1
            for d in Pos_a:
                sums_pos |= sums_pos << d
            
            sums_neg = 1
            for d in Neg_a:
                sums_neg |= sums_neg << d
            
            if (sums_pos & sums_neg) & ~1:
                return False
                
        return True

    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # Define several heuristic orders
    orders = []
    
    # 1. Ascending
    orders.append(candidates[:])
    # 2. Descending
    orders.append(candidates[::-1])
    # 3. Interleaved (small, large, small, large...)
    inter = []
    lo, hi = 0, len(candidates) - 1
    while lo <= hi:
        inter.append(candidates[lo])
        if lo != hi:
            inter.append(candidates[hi])
        lo += 1
        hi -= 1
    orders.append(inter)
    
    # 4. Random orders
    rng = random.Random(42)
    # We will generate more random orders in the main loop if time permits.
    
    for order in orders:
        if time.time() > deadline - 5:
            break
        
        S = []
        for x in order:
            if time.time() > deadline - 2:
                break
            if len(S) >= 48:
                break
            
            # Fast check
            if is_valid_addition(S, x):
                S.append(x)
                
        if len(S) > len(best_set):
            best_set = S[:]
            
        if len(best_set) >= 48:
            break
            
    # If we haven't reached 48, do random restarts
    while time.time() < deadline - 5:
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        
        S = []
        for x in rand_order:
            if time.time() > deadline - 2:
                break
            if len(S) >= 48:
                break
            
            if is_valid_addition(S, x):
                S.append(x)
                
        if len(S) > len(best_set):
            best_set = S[:]
            
        if len(best_set) >= 48:
            break
            
    return best_set
