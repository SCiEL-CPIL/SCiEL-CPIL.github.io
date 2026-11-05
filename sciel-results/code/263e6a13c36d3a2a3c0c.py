import time
import random
from itertools import combinations

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute conflict masks
    # conflict_mask[i][j] = bitmask of k such that adding i and j to a set
    # containing any element in the mask would create a violation.
    # Specifically, for a new element x, we need to check:
    # 1. x as 'a': for all b, c in S with b > x, c >= b: x !| b+c
    #    This means if x is in S, then for any pair (b,c) in S with b>x, c>=b and x|(b+c), it's invalid.
    #    So for a fixed x, the "conflict" is determined by pairs (b,c) that are multiples of x.
    # 2. x as 'b': for all a, c in S with a < x, c >= x: a !| x+c
    #    This means if x is in S, then for any pair (a,c) in S with a<x, c>=x and a|(x+c), it's invalid.
    # 3. x as 'c': for all a, b in S with a < b <= x: a !| b+x
    #    This means if x is in S, then for any pair (a,b) in S with a<b<=x and a|(b+x), it's invalid.
    
    # We can represent the set S as a bitmask of 256 bits.
    # We can precompute for each x a bitmask "forbidden[x]" such that
    # if (S & forbidden[x]) != 0, then x cannot be added?
    # No, because the condition depends on pairs in S, not single elements.
    
    # Let's use a different approach:
    # For each x, precompute a list of "conflict pairs".
    # conflict_pairs[x] = list of (mask1, mask2) such that if S contains an element from mask1 and an element from mask2, then x cannot be added.
    # This is too complex.
    
    # Simpler: Use a greedy + local search with efficient validity checking.
    
    def can_add(x, S_mask):
        """
        Check if x can be added to the set represented by S_mask.
        S_mask is an integer where bit i is set if i+1 is in S.
        We assume x is NOT in S_mask.
        """
        # 1. x is 'a': for all b, c in S with b > x, c >= b: x !| b+c
        #    We need to check if there exist b, c in S such that b > x, c >= b, and (b+c) % x == 0.
        #    Since b, c > x, let b = x + u, c = x + v with u, v >= 0 and u <= v.
        #    (x+u + x+v) % x == (2x + u + v) % x == (u + v) % x.
        #    So we need (u+v) % x == 0.
        
        # 2. x is 'b': for all a, c in S with a < x, c >= x: a !| x+c
        #    Let a be in S with a < x. Let c be in S with c >= x.
        #    Check if (x+c) % a == 0.
        
        # 3. x is 'c': for all a, b in S with a < b <= x: a !| b+x
        #    Let a, b be in S with a < b <= x.
        #    Check if (b+x) % a == 0.
        
        # Optimization:
        # Instead of iterating all pairs, we can iterate over elements in S.
        
        # Get list of elements in S
        # This is slow if done naively.
        # Let's precompute for each x the set of "bad" pairs?
        
        # Let's try a direct check with bit operations where possible.
        
        # 1. x as 'a'
        # We need to find b, c in S such that b > x, c >= b, and (b+c) % x == 0.
        # Let b = x + u. Then c = k*x - u for some k.
        # Since c >= b = x+u, k*x - u >= x+u => k*x >= x + 2u => (k-1)x >= 2u.
        # Also c <= 256.
        
        # This seems hard to vectorize easily without numpy.
        # Let's stick to a reasonably efficient Python loop, but optimize the inner loops.
        
        S_list = []
        # Extract elements from mask
        # This is O(n) per call, which is 256.
        # Total calls in local search could be many.
        
        # Alternative: maintain a sorted list S alongside the mask?
        # Or just use a set?
        
        # Let's use a set for S in the main loop, and only convert to mask for final output if needed?
        # The task asks for a subset, so a list or set is fine.
        
        # Let's rewrite construct to use a set S, and optimize can_add(x, S_set).
        
        pass # Placeholder, will rewrite below

    def can_add_set(x, S):
        """
        Check if x can be added to set S.
        S is a set of integers.
        """
        # 1. x is 'a'
        # b, c in S, b > x, c >= b, (b+c) % x == 0
        # For each b in S with b > x:
        #   We need c in S such that c >= b and (b+c) % x == 0.
        #   (b+c) % x == 0 => c % x == (-b) % x.
        #   Let rem = (-b) % x.
        #   We need c in S, c >= b, c % x == rem.
        
        for b in S:
            if b <= x:
                continue
            rem = (-b) % x
            # Find c in S such that c >= b and c % x == rem
            # We can iterate c in S? That's O(|S|).
            # Total O(|S|^2).
            for c in S:
                if c >= b and c % x == rem:
                    return False
        
        # 2. x is 'b'
        # a, c in S, a < x, c >= x, (x+c) % a == 0
        # For each a in S with a < x:
        #   We need c in S such that c >= x and (x+c) % a == 0.
        #   (x+c) % a == 0 => c % a == (-x) % a.
        #   Let rem = (-x) % a.
        #   We need c in S, c >= x, c % a == rem.
        
        for a in S:
            if a >= x:
                continue
            rem = (-x) % a
            for c in S:
                if c >= x and c % a == rem:
                    return False
        
        # 3. x is 'c'
        # a, b in S, a < b <= x, (b+x) % a == 0
        # For each a in S with a < x:
        #   We need b in S such that a < b <= x and (b+x) % a == 0.
        #   (b+x) % a == 0 => b % a == (-x) % a.
        #   Let rem = (-x) % a.
        #   We need b in S, a < b <= x, b % a == rem.
        
        for a in S:
            if a >= x:
                continue
            rem = (-x) % a
            for b in S:
                if a < b <= x and b % a == rem:
                    return False
        
        return True

    # Greedy Initialization
    # Start with a large set, e.g., [129, 256]
    S = set(range(129, n + 1))
    
    # Try to add smaller elements
    for x in range(128, 0, -1):
        if time.time() > deadline - 5:
            break
        if can_add_set(x, S):
            S.add(x)
    
    best_S = set(S)
    best_size = len(S)
    
    # Local Search
    # Try to swap elements: remove s from S, add x not in S.
    # Or just add x if possible (shouldn't happen after greedy).
    # Or remove s to allow adding x.
    
    R = [x for x in range(1, n + 1) if x not in S]
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Shuffle R to avoid bias?
        # random.shuffle(R) # Maybe deterministic is better for reproducibility?
        
        for x in R:
            if time.time() > deadline - 10:
                break
            
            # Find blockers for x
            # Blockers are elements s in S such that removing s allows adding x?
            # Actually, we need to identify which elements in S cause conflicts with x.
            
            # Let's find all s in S that are part of a conflict with x.
            # Conflict 1: x is 'a'. b, c in S, b > x, c >= b, x | b+c.
            #   s can be b or c.
            # Conflict 2: x is 'b'. a, c in S, a < x, c >= x, a | x+c.
            #   s can be a or c.
            # Conflict 3: x is 'c'. a, b in S, a < b <= x, a | b+x.
            #   s can be a or b.
            
            blockers = set()
            
            # 1. x is 'a'
            for b in S:
                if b <= x:
                    continue
                rem = (-b) % x
                for c in S:
                    if c >= b and c % x == rem:
                        blockers.add(b)
                        blockers.add(c)
            
            # 2. x is 'b'
            for a in S:
                if a >= x:
                    continue
                rem = (-x) % a
                for c in S:
                    if c >= x and c % a == rem:
                        blockers.add(a)
                        blockers.add(c)
            
            # 3. x is 'c'
            for a in S:
                if a >= x:
                    continue
                rem = (-x) % a
                for b in S:
                    if a < b <= x and b % a == rem:
                        blockers.add(a)
                        blockers.add(b)
            
            if not blockers:
                # Should not happen
                S.add(x)
                R.remove(x)
                improved = True
                break
            
            # Try removing each blocker s
            # To make it faster, we can limit the number of swaps we try.
            for s in blockers:
                if time.time() > deadline - 10:
                    break
                
                S_temp = S - {s}
                S_temp.add(x)
                
                # Check if S_temp is valid.
                # We know S was valid.
                # We removed s and added x.
                # We need to check if x can be added to S - {s}.
                # But can_add_set(x, S - {s}) is O(|S|^2).
                # Since s is a blocker, it might work.
                
                if can_add_set(x, S - {s}):
                    # S_temp is valid.
                    # Size is same.
                    # Does this help?
                    # Maybe we can add more elements?
                    # Let's check if we can add any element from R to S_temp.
                    added = False
                    for r2 in R:
                        if r2 == x:
                            continue
                        if can_add_set(r2, S_temp):
                            S_temp.add(r2)
                            R.remove(r2)
                            R.remove(x)
                            S = S_temp
                            improved = True
                            added = True
                            break
                    if added:
                        break
                    # If we couldn't add r2, we can still accept the swap if it allows future improvements?
                    # For now, let's only accept if size increases.
                    # Or accept swap if it seems to open up more options?
                    # Let's stick to size increase for safety.
            
            if improved:
                break
    
    return sorted(best_S if len(best_S) > len(S) else S)

# Test harness
if __name__ == "__main__":
    t0 = time.time()
    S = construct()
    t1 = time.time()
    print(f"Size: {len(S)}, Time: {t1-t0:.4f}s")
    # Verify
    valid = True
    L = len(S)
    for i in range(L):
        a = S[i]
        for j in range(i+1, L):
            b = S[j]
            for k in range(j, L):
                c = S[k]
                if (b+c) % a == 0:
                    valid = False
                    print(f"Invalid: {a} divides {b}+{c}")
                    break
            if not valid: break
        if not valid: break
    print("Valid" if valid else "Invalid")
