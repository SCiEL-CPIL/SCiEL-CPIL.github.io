"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 85.0  # Leave some buffer for 90s limit
    max_val = 4096
    
    def verify(s):
        """Verify that the set has unique triple sums."""
        if len(s) != len(set(s)):
            return False
        if any(x < 1 or x > 4096 for x in s):
            return False
        sums = [sum(t) for t in combinations_with_replacement(s, 3)]
        return len(sums) == len(set(sums))

    def get_sums_for(x, S_old):
        """Returns list of sums contributed by x given S_old (which does not contain x)."""
        sums = [3 * x]
        for s in S_old:
            sums.append(2 * x + s)
        for i in range(len(S_old)):
            for j in range(i, len(S_old)):
                sums.append(x + S_old[i] + S_old[j])
        return sums

    def get_pair_sums_for(x, S_old):
        """Returns set of pair sums involving x, given S_old."""
        ps = {2 * x}
        for s in S_old:
            ps.add(x + s)
        return ps

    def can_add(S, used_sums, pair_sums, x):
        """
        Check if x can be added to S.
        S: list of current elements
        used_sums: set of all triple sums in S
        pair_sums: set of all pair sums s_i + s_j for i <= j in S
        """
        # 1. x + x + x
        s1 = 3 * x
        if s1 in used_sums:
            return False
        
        # 2. x + x + s for s in S
        # Collision with existing used_sums?
        # Collision with other new sums?
        # New sums are:
        # A: 3x
        # B: 2x + s
        # C: x + s_i + s_j
        
        # Check B against used_sums
        for s in S:
            s2 = 2 * x + s
            if s2 in used_sums:
                return False
            # Check B against A (already done: 2x+s == 3x => s==x, but s!=x)
            # Check B against B (2x+s1 == 2x+s2 => s1==s2, distinct)
            
        # Check C against used_sums
        # C = x + p where p is in pair_sums (old)
        # But wait, pair_sums contains sums from S.
        # We need to check if x + p is in used_sums.
        for p in pair_sums:
            s3 = x + p
            if s3 in used_sums:
                return False
        
        # Check internal collisions among new sums (A, B, C)
        # A = 3x
        # B_s = 2x + s
        # C_ij = x + S[i] + S[j]
        
        # A vs B: 3x = 2x + s => x = s. Since x not in S, no collision.
        # A vs C: 3x = x + S[i] + S[j] => 2x = S[i] + S[j].
        # This means 2x must not be in pair_sums.
        if 2 * x in pair_sums:
            return False
        
        # B_s1 vs B_s2: 2x + s1 = 2x + s2 => s1 = s2. No collision.
        # B_s vs C_ij: 2x + s = x + S[i] + S[j] => x + s = S[i] + S[j].
        # This means for any s in S, x + s must not be in pair_sums.
        for s in S:
            if x + s in pair_sums:
                return False
        
        # C_ij1 vs C_ij2: x + S[i] + S[j] = x + S[k] + S[l] => S[i]+S[j] = S[k]+S[l].
        # Since pair_sums are unique for the set S? No, pair_sums is a set, so if S[i]+S[j] == S[k]+S[l] with {i,j} != {k,l}, 
        # then the sum value is the same. 
        # Wait, if S is a valid B_3[1] set, are pair sums unique?
        # No. B_3[1] requires triple sums to be unique.
        # Example: 1,2,4. Pair sums: 2,3,4,5,6,8. All unique.
        # Example: 1,3,4. Pair sums: 2,4,5,7,8,10. All unique.
        # Is it possible to have S[i]+S[j] = S[k]+S[l] with distinct pairs?
        # If S[i]+S[j] = S[k]+S[l], then consider triples.
        # Actually, if pair sums collide, it doesn't immediately violate B_3[1].
        # However, if x + S[i] + S[j] == x + S[k] + S[l], then the triple sums are equal.
        # So we need x + p to be unique for distinct p in pair_sums?
        # No, we need the set of values {x + p | p in pair_sums} to have unique values.
        # This is true if pair_sums has unique values.
        # But pair_sums is a set, so it only stores unique values.
        # If S has colliding pair sums, pair_sums will have fewer elements than number of pairs.
        # Does B_3[1] imply unique pair sums?
        # If S[i]+S[j] = S[k]+S[l] with {i,j} != {k,l}, say 1+4 = 2+3.
        # S={1,2,3,4}.
        # Triples:
        # 1+1+1=3, 1+1+2=4, 1+1+3=5, 1+1+4=6
        # 2+2+2=6. Collision! 1+1+4 = 2+2+2.
        # So yes, B_3[1] sets generally have unique pair sums?
        # Let's check.
        # If a+b = c+d with a,b,c,d in S.
        # If a=b=c=d, trivial.
        # If a=b != c=d, then 2a = 2c => a=c. Trivial.
        # If a=b, c!=d, then 2a = c+d.
        # Then triple (a,a,a) sum 3a. Triple (c,d, ?)
        # Consider (a,a,c) sum 2a+c = (c+d)+c = 2c+d.
        # And (a,a,d) sum 2a+d = 2c+2d? No.
        # Let's just trust that if pair sums collide, triple sums likely collide.
        # Specifically, if x + s is in pair_sums, we returned False.
        # This handles B vs C collisions.
        
        # What about C_ij collisions among themselves?
        # x + S[i] + S[j] = x + S[k] + S[l] => S[i]+S[j] = S[k]+S[l].
        # If S[i]+S[j] = S[k]+S[l] for distinct pairs, then the sum value is the same.
        # So we would add the same sum to used_sums twice?
        # No, used_sums is a set.
        # But the requirement is that ALL unordered triples have DIFFERENT sums.
        # So if two different triples have the same sum, it's invalid.
        # Triple (x, S[i], S[j]) and (x, S[k], S[l]).
        # If S[i]+S[j] = S[k]+S[l] and the pairs are distinct, we have a collision.
        # So we must ensure that for the new element x, the values x + p for p in pair_sums are unique.
        # This is guaranteed if pair_sums contains unique values.
        # But does S being a partial B_3[1] set guarantee unique pair sums?
        # As shown above, 1,2,3,4 is not B_3[1].
        # It is a known result that Sidon sets (B_2[1]) are a subset of B_3[1]? No.
        # But B_3[1] implies B_2[1] (unique pair sums) ?
        # If a+b = c+d, then consider triples (a,a,b) and (c,c,d)?
        # Sum1 = 2a+b. Sum2 = 2c+d.
        # Not necessarily equal.
        # However, if we have a+b=c+d, and we pick x such that x is not involved, we don't care.
        # But when adding x, we form triples with x.
        # The collision (x, S[i], S[j]) vs (x, S[k], S[l]) depends on S[i]+S[j] vs S[k]+S[l].
        # If S has non-unique pair sums, then adding ANY x will cause a collision for those pairs?
        # Yes. If S[i]+S[j] = S[k]+S[l] with distinct pairs, then for ANY x,
        # x+S[i]+S[j] = x+S[k]+S[l].
        # So a valid B_3[1] set MUST have unique pair sums.
        # Therefore, pair_sums will have size n(n+1)/2.
        # And the values x+p will be distinct for distinct p.
        
        return True

    def add_to_state(S, S_set, used_sums, pair_sums, x):
        S_old = S[:]
        S.append(x)
        S_set.add(x)
        
        # Update used_sums
        for s_val in get_sums_for(x, S_old):
            used_sums.add(s_val)
            
        # Update pair_sums
        for p_val in get_pair_sums_for(x, S_old):
            pair_sums.add(p_val)

    def remove_element(S, S_set, used_sums, pair_sums, idx):
        y = S[idx]
        S_rest = S[:idx] + S[idx+1:]
        
        S.pop(idx)
        S_set.remove(y)
        
        for s_val in get_sums_for(y, S_rest):
            used_sums.remove(s_val)
            
        for p_val in get_pair_sums_for(y, S_rest):
            pair_sums.remove(p_val)
            
        return y, S_rest

    # Strategy 1: Fast Greedy with Pair Sums optimization
    S = []
    S_set = set()
    used_sums = set()
    pair_sums = set()
    
    for x in range(1, max_val + 1):
        if time.time() - start_time > time_limit - 5:
            break
        if can_add(S, used_sums, pair_sums, x):
            add_to_state(S, S_set, used_sums, pair_sums, x)
            if len(S) >= 40:
                break
                
    best_set = list(S)
    best_len = len(S)
    
    # Strategy 2: Local Search / Swap
    # We have a set S. We try to swap elements to allow adding more.
    # Or just random restarts with different orderings.
    
    random.seed(42)
    for trial in range(50):
        if time.time() - start_time > time_limit - 10:
            break
            
        S = []
        S_set = set()
        used_sums = set()
        pair_sums = set()
        
        order = list(range(1, max_val + 1))
        random.shuffle(order)
        
        for x in order:
            if time.time() - start_time > time_limit - 10:
                break
            if len(S) >= 40:
                break
            if can_add(S, used_sums, pair_sums, x):
                add_to_state(S, S_set, used_sums, pair_sums, x)
                
        if len(S) > best_len:
            best_set = list(S)
            best_len = len(S)
            
    # Strategy 3: Hill Climbing / Swap to escape local optima
    # Start with best_set found so far
    if best_len < 40 and time.time() - start_time < time_limit - 15:
        S = list(best_set)
        S_set = set(S)
        used_sums = set()
        pair_sums = set()
        
        # Rebuild state
        # This is O(n^3) to rebuild, but n is small.
        # Actually, we can just rebuild from scratch for correctness.
        S_temp = []
        for x in S:
            S_temp.append(x)
            # We need to update state incrementally for this specific S
            # Let's just use the add_to_state logic carefully
            # But S is not built in sorted order necessarily?
            # The order doesn't matter for the final state, but add_to_state assumes S is the current state.
            # So we rebuild.
            pass
            
        # Rebuild properly
        S = []
        S_set = set()
        used_sums = set()
        pair_sums = set()
        for x in best_set:
            add_to_state(S, S_set, used_sums, pair_sums, x)
            
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            
            # Try to add any x
            for x in range(1, max_val + 1):
                if x not in S_set:
                    if can_add(S, used_sums, pair_sums, x):
                        add_to_state(S, S_set, used_sums, pair_sums, x)
                        improved = True
                        if len(S) > best_len:
                            best_set = list(S)
                            best_len = len(S)
                        break # Restart scan? Or continue? Continue is faster.
                        
            if len(S) >= 40:
                break
                
            if not improved:
                # Try a swap
                if len(S) > 0:
                    # Pick random element to remove
                    idx = random.randint(0, len(S) - 1)
                    y, S_rest = remove_element(S, S_set, used_sums, pair_sums, idx)
                    
                    # Try to find x to replace y
                    swapped = False
                    # Randomize search order for x?
                    candidates = [x for x in range(1, max_val + 1) if x != y and x not in S_set]
                    random.shuffle(candidates)
                    
                    for x in candidates:
                        if can_add(S, used_sums, pair_sums, x):
                            add_to_state(S, S_set, used_sums, pair_sums, x)
                            swapped = True
                            break
                            
                    if not swapped:
                        # Restore y
                        # We need to add y back to S (which is S_rest + y)
                        # But S is currently S_rest.
                        # add_to_state appends to S.
                        # So we just call add_to_state(S, ..., y)
                        add_to_state(S, S_set, used_sums, pair_sums, y)
                    else:
                        # State changed, might allow growth
                        # Check if size increased? No, swap is 1-for-1.
                        # But it might allow adding more in the next loop iteration.
                        # We set improved=True to trigger the "Try to add" loop again.
                        improved = True
                        
    # Verify final result
    if not verify(best_set):
        # Fallback
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    return best_set
# EVOLVE-BLOCK-END
