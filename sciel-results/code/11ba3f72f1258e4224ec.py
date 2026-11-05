import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    N = 192
    # Initial greedy construction: Descending order
    # This is a strong baseline because large numbers are "safer"
    selected = []
    for x in range(N, 0, -1):
        # Check if x can be added to selected
        # Condition 1: x does not divide sum of any 3 distinct in selected
        # Condition 2: No a in selected divides sum of any 3 distinct in selected + [x] involving x
        
        # Optimization: 
        # If selected is small, just brute force.
        # As selected grows, we need efficiency.
        
        valid = True
        
        # Check 1: x divides sum of 3 from selected?
        # Sums of 3 from selected.
        # We can precompute sums of pairs?
        # For now, brute force with early exit.
        if len(selected) >= 3:
            # Check if x divides any sum of 3 distinct from selected
            # To speed up, we can check modulo x.
            # We need x | (a+b+c).
            # This is equivalent to (a+b+c) % x == 0.
            
            # We can iterate over pairs (a,b) and check if there is c such that (a+b+c)%x==0.
            # c % x == -(a+b) % x.
            # Precompute counts of residues for selected?
            # Residue counts change as we add elements.
            
            # Let's just do a nested loop but break early.
            # Since we add descending, selected contains large numbers.
            # x is smaller than all in selected.
            # Sums are large.
            
            pass # Placeholder for optimized check
        
        # For the first version, let's use the same check as seed but maybe faster?
        # The seed check:
        # all(sum(t)%a for a in candidate for t in combinations([v for v in candidate if v!=a],3))
        # This is O(k^4) per step.
        
        # Let's try a heuristic check first.
        # If x is large, it's likely safe.
        
        # Actually, let's just use the seed logic but add a swap phase.
        
        if _is_valid_addition(x, selected):
            selected.append(x)
            
        if time.time() > deadline:
            break
            
    # Swap Phase
    # Try to improve by swapping smaller elements with larger unpicked ones
    # Unpicked: 1..192 not in selected
    # Sort selected to find smallest
    # This phase is risky for time, so keep it simple.
    
    # Sort selected descending for return? No, set is fine.
    # The problem asks for distinct positive integers.
    
    return selected

def _is_valid_addition(x, S):
    """
    Check if x can be added to S such that the new set is valid.
    S is a list of selected numbers.
    """
    # 1. x must not divide sum of any 3 distinct in S
    if len(S) >= 3:
        # Check if exists a,b,c in S distinct such that (a+b+c) % x == 0
        # Optimization: 
        # We can iterate a, b and check if there exists c.
        # But S is small enough (<= 100) that O(N^3) is ~ 1e6.
        # 192 steps * 1e6 = 1e8. Might be slow in Python.
        
        # Faster check:
        # Precompute pair sums?
        # Let's try a set of sums.
        # Sums of 3 elements from S.
        # If we maintain a set of all triplet sums, we can just check `0 in [s % x for s in triplet_sums]`? No.
        # We need `s % x == 0`.
        
        # Let's use a randomized check? No, must be exact.
        
        # We'll use the nested loop but with pruning.
        # Since S contains large numbers, and x is smaller (in descending greedy),
        # a+b+c is likely > x.
        
        # Actually, let's just use the combinations generator but it's slow.
        # Let's try to optimize by checking only if x is "small" enough to divide sums.
        # Max sum ~ 600. If x > 600, it can't divide (except 0, but sums > 0).
        # Here x <= 192.
        
        # We'll do:
        # for i in range(len(S)):
        #   for j in range(i+1, len(S)):
        #     target = (- (S[i] + S[j])) % x
        #     # Check if any S[k] (k != i, j) has S[k] % x == target
        #     # This is O(N^2 * N) = O(N^3).
        #     # We can precompute a set of residues?
        #     # Residues of S mod x.
        #     # But x changes.
        
        # Let's just do the O(N^3) check.
        n = len(S)
        for i in range(n):
            for j in range(i+1, n):
                s2 = S[i] + S[j]
                # We need S[k] such that (s2 + S[k]) % x == 0
                # S[k] % x == (-s2) % x
                target_res = (-s2) % x
                # Check if any S[k] (k != i, j) has this residue
                # We can just loop k.
                for k in range(j+1, n):
                    if (S[i] + S[j] + S[k]) % x == 0:
                        return False
                # Also check k < i? No, we just need existence of distinct i,j,k.
                # The loop above covers k > j. 
                # We also need to check if k < i or i < k < j.
                # My loop structure: i < j < k. This covers all distinct triples.
                
    # 2. No a in S divides sum of any 3 distinct in S + [x] involving x
    # Sums involving x are x + S[i] + S[j] for distinct i, j in S.
    # We need to check for each a in S:
    # Does a divide (x + S[i] + S[j]) for any distinct i, j?
    
    if len(S) >= 2:
        for a in S:
            # Check if exists i, j distinct in S such that (x + S[i] + S[j]) % a == 0
            # (S[i] + S[j]) % a == (-x) % a
            target_res = (-x) % a
            n = len(S)
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    if (S[i] + S[j]) % a == target_res:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
                
    return True

# EVOLVE-BLOCK-END
