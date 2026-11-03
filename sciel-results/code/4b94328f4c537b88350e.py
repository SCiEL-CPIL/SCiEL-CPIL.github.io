"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def score_set(U):
        U_set = set(U)
        max_U = max(U)
        if max_U == 0:
            return 0.0
        # Compute sumset and difference set
        sumset = set()
        diffset = set()
        # Optimization: Use list for faster iteration if U is small, 
        # but for large U, we must be careful. 
        # Given the 120s limit, we cannot afford O(N^2) for very large N.
        # However, the constructed sets are likely < 1000 elements if we use 
        # geometric progression, so O(N^2) is ~10^6 ops, which is fast.
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n = len(U_set)
        if n == 0:
            return 0.0
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except:
            return 0.0
        return exponent

    # Strategy: Construct sets using Base-B expansion.
    # U = { sum_{i=0}^{k-1} d_i B^i } where d_i in D.
    # To maximize |U-U|/|U+U|, we want the "difference" to be as large as possible
    # relative to the "sum". 
    # If B is large, sumset is roughly the Cartesian product of D+D in base B.
    # Difference set is roughly Cartesian product of D-D in base B.
    # Ratio is approx (|D-D| / |D+D|)^k.
    # We want to maximize |D-D|/|D+D|.
    
    # Let's test small digit sets D.
    # D = {0, 1}: |D+D|=3 (0,1,2), |D-D|=3 (-1,0,1). Ratio = 1.
    # D = {0, 1, 3}: |D+D|=6 (0,1,2,3,4,6), |D-D|=7 (-3,-2,-1,0,1,2,3). Ratio = 7/6 > 1.
    # D = {0, 1, 3, 9}: |D+D|? 
    #    Sums: 0,1,2,3,4,6,9,10,12,15,18 -> 11 elements.
    #    Diffs: -9..9? 
    #    D-D: 
    #    0: 0
    #    1: 1, -1
    #    3: 3, 2, 1, -1, -2, -3
    #    9: 9, 8, 6, 5, 4, 2, 1, -1, -2, -3, -5, -6, -8, -9
    #    Unique diffs: -9, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 8, 9. (17 elements)
    #    Ratio = 17/11 ~ 1.54.
    
    # Generalizing: D = {0, 1, 3, 9, ...} is powers of 3.
    # For D = {0, 3^0, 3^1, ..., 3^{k-1}}, 
    # This is the standard "Salem-Spencer" type set but with base 3.
    # Actually, if we use base B > 2*max(D), the representations are unique.
    # Let D = {0, 1, 3, 9, ... 3^{m-1}}.
    # The set U is all numbers with digits 0, 1, 3, 9... in base 10? No, base B.
    # If we pick B=10, and D={0,1,3,9}, the elements are distinct.
    
    # Let's try to construct U using a base B and digit set D.
    # We want to maximize the exponent:
    # E = 1 + log( (|D-D|/|D+D|)^K ) / log(2 * (B^K - 1) + 1)  approx
    # E = 1 + K * log(R_D) / (K * log(B)) = 1 + log(R_D) / log(B)
    # Where R_D = |D-D| / |D+D|.
    # To maximize E, we need to maximize log(R_D) / log(B).
    # Since R_D is determined by D, and B must be large enough to avoid carries (B > 2*max(D) is safe, B > max(D)+max(D) is strict).
    # Actually, if B > 2*max(D), then |U+U| = |D+D|^K and |U-U| = |D-D|^K.
    # So the ratio is (|D-D|/|D+D|)^K.
    # The denominator in the exponent is log(2*max(U)+1) ~ K*log(B).
    # So Exponent ~ 1 + K * log(R_D) / (K * log(B)) = 1 + log(R_D)/log(B).
    # This suggests K (length of the set) cancels out in the leading term?
    # Wait, max(U) is roughly B^{K-1} * max(D).
    # log(2*max(U)+1) ~ K*log(B).
    # log(|U-U|/|U+U|) = K * log(|D-D|/|D+D|).
    # So Exponent = 1 + log(|D-D|/|D+D|) / log(B).
    
    # This implies we should choose D to maximize R_D = |D-D|/|D+D|, and B to be as small as possible (to minimize log(B) in denominator) while ensuring uniqueness (B > 2*max(D)).
    
    # Let's evaluate R_D for various D.
    # D = {0, 1}: R=1. log(1)=0. Exponent=1.
    # D = {0, 1, 3}: 
    #   D+D = {0,1,2,3,4,6} (6)
    #   D-D = {-3,-2,-1,0,1,2,3} (7)
    #   R = 7/6 = 1.166. log(1.166) = 0.154.
    #   min B = 2*3 + 1 = 7. log(7) = 1.945.
    #   Exponent = 1 + 0.154/1.945 = 1.079.
    
    # D = {0, 1, 3, 9}:
    #   R = 17/11 = 1.545. log(1.545) = 0.435.
    #   min B = 2*9 + 1 = 19. log(19) = 2.944.
    #   Exponent = 1 + 0.435/2.944 = 1.147.
    
    # D = {0, 1, 3, 9, 27}:
    #   Let's calculate R.
    #   This looks like it might increase.
    #   Generally, for D = {0, 3^0, ..., 3^{m-1}}, 
    #   |D+D| grows like 3^m? No, |D+D| for {0,1,3,9} was 11. 3^4=81. It's much smaller.
    #   Actually, D is a Sidon set.
    #   Let's just compute R for a few candidates and pick the best B.
    
    candidates_D = [
        [0, 1],
        [0, 1, 3],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27],
        [0, 1, 3, 9, 27, 81],
        [0, 1, 2, 4, 8], # Powers of 2
        [0, 1, 2, 5, 10], # Randomish
        [0, 1, 4, 9, 16], # Squares
    ]
    
    best_exponent = 0.0
    best_U = [0]
    
    for D in candidates_D:
        # Calculate |D+D| and |D-D|
        sumset_D = set()
        diffset_D = set()
        for a in D:
            for b in D:
                sumset_D.add(a+b)
                diffset_D.add(a-b)
        
        R = len(diffset_D) / len(sumset_D)
        if R <= 1:
            continue
            
        # Minimum B to avoid carries: B > 2*max(D)
        # Actually, we need B > max(D+D) to ensure unique representation of sums?
        # If B > max(D+D), then digits of sum are just sums of digits.
        # So B = max(D+D) + 1 is sufficient.
        B = max(sumset_D) + 1
        
        # Theoretical exponent
        try:
            exp_val = 1 + math.log(R) / math.log(B)
        except:
            continue
        
        # We need to construct U to verify.
        # U = { sum d_i B^i }
        # We can choose K (number of digits) to make the set size reasonable.
        # Larger K doesn't change the asymptotic exponent, but finite size effects matter.
        # Also, we must finish in time.
        # Let's try K=1, 2, 3, 4... up to a limit.
        
        # Actually, the formula 1 + log(R)/log(B) assumes K is large.
        # For small K, we should just compute the actual score.
        
        # Let's generate a few sets with this D and B.
        # We want to maximize the actual score.
        # The score depends on the exact sizes.
        
        # To keep computation fast, we limit the size of U.
        # If |D|=m, |U| = m^K.
        # If m=4, K=5 -> 1024 elements. O(N^2) = 10^6. Fast.
        # If m=6, K=4 -> 1296 elements. Fast.
        
        max_K = 10
        for K in range(1, max_K + 1):
            if time.time() > deadline:
                break
                
            # Generate U
            U = [0]
            # Use product of D repeated K times
            # But we need to be careful with memory/time.
            # If m^K is too big, skip.
            if len(D) ** K > 2000:
                continue
                
            # Generate all combinations
            U_set = set()
            # Recursive generation or iterative
            # Iterative:
            current = [0]
            for i in range(K):
                new_current = []
                for x in current:
                    for d in D:
                        new_current.append(x + d * (B ** i))
                current = new_current
            
            U = list(set(current))
            
            # Score it
            s = score_set(U)
            if s > best_exponent:
                best_exponent = s
                best_U = U
    
    # If the algebraic construction didn't beat the seed (which is 1.0), 
    # we might want to try a simple local search on the best algebraic set 
    # or just return the best algebraic set.
    
    # However, the seed program starts with [0,1,3] and does local search.
    # Our algebraic construction for D={0,1,3}, B=7, K=1 is [0,1,3]. Score 1.0?
    # Let's check D={0,1,3}, B=7.
    # U=[0,1,3]. 
    # Sum: 0,1,2,3,4,6 (6)
    # Diff: -3..3 (7)
    # Ratio 7/6.
    # Max U = 3. 2*3+1=7.
    # Exponent = 1 + log(7/6)/log(7) = 1 + 0.154/1.945 = 1.079.
    # So this already beats 1.0.
    
    # What if we increase K?
    # D={0,1,3}, B=7, K=2.
    # U = {0,1,3, 7,8,10, 14,15,17}. Size 9.
    # Max U = 17. 2*17+1 = 35. log(35)=3.55.
    # Sumset size: (|D+D|)^2 = 6^2 = 36?
    # Since B=7 > max(D+D)=6, yes, unique.
    # Diffset size: (|D-D|)^2 = 7^2 = 49.
    # Ratio = 49/36 = 1.361.
    # Exponent = 1 + log(1.361)/log(35) = 1 + 0.308/3.55 = 1.086.
    # It increased!
    
    # K=3.
    # Max U ~ 7^2*3 = 147. 2*147+1=295. log(295)=5.68.
    # Ratio = (7/6)^3 = 1.48.
    # Exponent = 1 + log(1.48)/log(295) = 1 + 0.392/5.68 = 1.069.
    # It decreased!
    
    # So there is an optimal K.
    # The loop above tries K=1..10 and keeps the best.
    
    return best_U
# EVOLVE-BLOCK-END
