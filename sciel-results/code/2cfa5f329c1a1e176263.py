"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def check_time():
        return time.time() > deadline

    def score_set(U):
        """
        Compute the score for a set U.
        Score = 1 + log(|U-U| / |U+U|) / log(2*max(U)+1)
        """
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        
        n = len(U_set)
        if n > 10000:
            return 0.0 # Too large for exact set computation in Python within budget if called frequently
            
        sumset = set()
        diffset = set()
        
        # Optimization for small/medium sets
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
                
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except:
            return 0.0
        return exponent

    def score_structured(D, B):
        """
        Compute the score for U = { d0 + d1*B | d0, d1 in D } efficiently.
        Uses combinatorial logic for sum/diff sets to handle larger |D| than O(n^2) set construction.
        """
        if not D:
            return 0.0
        
        D_set = set(D)
        if not D_set:
            return 0.0
        
        # U = { d0 + d1*B }
        # max_U = max(D) + max(D)*B
        max_d = max(D_set)
        max_U = max_d + max_d * B
        
        if max_U == 0:
            return 0.0
            
        # Compute |U+U|
        # u = d0 + d1*B, v = e0 + e1*B
        # u+v = (d0+e0) + (d1+e1)*B
        # Let S0 = {d+e | d,e in D}, S1 = {d+e | d,e in D}
        # Since B is large (we will enforce B > 2*max(D) + 1), there are no carries between base-B digits 
        # IF we view the sum in base B? 
        # Wait. d0+e0 might exceed B. 
        # If B > 2*max(D), then d0+e0 < B. So the base-B representation of u+v is unique 
        # as (d0+e0) + (d1+e1)*B ONLY IF d1+e1 doesn't cause a carry into higher bits? 
        # Actually, u+v = d0+e0 + (d1+e1)B.
        # If B > 2*max(D), then d0+e0 < B.
        # The value is determined by the pair (d0+e0, d1+e1).
        # Are all pairs (s0, s1) distinct values?
        # Value = s0 + s1*B.
        # If B > max(s0) + max(s1)*B? No.
        # If B > max(s0) + max(s1)*B is impossible.
        # We need B > max(s0) + max(s1)*B? No.
        # We need the map (s0, s1) -> s0 + s1*B to be injective.
        # s0, s1 are sums of digits.
        # If B > max(s0) + max(s1)*B? No.
        # If B > max(s0) + max(s1)*B is not the condition.
        # The condition for injectivity of x + y*B for x in [0, X_max], y in [0, Y_max] is B > X_max.
        # Here x = d0+e0, y = d1+e1.
        # Max x = 2*max(D). Max y = 2*max(D).
        # So if B > 2*max(D), the representation is unique.
        # Thus |U+U| = |S0| * |S1| = |S0|^2 since S0=S1.
        
        # Similarly for U-U:
        # u-v = (d0-e0) + (d1-e1)*B
        # Let D0 = {d-e | d,e in D}, D1 = D0.
        # Value = d0_diff + d1_diff*B.
        # Range of d0_diff: [-maxD, maxD].
        # Range of d1_diff: [-maxD, maxD].
        # For injectivity of x + y*B with x in [-A, A], y in [-B_range, B_range]:
        # We need B > 2*maxD (the width of the lower digit range).
        # If B > 2*max(D), then |U-U| = |D0|^2.
        
        # So, if we enforce B > 2*max(D), the score calculation is O(|D|^2) to compute the sets of sums/diffs of D, 
        # which is much faster than O(|U|^2) where |U|=|D|^2.
        
        if B <= 2 * max_d + 1:
            # Fallback to exact set computation if base is too small (rare in our search)
            U = []
            for d0 in D_set:
                for d1 in D_set:
                    U.append(d0 + d1 * B)
            return score_set(U)
        
        # Compute sumset of D
        sum_D = set()
        diff_D = set()
        for a in D_set:
            for b in D_set:
                sum_D.add(a + b)
                diff_D.add(a - b)
        
        size_sum = len(sum_D)
        size_diff = len(diff_D)
        
        # |U+U| = size_sum^2
        # |U-U| = size_diff^2
        # Ratio = (size_diff / size_sum)^2
        
        ratio = (size_diff / size_sum) ** 2
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except:
            return 0.0
        return exponent

    def generate_set_from_D(D, B):
        """Generate the actual list of integers for U."""
        U = []
        D_set = set(D)
        for d0 in D_set:
            for d1 in D_set:
                U.append(d0 + d1 * B)
        return sorted(U)

    best_score = 0.0
    best_D = [0, 1]
    best_B = 10

    # Initial candidates
    # 1. Geometric progression (current best strategy)
    # Try to fit geometric progression into the D,B framework?
    # U = {b^k}. This is not a 2D grid structure.
    # We can run the old search briefly to get a baseline.
    
    def run_geometric_search():
        best_s = 0.0
        best_b = 1.5
        best_m = 10
        # Quick search
        for m in range(5, 20):
            for b in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
                U = []
                val = 1.0
                for k in range(m + 1):
                    U.append(int(val))
                    val *= b
                U = sorted(set(U))
                s = score_set(U)
                if s > best_s:
                    best_s = s
                    best_b = b
                    best_m = m
        return best_s, best_b, best_m

    if not check_time():
        s_geo, b_geo, m_geo = run_geometric_search()
        if s_geo > best_score:
            best_score = s_geo
            # We need to return a list. If we find a better geo set, we should remember it.
            # But our main strategy is D,B. Let's keep the geo result in a separate variable.
            # Actually, let's just use the D,B strategy. It's structurally different.
            # However, to be safe, if D,B fails to beat geo, we should return geo.
            # Let's store the best U explicitly.
            
    best_U = None
    
    if not check_time():
        s_geo, b_geo, m_geo = run_geometric_search()
        if s_geo > best_score:
            best_score = s_geo
            U_geo = []
            val = 1.0
            for k in range(m_geo + 1):
                U_geo.append(int(val))
                val *= b_geo
            best_U = sorted(set(U_geo))
    
    # 2. Structured D,B Search
    # We want to maximize 1 + 2*log(size_diff/size_sum)/log(2*(maxD + maxD*B)+1)
    # Let R = size_diff/size_sum.
    # We want R large.
    # For a set D, size_diff is at most 2*maxD+1 (if D is arithmetic progression).
    # size_sum is at most 2*maxD (if D is arithmetic progression).
    # Ratio for AP is ~1.
    # We want D to be "sum-closed" but "diff-spread"?
    # Actually, we want |D-D| large relative to |D+D|.
    # This happens if D is sparse?
    # If D is sparse, |D+D| ~ |D|^2, |D-D| ~ |D|^2. Ratio ~1.
    # If D is dense (AP), |D+D| ~ 2N, |D-D| ~ 2N. Ratio ~1.
    # Is it possible to get Ratio > 1?
    # Yes, if D has elements that sum to few values but difference to many.
    # Example: D = {0, 1, 3, 7, ...} (Mersenne-like)?
    # Let's search for D.
    
    def optimize_D_B():
        nonlocal best_score, best_U
        # Grid search for B and D structure
        # Let's try D as a subset of {0, ..., M}
        # We will use a random restart hill climbing on D.
        
        max_D_val = 100 # Max digit value
        max_B = 10000 # Max base
        
        for restart in range(20):
            if check_time():
                break
            
            # Initialize D randomly
            # Try different sizes of D
            size_D = random.randint(3, 15)
            # Initialize D as random subset of [0, max_D_val]
            D = set(random.sample(range(0, max_D_val + 1), size_D))
            if 0 not in D:
                D.add(0)
                D.discard(random.choice(list(D - {0}))) # Keep size roughly constant
            D = sorted(list(D))
            
            # Initialize B
            # B must be > 2*max(D)
            min_B = 2 * max(D) + 2
            B = min_B + random.randint(0, 100)
            
            current_score = score_structured(D, B)
            
            # Local search
            for step in range(100):
                if check_time():
                    break
                
                # Randomly modify D or B
                if random.random() < 0.5:
                    # Modify D
                    if random.random() < 0.5:
                        # Add element
                        if len(D) < 20:
                            new_d = random.randint(0, max_D_val + 10)
                            if new_d not in D:
                                D_new = D + [new_d]
                                s_new = score_structured(D_new, B)
                                if s_new > current_score or random.random() < 0.1: # Simulated annealing-like
                                    D = D_new
                                    current_score = s_new
                    else:
                        # Remove element (keep 0)
                        if len(D) > 2:
                            candidates = [d for d in D if d != 0]
                            if candidates:
                                rem_d = random.choice(candidates)
                                D_new = [d for d in D if d != rem_d]
                                s_new = score_structured(D_new, B)
                                if s_new > current_score or random.random() < 0.1:
                                    D = D_new
                                    current_score = s_new
                else:
                    # Modify B
                    # B must be > 2*max(D)
                    min_B = 2 * max(D) + 2
                    # Multiplicative change
                    B_new = int(B * random.uniform(0.8, 1.2))
                    if B_new < min_B:
                        B_new = min_B
                    # Cap B to avoid huge numbers? 
                    # log(2*max_U+1) grows slowly.
                    # But max_U = maxD*(1+B).
                    # If B is huge, log(B) is in denominator.
                    # Ratio is fixed by D.
                    # So we want B small enough to keep log(B) small, but large enough to avoid carries.
                    # So optimal B is close to min_B.
                    B_new = max(min_B, min(B_new, min_B + 1000))
                    
                    s_new = score_structured(D, B_new)
                    if s_new > current_score or random.random() < 0.1:
                        B = B_new
                        current_score = s_new
                
                # Update best
                if current_score > best_score:
                    best_score = current_score
                    best_U = generate_set_from_D(D, B)

    if not check_time():
        optimize_D_B()
        
    # Fallback
    if best_U is None:
        # Return a simple set
        best_U = [0, 1, 2]
        
    # Ensure 0 is in best_U
    if 0 not in best_U:
        best_U.insert(0, 0)
        
    return best_U

# EVOLVE-BLOCK-END
