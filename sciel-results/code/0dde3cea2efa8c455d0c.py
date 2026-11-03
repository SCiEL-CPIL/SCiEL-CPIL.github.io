"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START
def _score_set(U):
    """Compute the score for a given set U."""
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute |U+U| and |U-U|
    # For efficiency with small sets, use sets
    sum_set = set()
    diff_set = set()
    for i in range(n):
        for j in range(n):
            sum_set.add(U[i] + U[j])
            diff_set.add(U[i] - U[j])
    num = len(diff_set)
    den = len(sum_set)
    if den == 0:
        return 0.0
    log_ratio = np.log(num / den)
    log_max = np.log(2 * max_u + 1)
    if log_max == 0:
        return 0.0
    return 1.0 + log_ratio / log_max

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds, with a safety margin
    # We know [0, 1] gives a high score. Let's see if we can do better.
    # The score is 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    # For U = [0, 1], |U-U| = 3, |U+U| = 3, ratio = 1, log(1) = 0. Score = 1.0.
    # Wait, my previous analysis was wrong. Let's re-evaluate.
    # U = [0, 1]:
    # Sums: 0, 1, 1, 2 -> {0, 1, 2} -> |U+U| = 3
    # Diffs: 0, -1, 1, 0 -> {0, -1, 1} -> |U-U| = 3
    # Ratio = 1. Score = 1.0.
    
    # U = [0, 1, 3]:
    # Sums: 0,1,3, 1,2,4, 3,4,6 -> {0,1,2,3,4,6} -> |U+U| = 6
    # Diffs: 0,-1,-3, 1,0,-2, 3,2,0 -> {0,1,2,3,-1,-2,-3} -> |U-U| = 7
    # Ratio = 7/6. Score = 1 + log(7/6)/log(7) = 1 + 0.154/1.946 = 1.079.
    
    # U = [0, 1, 4]:
    # Sums: 0,1,4, 1,2,5, 4,5,8 -> {0,1,2,4,5,8} -> |U+U| = 6
    # Diffs: 0,-1,-4, 1,0,-3, 4,3,0 -> {0,1,3,4,-1,-3,-4} -> |U-U| = 7
    # Ratio = 7/6. Max U = 4. Score = 1 + log(7/6)/log(9) = 1 + 0.154/2.197 = 1.070.
    # So [0,1,3] is better than [0,1,4].
    
    # What about U = [0, 2, 3]?
    # Sums: 0,2,3, 2,4,5, 3,5,6 -> {0,2,3,4,5,6} -> |U+U| = 6
    # Diffs: 0,-2,-3, 2,0,-1, 3,1,0 -> {0,1,2,3,-1,-2,-3} -> |U-U| = 7
    # Ratio = 7/6. Max U = 3. Score = 1 + log(7/6)/log(7) = 1.079. Same as [0,1,3].
    
    # It seems the score depends on the ratio and the max element.
    # To maximize score, we want a high ratio (close to 2(n-1)+1 / (2n-1) for Sidon sets) and a small max element.
    # Sidon sets (B2 sequences) have |U+U| = n(n+1)/2.
    # For n=3, max ratio is 7/6.
    # For n=4, max ratio is (2*3+1)/(2*4-1) = 7/7 = 1? No.
    # Max |U-U| is 2n(n-1)+1.
    # Min |U+U| is 2n-1 (for AP).
    # For AP, ratio is 1.
    # For Sidon, |U+U| = n(n+1)/2.
    # Ratio = (2n^2-2n+1) / (n^2+n/2) approx 2 for large n.
    # But max U grows quickly.
    
    # Let's search for small sets.
    # n=2: Max ratio 1. Score 1.0.
    # n=3: Max ratio 7/6. Min max U for ratio 7/6 is 3 ([0,1,3]). Score 1.079.
    # n=4: Can we get a good ratio with small max U?
    # Try [0,1,3,7]:
    # Sums:
    # 0,1,3,7
    # 1,2,4,8
    # 3,4,6,10
    # 7,8,10,14
    # Unique sums: 0,1,2,3,4,6,7,8,10,14. Size 10.
    # Diffs:
    # 0,-1,-3,-7
    # 1,0,-2,-6
    # 3,2,0,-4
    # 7,6,4,0
    # Unique diffs: 0,1,2,3,4,6,7,-1,-2,-3,-4,-6,-7. Size 13.
    # Ratio = 13/10 = 1.3.
    # Max U = 7. Score = 1 + log(1.3)/log(15) = 1 + 0.262/2.708 = 1.097.
    # This is better than 1.079!
    
    # Let's try to find an even better set with n=4 or n=5.
    # We want to maximize log(ratio) / log(2*max+1).
    # This is equivalent to maximizing ratio^(1/log(2*max+1)) = (2*max+1)^(log(ratio)/log(2*max+1))... no.
    # Let's just search.
    
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    # Search n=4
    # Max element M. We can try M up to 10 or so.
    # Combinations of 3 other elements from 1..M.
    # To keep it fast, we can iterate M and use combinations.
    
    def search_n(n, max_m, time_limit):
        nonlocal best_set, best_score
        if n == 1:
            return
        # We need to pick n-1 elements from 1..max_m
        # This is C(max_m, n-1).
        # For n=4, max_m=10, C(10,3)=120. Fast.
        # For n=5, max_m=10, C(10,4)=210. Fast.
        # For n=6, max_m=10, C(10,5)=252. Fast.
        
        for m in range(n-1, max_m + 1):
            # Elements are 0 and n-1 others from 1..m
            # To ensure 0 is the min, we pick from 1..m.
            # Actually, we can just pick n-1 elements from 1..m, and add 0.
            # But the set doesn't have to contain 0 as the min?
            # The problem says "containing 0". It doesn't say 0 is the min.
            # But nonnegative integers. So min is 0.
            # So we pick n-1 distinct elements from 1..m.
            for combo in itertools.combinations(range(1, m+1), n-1):
                if time.time() > time_limit:
                    return
                U = [0] + list(combo)
                s = _score_set(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    # Search n=4,5,6 with small max_m
    # n=4, max_m=15
    search_n(4, 15, deadline - 10)
    # n=5, max_m=10
    if time.time() < deadline - 20:
        search_n(5, 10, deadline - 10)
    # n=6, max_m=8
    if time.time() < deadline - 30:
        search_n(6, 8, deadline - 10)
        
    # Let's also try specific known good sets or variations.
    # [0, 1, 3, 7] was 1.097.
    # What about [0, 1, 4, 6]?
    # Sums:
    # 0,1,4,6
    # 1,2,5,7
    # 4,5,8,10
    # 6,7,10,12
    # Unique: 0,1,2,4,5,6,7,8,10,12. Size 10.
    # Diffs:
    # 0,-1,-4,-6
    # 1,0,-3,-5
    # 4,3,0,-2
    # 6,5,2,0
    # Unique: 0,1,2,3,4,5,6,-1,-2,-3,-4,-5,-6. Size 13.
    # Ratio 13/10 = 1.3. Max U=6.
    # Score = 1 + log(1.3)/log(13) = 1 + 0.262/2.565 = 1.102.
    # Better!
    
    # It seems we should update the best set with these.
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 3, 7],
        [0, 1, 4, 5],
        [0, 1, 4, 7],
        [0, 1, 3, 4],
    ]
    for c in candidates:
        s = _score_set(c)
        if s > best_score:
            best_score = s
            best_set = c
            
    return best_set
# EVOLVE-BLOCK-END
