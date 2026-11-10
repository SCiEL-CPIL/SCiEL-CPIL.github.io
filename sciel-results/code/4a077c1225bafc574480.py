"""
Strategy: Backtracking Search with Constraint Propagation
Instead of greedy construction, we use a systematic search to find a valid set 
of size K. We iterate K from 48 downwards. For each K, we attempt to find a 
valid subset of size K using backtracking. If a valid set is found, we return it.
This explores the solution space more thoroughly than greedy, potentially finding 
larger sets that greedy misses due to early suboptimal choices.
"""
import time
import sys

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 55.0  # Safety margin for 60s limit

def _check_candidate_vs_set(candidate, selected):
    """
    Check if 'candidate' can be added to 'selected' without violating constraints.
    1. candidate must not divide sum of any 3 distinct elements in selected.
    2. For each a in selected, a must not divide (candidate + x + y) for any 
       distinct x, y in selected \ {a}.
    """
    n = len(selected)
    
    # 1. Check candidate against existing triples in selected
    if n >= 3:
        # Optimization: If candidate is large, it's unlikely to divide small sums?
        # Sum of 3 distinct positives is at least 1+2+3=6.
        # If candidate > max_sum, it can't divide the sum (unless sum is 0, which is impossible).
        # Max sum of 3 from selected <= 192*3 = 576.
        # So if candidate > 576, check 1 is always false (safe).
        # But candidate <= 192, so this doesn't help much directly.
        
        # We need to check if any triple sum is a multiple of candidate.
        # Sums are in range [1+2+3, 190+191+192] approx [6, 573].
        # Multiples of candidate in this range: candidate, 2*candidate, ...
        # We can iterate through possible multiples and check if any triple sums to it.
        # Or just iterate triples. n is small (<=48). n^3/6 ~ 18k ops.
        # Let's just iterate triples for simplicity and early exit.
        sel = selected
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    s = sel[i] + sel[j] + sel[k]
                    if s % candidate == 0:
                        return False
    
    # 2. Check existing 'a' against new sums involving candidate
    # For each a in selected:
    #   Check if a divides (candidate + x + y) for distinct x, y in selected \ {a}
    # (candidate + x + y) % a == 0  =>  (x + y) % a == (-candidate) % a
    
    for idx_a, a in enumerate(selected):
        target = (-candidate) % a
        # We need distinct x, y from selected \ {a} such that (x+y) % a == target
        # And x != y.
        
        # Optimization:
        # If a is large, the modulo space is large, so collisions are rare?
        # Actually, we can just check pairs.
        # Number of pairs in selected \ {a} is (n-1)*(n-2)/2.
        # Total ops ~ n * n^2 = n^3.
        
        # To speed up, we can precompute residues?
        # But 'a' changes.
        
        # Let's use a set of residues found so far?
        # For a fixed 'a', we want to know if there exist x, y in S' (S' = selected \ {a})
        # such that (x+y) % a == target.
        
        # We can iterate x in S'. For each x, we need y in S' \ {x} such that
        # y % a == (target - x) % a.
        
        # Let's create a list of elements in S' with their residues mod a.
        # Since n is small, we can just do a double loop.
        # But we can optimize by breaking early.
        
        for i in range(n):
            if i == idx_a:
                continue
            x = selected[i]
            needed_residue = (target - x) % a
            
            for j in range(n):
                if j == idx_a or j == i:
                    continue
                y = selected[j]
                if y % a == needed_residue:
                    return False
                    
    return True

def _backtrack_find_set(K, start_idx, current_set, candidates, deadline):
    """
    Recursively find a valid set of size K from candidates[start_idx:].
    Returns True if found, and current_set is modified to contain the set.
    """
    if time.time() > deadline:
        return False
        
    if len(current_set) == K:
        return True
        
    # Pruning: if not enough candidates left to reach K
    if len(candidates) - start_idx < K - len(current_set):
        return False
        
    for i in range(start_idx, len(candidates)):
        c = candidates[i]
        
        # Try including c
        if _check_candidate_vs_set(c, current_set):
            current_set.append(c)
            if _backtrack_find_set(K, i + 1, current_set, candidates, deadline):
                return True
            current_set.pop()
            
        # Pruning: if we skip c, can we still reach K?
        # We need K - len(current_set) more from candidates[i+1:]
        # if len(candidates) - (i + 1) < K - len(current_set):
        #     return False # This is already handled by the loop structure mostly, 
        #                  # but explicit check helps if we break early.
        
    return False

def construct():
    start_time = time.time()
    candidates = list(range(1, 193))
    
    # We want to maximize cardinality.
    # Try K from 48 down to 1.
    # Note: Backtracking can be slow. We allocate a time budget for each K.
    # If K=48 takes too long, we move to K=47.
    
    # Heuristic: Larger K is harder. 
    # We will try a few random restarts for each K if the first deterministic order fails?
    # Actually, the order of candidates matters.
    # Smaller numbers are more "constraining" (they divide more things).
    # It might be better to decide on small numbers first?
    # If we include 1, it divides everything. So 1 can NEVER be in the set?
    # Wait. "No selected integer may divide the sum..."
    # If 1 is selected, it divides ANY sum. So if there are 3 other selected integers, 
    # their sum is divisible by 1. 
    # So if |S| >= 4, 1 cannot be in S.
    # If |S| <= 3, there are no 3 DISTINCT OTHER integers. So 1 is safe if |S|<=3.
    # Since we want K up to 48, 1 is likely excluded.
    
    # Similarly, 2 divides any even sum.
    
    # Let's stick to the standard order 1..192 for the backtracking variable assignment.
    # But we can sort candidates by "constraint power" or just use natural order.
    
    best_set = []
    
    # Time management:
    # We have ~55 seconds.
    # We don't know how long each K takes.
    # Let's try K=48 with a 10s limit. If fail, K=47 with 10s limit, etc.
    # This might waste time on high K if they are impossible.
    # However, finding a valid set of size 48 is the goal.
    
    # Alternative: Use a greedy result as a lower bound?
    # The seed program finds some set. Let's assume we can find at least 10-20 easily.
    # But we want to beat 1.0 (which is the seed's score).
    # The seed's score is 1.0. The "value" is the cardinality.
    # If the seed finds a set of size S_seed, score = S_seed / S_seed = 1.0?
    # No, "search score is value/seed". 
    # Usually "seed" refers to the baseline value. 
    # If the baseline program returns a set of size N, the score is 1.0.
    # If we return a set of size N+1, score is (N+1)/N > 1.0.
    
    # So we just need to find a VALID set with MAXIMUM cardinality.
    
    # Let's run backtracking for K from 48 down.
    # We will use a global deadline.
    
    # To improve speed, we can pre-filter candidates.
    # As noted, 1 is very restrictive. 
    # In backtracking, if we try to include 1, and we already have 3 others, we fail.
    # So 1 will be quickly pruned if we have 3 others.
    
    # Let's try K=48.
    # We will use a time slice for each K attempt.
    # If we run out of time, we return the best set found so far (which might be empty if we haven't finished K=48).
    # To ensure we return *something* valid, we should keep a fallback greedy set.
    
    # Fallback: Greedy from 192 down to 1.
    def _greedy_fallback():
        selected = []
        for c in range(192, 0, -1):
            if _check_candidate_vs_set(c, selected):
                selected.append(c)
        return selected
    
    fallback_set = _greedy_fallback()
    best_set = fallback_set
    best_K = len(fallback_set)
    
    # Now try to improve with backtracking.
    # We try K from 48 down to best_K + 1.
    
    # To make backtracking faster, we can shuffle candidates? 
    # No, natural order is deterministic. 
    # However, the choice of which candidates to include first matters.
    # If we put large numbers first, they are less likely to divide sums of smaller numbers?
    # Actually, large numbers are less likely to be DIVISORS of sums (since sums are bounded).
    # Small numbers are more likely to be divisors.
    # So, deciding on small numbers first might prune the tree earlier?
    # If we decide to INCLUDE a small number, we constrain many others.
    # If we decide to EXCLUDE a small number, we free up others.
    # Standard backtracking tries Include first.
    # If we include 1, we fail immediately if we have 3 others.
    # So for K>=4, 1 will be excluded quickly.
    
    # Let's try K values.
    # We allocate time dynamically.
    time_per_K = 5.0 # seconds per K attempt
    
    for K in range(48, best_K, -1):
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        deadline_for_K = time.time() + time_per_K
        current_set = []
        
        # Try to find a set of size K
        if _backtrack_find_set(K, 0, current_set, candidates, deadline_for_K):
            best_set = current_set
            best_K = K
            # If we found K, we can try K+1? No, we are going down.
            # If we found 48, we are done (max possible).
            if K == 48:
                break
                
            # If we found a smaller K, we can continue to see if a larger one was missed?
            # No, we are iterating down from 48. If 48 failed (timeout or no solution), we try 47.
            # If 47 succeeds, we have a set of 47. We don't need to check 48 again unless we had more time.
            # But wait, if 48 timed out, it might have a solution. 
            # We should probably stick to the first valid K found from the top.
            
    return best_set

# EVOLVE-BLOCK-END
