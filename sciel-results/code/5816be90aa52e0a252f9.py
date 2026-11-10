import time
import sys
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    def is_valid_dp(S):
        """
        Check if subset sums of different cardinalities are disjoint.
        Uses DP to track achievable sums for each cardinality.
        """
        n = len(S)
        max_sum = sum(S)
        # dp[k][s] is True if sum s is achievable with k elements
        # We use a list of sets for efficiency if sums are sparse, 
        # but for n=32, max_sum ~ 8000, bitsets or boolean arrays are better.
        # Using list of sets of integers.
        dp = [set() for _ in range(n + 1)]
        dp[0].add(0)
        
        for x in S:
            # Iterate backwards to avoid using x multiple times in same step
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    new_sums = {s + x for s in dp[k]}
                    dp[k + 1].update(new_sums)
        
        # Check for collisions between different cardinalities
        # We need to ensure that for any s, it appears in at most one dp[k]
        # Actually, we can just check if any sum appears in dp[k] and dp[j] for k != j.
        # Efficient way: keep a dict sum -> list of k's.
        # Or simply: since we just need validity, we can check intersections.
        # But intersections of sets of size up to binom(n, k/2) are expensive.
        
        # Alternative: The condition "subset sums with different cardinalities are distinct"
        # is equivalent to: for any s, the set of k such that s in dp[k] has size <= 1.
        
        # Let's collect all sums and their cardinalities.
        sum_to_k = {}
        valid = True
        for k in range(n + 1):
            for s in dp[k]:
                if s in sum_to_k:
                    if sum_to_k[s] != k:
                        valid = False
                        break
                else:
                    sum_to_k[s] = k
            if not valid:
                break
        return valid

    def try_find_valid_set(target_size):
        """
        Try to find a valid set of given size.
        """
        # Strategy: Start with AP near the top, try to compress or adjust.
        # For size n, max element 256.
        # AP: a, a+1, ..., a+n-1.
        # a + n - 1 <= 256 => a <= 257 - n.
        # Try AP first.
        max_a = 257 - target_size
        if max_a < 1:
            return None
        
        # Check AP
        S_ap = list(range(max_a, max_a + target_size))
        if is_valid_dp(S_ap):
            return S_ap
        
        # If AP fails, try to find a set satisfying M_{k+1} > N_k using LP.
        # Variables x_1 ... x_n
        # Constraints:
        # x_{i+1} - x_i >= 1
        # x_n <= 256
        # x_1 >= 1
        # D_k >= 1 for all k
        
        c = [0] * target_size
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        bounds = [(1, 256) for _ in range(target_size)]
        
        # x_{i+1} - x_i >= 1  =>  x_i - x_{i+1} <= -1
        for i in range(target_size - 1):
            row = [0] * target_size
            row[i] = 1
            row[i+1] = -1
            A_ub.append(row)
            b_ub.append(-1)
        
        # D_k >= 1 => -D_k <= -1
        # D_k = sum_{i=1}^{k+1} x_i - sum_{i=n-k}^{n} x_i
        # Note: indices in 0-based python: 0..k and n-k..n-1
        for k in range(1, target_size):
            row = [0] * target_size
            # Add 1 to x_0 ... x_k
            for i in range(k + 1):
                row[i] = 1
            # Subtract 1 from x_{n-k} ... x_{n-1}
            for i in range(target_size - k, target_size):
                row[i] = -1
            # We want D_k >= 1 => -D_k <= -1
            # So we add -row to A_ub and -1 to b_ub?
            # linprog minimizes c^T x subject to A_ub x <= b_ub.
            # We want -D_k <= -1.
            A_ub.append([-v for v in row])
            b_ub.append(-1)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.status == 0:
            # Round the solution
            S_lp = [int(round(x)) for x in res.x]
            # Ensure distinct and in range
            # Rounding might break distinctness or bounds
            # Sort and check
            S_lp.sort()
            if S_lp[-1] > 256 or S_lp[0] < 1:
                return None
            if len(set(S_lp)) < target_size:
                return None
            
            if is_valid_dp(S_lp):
                return S_lp
            
            # If rounding failed validity, try to repair
            # Simple repair: if invalid, try shifting or small perturbations
            # For now, just return None if not immediately valid
            return None
        
        return None

    # Try sizes from 60 down to 31
    # But 60 is impossible. Max possible is likely small.
    # We know 31 is possible.
    # Try 32, 33, ... up to some limit?
    # LP for large n might be slow or infeasible.
    # Let's try 32 first.
    
    for size in range(32, 31, -1): # Just 32 for now?
        if time.time() > deadline:
            break
        S = try_find_valid_set(size)
        if S:
            return S
            
    # Fallback to 31
    S_31 = list(range(226, 257))
    return S_31
