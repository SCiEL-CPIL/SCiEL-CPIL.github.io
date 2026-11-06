"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Greedy approach: try to add numbers in a strategic order
    # Start with numbers that are "far apart" to avoid mean conditions
    
    selected = []
    
    # First, add powers of 2 - they have good separation properties
    powers_of_2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for x in powers_of_2:
        if time.time() > deadline:
            break
        if _is_valid(selected, x):
            selected.append(x)
    
    # Now try to add other numbers, prioritizing those that are less likely to be means
    # Sort candidates by "risk" - numbers in the middle are more likely to be means
    candidates = [x for x in range(1, 513) if x not in selected]
    
    # Prioritize numbers that are far from existing elements
    def risk_score(x, sel):
        if not sel:
            return 0
        # Count how many pairs/triples could potentially average to x
        risk = 0
        for i in range(len(sel)):
            for j in range(i+1, len(sel)):
                if (sel[i] + sel[j]) // 2 == x:
                    risk += 1
        return risk
    
    candidates.sort(key=lambda x: risk_score(x, selected))
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _is_valid(selected, x):
            selected.append(x)
    
    return selected

def _is_valid(selected, new_elem):
    """Check if adding new_elem to selected maintains the no-mean property."""
    candidate = selected + [new_elem]
    n = len(candidate)
    
    for a in candidate:
        # Check if a is the mean of any subset of other elements
        others = [x for x in candidate if x != a]
        m = len(others)
        
        # Use bitset DP to check all subset sizes
        # For each target sum = k * a, check if it's achievable with exactly k elements
        # from 'others'
        
        # We need to check for k from 2 to m
        # DP: dp[k][s] = can we pick k elements with sum s?
        # But s can be up to 512 * 48 = 24576, and k up to 48
        # This is too large for full DP. Instead, use a different approach.
        
        # Alternative: For each k, check if sum = k*a is achievable
        # Using meet-in-the-middle or clever DP
        
        # For small m, we can use combinations, but for larger m we need DP
        if m <= 20:
            # Use combinations for small sets
            from itertools import combinations
            for k in range(2, m + 1):
                target = k * a
                for combo in combinations(others, k):
                    if sum(combo) == target:
                        return False
        else:
            # Use DP with bitsets
            # dp[k] is a bitset where bit s is set if we can achieve sum s with k elements
            # Max sum is 512 * 48 = 24576
            max_sum = 512 * 48
            dp = [0] * (m + 1)
            dp[0] = 1  # bit 0 set: sum 0 with 0 elements
            
            for elem in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        # Shift dp[k] by elem and OR into dp[k+1]
                        dp[k+1] |= (dp[k] << elem)
            
            # Check if for any k >= 2, bit (k*a) is set in dp[k]
            for k in range(2, m + 1):
                target = k * a
                if target <= max_sum and (dp[k] >> target) & 1:
                    return False
    
    return True
# EVOLVE-BLOCK-END
