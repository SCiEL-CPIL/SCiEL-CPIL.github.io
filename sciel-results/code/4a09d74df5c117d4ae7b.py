import time

def construct():
    deadline = time.time() + 85.0  # 85s safety margin
    
    selected = []
    
    # Try candidates in an order that might yield more elements
    # Start with smaller numbers, but skip if invalid
    # Actually, let's try a different order: interleave or use a heuristic
    
    # Greedy: try each number from 1 to 512, add if valid
    # But the seed already does 1..128. Let's try 1..512 but with faster check.
    
    def is_valid(candidate_set, new_elem):
        """Check if adding new_elem to candidate_set keeps the property."""
        full_set = candidate_set + [new_elem]
        n = len(full_set)
        
        # For each element a in full_set, check if a is the mean of any subset of size >=2 from full_set \ {a}
        for a in full_set:
            others = [x for x in full_set if x != a]
            m = len(others)
            if m < 2:
                continue
            
            # DP: dp[k][s] = True if we can pick k elements from others summing to s
            # Max sum needed: for k elements, max sum is k * max(others)
            # We need to check if dp[k][k*a] is True for any k in [2, m]
            
            max_sum = sum(others)
            # But we only care about sums up to k*a for various k
            # Let's do DP up to max needed sum
            max_target = max(k * a for k in range(2, m + 1))
            if max_target > max_sum:
                max_target = max_sum
            
            # dp[k] = set of reachable sums with k elements
            # Use a list of sets or bitsets
            # Since m <= 48 and sums <= 24576, use list of sets
            dp = [set() for _ in range(m + 1)]
            dp[0].add(0)
            
            for x in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        for s in list(dp[k]):
                            new_s = s + x
                            if new_s <= max_target:
                                dp[k + 1].add(new_s)
            
            for k in range(2, m + 1):
                if k * a in dp[k]:
                    return False
        
        return True
    
    # Try adding numbers in order 1 to 512
    for x in range(1, 513):
        if time.time() > deadline:
            break
        if is_valid(selected, x):
            selected.append(x)
            if len(selected) >= 48:
                break
    
    return selected
