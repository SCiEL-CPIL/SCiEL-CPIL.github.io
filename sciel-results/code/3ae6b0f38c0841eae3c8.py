import time
import random

def construct():
    start = time.time()
    deadline = start + 82.0

    def valid(S, x):
        # x must not be mean of any subset of S (size >=2)
        # and for each a in S, a must not be mean of any subset of S\{a} U {x}
        # For the second: subset T of (S\{a} U {x}) with |T|>=2 and mean a.
        # Either x in T or not.
        # If x not in T: already checked by previous iterations (a not mean of subset of S)
        # If x in T: sum(T) = |T|*a, T = U U {x} where U subset of S\{a}
        # sum(U) + x = |U|*a + a => sum(U) = (|U|+1)*a - x
        # |T| = |U|+1 >= 2 => |U| >= 1
        # Also need to check that a is not mean of {x} union subset of S\{a}
        
        n = len(S)
        if n < 2:
            # x can't be mean of subset of S with size >=2 if S has <2 elements
            # Check a in S: a mean of subset of S\{a} U {x}
            # For a in S (only if n>=1): need subset of {x} (size 1) - not >=2
            # So valid
            return True
        
        # Check x not mean of subset of S
        # sum(subset) = k*x, k>=2
        # DP: dp[k] = set of sums of k elements from S
        # Use bitset
        max_sum = sum(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        for e in S:
            for k in range(n, 0, -1):
                dp[k] |= (dp[k-1] << e)
        
        for k in range(2, n + 1):
            target = k * x
            if target <= max_sum and (dp[k] >> target) & 1:
                return False
        
        # Check each a in S: a not mean of subset of S\{a} U {x} that contains x
        # sum(U) = (j+1)*a - x where U subset of S\{a}, j = |U| >= 1
        # |T| = j+1 >= 2
        for a in S:
            others = [v for v in S if v != a]
            m = len(others)
            if m < 1:
                continue
            
            # DP for others
            dp2 = [0] * (m + 1)
            dp2[0] = 1
            for e in others:
                for k in range(m, 0, -1):
                    dp2[k] |= (dp2[k-1] << e)
            
            max_sum2 = sum(others)
            for j in range(1, m + 1):
                target = (j + 1) * a - x
                if 0 <= target <= max_sum2 and (dp2[j] >> target) & 1:
                    return False
        
        return True

    # Pre-filter: for each x, check if 2x can be expressed as sum of two distinct elements in [1,512]
    # This is a necessary condition for x to be mean of a pair. If 2x = a+b for distinct a,b in [1,512],
    # then x could potentially be mean of {a,b} if both are in our set.
    # But we're building the set, so this pre-filter is weak. Let's skip it.
    
    # Strategy: use a more structured search
    # Key insight: if we pick numbers that are "spread out", it's harder to form means.
    # Try: pick from a geometric-like sequence, or use modular constraints.
    
    # Let's try a different approach: build the set by picking numbers that are
    # far from being means of existing elements.
    
    best_set = []
    best_len = 0
    
    # Try multiple strategies
    # Strategy 1: Greedy from large to small, with randomization
    random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        if trial == 0:
            # Deterministic descending
            candidates = list(range(512, 0, -1))
        else:
            # Randomized with bias
            candidates = list(range(1, 513))
            # Weight larger numbers more
            weights = [c for c in candidates]
            random.shuffle(candidates)
        
        current = []
        for x in candidates:
            if time.time() > deadline:
                break
            if valid(current, x):
                current.append(x)
        
        if len(current) > best_len:
            best_len = len(current)
            best_set = current[:]
        
        if best_len >= 48:
            break
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(result[:10], "...", result[-10:])
