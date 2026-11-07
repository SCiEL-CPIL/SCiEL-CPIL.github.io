"""Construct a large set with cardinality-separated subset sums."""
import time
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Start with the known valid set of 8 (powers of 2 scaled)
    best = [128, 136, 144, 152, 160, 168, 176, 184]
    best_size = len(best)
    
    # Try to build larger sets using greedy + verification
    # Strategy: use elements that are multiples of a large base to separate cardinality ranges
    # If all elements are in [B, B+R] where B >> R*k_max, then sum ranges for
    # cardinality k and k+1 don't overlap.
    
    # For elements in [a, a+d] with d small relative to a:
    # Sum range for cardinality k: [k*a, k*(a+d)]
    # Sum range for cardinality k+1: [(k+1)*a, (k+1)*(a+d)]
    # Need: k*(a+d) < (k+1)*a => k*d < a => d < a/k
    # For max cardinality K: d < a/K
    
    # Let's try: elements = base + offsets where offsets are small
    # If base = 200, max K = 60, then d < 200/60 ≈ 3.3, so d <= 3
    # That gives at most 4 distinct offsets -> only 4 elements. Not enough.
    
    # Better approach: use a geometric/spread sequence
    # Elements: e_i such that the minimum sum of (k+1) smallest > maximum sum of k largest
    # min_sum(k+1) = sum of (k+1) smallest elements
    # max_sum(k) = sum of k largest elements
    # Need: sum of (k+1) smallest > sum of k largest for all k >= 0
    
    # This is satisfied if elements grow fast enough.
    # For a set sorted a_1 < a_2 < ... < a_n:
    # Need: a_1 + ... + a_{k+1} > a_{n-k+1} + ... + a_n for all k
    
    # Simple sufficient condition: a_{i+1} > a_i + a_{i-1} + ... + a_1 (Fibonacci-like)
    # But we're bounded by 256, so let's compute what's achievable.
    
    # Let's try a constructive approach:
    # Pick elements greedily to maximize size while maintaining separation.
    
    def is_valid(elements):
        """Check if a set of elements has cardinality-separated subset sums."""
        n = len(elements)
        if n == 0:
            return True
        
        # For each pair of cardinalities (i, j) with i != j, check no common sum
        # Using bitset DP for efficiency
        # Max possible sum: n * 256 = 60 * 256 = 15360
        
        max_sum = sum(elements)
        
        # dp[k] = set of achievable sums with exactly k elements
        # Use bitsets (Python ints) for efficiency
        # dp[k] is an integer where bit s is set if sum s is achievable with k elements
        
        dp = [0] * (n + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for e in elements:
            # Update in reverse to avoid using e multiple times
            for k in range(n, 0, -1):
                # dp[k] |= dp[k-1] shifted by e
                dp[k] |= (dp[k-1] << e)
        
        # Check: for each pair (i, j) with i < j, dp[i] & dp[j] should be 0
        # (excluding sum 0 which only appears in dp[0])
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if dp[i] & dp[j]:
                    return False
        return True
    
    def try_greedy(base_elements, candidates):
        """Try to add elements from candidates to base_elements while maintaining validity."""
        current = list(base_elements)
        for c in candidates:
            if c in current:
                continue
            new_set = current + [c]
            if is_valid(new_set):
                current = new_set
        return current
    
    # Strategy 1: Start with a geometric sequence and try to fill in
    # Elements: 1, 2, 4, 8, 16, 32, 64, 128 -> 8 elements, all valid
    # Can we add more? 128+64+32+16+8+4+2+1 = 255, so next element would need > 255.
    # So powers of 2 give exactly 8.
    
    # Strategy 2: Use a less aggressive growth to fit more elements
    # Let's try: elements where each is slightly more than the sum of all previous
    # a_1 = 1, a_2 > 1, a_3 > 1+a_2, etc.
    # This gives Fibonacci-like growth, which is slower than powers of 2.
    
    # Actually, the condition is weaker: we just need separation between cardinality sums,
    # not unique subset sums. Let me reconsider.
    
    # Key insight: if all elements are in [L, U] with U < L/2 (approximately),
    # then sum of k elements is in [kL, kU] and sum of (k+1) elements is in [(k+1)L, (k+1)U].
    # These ranges don't overlap if kU < (k+1)L, i.e., U/L < (k+1)/k = 1 + 1/k.
    # For the worst case k=1: U/L < 2, i.e., U < 2L.
    # So if all elements are in [L, 2L), cardinality ranges are separated!
    
    # With elements in [L, 2L), we can have at most 2L - L = L distinct values.
    # Max element is 256, so L can be up to 128, giving elements in [128, 255].
    # That's 128 distinct values! But we need them to be integers in 1..256.
    
    # Wait, but we also need the ranges to not overlap. Let me verify:
    # If elements are in [L, U] with U < 2L:
    # Sum of k elements: [kL, kU]
    # Sum of (k+1) elements: [(k+1)L, (k+1)U]
    # Need kU < (k+1)L for all k >= 1
    # Worst case: k=1: U < 2L ✓ (given)
    # For larger k: kU < kL + L => k(U-L) < L => U-L < L/k <= L
    # Since U-L < U < 2L, and L/k <= L, this holds when U-L < L/k.
    # For k=1: U-L < L => U < 2L ✓
    # For k=2: U-L < L/2
    # ...
    # For k=n: U-L < L/n
    
    # So the tightest constraint is for large k. If we have n elements all in [L, U],
    # we need U-L < L/n, i.e., U < L(1 + 1/n).
    
    # For n=60: U < L * 61/60, so the range width is L/60.
    # If L=128: width = 128/60 ≈ 2.13, so at most 3 distinct values. Not enough.
    
    # This approach only works for small n. Let me think differently.
    
    # Better: use elements that are spread out. The condition is:
    # For any k < m: max sum of k elements < min sum of m elements
    # i.e., sum of k largest < sum of m smallest
    
    # Let's just do a search: try different element sets and verify.
    
    # Approach: use a parametric family and search for good parameters.
    # Family: elements = floor(base * r^i) for i=0..n-1, with r slightly > 1.
    
    # Or: elements are arithmetic progression with large gap.
    # a, a+d, a+2d, ..., a+(n-1)d where d is large enough.
    
    # For AP: elements a, a+d, ..., a+(n-1)d
    # Sum of k largest: k*(a+(n-1)d) - d*k*(k-1)/2
    # Sum of (k+1) smallest: (k+1)*a + d*(k+1)*k/2
    # Need: k*(a+(n-1)d) - d*k*(k-1)/2 < (k+1)*a + d*(k+1)*k/2
    # This is complex. Let's just try it empirically.
    
    # Let's try: start with a base and step, generate candidates, verify.
    
    random.seed(42)
    
    # Try various (start, step, count) combinations
    best_found = list(best)
    
    # First, try the simple valid set of 8
    # Then try to extend or find better
    
    # Try powers of 2: 1,2,4,8,16,32,64,128 -> valid, size 8
    # Try 2,4,8,...,128 -> 7 elements
    # Try 1,3,9,27,81,243 -> 6 elements (geometric r=3)
    
    # Let's try to find the maximum size by searching over structured families.
    
    # Family 1: Geometric sequence a*r^i
    for a in range(1, 30):
        for r_num in range(2, 10):
            for r_den in range(1, r_num):
                r = r_num / r_den
                elements = []
                val = float(a)
                while val <= 256:
                    iv = int(round(val))
                    if 1 <= iv <= 256 and iv not in elements:
                        elements.append(iv)
                    val *= r
                if len(elements) > len(best_found):
                    if is_valid(elements):
                        best_found = list(elements)
    
    # Family 2: Arithmetic progression
    for start in range(1, 50):
        for step in range(1, 20):
            elements = []
            val = start
            while val <= 256:
                elements.append(val)
                val += step
            if len(elements) > len(best_found):
                if is_valid(elements):
                    best_found = list(elements)
    
    # Family 3: Mix - start with large gap, then smaller gaps
    # e.g., 1, 10, 20, 30, 40, ... 
    for start in range(1, 20):
        for step in range(5, 30):
            elements = [start]
            val = start
            while val + step <= 256:
                val += step
                elements.append(val)
            # Try adding some in between
            if len(elements) > len(best_found):
                if is_valid(elements):
                    best_found = list(elements)
    
    # Try a more systematic approach: greedy with verification
    # Start with smallest possible, add next element that maintains validity
    def greedy_construct(start_val, end_val, candidates_order):
        current = []
        for c in candidates_order:
            if c < start_val or c > end_val:
                continue
            if c in current:
                continue
            new_set = current + [c]
            if is_valid(new_set):
                current = new_set
        return current
    
    # Try greedy from different starting points
    for start in [1, 2, 5, 10, 20, 50]:
        # Candidates in increasing order
        candidates = list(range(start, 257))
        result = greedy_construct(start, 256, candidates)
        if len(result) > len(best_found):
            best_found = result
        
        # Candidates in decreasing order
        candidates = list(range(256, start - 1, -1))
        result = greedy_construct(start, 256, candidates)
        if len(result) > len(best_found):
            best_found = result
    
    # Try random restarts
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        # Random candidate order
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        result = greedy_construct(1, 256, candidates)
        if len(result) > len(best_found):
            best_found = result
    
    # Try to extend the best found set
    if time.time() < deadline - 5:
        extended = try_greedy(best_found, list(range(1, 257)))
        if len(extended) > len(best_found):
            best_found = extended
    
    # Ensure we return at least the known valid set
    if len(best_found) < 8:
        best_found = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Final verification
    if not is_valid(best_found):
        # Fallback to known valid
        best_found = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Ensure all are Python ints
    return [int(x) for x in best_found]


# Test at module level to ensure it works
if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}: {result}")
    # Verify
    n = len(result)
    max_sum = sum(result)
    dp = [0] * (n + 1)
    dp[0] = 1
    for e in result:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k-1] << e)
    valid = True
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if dp[i] & dp[j]:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
