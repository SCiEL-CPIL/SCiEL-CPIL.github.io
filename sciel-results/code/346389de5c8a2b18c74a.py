"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _is_valid(set_arr):
    """Check no element is the mean of any subset of >=2 OTHER elements.
    Uses DP: for each target a, check if any subset of others (size>=2) sums to len(a)*a.
    Returns True if valid.
    """
    n = len(set_arr)
    if n < 3:
        return True
    # Sort for determinism
    arr = np.sort(set_arr)
    max_sum = int(arr.sum())
    for i in range(n):
        a = int(arr[i])
        target_sum = a * (n - 1)  # sum of all others
        others = np.concatenate([arr[:i], arr[i+1:]])
        m = n - 1
        # DP: reachable sums with at least 2 elements from others
        # dp[s] = bitset of subset sizes (bit k = size k achievable)
        # We only need to know if target_sum is reachable with size >= 2
        # Use a 2D-ish approach: dp[s] = bitmask of achievable sizes
        # But target_sum can be up to ~48*512 = 24576, and we need size info.
        # Alternative: for each size k from 2 to m, check if target_sum reachable with exactly k elements.
        # That's expensive. Instead, track min/max size for each sum? No, need exact.
        # Use: dp[s] = (min_size, max_size) doesn't work for exact size.
        # Better: for each k in [2..m], do a subset-sum with exactly k elements.
        # But m can be up to 47, and target_sum up to 24576.
        # Optimize: use numpy bitset for subset sum, tracking size via multiple passes.
        # Actually, let's use a different approach: 
        # dp[s] = bitmask where bit k is set if sum s is achievable with k elements.
        # This is too memory heavy. 
        # Alternative: since we just need to know if ANY subset of size >= 2 sums to target,
        # and target = a*(n-1), note that the sum of ALL others is target.
        # So we need: is there a proper subset of others (size 2 to m-1... wait, size m is all others)
        # that sums to target? If all others sum to target, then the full set of others works.
        # But wait: target = a * (n-1), and sum(others) = sum(arr) - a.
        # So we need sum(others) == a*(n-1), i.e., sum(arr) - a == a*(n-1), i.e., sum(arr) == a*n.
        # So the full set of others is a valid subset iff sum(arr) == a*n.
        # For proper subsets: we need a subset of others of size k (2<=k<=m) summing to a*n... 
        # No wait, the condition is: sum(subset) == len(subset) * a.
        # So for a subset S of others with |S|=k, we need sum(S) = k*a.
        # Let me redo: for target element a, a subset T of OTHERS with |T|>=2 violates if sum(T) = |T|*a.
        # So I need to check: for each k in [2..m], is there a subset of others of size k summing to k*a?
        # This is the "exactly k elements summing to k*a" problem.
        # Use DP: dp[k][s] = can we pick k elements summing to s?
        # But k up to 47, s up to 47*512=24064. That's 47*24064 ~ 1.1M booleans, feasible.
        # But doing this for each of 48 elements is 48 * 1.1M = 53M, too slow in Python.
        # Use numpy bitset: for each k, maintain a bitset of achievable sums.
        # dp_k is a bitset (integer or numpy array) where bit s is set if sum s achievable with k elements.
        # Transition: dp_{k+1} = OR over each element e of (dp_k << e)
        # This is O(m * max_sum / word_size) per element, which is fast with numpy.
        
        # Let's use numpy bitsets. max_sum = int(others.max()) * m
        max_s = int(others.max()) * m
        # dp[k] = bitset of sums achievable with exactly k elements
        # dp[0] = 1 (sum 0 with 0 elements)
        dp = [0] * (m + 1)
        dp[0] = 1  # bit 0 set
        for e in others:
            e = int(e)
            for k in range(m, 0, -1):
                # dp[k] |= dp[k-1] << e
                dp[k] |= (dp[k-1] << e)
            # Check: does dp[k] have bit (k*a) set?
            for k in range(2, m + 1):
                if (dp[k] >> (k * a)) & 1:
                    return False
        # Also check full set: if sum(others) == m*a, that's covered by k=m
    return True

def _is_valid_fast(set_arr):
    """Faster validity check using numpy bitsets with early termination."""
    n = len(set_arr)
    if n < 3:
        return True
    arr = np.sort(set_arr)
    for i in range(n):
        a = int(arr[i])
        others = np.concatenate([arr[:i], arr[i+1:]])
        m = n - 1
        if m < 2:
            continue
        max_s = int(others.max()) * m
        # dp[k] = bitset (Python int) of sums achievable with exactly k elements
        dp = [0] * (m + 1)
        dp[0] = 1
        for e in others:
            e = int(e)
            for k in range(m, 0, -1):
                dp[k] |= (dp[k-1] << e)
            # Early check for this element
            for k in range(2, m + 1):
                if (dp[k] >> (k * a)) & 1:
                    return False
    return True

def construct():
    deadline = time.time() + 80  # safety margin
    best = []
    
    # Parity-lattice: use numbers congruent to 0 mod 4
    # This ensures that for any subset sum S, S is divisible by 4.
    # For a to be a mean: sum(T) = |T| * a. Since a is divisible by 4, |T|*a is divisible by 16.
    # So we need sum(T) divisible by 16. Since each element is 0 mod 4, sum(T) is 0 mod 4.
    # This doesn't fully eliminate, but restricts.
    # Actually, let's think: if all elements are 0 mod 4, write a_i = 4*b_i.
    # Then sum(T) = 4 * sum(b_i for i in T). We need 4 * sum(b) = |T| * 4 * b_a, i.e., sum(b) = |T| * b_a.
    # So it reduces to the same problem on b_i. Not helpful structurally.
    # 
    # Alternative: use numbers that are all 0 mod 3. Then sum(T) is 0 mod 3.
    # Need sum(T) = |T| * a. If a is 0 mod 3, then |T|*a is 0 mod 3. OK, same reduction.
    #
    # Better structural approach: use powers of 2 scaled, or use a Sidon-like set.
    # Actually, the key violation for small subsets is 3-term AP: a, b, c with b = (a+c)/2.
    # To avoid 3-AP, use a set with no 3-term AP.
    # Then for larger subsets, the condition is harder to violate.
    #
    # Let's use a greedy approach with fast validation, trying to maximize cardinality.
    # Start with a known good set (e.g., powers of 2 up to 256: 1,2,4,8,16,32,64,128,256 - 9 elements)
    # Then greedily add more elements.
    
    # Actually, let's just do a greedy fill with fast validation, trying candidates in a smart order.
    # Candidate pool: all numbers 1..512
    # Try adding in order that minimizes conflicts.
    
    selected = []
    candidates = list(range(1, 513))
    
    # Greedy: try each candidate, add if valid
    # To be smarter, try in an order that spreads out the values
    # Use a heuristic: try numbers with fewer "mean" relationships first
    
    # Simple greedy: iterate through candidates, add if valid
    for c in candidates:
        if time.time() > deadline - 10:
            break
        test = selected + [c]
        if _is_valid_fast(np.array(test)):
            selected.append(c)
        if len(selected) >= 48:
            break
    
    # Try to improve with local search if time permits
    if time.time() < deadline - 5 and len(selected) < 48:
        # Try swapping: for each element, try replacing with a larger unused number
        used = set(selected)
        all_nums = set(range(1, 513))
        unused = sorted(all_nums - used)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(selected)):
                for c in unused:
                    if c <= selected[i]:
                        continue
                    test = selected[:i] + [c] + selected[i+1:]
                    if _is_valid_fast(np.array(test)):
                        selected[i] = c
                        used.discard(selected[i])
                        used.add(c)
                        unused.remove(c)
                        improved = True
                        break
                if improved:
                    break
    
    return selected
# EVOLVE-BLOCK-END
