import time
import sys

N = 256
MAX_SUM = 2 * N  # Sums range from 2 to 512
# We need a bitmask for sums. Index s corresponds to bit (s-2) or similar.
# Let's use 0-indexed for sums: sum value s -> bit index s-2.
# Range of sums: [2, 512]. Size 511 bits.
# We can use a Python int as the bitset.

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute masks for elements
    # elem_mask[x] has bit x set. We use 1..256, so bit x.
    # Sum mask: bit s-2 represents sum s.
    
    # Lower bound heuristic: 
    # A Sidon set of size k has k(k+1)/2 distinct sums.
    # Max possible sums is 2N-1 = 511.
    # k(k+1)/2 <= 511 => k ~ 31.
    # However, maximality imposes stronger constraints.
    # Known results for N=256 suggest minimal maximal Sidon sets are around 20-25.
    
    best_size = float('inf')
    best_set = None
    
    # Bitmask utilities
    def get_elem_mask(x):
        return 1 << x
        
    def get_sum_mask(s):
        # s is in [2, 512]
        return 1 << (s - 2)

    # State:
    # S: list of selected elements
    # sum_bits: int representing all sums a+b for a,b in S, a<=b
    # blocked_bits: int representing elements x in [1, N] that are blocked by S
    #   x is blocked if 2x is a sum OR x+a is a sum for some a in S.
    #   Actually, for maximality, we just need to know if [1, N] \ S is subset of Blocked.
    #   Blocked(x) = (sum_bits has bit for 2x) OR (exists a in S such that sum_bits has bit for x+a).
    
    # Optimization: 
    # Instead of recalculating blocked_bits from scratch, we can update it.
    # When adding x:
    # New sums: 2x, x+a for all a in S.
    # These new sums block new elements.
    # A sum s blocks y if 2y=s (y=s/2) or y+a=s for some a in S (y=s-a).
    # This is complex to update incrementally because "y+a=s" depends on S.
    
    # Simpler Approach for B&B:
    # Since N=256 is small, we can check maximality at the leaves or prune branches.
    # Pruning:
    # 1. If |S| >= best_size, prune.
    # 2. If sum_bits covers all possible sums (impossible for small k), prune.
    # 3. If there exists an element y not in S and not blocked, we MUST add it or another element that blocks it.
    #    Actually, if y is not blocked, we can add y. If we don't add y, we must add some z such that z blocks y.
    #    This leads to a "must add" set.
    
    # Let's use a recursive DFS with memoization or just simple B&B.
    # To speed up, we will precompute conflicts.
    
    # Conflict check for adding x to S:
    # 1. 2x not in sum_bits
    # 2. For all a in S, x+a not in sum_bits
    
    # Maximality check for S:
    # For all y in [1, N] \ S:
    #   y is blocked if:
    #   (sum_bits & get_sum_mask(2*y)) != 0  OR
    #   (exists a in S such that (sum_bits & get_sum_mask(y+a)) != 0)
    
    # We will search for the minimum size.
    # Start with k=1, k=2, ... until we find a valid set.
    # Since we want to minimize, we can try to construct sets of size k.
    
    # However, B&B is better.
    # Let's order elements by some heuristic.
    # Elements that create "central" sums might be better?
    # Or just iterate 1..N.
    
    # To make it fast, we will use a greedy initialization to get a good upper bound (best_size).
    
    def greedy_init():
        S = []
        sum_bits = 0
        for x in range(1, N + 1):
            # Check if x can be added
            valid = True
            if sum_bits & get_sum_mask(2 * x):
                valid = False
            else:
                for a in S:
                    if sum_bits & get_sum_mask(x + a):
                        valid = False
                        break
            if valid:
                S.append(x)
                sum_bits |= get_sum_mask(2 * x)
                for a in S[:-1]:
                    sum_bits |= get_sum_mask(x + a)
        return S, sum_bits

    # Run greedy init to get a starting bound
    init_S, init_sum_bits = greedy_init()
    best_size = len(init_S)
    best_set = list(init_S)
    
    # Now, perform B&B to find smaller sets.
    # We will try to remove elements or search for smaller constructions.
    # Actually, a direct B&B from empty set is risky if the search space is large.
    # But with the sum constraint, it's very tight.
    
    # Let's try a different approach: Local Search on the Greedy Result.
    # The greedy result is maximal.
    # We try to remove elements.
    # If removing x keeps it maximal, we remove it.
    # Repeat until no more removals.
    # Then try to swap elements?
    
    # Given the time limit, a robust Local Search on a good Greedy seed is often faster than full B&B for N=256 in Python.
    # But the prompt asks for a STRUCTURAL change.
    # Let's use **Bitmask-based Local Search with Random Restarts**.
    # This is structurally different from the "Slow Set Operations" of the current program.
    # The current program uses `set` objects. This uses `int` bitmasks.
    
    import random
    
    def is_valid_sidon(S):
        sum_bits = 0
        for i, a in enumerate(S):
            if sum_bits & get_sum_mask(2 * a):
                return False
            for b in S[i+1:]:
                if sum_bits & get_sum_mask(a + b):
                    return False
            # Update sum_bits? No, this check is O(k^2).
            # For k~20, this is fast.
        return True

    def get_sum_bits(S):
        bits = 0
        for i, a in enumerate(S):
            bits |= get_sum_mask(2 * a)
            for b in S[i+1:]:
                bits |= get_sum_mask(a + b)
        return bits

    def is_maximal(S, sum_bits):
        # Check if any y in [1, N] \ S is NOT blocked.
        S_set = set(S)
        for y in range(1, N + 1):
            if y in S_set:
                continue
            # Check if y is blocked
            if sum_bits & get_sum_mask(2 * y):
                continue
            # Check if y+a in sums for any a in S
            blocked = False
            for a in S:
                if sum_bits & get_sum_mask(y + a):
                    blocked = True
                    break
            if not blocked:
                return False
        return True

    def local_search(S, sum_bits):
        S = list(S)
        sum_bits = get_sum_bits(S)
        improved = True
        while improved:
            improved = False
            # Try removing each element
            # Shuffle order for diversity? Or deterministic?
            # Deterministic first, then random if stuck?
            # Let's try all removals.
            for i in range(len(S)):
                x = S[i]
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sum_bits = get_sum_bits(new_S)
                if is_maximal(new_S, new_sum_bits):
                    S = new_S
                    sum_bits = new_sum_bits
                    improved = True
                    break # Restart loop
        return S, sum_bits

    def random_greedy(seed):
        random.seed(seed)
        S = []
        sum_bits = 0
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Check validity
            if sum_bits & get_sum_mask(2 * x):
                continue
            valid = True
            for a in S:
                if sum_bits & get_sum_mask(x + a):
                    valid = False
                    break
            if valid:
                S.append(x)
                sum_bits |= get_sum_mask(2 * x)
                for a in S[:-1]:
                    sum_bits |= get_sum_mask(x + a)
        
        # Ensure maximality (greedy usually produces maximal, but verify)
        sum_bits = get_sum_bits(S)
        if not is_maximal(S, sum_bits):
            # Extend
            for x in range(1, N + 1):
                if x in S:
                    continue
                if sum_bits & get_sum_mask(2 * x):
                    continue
                valid = True
                for a in S:
                    if sum_bits & get_sum_mask(x + a):
                        valid = False
                        break
                if valid:
                    S.append(x)
                    sum_bits |= get_sum_mask(2 * x)
                    for a in S[:-1]:
                        sum_bits |= get_sum_mask(x + a)
        
        return S, sum_bits

    # Main Loop
    trials = 0
    while time.time() < deadline:
        trials += 1
        # Use different seeds for diversity
        S, sum_bits = random_greedy(trials)
        
        # Local Search to minimize
        S_min, sum_bits_min = local_search(S, sum_bits)
        
        if len(S_min) < best_size:
            best_size = len(S_min)
            best_set = list(S_min)
            # If we reach a very low bound, we can stop early?
            # Theoretical min is ~ sqrt(2N) ~ 22.
            if best_size <= 20:
                break

    return best_set

# EVOLVE-BLOCK-END
