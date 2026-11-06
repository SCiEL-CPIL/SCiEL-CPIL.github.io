import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin for 60s limit

def construct():
    start_time = time.time()
    best_size = 0
    best_A = []
    best_B = []
    
    # Helper to check if adding x to a set is valid
    # set_elems: list of elements in the set
    # sum_bits: bitset of pair sums (a+b for a<=b)
    # diff_bits: bitset of absolute differences (|a-b| for a!=b)
    # other_diff_bits: bitset of differences of the OTHER set
    # x: candidate element
    
    def is_valid_add(x, elems, sum_bits, diff_bits, other_diff_bits):
        # Check sum collisions
        # x+x
        if (sum_bits >> (x + x)) & 1:
            return False
        # x + a for a in elems
        # We need to check if (x+a) is in sum_bits
        # Note: sum_bits contains sums of pairs from elems.
        # If we add x, new sums are x+x and x+a.
        # We must ensure x+x and x+a are not in current sum_bits.
        # Also, we must ensure that no two new sums collide with each other?
        # x+x is unique. x+a is unique for distinct a.
        # Do x+a collide with x+b? No, a!=b.
        # Do x+a collide with x+x? Only if a=x, but x not in elems.
        # So we just need to check against existing sum_bits.
        
        # Check diff collisions
        # |x-a| for a in elems
        # These new diffs must not be in current diff_bits.
        # And they must not be in other_diff_bits.
        
        # Optimization: Iterate through elems
        for a in elems:
            s = x + a
            if (sum_bits >> s) & 1:
                return False
            d = abs(x - a)
            if (diff_bits >> d) & 1:
                return False
            if (other_diff_bits >> d) & 1:
                return False
        
        # Check x+x sum collision again? 
        # The loop above checks x+a. It does NOT check x+x.
        # So we explicitly check x+x.
        if (sum_bits >> (x + x)) & 1:
            return False
            
        return True

    def add_element(x, elems, sum_bits, diff_bits):
        # Returns updated (elems, sum_bits, diff_bits)
        new_sum_bits = sum_bits
        new_diff_bits = diff_bits
        # Add x+x to sums
        new_sum_bits |= (1 << (x + x))
        for a in elems:
            new_sum_bits |= (1 << (x + a))
            d = abs(x - a)
            new_diff_bits |= (1 << d)
        return elems + [x], new_sum_bits, new_diff_bits

    def greedy_construct():
        A = []
        B = []
        sumA = 0
        diffA = 0
        sumB = 0
        diffB = 0
        
        # We iterate through all numbers 1..256
        # We can randomize the order to escape local optima in ILS
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try adding to A
            if len(A) < MAX_EACH and is_valid_add(x, A, sumA, diffA, diffB):
                A, sumA, diffA = add_element(x, A, sumA, diffA)
                continue
            
            # Try adding to B
            if len(B) < MAX_EACH and is_valid_add(x, B, sumB, diffB, diffA):
                B, sumB, diffB = add_element(x, B, sumB, diffB)
                continue
            
            # If neither, skip x for this iteration
        
        return A, B

    def local_search(A, B):
        # Try to improve by swapping or removing/adding
        # This is a simple hill climbing: try to add any skipped element
        # by removing one existing element.
        
        # Collect skipped elements
        used = set(A) | set(B)
        skipped = [x for x in range(1, N + 1) if x not in used]
        
        current_size = len(A) + len(B)
        
        # Reconstruct bitsets for A and B
        sumA = 0
        diffA = 0
        for a in A:
            for b in A:
                if a <= b:
                    sumA |= (1 << (a + b))
        for a in A:
            for b in A:
                if a != b:
                    diffA |= (1 << abs(a - b))
        
        sumB = 0
        diffB = 0
        for b in B:
            for c in B:
                if b <= c:
                    sumB |= (1 << (b + c))
        for b in B:
            for c in B:
                if b != c:
                    diffB |= (1 << abs(b - c))
        
        improved = True
        while improved:
            improved = False
            # Try to add a skipped element by removing one from A or B
            # We prioritize removing elements that "block" many others?
            # For simplicity, try all pairs (skip, remove)
            # This is O(|skipped| * (|A|+|B|)) which is ~ 200 * 128 = 25k checks per iteration.
            # Each check is O(|set|) ~ 64. Total ~ 1.6M ops. Fast enough.
            
            # To avoid O(N^3), we can be smarter, but let's try this first with a time check.
            
            for s in skipped:
                if time.time() - start_time > DEADLINE - 5:
                    return A, B, current_size
                
                # Try removing from A
                for i, a in enumerate(A):
                    # Remove a from A
                    new_A = A[:i] + A[i+1:]
                    # Recalculate sumA, diffA? 
                    # It's expensive to recalculate from scratch.
                    # Better: Just check if adding s to new_A is valid.
                    # But we need new_A's sum/diff bits.
                    # Let's just do a full validity check on new_A + [s] vs B.
                    
                    # Check if new_A + [s] is a valid Sidon set
                    # And if its diffs are disjoint from B's diffs.
                    
                    # This requires checking all pairs in new_A + [s].
                    # Size is ~64. Pairs ~ 2000.
                    # Doing this for every (s, a) pair is too slow.
                    
                    # Alternative: 
                    # The greedy construction already maximized for the given order.
                    # The main gain from ILS is re-shuffling the order.
                    # So, let's stick to ILS with different random orders.
                    
                    break # Skip this complex local search, rely on ILS re-greedy
                
                # Try removing from B
                for i, b in enumerate(B):
                    break # Skip
                    
            # Since simple local search is complex to implement efficiently without bitset updates,
            # we will rely on multiple greedy runs with different random seeds/orders.
            
            break # Exit while loop, just do multiple greedy runs
            
        return A, B, current_size

    # Main Loop: ILS
    best_A, best_B = [], []
    
    # Run multiple greedy constructions with different random orders
    # 256 elements. 
    # Time per greedy run: 
    # 256 iterations.
    # Each iteration: try A (check O(|A|)), try B (check O(|B|)).
    # Avg |A|, |B| ~ 30. 
    # 256 * 60 ops ~ 15k ops. Very fast.
    # We can do thousands of runs.
    
    runs = 0
    while time.time() - start_time < DEADLINE:
        A, B = greedy_construct()
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A
            best_B = B
            # print(f"New best: {size} at run {runs}")
        runs += 1
        # If we hit the max possible size (128), we can stop early
        if best_size >= 128:
            break

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2] # 1 and 2 have diff 1, sum 3. 
        # Check if valid.
        # If A=[1], B=[2]. 
        # A sums: 2. Diffs: {}.
        # B sums: 4. Diffs: {}.
        # Disjoint diffs? Yes.
        # But wait, if best_A was [1], best_B was empty, we add 2.
        # Is 2 valid in B?
        # diffB is empty. sumB is 0.
        # is_valid_add(2, [], 0, 0, diffA=0) -> True.
        
    return [best_A, best_B]

if __name__ == "__main__":
    # Quick test
    A, B = construct()
    print(f"Size A: {len(A)}, Size B: {len(B)}, Total: {len(A)+len(B)}")
    # Verify
    def verify(A, B):
        # Check Sidon
        def is_sidon(S):
            sums = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    s = S[i] + S[j]
                    if s in sums:
                        return False
                    sums.add(s)
            return True
        
        if not is_sidon(A) or not is_sidon(B):
            return False, "Sidon check failed"
            
        # Check Disjoint Diffs
        def get_diffs(S):
            return {abs(a-b) for i, a in enumerate(S) for b in S[i+1:]}
        
        dA = get_diffs(A)
        dB = get_diffs(B)
        
        if dA & dB:
            return False, f"Diff intersection: {dA & dB}"
            
        return True, "Valid"

    valid, msg = verify(A, B)
    print(f"Valid: {valid}, Msg: {msg}")
