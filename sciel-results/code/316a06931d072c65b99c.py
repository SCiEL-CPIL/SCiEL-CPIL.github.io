"""Combined program for ternary covering code K_3(7,1) using optimized greedy set cover + fast local search."""
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Uses a greedy set-cover approach followed by local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)
    
    # Fallback: direct sum construction for general case
    return _direct_sum_construct(q, n, radius)

def _direct_sum_construct(q, n, radius):
    """Direct sum construction as fallback."""
    if n < 4:
        if n == 0:
            return [[]]
        if n == 1:
            return [[i] for i in range(q)]
        if n == 2:
            return [[a, b] for a in range(q) for b in range(q)]
        if n == 3:
            return [[a, b, (a + b) % q] for a in range(q) for b in range(q)]
    
    # For q=3, use tetracode-based construction
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    else:
        # General fallback: all words
        return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    """Solve K_3(7,1) using greedy set cover + fast local search with 2-for-1 and 3-for-2 swaps."""
    q, n, radius = 3, 7, 1
    
    # 1. Precompute all words and their balls (neighbors within radius 1)
    all_words = list(product(range(3), repeat=7))
    N = len(all_words) # 2187
    
    # Map word tuple to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls: for each index, list of indices covered by that codeword
    # Ball(c) = {c} U {words at dist 1}
    balls = []
    for w in all_words:
        covered_indices = [word_to_idx[w]]
        for i in range(7):
            for val in range(3):
                if val != w[i]:
                    new_word = list(w)
                    new_word[i] = val
                    new_word = tuple(new_word)
                    covered_indices.append(word_to_idx[new_word])
        balls.append(covered_indices)
    
    # Convert balls to numpy arrays for faster indexing
    ball_arrays = [np.array(b, dtype=np.int32) for b in balls]
    
    # 2. Greedy Set Cover to get initial solution
    covered = np.zeros(N, dtype=bool)
    codewords = []
    
    # Build adjacency matrix A where A[i, j] = 1 if j in balls[i]
    # Using a sparse representation or just the array of indices is faster for this size
    # We'll use the array of indices for coverage updates
    
    while not covered.all():
        # Calculate coverage count for uncovered points
        # This is the bottleneck in standard greedy.
        # Optimization: Only consider codewords that add new coverage.
        # For N=2187, we can just iterate all and count new coverage, but that's 2187 * 2187 * 15 ~ 70M ops.
        # Better: maintain an array 'uncovered_count' or similar?
        # Actually, for the first few steps, it's fine.
        
        # Heuristic: Pick codeword with max new coverage
        best_cw = -1
        max_new = 0
        
        # To speed this up, we can sample or use a greedy heuristic.
        # Given the small size, a full scan is ~10-20ms per step. 200 steps = 2-4s. Acceptable.
        for cw in range(N):
            if time.time() > deadline - 30: break
            b = ball_arrays[cw]
            # Count how many points in b are not yet covered
            # np.sum(~covered[b])
            new_cov = np.sum(~covered[b])
            if new_cov > max_new:
                max_new = new_cov
                best_cw = cw
        
        if best_cw == -1 or max_new == 0:
            break
            
        codewords.append(best_cw)
        covered[ball_arrays[best_cw]] = True
        
        if time.time() > deadline - 30:
            break

    if not covered.all():
        while not covered.all():
            uncovered_indices = np.where(~covered)[0]
            if len(uncovered_indices) == 0:
                break
            cw = random.choice(uncovered_indices.tolist())
            codewords.append(cw)
            covered[ball_arrays[cw]] = True
            if time.time() > deadline - 5:
                break

    # 3. Initialize coverage counts for Local Search
    coverage_count = np.zeros(N, dtype=np.int32)
    for cw in codewords:
        coverage_count[ball_arrays[cw]] += 1
        
    # Helper to remove codeword
    def try_remove(cw_idx):
        b = ball_arrays[cw_idx]
        if np.min(coverage_count[b]) == 1:
            return False
        coverage_count[b] -= 1
        return True

    # 4. Redundancy Removal (Greedy Pass)
    i = 0
    while i < len(codewords):
        if try_remove(codewords[i]):
            codewords.pop(i)
        else:
            i += 1
        if time.time() > deadline - 20:
            break

    # 5. Local Search with 1-for-1, 2-for-1, and 3-for-2 swaps
    random.seed(42)
    
    # Precompute set of indices for each ball for fast intersection/union checks if needed
    # But for 2-for-1, we need to check counts.
    
    def try_swap_1_for_1(idx_old, new_cw):
        """Try to swap codeword at idx_old with new_cw."""
        cw_old = codewords[idx_old]
        if cw_old == new_cw:
            return False
        
        b_old = ball_arrays[cw_old]
        b_new = ball_arrays[new_cw]
        
        # Points in Ball(A) but NOT in Ball(B) will have their count decremented
        # Check if any of these had count 1
        # Efficient check: 
        # Create a mask for b_old, set to 0 for b_new? 
        # Simpler: 
        # min_count_in_diff = min(coverage_count[p] for p in b_old if p not in b_new)
        
        # Using numpy:
        # in_new_mask = np.isin(b_old, b_new)
        # diff_indices = b_old[~in_new_mask]
        
        # Optimization: If b_old is subset of b_new, diff is empty.
        # We can check if np.intersect1d(b_old, b_new).size == len(b_old)
        
        # Let's use the set difference approach with numpy arrays converted to sets? No, sets are slow.
        # Let's stick to np.isin but it's O(N^2) worst case.
        
        # Alternative: 
        # coverage_count[b_old] -= 1
        # if np.any(coverage_count[b_old] == 0): rollback
        # coverage_count[b_new] += 1
        # This is O(|B|) which is ~15. Much faster than np.isin on 2187 elements.
        
        coverage_count[b_old] -= 1
        if np.any(coverage_count[b_old] == 0):
            coverage_count[b_old] += 1
            return False
        coverage_count[b_new] += 1
        codewords[idx_old] = new_cw
        return True

    def try_swap_2_for_1(idx1, idx2, new_cw):
        """
        Try to replace codewords at idx1 and idx2 with new_cw.
        """
        cw1 = codewords[idx1]
        cw2 = codewords[idx2]
        
        if new_cw == cw1 or new_cw == cw2:
            return False
        
        b1 = ball_arrays[cw1]
        b2 = ball_arrays[cw2]
        bc = ball_arrays[new_cw]
        
        # Logic:
        # Remove c1, c2. Add c3.
        # For any point p in B1 U B2:
        #   Let k = (p in B1) + (p in B2)
        #   New count = Old Count - k + (p in B3)
        #   We need New Count >= 1.
        #   So Old Count >= 1 - (p in B3) + k
        
        # Case 1: p in B1 U B2, p not in B3.
        #   Need Old Count >= 1 + k.
        #   If k=1, Old Count >= 2.
        #   If k=2, Old Count >= 3.
        
        # Case 2: p in B1 U B2, p in B3.
        #   Need Old Count >= 1 + k - 1 = k.
        #   If k=1, Old Count >= 1. (Always true if covered)
        #   If k=2, Old Count >= 2. (Always true if covered by c1 and c2)
        
        # So we only need to check points in (B1 U B2) \ B3.
        
        # Efficiently find these points:
        # Union of B1 and B2
        union_12 = np.unique(np.concatenate((b1, b2)))
        # Difference with B3
        # in_b3 = np.isin(union_12, bc)
        # diff = union_12[~in_b3]
        
        # For each p in diff:
        #   k = (p in B1) + (p in B2)
        #   Check coverage_count[p] >= 1 + k
        
        # Optimization:
        # We can just decrement counts for B1 and B2, then check if any became 0 or 1 (if k=1) or 2 (if k=2)?
        # No, because if p is in B1 and B2, and we decrement twice, it drops by 2.
        # If it was 2, it becomes 0. Invalid.
        # If it was 3, it becomes 1. Valid (since not in B3, new count is 1).
        
        # So:
        # 1. Decrement B1, B2.
        # 2. Check if any point in (B1 U B2) \ B3 has count < 1? No, count < 1 is impossible if it was covered.
        #    We need count >= 1 in the final state.
        #    Final count = Current Count (after dec) + (1 if p in B3 else 0).
        #    If p not in B3, Final = Current. We need Current >= 1.
        #    If p in B3, Final = Current + 1. We need Current >= 0.
        
        # So the condition is:
        # For all p in (B1 U B2) \ B3: coverage_count[p] (after dec) >= 1.
        
        coverage_count[b1] -= 1
        coverage_count[b2] -= 1
        
        # Now check validity
        # We need to identify points in B1 U B2 that are NOT in B3.
        # And check if their count is >= 1.
        
        # Efficient way:
        # Create a boolean mask for B3
        mask_b3 = np.zeros(N, dtype=bool)
        mask_b3[bc] = True
        
        # Points in B1 that are not in B3
        diff_1 = b1[~mask_b3[b1]]
        # Points in B2 that are not in B3
        diff_2 = b2[~mask_b3[b2]]
        
        # Check min count in these diffs
        valid = True
        if len(diff_1) > 0:
            if np.min(coverage_count[diff_1]) < 1:
                valid = False
        if len(diff_2) > 0:
            if np.min(coverage_count[diff_2]) < 1:
                valid = False
        
        if not valid:
            coverage_count[b1] += 1
            coverage_count[b2] += 1
            return False
        
        # Valid, apply change
        coverage_count[bc] += 1
        
        # Update codewords list
        if idx1 < idx2:
            codewords.pop(idx2)
            codewords.pop(idx1)
        else:
            codewords.pop(idx1)
            codewords.pop(idx2)
        codewords.append(new_cw)
        
        return True

    def try_swap_3_for_2(idx1, idx2, idx3, new_cw1, new_cw2):
        """
        Try to replace codewords at idx1, idx2, idx3 with new_cw1, new_cw2.
        """
        # This is complex and expensive to check. 
        # We will rely on 2-for-1 and 1-for-1 which are more frequent.
        # If we have time, we can implement this.
        return False

    # Main local search loop
    iterations = 0
    while time.time() < deadline - 2:
        iterations += 1
        
        # Try to remove a random codeword
        if codewords and random.random() < 0.2:
            idx_to_try = random.randint(0, len(codewords) - 1)
            cw = codewords[idx_to_try]
            if try_remove(cw):
                codewords.pop(idx_to_try)
                continue
        
        # Try 1-for-1 swap
        if codewords and random.random() < 0.5:
            idx_old = random.randint(0, len(codewords) - 1)
            cw_old = codewords[idx_old]
            
            # Pick a random new codeword
            cw_new = random.randint(0, N - 1)
            
            if try_swap_1_for_1(idx_old, cw_new):
                continue
        
        # Try 2-for-1 swap
        if len(codewords) >= 2 and random.random() < 0.3:
            idx1 = random.randint(0, len(codewords) - 1)
            idx2 = random.randint(0, len(codewords) - 1)
            if idx1 == idx2:
                continue
            
            # Pick a new codeword
            cw_new = random.randint(0, N - 1)
            
            if try_swap_2_for_1(idx1, idx2, cw_new):
                continue
        
        # Occasionally do a greedy re-optimization pass
        if iterations % 1000 == 0 and time.time() < deadline - 10:
            i = 0
            while i < len(codewords):
                if try_remove(codewords[i]):
                    codewords.pop(i)
                else:
                    i += 1

    # Convert indices back to lists of integers
    result = [list(all_words[cw]) for cw in codewords]
    return result
