import time
import random

START = time.time()
DEADLINE = 55.0

def _validate(vals):
    n = len(vals)
    sums_by_k = [0] * (n + 1)
    sums_by_k[0] = 1
    for v in vals:
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                sums_by_k[k + 1] |= (sums_by_k[k] << v)
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_k[i] & sums_by_k[j]:
                return False
    return True

def _add_element(vals, sums_by_k, c):
    """
    Check if adding c to vals keeps the constraint valid.
    If valid, returns the updated sums_by_k (list of ints). Otherwise returns None.
    vals: list of current elements
    sums_by_k: list of bitsets for current elements
    c: candidate element
    """
    n = len(vals)
    new_sums = sums_by_k[:]
    
    # Compute the bits that will be added to each new_sums[k+1]
    # new_sums[k+1] |= (sums_by_k[k] << c)
    # We need to ensure that for all i, j, new_sums[i] & new_sums[j] == 0 (for i != j)
    
    # Optimization: 
    # 1. Check if the new bits intersect with any existing bits in different cardinality classes.
    # 2. Check if the new bits in class k+1 intersect with existing bits in class k+1? No, that's allowed (OR).
    #    Wait, the constraint is: "Whenever two subsets have different cardinalities their sums must differ."
    #    So sums_by_k[i] and sums_by_j must be disjoint for i != j.
    #    Within the same cardinality, sums can repeat (OR is fine).
    
    # So we need to check:
    # For each k (0..n):
    #   added_bits = sums_by_k[k] << c
    #   These bits go into new_sums[k+1].
    #   We must ensure added_bits does not intersect with new_sums[m] for any m != k+1.
    
    # Since new_sums[m] = sums_by_k[m] for m != k+1 (mostly), and new_sums[k+1] = sums_by_k[k+1] | added_bits,
    # we need to check:
    # A) added_bits & sums_by_k[m] == 0 for all m != k+1
    # B) (sums_by_k[k+1] | added_bits) & sums_by_k[m] == 0 for all m != k+1
    #    This implies:
    #    B1) sums_by_k[k+1] & sums_by_k[m] == 0 (already true by invariant)
    #    B2) added_bits & sums_by_k[m] == 0 (same as A)
    
    # So the only check is: for all k in 0..n, and for all m in 0..n+1, m != k+1:
    #   (sums_by_k[k] << c) & sums_by_k[m] == 0
    
    # Wait, is that sufficient?
    # Let's check the final state new_sums.
    # new_sums[i] = sums_by_k[i] | (sums_by_k[i-1] << c)  (with sums_by_k[-1]=0, sums_by_k[n+1]=0)
    # Actually, the update is:
    # for k in range(n, -1, -1):
    #   new_sums[k+1] |= (sums_by_k[k] << c)
    #   Note: new_sums[k+1] starts as sums_by_k[k+1].
    #   So new_sums[j] = sums_by_k[j] | (sums_by_k[j-1] << c)
    
    # We need new_sums[i] & new_sums[j] == 0 for i != j.
    # Let i < j.
    # new_sums[i] = sums_by_k[i] | (sums_by_k[i-1] << c)
    # new_sums[j] = sums_by_k[j] | (sums_by_k[j-1] << c)
    
    # Intersection:
    # 1. sums_by_k[i] & sums_by_k[j] == 0 (True by invariant)
    # 2. sums_by_k[i] & (sums_by_k[j-1] << c) == 0
    # 3. (sums_by_k[i-1] << c) & sums_by_k[j] == 0
    # 4. (sums_by_k[i-1] << c) & (sums_by_k[j-1] << c) == 0
    #    => (sums_by_k[i-1] & sums_by_k[j-1]) << c == 0
    #    => sums_by_k[i-1] & sums_by_k[j-1] == 0
    #    Since i-1 < j-1, this is True by invariant (unless i-1 == j-1 which implies i=j, but i<j).
    #    Wait, if i=1, j=2, then i-1=0, j-1=1. sums_by_k[0] & sums_by_k[1] == 0 is True.
    
    # So we only need to check:
    # For all i, j with i < j:
    #   (sums_by_k[i] & (sums_by_k[j-1] << c)) == 0
    #   AND
    #   ((sums_by_k[i-1] << c) & sums_by_k[j]) == 0
    
    # Let's simplify.
    # Check 2: sums_by_k[i] & (sums_by_k[j-1] << c) == 0
    #   Let p = j-1. Then p >= i (since j > i).
    #   If p == i, then sums_by_k[i] & (sums_by_k[i] << c) == 0. This is always true because shifting by c>=1 moves bits.
    #   If p > i, then sums_by_k[i] & (sums_by_k[p] << c) == 0.
    #   This is equivalent to: for all i < p, sums_by_k[i] & (sums_by_k[p] << c) == 0.
    
    # Check 3: (sums_by_k[i-1] << c) & sums_by_k[j] == 0
    #   Let q = i-1. Then q < j-1.
    #   This is equivalent to: for all q < p (where p=j), (sums_by_k[q] << c) & sums_by_k[p] == 0.
    #   This is the same condition as Check 2 with indices swapped/shifted!
    #   Check 2 was: sums_by_k[i] & (sums_by_k[p] << c) == 0 for i < p.
    #   Check 3 is: (sums_by_k[q] << c) & sums_by_k[p] == 0 for q < p.
    #   These are identical conditions: A & (B << c) == 0  <=>  (A << c) & B == 0 ?
    #   No! A & (B << c) checks if any bit in A is at position x such that B has a bit at x-c.
    #   (A << c) & B checks if any bit in A is at position x such that B has a bit at x+c.
    #   They are NOT the same.
    #   A & (B << c) == 0 means: No sum in A is equal to (sum in B) - c. i.e., No sum in A + c is in B? No.
    #   Let's stick to the explicit loop, it's safer and n is small (<= 12).
    
    # Efficient check:
    # For each k (0..n), let added = sums_by_k[k] << c.
    # This added goes into new_sums[k+1].
    # It must not intersect with new_sums[m] for m != k+1.
    # new_sums[m] = sums_by_k[m] | (sums_by_k[m-1] << c)
    
    # So for each k, we check added against new_sums[m] for all m != k+1.
    # This is O(n^2) bit ops. n is small.
    
    # Let's precompute new_sums first?
    # No, we need to check validity before committing.
    
    # Algorithm:
    # 1. Compute added_bits[k] = sums_by_k[k] << c for k=0..n.
    # 2. For each k, target_idx = k+1.
    # 3. For each m in 0..n+1, m != target_idx:
    #      new_m = sums_by_k[m] | (added_bits[m-1] if m>0 else 0)
    #      if added_bits[k] & new_m: return None
    
    # This is O(n^2).
    
    added_bits = [0] * (n + 1)
    for k in range(n + 1):
        if sums_by_k[k]:
            added_bits[k] = sums_by_k[k] << c
    
    for k in range(n + 1):
        if not added_bits[k]:
            continue
        target = k + 1
        for m in range(n + 2):
            if m == target:
                continue
            # Compute new_sums[m]
            nm = sums_by_k[m]
            if m > 0 and m <= n + 1:
                # m-1 is in 0..n
                if m-1 <= n:
                    nm |= added_bits[m-1]
            if added_bits[k] & nm:
                return None
    
    # If we reach here, it's valid. Update sums_by_k.
    for k in range(n, -1, -1):
        if sums_by_k[k]:
            new_sums[k+1] |= (sums_by_k[k] << c)
            
    return new_sums

def _solve_size(target_size):
    """
    Find a set of target_size distinct ints in 1..256 satisfying the constraint.
    Uses DFS with backtracking and bitsets.
    """
    # Use a list of candidates.
    # Heuristic: Try larger numbers first? Or smaller?
    # Larger numbers tend to push sums higher, potentially separating cardinalities better?
    # Or smaller numbers fill the lower range, allowing larger numbers later?
    # Let's try descending order first, as it's often good for packing/sum problems.
    
    avail_list = list(range(256, 0, -1))
    
    def dfs(idx, current_vals, current_sums, used):
        if time.time() - START > DEADLINE:
            return None
            
        if len(current_vals) == target_size:
            return list(current_vals)
            
        spots_left = target_size - len(current_vals)
        
        # Pruning: if not enough elements left
        # We are iterating from idx to end.
        # Remaining elements = len(avail_list) - idx
        # But we skip used ones.
        # Simple check:
        # if spots_left > (len(avail_list) - idx): return None # This is weak because of 'used'
        
        for i in range(idx, len(avail_list)):
            c = avail_list[i]
            if used[c]:
                continue
                
            new_sums = _add_element(current_vals, current_sums, c)
            if new_sums is not None:
                used[c] = True
                res = dfs(i + 1, current_vals + [c], new_sums, used)
                if res:
                    return res
                used[c] = False
                
        return None
        
    used = [False] * 257
    return dfs(0, [], [1] + [0]*target_size, used)

def _solve_size_random(target_size, seed):
    """
    Try solving with a shuffled candidate list.
    """
    rng = random.Random(seed)
    avail_list = list(range(1, 257))
    rng.shuffle(avail_list)
    
    def dfs(idx, current_vals, current_sums, used):
        if time.time() - START > DEADLINE:
            return None
            
        if len(current_vals) == target_size:
            return list(current_vals)
            
        for i in range(idx, len(avail_list)):
            c = avail_list[i]
            if used[c]:
                continue
                
            new_sums = _add_element(current_vals, current_sums, c)
            if new_sums is not None:
                used[c] = True
                res = dfs(i + 1, current_vals + [c], new_sums, used)
                if res:
                    return res
                used[c] = False
                
        return None
        
    used = [False] * 257
    return dfs(0, [], [1] + [0]*target_size, used)

def construct():
    global START, DEADLINE
    
    # The previous program scored 1.5, which implies it found a set of size 12?
    # Or maybe the seed score was 8?
    # "value/seed". If seed program returns 8, and we return 12, score is 1.5.
    # So we are currently finding size 12.
    # Can we find 13? 14?
    
    # Try sizes from high to low.
    # Start from a reasonable upper bound. 12 worked. Try 15, 14, 13.
    
    # First, try deterministic descending order for larger sizes.
    for size in [15, 14, 13]:
        if time.time() - START > DEADLINE - 5:
            break
        result = _solve_size(size)
        if result:
            if _validate(result):
                return [int(x) for x in result]
        
    # If those fail, try randomized search for 13, 14, 15.
    for size in [15, 14, 13]:
        if time.time() - START > DEADLINE - 2:
            break
        for seed in range(10):
            if time.time() - START > DEADLINE - 2:
                break
            result = _solve_size_random(size, seed)
            if result:
                if _validate(result):
                    return [int(x) for x in result]
                    
    # Fallback to 12 (which we know works)
    result = _solve_size(12)
    if result:
        if _validate(result):
            return [int(x) for x in result]
            
    # Final fallback
    return [1, 2, 4, 8, 16, 32, 64, 128]

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Found size: {len(res)}")
    print(res)
    print("Valid:", _validate(res))
