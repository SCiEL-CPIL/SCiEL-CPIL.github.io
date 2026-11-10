"""Improved program: Hypergraph-based greedy construction for 3-AP free set, followed by full constraint repair."""
import random

def _element_is_mean(a, S):
    """Return True if a is the exact arithmetic mean of some subset of S\\{a} of size >= 2.
    Uses a per-count bitset DP (Python big ints) over subset sums."""
    others = [x for x in S if x != a]
    n = len(others)
    if n < 2:
        return False
    max_sum = sum(others)
    mask = (1 << (max_sum + 1)) - 1
    dp = [0] * (n + 1)
    dp[0] = 1  # bit 0 set: empty subset sum 0
    for e in others:
        for c in range(n - 1, -1, -1):
            if dp[c]:
                dp[c + 1] |= (dp[c] << e) & mask
    for c in range(2, n + 1):
        target = c * a
        if target <= max_sum and ((dp[c] >> target) & 1):
            return True
    return False

def _is_fully_valid(S):
    for a in S:
        if _element_is_mean(a, S):
            return False
    return True

def _repair(S):
    """Remove any element that is the mean of a subset of the others, until valid."""
    S = set(S)
    guard = len(S) + 2
    while guard > 0:
        guard -= 1
        removed = False
        for a in list(S):
            if _element_is_mean(a, S):
                S.remove(a)
                removed = True
                break
        if not removed:
            break
    return S

def _greedy_3ap_free(n_max):
    """
    Greedy construction of a 3-AP free set using bitset conflicts.
    1. Precompute for each a, the set of b such that (a+b)/2 is integer and in range?
       No, 3-AP is (b, a, c) or (a, b, c) etc.
       Conflict: a and b are in conflict if there exists c such that 2a=b+c or 2b=a+c or 2c=a+b.
       Actually, for MIS on 3-AP hypergraph, an edge is (x,y,z) with x+z=2y.
       We want a set S such that no edge is fully contained in S.
       This is equivalent to S being 3-AP free.
       
       Greedy strategy:
       Sort candidates 1..n_max.
       Maintain a set `chosen`.
       For each candidate x in order:
           Check if adding x to `chosen` creates a 3-AP.
           A 3-AP is created if there exist y, z in `chosen` such that:
           1. x, y, z form a 3-AP.
           This means:
           - x is the middle: 2x = y+z.
           - y is the middle: 2y = x+z => z = 2y-x.
           - z is the middle: 2z = x+y => z = (x+y)/2 (if integer).
           
           So for a new x, we check all y in `chosen`:
           - If (x+y)/2 is integer and in `chosen`, conflict.
           - If 2y-x is in `chosen`, conflict.
           - If 2x-y is in `chosen`, conflict.
           
           If no conflict, add x.
    """
    chosen = set()
    # To make it faster, we can just iterate 1 to n_max.
    # "Greedy" usually implies some ordering. Smallest first is a common heuristic for 3-AP free sets?
    # Or random?
    # Let's try smallest first, then maybe largest first, or random, and keep the best.
    
    best_set = set()
    
    strategies = [
        list(range(1, n_max + 1)),
        list(range(n_max, 0, -1)),
        [x for x in range(1, n_max + 1) if x % 3 != 0], # Remove multiples of 3?
    ]
    
    # Add random permutations
    rng = random.Random(42)
    for _ in range(5):
        perm = list(range(1, n_max + 1))
        rng.shuffle(perm)
        strategies.append(perm)
        
    for order in strategies:
        chosen = set()
        for x in order:
            conflict = False
            for y in chosen:
                # Check if x, y, z form 3-AP
                # Case 1: x is middle. 2x = y + z => z = 2x - y.
                z = 2 * x - y
                if z in chosen:
                    conflict = True
                    break
                # Case 2: y is middle. 2y = x + z => z = 2y - x.
                z = 2 * y - x
                if z in chosen:
                    conflict = True
                    break
                # Case 3: z is middle. 2z = x + y => z = (x + y) / 2.
                if (x + y) % 2 == 0:
                    z = (x + y) // 2
                    if z in chosen:
                        conflict = True
                        break
            if not conflict:
                chosen.add(x)
        
        if len(chosen) > len(best_set):
            best_set = chosen
            
    return best_set

def construct():
    n_max = 512
    target_size = 48
    
    # 1. Generate a large 3-AP free set
    s_3ap = _greedy_3ap_free(n_max)
    
    # 2. Try to find a valid subset of size 48 from s_3ap
    # Since s_3ap is 3-AP free, violations are only for k >= 3.
    # We can greedily build a valid set from s_3ap.
    
    # Sort s_3ap to have a deterministic order for greedy selection
    candidates = sorted(s_3ap)
    
    valid_set = set()
    for x in candidates:
        if len(valid_set) >= target_size:
            break
        new_set = valid_set | {x}
        if _is_fully_valid(new_set):
            valid_set = new_set
            
    # If we didn't get 48, try to repair the whole s_3ap?
    # Or try a different subset of s_3ap?
    
    if len(valid_set) < target_size:
        # Try repairing a random subset of s_3ap of size 50
        rng = random.Random(42)
        for _ in range(20):
            if len(s_3ap) < 50:
                sub = set(s_3ap)
            else:
                sub = set(rng.sample(s_3ap, 50))
            repaired = _repair(sub)
            if len(repaired) > len(valid_set):
                valid_set = repaired
            if len(valid_set) >= target_size:
                break
                
    # If still not 48, fallback to Cantor set logic from previous program?
    # The previous program was robust. Let's keep it as a fallback.
    if len(valid_set) < target_size:
        def _generate_cantor_set(n_max):
            nums = []
            max_power = 0
            while (3 ** max_power) <= n_max:
                max_power += 1
            for mask in range(1 << max_power):
                val = 0
                for i in range(max_power):
                    if mask & (1 << i):
                        val += 3 ** i
                if val <= n_max and val > 0:
                    nums.append(val)
            return sorted(nums)
            
        cantor_nums = _generate_cantor_set(n_max)
        candidate_set = set(cantor_nums[:target_size])
        valid_set_cantor = _repair(candidate_set)
        if len(valid_set_cantor) > len(valid_set):
            valid_set = valid_set_cantor
            if len(valid_set) < target_size:
                remaining = [x for x in cantor_nums[target_size:] if x not in valid_set]
                for x in remaining:
                    if len(valid_set) >= target_size:
                        break
                    new_set = valid_set | {x}
                    if _is_fully_valid(new_set):
                        valid_set = new_set

    result = sorted(x for x in valid_set if 1 <= x <= n_max)[:target_size]
    return result
