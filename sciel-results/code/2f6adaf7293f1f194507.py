import time
import random
import sys

def _build_conflict_graph(n_max=512):
    """
    Precompute the conflict graph for all pairs (i, j) in [1, n_max].
    conflict[i][j] is True if i is the arithmetic mean of a subset of >= 2 elements from {1..n_max} \ {i, j}.
    This is independent of the current set, as long as the set is a subset of [1, n_max].
    """
    n = n_max
    # dp[k] is a bitset of achievable sums using exactly k elements from the candidate pool
    # We need to check for each pair (i, j) if i is mean of subset of (U \ {i, j})
    
    # Optimization: We can precompute the "all subset sums" for the full set [1..n_max]
    # But the condition is "subset of OTHER elements".
    # If we precompute dp_all[k] for the full set [1..n_max], we can't easily exclude i and j.
    
    # Alternative:
    # For each candidate mean m (1..n_max), we want to know if there exists a subset S of [1..n_max]\{m} such that sum(S) = k*m and |S| >= 2.
    # If such a subset S exists, then for ANY j in S, the pair (m, j) is a conflict?
    # No. The condition is: "No element a is the mean of a subset of at least two OTHER elements in the SET".
    # Let the set be A. For a in A, let B = A \ {a}. We need no subset S of B with sum(S) = k*a.
    
    # So, conflict depends on the specific set A.
    # However, we can precompute for each m, the set of "bad partners" J_m such that if j in J_m is in A, and m is in A, it might cause a conflict?
    # Actually, if m is in A, and there exists a subset S of A\{m} such that sum(S) = k*m, then m is invalid.
    # The subset S consists of elements from A\{m}.
    # So if we have a precomputed list of all "bad subsets" for each m, we just need to check if any bad subset is contained in A\{m}.
    
    # Let's precompute for each m, all minimal bad subsets?
    # Or simpler: For each m, compute the set of all possible sums of subsets of [1..n_max]\{m}.
    # This is too big.
    
    # Let's stick to the pair conflict idea but be careful.
    # Conflict(m, j) is True if there exists a subset S of [1..n_max]\{m, j} such that sum(S) = k*m and |S| >= 1 (so total subset size in A\{m} is >= 2).
    # If such an S exists in the full universe U=[1..n_max], it might not be in A.
    # So we can't just precompute a static graph.
    
    # HOWEVER, we can precompute the DP for the full set U=[1..n_max] once.
    # dp_full[k] = bitset of sums achievable by subsets of size k from U.
    # If we want to exclude m and j, we can't easily subtract from the bitset.
    
    # Let's try a different structural change:
    # Use a faster validity check that is still incremental but uses a global DP that is updated efficiently?
    # Or, use the "Simulated Annealing" with a validity check that is O(|A| * |A|^2) but optimized with bitsets for the *current set* only?
    # The current program's `_fast_validity_check_incr` is O(n * 2^n) effectively? No, it's O(n * max_sum) per check.
    # With n=48, max_sum=24576. 48 * 24576 ~ 1.2M ops per check.
    # SA might do 10,000-100,000 checks. 1.2M * 100k = 120B ops. Too slow.
    
    # We need a much faster validity check or a better precomputation.
    
    # Insight: The condition "a is mean of subset of others" is equivalent to:
    # There exist distinct x_1, ..., x_k in A\{a} such that sum(x_i) = k*a.
    # This is a subset sum problem.
    
    # Let's try to precompute for each a, the set of "obstructions".
    # An obstruction for a is a set of elements that cannot coexist with a?
    # No, it's a set of elements that *cause* a to be invalid.
    # If A contains a, and A contains a "bad subset" S for a, then A is invalid.
    # So for each a, we can precompute the list of all minimal bad subsets S_a.
    # Then, to check if A is valid, for each a in A, check if any S_a is a subset of A\{a}.
    
    # How many minimal bad subsets are there?
    # For a=1, sum(S) = k. Min size 2. {1, 1} not distinct. {2} size 1. {1, 2} sum 3 != 2*1.
    # For a=2, sum(S) = 2k. S subset of U\{2}.
    # k=2: sum=4. {1, 3}. So {1, 3} is a bad subset for 2.
    # k=3: sum=6. {1, 2, 3} (2 excluded), {1, 5}, {2, 4} (2 excluded). So {1, 5} is bad.
    # The number of minimal bad subsets might be manageable.
    
    # Let's generate all minimal bad subsets for each a in [1, 512].
    # A subset S is a bad subset for a if sum(S) = k*a, |S| >= 2, and a not in S.
    # Minimal means no proper subset of S is also a bad subset for a.
    
    pass

def _get_minimal_bad_subsets(a, n_max=512):
    """
    Find all minimal subsets S of [1..n_max]\{a} such that sum(S) = k*a for some k >= 2.
    """
    candidates = [x for x in range(1, n_max + 1) if x != a]
    # We want to find subsets with sum divisible by a.
    # DP over candidates?
    # Since n_max is 512, and a is small, we can use a bitset DP for sums modulo a?
    # No, we need exact sum = k*a.
    
    # Let's use a standard subset sum DP to find all achievable sums, and track the subsets?
    # Too expensive to track all subsets.
    
    # Alternative: Just check validity of the final set using the original bitset DP, but optimize the SA loop to be very fast.
    # Maybe the SA doesn't need to check *every* move?
    # Or maybe we can use a heuristic validity check that is fast but approximate, and only verify at the end?
    # Risky.
    
    # Let's try a different approach:
    # The current program scores 1.53. The max is 48.
    # If we can find a set of size 48, we win.
    # Is it possible to have size 48?
    # The constraint is very strong.
    
    # Let's look at the "powers of 2" set.
    # {1, 2, 4, 8, ..., 256}. Size 9.
    # Add more?
    # {1, 2, 4, 8, 16, 32, 64, 128, 256}
    # Can we add 3?
    # 3 is mean of {1, 5}? No 5 not in set.
    # 3 is mean of {1, 2, 4}? Sum 7, 3*? No.
    # 3 is mean of {1, 5}...
    # Actually, if we take all numbers of form 2^i * 3^j, they are "multiplicatively independent" in a sense that avoids additive relations?
    # No, 2+4=6, 6 is in the set? No, 6=2*3.
    # If we have 2, 4, 6. Is 4 the mean of 2 and 6? Yes. (2+6)/2 = 4.
    # So {2, 4, 6} is invalid.
    
    # So we cannot have arithmetic progressions.
    # Sidon sets? No, mean-free is stronger.
    
    # Let's try a random restart with a very fast local search.
    # The bottleneck is the validity check.
    # Can we make the validity check O(1) or O(log N) after precomputation?
    
    # Precompute for each pair (i, j) if they form a "forbidden pair" in the sense that i is mean of j and something else?
    # No, it's subsets of size >= 2.
    
    # Let's try to implement the "Minimal Bad Subsets" precomputation.
    # For each a, find all minimal S such that sum(S) = k*a.
    # If the number of such S is small, we can store them as bitmasks.
    # Then validity check: For each a in A, check if any bitmask S is a subset of A_mask ^ (1 << a).
    
    def _find_minimal_bad_subsets_for_a(a, n_max=512):
        bad_subsets = []
        # We need subsets of [1..n_max]\{a} with sum = k*a.
        # Max sum is ~ 512*513/2 = 131k.
        # k can be up to 131k / a.
        
        # Use DP to find if a sum is achievable, and backtrack to find one subset.
        # But we need ALL minimal subsets.
        # This might be exponential.
        
        # Heuristic: Only consider small k (2, 3, 4).
        # For k=2: sum = 2a. Pairs (x, y) such that x+y=2a, x!=y, x,y!=a.
        # For k=3: sum = 3a. Triples (x, y, z) distinct, not a, sum=3a.
        
        # Let's generate all pairs for k=2.
        for x in range(1, n_max + 1):
            if x == a: continue
            y = 2 * a - x
            if y == x or y == a or y < 1 or y > n_max: continue
            if x < y:
                bad_subsets.append((x, y))
        
        # For k=3:
        # This is O(N^2) to find all triples?
        # We can iterate x, y and check z = 3a - x - y.
        # To avoid duplicates, x < y < z.
        for x in range(1, n_max + 1):
            if x == a: continue
            for y in range(x + 1, n_max + 1):
                if y == a: continue
                z = 3 * a - x - y
                if z <= y or z == a or z < 1 or z > n_max: continue
                # Check if minimal: no pair in (x,y), (x,z), (y,z) sums to 2a?
                # Actually, minimal means no proper subset is a bad subset.
                # Proper subsets of size 2.
                # So if x+y=2a, or x+z=2a, or y+z=2a, then it's not minimal (or rather, the pair is the bad subset).
                # If none of the pairs sum to 2a, then the triple is minimal for k=3.
                if x + y != 2 * a and x + z != 2 * a and y + z != 2 * a:
                    bad_subsets.append((x, y, z))
                    
        return bad_subsets

    # This might be too slow or generate too many subsets.
    # Let's just use the bitset DP in the SA loop but optimize it heavily.
    # Or, use a "lazy" validity check: only check the changed elements?
    
    # In SA, we swap i and j.
    # Only validity of i and j and their neighbors might change?
    # No, adding j might make some other element k invalid if j is part of a bad subset for k.
    # So we need to check all elements?
    
    # Let's try a different strategy:
    # Use the current program's `_valid_mean_free` but call it less frequently?
    # Or, use a faster check:
    # For a set A, precompute the DP bitsets for A.
    # Then checking if a is mean of subset of A\{a} is O(|A|) lookups in the DP?
    # No, we need to exclude a.
    
    # Let's try to implement a fast "is_valid" that uses the precomputed DP of the *full set* [1..512]?
    # No, that doesn't work.
    
    # Okay, I will implement a Simulated Annealing with a fast incremental check that reuses DP where possible, or just accepts the O(N^2 * MaxSum) cost but optimizes the constant factor.
    # With N=48, MaxSum=24576.
    # 48 * 24576 = 1.2M bit operations.
    # In Python, bit operations are fast.
    # 1.2M operations per check.
    # If we do 10,000 checks, that's 12 billion operations. Too slow.
    
    # We need to reduce the cost of the check.
    # Idea: Precompute for each a, the set of "bad pairs" (x, y) such that x+y=2a.
    # If A contains a, and A contains x and y, then A is invalid.
    # This handles k=2.
    # For k=3, precompute bad triples?
    # If we only check k=2, we might miss violations.
    # But maybe most violations are k=2?
    # Let's try a hybrid:
    # 1. Precompute all bad pairs for all a.
    # 2. In the SA loop, check only k=2 violations using a fast graph check.
    # 3. At the end, run the full check.
    # This might allow SA to find a set that is "mostly" valid, and then we polish it.
    
    # Let's implement this "Approximate SA" + "Full Verification" strategy.
    
    pass

def _precompute_bad_pairs(n_max=512):
    """
    bad_pairs[a] is a set of frozensets of size 2, i.e., {x, y}, such that x+y=2a.
    """
    bad_pairs = [set() for _ in range(n_max + 1)]
    for a in range(1, n_max + 1):
        for x in range(1, n_max + 1):
            if x == a: continue
            y = 2 * a - x
            if y == x or y == a or y < 1 or y > n_max: continue
            if x < y:
                bad_pairs[a].add(frozenset((x, y)))
    return bad_pairs

def _is_valid_approximate(A, bad_pairs):
    """
    Check if A violates any k=2 mean condition.
    Returns True if valid (no k=2 violations).
    """
    A_set = set(A)
    for a in A:
        for pair in bad_pairs[a]:
            if pair.issubset(A_set):
                return False
    return True

def _is_valid_full(A):
    """
    Full check using bitset DP.
    """
    n = len(A)
    if n < 3:
        return True
    for i, a in enumerate(A):
        others = [x for j, x in enumerate(A) if j != i]
        m = len(others)
        if m < 2:
            continue
        max_sum = sum(others)
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << x)
        for k in range(2, m + 1):
            target = k * a
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n_max = 512
    target_size = 48
    
    bad_pairs = _precompute_bad_pairs(n_max)
    
    # Initial solution: Random set of size 48
    # Or start with a known good small set and expand?
    # Let's start with a random set of size 48.
    initial_set = random.sample(range(1, n_max + 1), target_size)
    
    current_set = set(initial_set)
    current_valid = _is_valid_full(current_set)
    
    best_set = set(initial_set)
    best_valid = current_valid
    
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    
    steps = 0
    while time.time() < deadline and T > min_T:
        steps += 1
        if steps % 1000 == 0:
            # Check time and update best
            if current_valid and len(current_set) > len(best_set):
                best_set = set(current_set)
                best_valid = True
            
        # Generate neighbor: Swap one element
        # Pick a random element in current_set to remove
        elem_to_remove = random.choice(list(current_set))
        
        # Pick a random element not in current_set to add
        # To keep size constant, we must add one.
        # We can pick from a pool of candidates.
        # Or just pick randomly from [1..512] \ current_set.
        candidates = [x for x in range(1, n_max + 1) if x not in current_set]
        elem_to_add = random.choice(candidates)
        
        new_set = set(current_set)
        new_set.remove(elem_to_remove)
        new_set.add(elem_to_add)
        
        # Check validity of new_set
        # Use approximate check first?
        # If approximate is invalid, it might still be fully invalid.
        # If approximate is valid, it might still be fully invalid (k>=3).
        
        # To save time, let's just use the full check for the SA step?
        # It might be too slow.
        
        # Let's try to optimize the full check by only checking the changed elements?
        # No, adding elem_to_add might invalidate others.
        
        # Let's use the approximate check as a filter.
        if not _is_valid_approximate(new_set, bad_pairs):
            # Likely invalid, skip full check?
            # But it might be valid if the violation is k>=3? No, if k=2 violation exists, it's invalid.
            # So if approximate says invalid, it IS invalid.
            delta_energy = 1000.0 # Penalty
        else:
            # Approximate says valid. Check full.
            if _is_valid_full(new_set):
                delta_energy = 0.0
            else:
                delta_energy = 1000.0
        
        # Metropolis step
        if delta_energy < 0 or random.random() < pow(2.718281828, -delta_energy / T):
            current_set = new_set
            current_valid = (delta_energy == 0.0)
            
        T *= cooling
        
    # Final polish: try to improve best_set
    if best_valid:
        # Try to add more elements? No, size is fixed at 48.
        # Try to fix any remaining issues? It's already valid.
        pass
    
    # If best_set is not valid, try to find a valid one by random restarts
    if not best_valid:
        for _ in range(100):
            if time.time() > deadline: break
            s = set(random.sample(range(1, n_max + 1), target_size))
            if _is_valid_full(s):
                best_set = s
                best_valid = True
                break
    
    return list(best_set)

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _is_valid_full(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"

