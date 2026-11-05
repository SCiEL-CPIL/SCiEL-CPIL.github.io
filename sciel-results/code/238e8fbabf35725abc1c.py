"""Backtracking search for a Sidon set containing 0 realizing differences 1..30."""
import time
import sys

DEADLINE = 130  # safety margin before 150s kill


def construct():
    start = time.time()
    best = [None]
    best_max = [10**6]

    def elapsed():
        return time.time() - start

    def check_deadline():
        return elapsed() > DEADLINE

    # Backtracking search: build a sorted list, adding elements one at a time.
    # We need a Sidon set (all positive differences distinct) containing 0,
    # with every d in 1..30 appearing as a difference, and at most 32 elements.
    # Minimize max element.

    # Strategy: we must include 0. We'll search for the smallest possible max.
    # Use iterative deepening on the max value, or directly backtrack.

    # Key insight: with at most 32 elements, we have at most 32*31/2 = 496 differences.
    # We need 30 specific differences. The Sidon constraint means all differences are distinct.

    # Approach: backtracking with pruning.
    # State: current sorted set S, used differences set, missing differences set.
    # We add elements in increasing order. 0 is always included.

    # For efficiency, we track:
    # - S: current set (sorted list)
    # - used_diffs: set of all differences present
    # - missing: set of differences from 1..30 not yet covered
    # - next_min: the minimum value we can add next (must be > S[-1])

    # Pruning:
    # 1. If len(S) > 32: backtrack
    # 2. If max(S) > current_best_max: backtrack
    # 3. If missing is non-empty but we can't possibly cover it (not enough room): backtrack
    # 4. Forward checking: when adding x, check no duplicate differences

    # Let's use a more structured approach:
    # We know 0 must be in the set. Let's say the set is {0, a1, a2, ..., ak} with 0 < a1 < ... < ak.
    # We want to minimize ak.

    # For a given upper bound U, can we find a valid set with max <= U?
    # Use backtracking to find the smallest U.

    # Actually, let's just do a direct backtracking search trying to minimize max.
    # We'll search element by element, trying small values first.

    # To make this tractable, note that we need 30 differences covered.
    # Each new element x added to set {0, s1, ..., si} creates i+1 new differences: x, x-s1, ..., x-si.
    # We need these to be distinct from each other and from all previously used differences.

    # Let's use a recursive backtracking with memoization-like pruning.

    # For speed, I'll use an iterative approach with a stack.

    # Actually, let me use a cleaner recursive approach with good pruning.

    # Key optimization: we don't need to try all values. For each position, we can
    # focus on values that either cover a missing difference or are "forced".

    # Let me think about this differently. We need at least enough elements to cover 30 diffs.
    # With k elements (including 0), we have k*(k-1)/2 differences. We need at least 30,
    # so k*(k-1)/2 >= 30, meaning k >= 9 (9*8/2=36).

    # Let's search for the minimal maximum element using backtracking.

    def backtrack(S, used_diffs, missing, max_allowed):
        """
        S: sorted list of elements (0 is always first)
        used_diffs: set of all positive differences in S
        missing: set of differences from 1..30 not yet in used_diffs
        max_allowed: current upper bound on max element
        Returns True if a valid solution is found, updates best.
        """
        if check_deadline():
            return False

        if not missing:
            # All differences covered! Check if we have a valid Sidon set.
            # It's valid by construction (we only add elements that don't create dup diffs).
            if len(S) <= 32:
                m = S[-1]
                if m < best_max[0]:
                    best_max[0] = m
                    best[0] = list(S)
                    return True
            return False

        # Pruning: not enough elements left to cover missing differences
        # Each new element can cover at most (len(S)) new differences (one with each existing element)
        # Actually, element x added to S creates diffs: x - s for s in S, plus x itself if 0 not in S (but 0 is in S)
        # So x creates len(S) new differences (x - s for each s in S, since x > all s)
        # Wait, differences are |x - s| for all pairs. Since x > s for all s in S, diffs are x - s.
        # So adding x creates len(S) new differences.
        # We need to cover len(missing) more differences.
        # Max elements we can add: 32 - len(S)
        # Max new diffs from those: sum of (len(S) + i) for i in range(32 - len(S))
        # = (32-len(S))*len(S) + (32-len(S))*(32-len(S)-1)/2
        
        remaining_slots = 32 - len(S)
        if remaining_slots <= 0:
            return False
        
        # Max new differences we can get
        max_new_diffs = 0
        for i in range(remaining_slots):
            max_new_diffs += len(S) + i
        
        if max_new_diffs < len(missing):
            return False

        # Try adding new elements
        # The next element must be > S[-1] and <= max_allowed
        # To cover missing differences, for each missing d, we could add x = S[-1] + d or x = s + d for some s in S
        # But also x could be any value that doesn't create duplicate differences.

        # Strategy: try candidate values that are likely to cover missing differences
        # Candidate: for each missing d and each s in S, candidate = s + d (if > S[-1])
        # Also try S[-1] + 1, S[-1] + 2, etc. for small values

        candidates = set()
        # Candidates that cover at least one missing difference
        for d in missing:
            for s in S:
                c = s + d
                if c > S[-1] and c <= max_allowed:
                    candidates.add(c)
        
        # Also try sequential values near the end
        for c in range(S[-1] + 1, min(S[-1] + 20, max_allowed + 1)):
            candidates.add(c)

        # Sort candidates for deterministic search, prefer smaller values
        for x in sorted(candidates):
            if check_deadline():
                return False
            
            if x > max_allowed:
                continue
            
            # Check if adding x creates any duplicate difference
            new_diffs = []
            conflict = False
            for s in S:
                d = x - s
                if d in used_diffs:
                    conflict = True
                    break
                new_diffs.append(d)
            
            if conflict:
                continue
            
            # Also check new diffs among themselves (they're all distinct since x-s1 != x-s2 for s1 != s2)
            # So no conflict among new diffs.
            
            new_used = used_diffs | set(new_diffs)
            new_missing = missing - set(new_diffs)
            new_S = S + [x]
            
            # Pruning: if max element already exceeds best
            if x > best_max[0]:
                continue
            
            if backtrack(new_S, new_used, new_missing, max_allowed):
                return True

        return False

    # Initial call: S = [0], used_diffs = set(), missing = set(1..30)
    # We'll try increasing max_allowed to find the minimum.
    # But for efficiency, let's start with a reasonable upper bound and search.
    
    # Lower bound on max: we need 30 differences. The smallest possible max for a Sidon set
    # with 30 specific differences... Let's just search.
    
    # Start with a generous upper bound and try to find any solution, then tighten.
    # Actually, let's do iterative deepening on max_allowed.
    
    # First, find any valid solution to establish an upper bound
    # Then try to improve.
    
    # Let's start with max_allowed = 1000 and search
    # If found, try smaller values.
    
    # Actually, for efficiency, let's just do a single backtracking search
    # that naturally finds small values first (since we try candidates in order).
    
    # Set a high initial bound
    initial_max = 5000
    
    # First search: find any solution
    S0 = [0]
    used0 = set()
    missing0 = set(range(1, 31))
    
    # Try to find a solution
    found = backtrack(S0, used0, missing0, initial_max)
    
    if best[0] is not None:
        # We have a solution. Try to improve by searching with tighter bounds.
        current_best = best_max[0]
        
        # Try to find better solutions with lower max
        # Binary search on max_allowed
        lo = 1
        hi = current_best - 1
        
        while lo <= hi and not check_deadline():
            mid = (lo + hi) // 2
            # Try to find a solution with max <= mid
            # Reset best for this search
            old_best = best[0]
            old_best_max = best_max[0]
            best[0] = None
            best_max[0] = mid + 1  # So any solution with max <= mid is better
            
            found = backtrack(S0, used0, missing0, mid)
            
            if best[0] is not None:
                hi = mid - 1
            else:
                lo = mid + 1
                # Restore
                best[0] = old_best
                best_max[0] = old_best_max
        
        # After binary search, do one more search at the optimal bound to get the actual set
        if best[0] is None:
            # Re-run at the best known max
            best[0] = None
            best_max[0] = 10**6
            backtrack(S0, used0, missing0, initial_max)
    
    if best[0] is None:
        # Fallback: use a simple construction
        # Greedy approach as fallback
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
            else:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a
    
    return best[0]


if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Max: {max(result)}")
    print(f"Size: {len(result)}")
    # Verify
    diffs = set()
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            d = result[j] - result[i]
            if d in diffs:
                print(f"ERROR: duplicate difference {d}")
            diffs.add(d)
    missing = set(range(1, 31)) - diffs
    if missing:
        print(f"ERROR: missing differences: {missing}")
    else:
        print("All differences 1..30 covered!")
