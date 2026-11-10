# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # Safety margin before 90s kill
    N = 512
    MAXC = 48
    rng = random.Random(42)

    def _deadline_exceeded():
        return time.time() >= DEADLINE

    def _is_mean_free(s):
        """
        Exact check using integer bitset DP.
        """
        n = len(s)
        if n < 3:
            return True
        for i in range(n):
            a = s[i]
            dp = [0] * (n + 1)
            dp[0] = 1
            for j in range(n):
                if j == i:
                    continue
                v = s[j]
                for k in range(n, 0, -1):
                    dp[k] |= (dp[k - 1] << v)
            for k in range(2, n):
                if (dp[k] >> (k * a)) & 1:
                    return False
        return True

    def _find_violators(s):
        """Return set of indices that are violators."""
        n = len(s)
        if n < 3:
            return set()
        violators = set()
        for i in range(n):
            a = s[i]
            dp = [0] * (n + 1)
            dp[0] = 1
            for j in range(n):
                if j == i:
                    continue
                v = s[j]
                for k in range(n, 0, -1):
                    dp[k] |= (dp[k - 1] << v)
            for k in range(2, n):
                if (dp[k] >> (k * a)) & 1:
                    violators.add(i)
                    break
        return violators

    def _quick_2_conflict(s, x, s_set):
        """Check if adding x creates any 2-element mean conflict."""
        for a in s:
            b = 2 * x - a
            if b in s_set and b != a and b != x:
                return False
            b2 = 2 * a - x
            if b2 in s_set and b2 != a and b2 != x:
                return False
        return True

    def _greedy_build(seed_rng):
        """Build a set using greedy with random tie-breaking."""
        s = []
        s_set = set()
        remaining = list(range(1, N + 1))
        seed_rng.shuffle(remaining)
        
        # Start with some spread-out values
        for v in [1, 3, 7, 15, 31, 63, 127, 255]:
            if v <= N and v not in s_set:
                s.append(v)
                s_set.add(v)
        
        while len(s) < MAXC and not _deadline_exceeded():
            best_x = None
            best_score = None
            for x in remaining:
                if x in s_set:
                    continue
                if not _quick_2_conflict(s, x, s_set):
                    continue
                # Score: prefer larger gaps to existing elements
                score = 0
                for a in s:
                    score += abs(x - a)
                if best_score is None or score > best_score:
                    best_score = score
                    best_x = x
                    if score > 200:
                        break
            if best_x is None:
                # Try without 2-conflict check
                for x in remaining:
                    if x in s_set:
                        continue
                    test_s = s + [x]
                    if _is_mean_free(test_s):
                        best_x = x
                        break
            if best_x is None:
                break
            s.append(best_x)
            s_set.add(best_x)
            remaining.remove(best_x)
        s.sort()
        return s

    def _local_search(s, s_set):
        """Hill-climbing with add and swap moves."""
        s = list(s)
        s_set = set(s)
        
        # Phase 1: Add elements
        improved = True
        while improved and not _deadline_exceeded():
            improved = False
            for x in range(1, N + 1):
                if _deadline_exceeded():
                    break
                if x in s_set:
                    continue
                if len(s) >= MAXC:
                    break
                if not _quick_2_conflict(s, x, s_set):
                    continue
                test_s = s + [x]
                if _is_mean_free(test_s):
                    s = test_s
                    s_set.add(x)
                    improved = True
                    break
        
        # Phase 2: Swap moves (replace one element with another)
        improved = True
        while improved and not _deadline_exceeded():
            improved = False
            for i in range(len(s)):
                if _deadline_exceeded():
                    break
                old = s[i]
                for x in range(1, N + 1):
                    if _deadline_exceeded():
                        break
                    if x in s_set:
                        continue
                    new_s = [v for j, v in enumerate(s) if j != i] + [x]
                    new_s.sort()
                    if not _is_mean_free(new_s):
                        continue
                    # Accept if same size (swap) - we want to maintain size
                    s = new_s
                    s_set = set(s)
                    improved = True
                    break
                if improved:
                    break
        
        return s, s_set

    best_s = []
    best_set = set()
    
    # Multiple random restarts
    num_restarts = 0
    while not _deadline_exceeded():
        num_restarts += 1
        seed_rng = random.Random(1000 + num_restarts * 7919)
        s = _greedy_build(seed_rng)
        
        # Local search
        s, s_set = _local_search(s, set(s))
        
        # Ensure validity
        if not _is_mean_free(s):
            violators = _find_violators(s)
            for v in sorted(violators, reverse=True):
                s.pop(v)
            s_set = set(s)
        
        if len(s) > len(best_s):
            best_s = s
            best_set = set(s)
        
        if len(best_s) >= MAXC:
            break
        
        # If we've been stuck, try a different strategy
        if num_restarts % 5 == 0 and not _deadline_exceeded():
            # Try building from a different base
            base = [2**i for i in range(1, 9) if 2**i <= N]
            s = list(base)
            s_set = set(s)
            # Greedy add
            for x in range(1, N + 1):
                if _deadline_exceeded():
                    break
                if len(s) >= MAXC:
                    break
                if x in s_set:
                    continue
                if not _quick_2_conflict(s, x, s_set):
                    continue
                test_s = s + [x]
                if _is_mean_free(test_s):
                    s = test_s
                    s_set.add(x)
            s.sort()
            s, s_set = _local_search(s, s_set)
            if not _is_mean_free(s):
                violators = _find_violators(s)
                for v in sorted(violators, reverse=True):
                    s.pop(v)
                s_set = set(s)
            if len(s) > len(best_s):
                best_s = s
                best_set = set(s)
    
    # Final validation
    if not _is_mean_free(best_s):
        while not _is_mean_free(best_s) and not _deadline_exceeded():
            violators = _find_violators(best_s)
            if not violators:
                break
            for v in sorted(violators, reverse=True):
                best_s.pop(v)
    
    return best_s

# EVOLVE-BLOCK-END
