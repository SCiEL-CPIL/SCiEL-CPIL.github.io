# EVOLVE-BLOCK-START
import time
import random
import sys

def _check_sidon(S):
    """Check Sidon property: all pair sums (with replacement) are unique."""
    seen = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _diffs(S):
    """Return set of absolute nonzero differences."""
    d = set()
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(S[i] - S[j])
            if diff > 0:
                d.add(diff)
    return d

def _validate(A, B, max_n=256):
    """Validate solution. Returns (is_valid, score)."""
    if not A or not B:
        return False, 0
    if len(A) > 64 or len(B) > 64:
        return False, 0
    for x in A:
        if x < 1 or x > max_n:
            return False, 0
    for x in B:
        if x < 1 or x > max_n:
            return False, 0
    if not _check_sidon(A):
        return False, 0
    if not _check_sidon(B):
        return False, 0
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False, 0
    return True, len(A) + len(B)

def _build_sidon_with_diffs(allowed_diffs, max_n, max_size, rng=None):
    """
    Build a Sidon set using only differences in allowed_diffs.
    Greedy: iterate 1..max_n, add x if it maintains Sidon and all
    new differences are in allowed_diffs.
    """
    S = []
    sums = set()
    diffs = set()
    
    for x in range(1, max_n + 1):
        if len(S) >= max_size:
            break
        # Check Sidon
        ok = True
        new_sums = set()
        for a in S:
            s = x + a
            if s in sums:
                ok = False
                break
            new_sums.add(s)
        if ok:
            sx = x + x
            if sx in sums:
                ok = False
            else:
                new_sums.add(sx)
        if not ok:
            continue
        # Check differences
        new_diffs = set()
        for a in S:
            d = abs(x - a)
            if d > 0:
                if d not in allowed_diffs:
                    ok = False
                    break
                new_diffs.add(d)
        if not ok:
            continue
        # All good, add x
        S.append(x)
        sums = sums | new_sums
        diffs = diffs | new_diffs
    
    return S

def _build_sidon_with_diffs_random_order(allowed_diffs, max_n, max_size, rng):
    """
    Build a Sidon set using only differences in allowed_diffs,
    but iterate in random order for diversity.
    """
    candidates = list(range(1, max_n + 1))
    rng.shuffle(candidates)
    
    S = []
    sums = set()
    diffs = set()
    
    for x in candidates:
        if len(S) >= max_size:
            break
        # Check Sidon
        ok = True
        new_sums = set()
        for a in S:
            s = x + a
            if s in sums:
                ok = False
                break
            new_sums.add(s)
        if ok:
            sx = x + x
            if sx in sums:
                ok = False
            else:
                new_sums.add(sx)
        if not ok:
            continue
        # Check differences
        new_diffs = set()
        for a in S:
            d = abs(x - a)
            if d > 0:
                if d not in allowed_diffs:
                    ok = False
                    break
                new_diffs.add(d)
        if not ok:
            continue
        # All good, add x
        S.append(x)
        sums = sums | new_sums
        diffs = diffs | new_diffs
    
    S.sort()
    return S

def _partition_greedy(max_n=256, max_size=64, rng=None):
    """
    Greedy partition of differences: assign each d to A or B based on
    which set currently "needs" it more, or randomly.
    Then build Sidon sets from each partition.
    """
    if rng is None:
        rng = random.Random(0)
    
    all_diffs = list(range(1, max_n))
    rng.shuffle(all_diffs)
    
    # Start with a random 50/50 split
    half = len(all_diffs) // 2
    diff_A = set(all_diffs[:half])
    diff_B = set(all_diffs[half:])
    
    A = _build_sidon_with_diffs(diff_A, max_n, max_size, rng)
    B = _build_sidon_with_diffs(diff_B, max_n, max_size, rng)
    
    return A, B, diff_A, diff_B

def _partition_random(max_n=256, max_size=64, rng=None):
    """
    Random partition of differences, then build Sidon sets.
    Try multiple random splits.
    """
    if rng is None:
        rng = random.Random(0)
    
    best_A, best_B = [], []
    best_score = 0
    best_dA, best_dB = set(), set()
    
    for trial in range(20):
        all_diffs = list(range(1, max_n))
        rng.shuffle(all_diffs)
        
        # Random split ratio between 0.3 and 0.7
        ratio = rng.uniform(0.3, 0.7)
        split = int(len(all_diffs) * ratio)
        
        diff_A = set(all_diffs[:split])
        diff_B = set(all_diffs[split:])
        
        A = _build_sidon_with_diffs_random_order(diff_A, max_n, max_size, rng)
        B = _build_sidon_with_diffs_random_order(diff_B, max_n, max_size, rng)
        
        valid, score = _validate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
            best_dA, best_dB = set(diff_A), set(diff_B)
    
    return best_A, best_B, best_dA, best_dB, best_score

def _local_search_partition(A, B, diff_A, diff_B, max_n=256, max_size=64, rng=None, time_budget=10.0):
    """
    Local search on the difference partition: move a difference from one pool
    to another, rebuild, and accept if it improves.
    """
    if rng is None:
        rng = random.Random(0)
    
    start = time.time()
    A = list(A)
    B = list(B)
    diff_A = set(diff_A)
    diff_B = set(diff_B)
    
    valid, best_score = _validate(A, B)
    best_A, best_B = list(A), list(B)
    
    while time.time() - start < time_budget:
        # Pick a random difference to reassign
        all_diffs = list(diff_A | diff_B)
        if not all_diffs:
            break
        d = rng.choice(all_diffs)
        
        # Move d from its current pool to the other
        if d in diff_A:
            diff_A.remove(d)
            diff_B.add(d)
        else:
            diff_B.remove(d)
            diff_A.add(d)
        
        # Rebuild
        A_new = _build_sidon_with_diffs(diff_A, max_n, max_size, rng)
        B_new = _build_sidon_with_diffs(diff_B, max_n, max_size, rng)
        
        valid_new, score_new = _validate(A_new, B_new)
        if valid_new and score_new >= best_score:
            A, B = A_new, B_new
            best_A, best_B = list(A), list(B)
            best_score = score_new
        else:
            # Revert
            if d in diff_B and d not in diff_A:
                diff_B.remove(d)
                diff_A.add(d)
            else:
                diff_A.remove(d)
                diff_B.add(d)
    
    return best_A, best_B, best_score

def _greedy_fallback(max_n=256, max_size=64):
    """Simple greedy fallback if partitioning fails."""
    A = []
    B = []
    diff_A = set()
    diff_B = set()
    sums_A = set()
    sums_B = set()
    
    for x in range(1, max_n + 1):
        # Try A
        if len(A) < max_size:
            ok = True
            new_sums = set()
            for a in A:
                s = x + a
                if s in sums_A:
                    ok = False
                    break
                new_sums.add(s)
            if ok:
                sx = x + x
                if sx in sums_A:
                    ok = False
                else:
                    new_sums.add(sx)
            if ok:
                new_diffs = set()
                for a in A:
                    d = abs(x - a)
                    if d > 0:
                        if d in diff_A or d in diff_B:
                            ok = False
                            break
                        new_diffs.add(d)
            if ok:
                A.append(x)
                sums_A = sums_A | new_sums
                diff_A = diff_A | new_diffs
                continue
        
        # Try B
        if len(B) < max_size:
            ok = True
            new_sums = set()
            for b in B:
                s = x + b
                if s in sums_B:
                    ok = False
                    break
                new_sums.add(s)
            if ok:
                sx = x + x
                if sx in sums_B:
                    ok = False
                else:
                    new_sums.add(sx)
            if ok:
                new_diffs = set()
                for b in B:
                    d = abs(x - b)
                    if d > 0:
                        if d in diff_A or d in diff_B:
                            ok = False
                            break
                        new_diffs.add(d)
            if ok:
                B.append(x)
                sums_B = sums_B | new_sums
                diff_B = diff_B | new_diffs
    
    return A, B

def _local_search_elements(A, B, max_n=256, max_size=64, rng=None, time_budget=5.0):
    """
    Element-level local search: try swapping elements to improve total size.
    """
    if rng is None:
        rng = random.Random(0)
    
    start = time.time()
    A = list(A)
    B = list(B)
    best_A, best_B = list(A), list(B)
    best_score = len(A) + len(B)
    
    used = set(A) | set(B)
    available = [x for x in range(1, max_n + 1) if x not in used]
    
    while time.time() - start < time_budget:
        # Try adding to A
        if available and len(A) < max_size:
            for _ in range(5):  # try a few random candidates
                if not available:
                    break
                idx = rng.randrange(len(available))
                x = available[idx]
                A_new = A + [x]
                A_new.sort()
                if _check_sidon(A_new):
                    dA_new = _diffs(A_new)
                    dB = _diffs(B)
                    if not (dA_new & dB):
                        A = A_new
                        best_A = list(A)
                        best_score = len(A) + len(B)
                        available.pop(idx)
                        used.add(x)
                        break
        
        # Try adding to B
        if available and len(B) < max_size:
            for _ in range(5):
                if not available:
                    break
                idx = rng.randrange(len(available))
                x = available[idx]
                B_new = B + [x]
                B_new.sort()
                if _check_sidon(B_new):
                    dB_new = _diffs(B_new)
                    dA = _diffs(A)
                    if not (dB_new & dA):
                        B = B_new
                        best_B = list(B)
                        best_score = len(A) + len(B)
                        available.pop(idx)
                        used.add(x)
                        break
        
        # Try swap in A
        if A and available:
            i = rng.randrange(len(A))
            old = A[i]
            A_tmp = A[:i] + A[i+1:]
            for _ in range(3):
                if not available:
                    break
                idx = rng.randrange(len(available))
                x = available[idx]
                if x == old:
                    continue
                A_new = A_tmp + [x]
                A_new.sort()
                if _check_sidon(A_new):
                    dA_new = _diffs(A_new)
                    dB = _diffs(B)
                    if not (dA_new & dB):
                        A = A_new
                        best_A = list(A)
                        best_score = len(A) + len(B)
                        available.pop(idx)
                        used.discard(old)
                        used.add(x)
                        available.append(old)
                        break
        
        # Try swap in B
        if B and available:
            i = rng.randrange(len(B))
            old = B[i]
            B_tmp = B[:i] + B[i+1:]
            for _ in range(3):
                if not available:
                    break
                idx = rng.randrange(len(available))
                x = available[idx]
                if x == old:
                    continue
                B_new = B_tmp + [x]
                B_new.sort()
                if _check_sidon(B_new):
                    dB_new = _diffs(B_new)
                    dA = _diffs(A)
                    if not (dB_new & dA):
                        B = B_new
                        best_B = list(B)
                        best_score = len(A) + len(B)
                        available.pop(idx)
                        used.discard(old)
                        used.add(x)
                        available.append(old)
                        break
    
    return best_A, best_B, best_score

def construct():
    """
    Main function: difference-partitioning model with local search.
    """
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    best_A, best_B = [1], [3]
    best_score = 2
    
    # Phase 1: Multiple random difference partitions
    for seed in range(15):
        if time.time() > deadline - 15.0:
            break
        rng = random.Random(seed)
        A, B, dA, dB, score = _partition_random(256, 64, rng)
        valid, score = _validate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
    
    # Phase 2: Local search on difference partition for best solution
    if time.time() < deadline - 10.0:
        rng = random.Random(100)
        # Reconstruct partition from best solution
        dA = _diffs(best_A)
        dB = _diffs(best_B)
        A_ls, B_ls, score_ls = _local_search_partition(
            best_A, best_B, dA, dB, max_n=256, max_size=64,
            rng=rng, time_budget=min(10.0, deadline - 10.0 - time.time())
        )
        valid, score = _validate(A_ls, B_ls)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A_ls), list(B_ls)
    
    # Phase 3: Element-level local search
    if time.time() < deadline - 5.0:
        for seed in range(5):
            if time.time() > deadline - 3.0:
                break
            rng = random.Random(200 + seed)
            A_ls, B_ls, score_ls = _local_search_elements(
                best_A, best_B, max_n=256, max_size=64,
                rng=rng, time_budget=min(5.0, deadline - 3.0 - time.time())
            )
            valid, score = _validate(A_ls, B_ls)
            if valid and score > best_score:
                best_score = score
                best_A, best_B = list(A_ls), list(B_ls)
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
