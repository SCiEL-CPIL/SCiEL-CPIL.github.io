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

def _greedy_construct(pool_A, pool_B, max_n=256, max_size=64, rng=None):
    """
    Greedy construction given difference pools.
    Returns (A, B) or (None, None) if fails.
    """
    if rng is None:
        rng = random.Random()
    
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    used_diffs_A = set()
    used_diffs_B = set()
    
    def can_add_A(x):
        if len(A) >= max_size:
            return False
        # Check Sidon sums
        for a in A:
            if x + a in sums_A:
                return False
        if 2 * x in sums_A:
            return False
        # Check differences
        for a in A:
            d = abs(x - a)
            if d > 0:
                if d not in pool_A or d in used_diffs_A:
                    return False
        return True
    
    def can_add_B(x):
        if len(B) >= max_size:
            return False
        for b in B:
            if x + b in sums_B:
                return False
        if 2 * x in sums_B:
            return False
        for b in B:
            d = abs(x - b)
            if d > 0:
                if d not in pool_B or d in used_diffs_B:
                    return False
        return True
    
    def add_A(x):
        A.append(x)
        for a in A[:-1]:
            sums_A.add(x + a)
        sums_A.add(2 * x)
        for a in A[:-1]:
            d = abs(x - a)
            if d > 0:
                used_diffs_A.add(d)
    
    def add_B(x):
        B.append(x)
        for b in B[:-1]:
            sums_B.add(x + b)
        sums_B.add(2 * x)
        for b in B[:-1]:
            d = abs(x - b)
            if d > 0:
                used_diffs_B.add(d)
    
    # Try adding elements in increasing order, with some randomization
    order = list(range(1, max_n + 1))
    rng.shuffle(order)
    
    for x in order:
        # Try A first if it's smaller, otherwise B
        if len(A) < len(B) or (len(A) == len(B) and rng.random() < 0.5):
            if can_add_A(x):
                add_A(x)
                continue
        if can_add_B(x):
            add_B(x)
    
    if not A or not B:
        return None, None
    
    # Final check
    if not _check_sidon(A) or not _check_sidon(B):
        return None, None
    if _diffs(A) & _diffs(B):
        return None, None
    
    return A, B

def _backtrack_construct(max_n=256, max_size=64, deadline=None, rng=None):
    """
    Backtracking search assigning each x in 1..max_n to A, B, or Neither.
    Uses forward checking on Sidon sums and difference disjointness.
    """
    if rng is None:
        rng = random.Random()
    
    A = []
    B = []
    sums_A = set()
    sums_B = set()
    used_diffs_A = set()
    used_diffs_B = set()
    
    def can_add_A(x):
        if len(A) >= max_size:
            return False
        for a in A:
            if x + a in sums_A:
                return False
        if 2 * x in sums_A:
            return False
        for a in A:
            d = abs(x - a)
            if d > 0 and d in used_diffs_B:
                return False
        return True
    
    def can_add_B(x):
        if len(B) >= max_size:
            return False
        for b in B:
            if x + b in sums_B:
                return False
        if 2 * x in sums_B:
            return False
        for b in B:
            d = abs(x - b)
            if d > 0 and d in used_diffs_A:
                return False
        return True
    
    def add_A(x):
        A.append(x)
        for a in A[:-1]:
            sums_A.add(x + a)
        sums_A.add(2 * x)
        for a in A[:-1]:
            d = abs(x - a)
            if d > 0:
                used_diffs_A.add(d)
    
    def add_B(x):
        B.append(x)
        for b in B[:-1]:
            sums_B.add(x + b)
        sums_B.add(2 * x)
        for b in B[:-1]:
            d = abs(x - b)
            if d > 0:
                used_diffs_B.add(d)
    
    def backtrack(x, depth):
        if deadline and time.time() > deadline:
            return False
        if x > max_n:
            return len(A) > 0 and len(B) > 0
        
        # Try A
        if can_add_A(x):
            add_A(x)
            if backtrack(x + 1, depth + 1):
                return True
            A.pop()
            for a in A:
                sums_A.discard(x + a)
            sums_A.discard(2 * x)
            for a in A:
                d = abs(x - a)
                if d > 0:
                    used_diffs_A.discard(d)
        
        # Try B
        if can_add_B(x):
            add_B(x)
            if backtrack(x + 1, depth + 1):
                return True
            B.pop()
            for b in B:
                sums_B.discard(x + b)
            sums_B.discard(2 * x)
            for b in B:
                d = abs(x - b)
                if d > 0:
                    used_diffs_B.discard(d)
        
        # Try Neither
        return backtrack(x + 1, depth + 1)
    
    # Use randomized order for backtracking
    order = list(range(1, max_n + 1))
    rng.shuffle(order)
    
    # Map original indices to shuffled order
    # We'll process in shuffled order but track which values we've assigned
    assigned = set()
    
    def backtrack_shuffled(idx):
        if deadline and time.time() > deadline:
            return False
        if idx >= len(order):
            return len(A) > 0 and len(B) > 0
        
        x = order[idx]
        
        # Try A
        if can_add_A(x):
            add_A(x)
            assigned.add(x)
            if backtrack_shuffled(idx + 1):
                return True
            assigned.discard(x)
            A.pop()
            for a in A:
                sums_A.discard(x + a)
            sums_A.discard(2 * x)
            for a in A:
                d = abs(x - a)
                if d > 0:
                    used_diffs_A.discard(d)
        
        # Try B
        if can_add_B(x):
            add_B(x)
            assigned.add(x)
            if backtrack_shuffled(idx + 1):
                return True
            assigned.discard(x)
            B.pop()
            for b in B:
                sums_B.discard(x + b)
            sums_B.discard(2 * x)
            for b in B:
                d = abs(x - b)
                if d > 0:
                    used_diffs_B.discard(d)
        
        # Try Neither
        return backtrack_shuffled(idx + 1)
    
    success = backtrack_shuffled(0)
    if success and A and B:
        return A, B
    return None, None

def _local_search_improve(A, B, deadline, rng):
    """
    Local search: try to add elements, swap elements between A and B.
    """
    A = list(A)
    B = list(B)
    max_n = 256
    max_size = 64
    
    def get_sums(S):
        s = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s.add(S[i] + S[j])
        return s
    
    def get_diffs(S):
        return _diffs(S)
    
    sums_A = get_sums(A)
    sums_B = get_sums(B)
    diffs_A = get_diffs(A)
    diffs_B = get_diffs(B)
    all_A = set(A)
    all_B = set(B)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding to A
        candidates = [x for x in range(1, max_n + 1) if x not in all_A and x not in all_B]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(A) >= max_size:
                break
            
            # Check if x can be added to A
            ok = True
            new_sums = set(sums_A)
            new_diffs = set(diffs_A)
            for a in A:
                if x + a in new_sums:
                    ok = False
                    break
                new_sums.add(x + a)
                d = abs(x - a)
                if d > 0:
                    if d in diffs_B:
                        ok = False
                        break
                    new_diffs.add(d)
            if ok:
                if 2 * x in new_sums:
                    ok = False
                else:
                    new_sums.add(2 * x)
            
            if ok:
                A.append(x)
                sums_A = new_sums
                diffs_A = new_diffs
                all_A.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Try adding to B
        candidates = [x for x in range(1, max_n + 1) if x not in all_A and x not in all_B]
        rng.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(B) >= max_size:
                break
            
            ok = True
            new_sums = set(sums_B)
            new_diffs = set(diffs_B)
            for b in B:
                if x + b in new_sums:
                    ok = False
                    break
                new_sums.add(x + b)
                d = abs(x - b)
                if d > 0:
                    if d in diffs_A:
                        ok = False
                        break
                    new_diffs.add(d)
            if ok:
                if 2 * x in new_sums:
                    ok = False
                else:
                    new_sums.add(2 * x)
            
            if ok:
                B.append(x)
                sums_B = new_sums
                diffs_B = new_diffs
                all_B.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap: remove from A, add to B
        if len(A) > 1:
            idx = rng.randint(0, len(A) - 1)
            removed = A[idx]
            A_temp = A[:idx] + A[idx+1:]
            sums_temp = get_sums(A_temp)
            diffs_temp = get_diffs(A_temp)
            all_A_temp = set(A_temp)
            
            for x in range(1, max_n + 1):
                if x in all_A_temp or x in all_B:
                    continue
                if len(B) >= max_size:
                    break
                
                ok = True
                new_sums = set(sums_B)
                new_diffs = set(diffs_B)
                for b in B:
                    if x + b in new_sums:
                        ok = False
                        break
                    new_sums.add(x + b)
                    d = abs(x - b)
                    if d > 0:
                        if d in diffs_temp:
                            ok = False
                            break
                        new_diffs.add(d)
                if ok:
                    if 2 * x in new_sums:
                        ok = False
                    else:
                        new_sums.add(2 * x)
                
                if ok:
                    B.append(x)
                    sums_B = new_sums
                    diffs_B = new_diffs
                    all_B.add(x)
                    A = A_temp
                    sums_A = sums_temp
                    diffs_A = diffs_temp
                    all_A = all_A_temp
                    improved = True
                    break
            
            if not improved:
                A = A_temp + [removed]
                A.sort()
                all_A = set(A)
                sums_A = get_sums(A)
                diffs_A = get_diffs(A)
        
        # Try swap: remove from B, add to A
        if len(B) > 1 and not improved:
            idx = rng.randint(0, len(B) - 1)
            removed = B[idx]
            B_temp = B[:idx] + B[idx+1:]
            sums_temp = get_sums(B_temp)
            diffs_temp = get_diffs(B_temp)
            all_B_temp = set(B_temp)
            
            for x in range(1, max_n + 1):
                if x in all_A or x in all_B_temp:
                    continue
                if len(A) >= max_size:
                    break
                
                ok = True
                new_sums = set(sums_A)
                new_diffs = set(diffs_A)
                for a in A:
                    if x + a in new_sums:
                        ok = False
                        break
                    new_sums.add(x + a)
                    d = abs(x - a)
                    if d > 0:
                        if d in diffs_temp:
                            ok = False
                            break
                        new_diffs.add(d)
                if ok:
                    if 2 * x in new_sums:
                        ok = False
                    else:
                        new_sums.add(2 * x)
                
                if ok:
                    A.append(x)
                    sums_A = new_sums
                    diffs_A = new_diffs
                    all_A.add(x)
                    B = B_temp
                    sums_B = sums_temp
                    diffs_B = diffs_temp
                    all_B = all_B_temp
                    improved = True
                    break
            
            if not improved:
                B = B_temp + [removed]
                B.sort()
                all_B = set(B)
                sums_B = get_sums(B)
                diffs_B = get_diffs(B)
    
    return A, B

def construct():
    """
    Main entry point. Uses backtracking + greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    best_A, best_B = [1], [3]
    best_score = 2
    max_n = 256
    max_size = 64
    
    # Strategy 1: Greedy with random difference pools
    for rnd in range(15):
        if time.time() > deadline - 10.0:
            break
        
        rng = random.Random(rnd * 104729 + 7)
        
        # Random partition of differences
        pool_A = set()
        pool_B = set()
        for d in range(1, max_n):
            r = rng.random()
            if r < 0.45:
                pool_A.add(d)
            elif r < 0.90:
                pool_B.add(d)
        
        if not pool_A:
            pool_A.add(1)
        if not pool_B:
            pool_B.add(2)
        
        A, B = _greedy_construct(pool_A, pool_B, max_n, max_size, rng)
        
        if A and B:
            # Local search
            ls_deadline = min(deadline - 5.0, time.time() + 3.0)
            A, B = _local_search_improve(A, B, ls_deadline, rng)
            
            valid, score = _validate(A, B)
            if valid and score > best_score:
                best_score = score
                best_A, best_B = list(A), list(B)
        
        if best_score >= 40:
            break
    
    # Strategy 2: Backtracking with different random seeds
    for seed in range(5):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(seed * 31337 + 42)
        bt_deadline = min(deadline - 2.0, time.time() + 5.0)
        
        A, B = _backtrack_construct(max_n, max_size, bt_deadline, rng)
        
        if A and B:
            valid, score = _validate(A, B)
            if valid and score > best_score:
                best_score = score
                best_A, best_B = list(A), list(B)
        
        if best_score >= 40:
            break
    
    # Final local search on best solution
    if time.time() < deadline - 2.0:
        rng = random.Random(999)
        A, B = _local_search_improve(best_A, best_B, deadline - 1.0, rng)
        valid, score = _validate(A, B)
        if valid and score > best_score:
            best_score = score
            best_A, best_B = list(A), list(B)
    
    # Final validation
    if not best_A or not best_B:
        best_A, best_B = [1], [3]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
