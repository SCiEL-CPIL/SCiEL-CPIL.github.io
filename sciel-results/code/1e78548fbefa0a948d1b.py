import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 80.0  # seconds, safety margin before 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _evaluate(a):
    """Return (|A+A|, |A*A|, max)."""
    a = np.asarray(a, dtype=np.int64)
    n = len(a)
    s = set()
    for i in range(n):
        for j in range(i, n):
            s.add(a[i] + a[j])
    p = set()
    for i in range(n):
        for j in range(i, n):
            p.add(a[i] * a[j])
    return len(s), len(p), max(len(s), len(p))

def _evaluate_np(a):
    """Faster evaluate using numpy for sums."""
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    # Sumset
    s = set()
    for i in range(n):
        vals = a[i] + a[i:]
        s.update(vals.tolist())
    # Productset
    p = set()
    for i in range(n):
        vals = a[i] * a[i:]
        p.update(vals.tolist())
    return len(s), len(p), max(len(s), len(p))

def _is_valid(a, n=20, maximum=10000):
    if len(a) != n:
        return False
    if len(set(a)) != n:
        return False
    if min(a) < 1:
        return False
    if max(a) > maximum:
        return False
    return True

def _generate_ap_union(n=20, max_val=10000, seed=0):
    """Generate sets that are unions of two APs."""
    rng = np.random.RandomState(seed)
    best = None
    best_score = float('inf')
    
    # Try various combinations of two APs
    # A1: start1, diff1, len1
    # A2: start2, diff2, len2
    # Total length n
    
    for trial in range(200):
        len1 = rng.randint(5, n - 4)
        len2 = n - len1
        
        diff1 = rng.randint(1, 10)
        diff2 = rng.randint(1, 10)
        
        start1 = rng.randint(1, max(1, max_val // (len1 * diff1 + 1)))
        
        # Ensure A1 fits
        end1 = start1 + (len1 - 1) * diff1
        if end1 > max_val:
            continue
            
        A1 = [start1 + i * diff1 for i in range(len1)]
        
        # For A2, try to overlap or be close to A1 to encourage collisions?
        # Or just random.
        start2 = rng.randint(1, max(1, max_val // (len2 * diff2 + 1)))
        
        end2 = start2 + (len2 - 1) * diff2
        if end2 > max_val:
            continue
            
        A2 = [start2 + i * diff2 for i in range(len2)]
        
        vals = A1 + A2
        if len(set(vals)) != n:
            # If duplicates, try to fix by shifting A2
            # Or just skip
            # Let's try to resolve duplicates by shifting
            used = set(A1)
            new_A2 = []
            cur_start2 = start2
            attempts = 0
            while len(new_A2) < len2 and attempts < 50:
                val = cur_start2 + len(new_A2) * diff2
                if val not in used and 1 <= val <= max_val:
                    new_A2.append(val)
                    used.add(val)
                else:
                    cur_start2 += 1
                    if cur_start2 + (len2 - len(new_A2) - 1) * diff2 > max_val:
                        break
                    # Reset new_A2? No, just try next slot
                    # Actually, if we shift start, we should restart A2 construction
                    new_A2 = []
                    attempts += 1
            if len(new_A2) == len2:
                vals = A1 + new_A2
            else:
                continue
                
        if len(set(vals)) != n:
            continue
            
        s, p, sc = _evaluate_np(vals)
        if sc < best_score:
            best_score = sc
            best = list(vals)
            
    return best, best_score

def _local_search(a, n=20, max_val=10000, time_budget=50.0):
    """Local search to improve the set."""
    a = list(a)
    s, p, best_score = _evaluate_np(a)
    best_a = list(a)
    deadline = time.time() + time_budget
    
    # Hill climbing with random restarts
    while time.time() < deadline:
        improved = False
        # Try swapping or modifying elements
        for i in range(n):
            if time.time() > deadline:
                break
            # Try replacing a[i] with nearby values
            current = a[i]
            # Try small perturbations
            candidates = []
            for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10]:
                nv = current + delta
                if 1 <= nv <= max_val and nv not in a[:i] and nv not in a[i+1:]:
                    candidates.append(nv)
            # Also try some random values
            for _ in range(10):
                nv = int(np.random.randint(1, max_val + 1))
                if nv not in a[:i] and nv not in a[i+1:]:
                    candidates.append(nv)
            
            for nv in candidates:
                a[i] = nv
                _, _, sc = _evaluate_np(a)
                if sc < best_score:
                    best_score = sc
                    best_a = list(a)
                    improved = True
                    break
                a[i] = current
        
        if not improved:
            # Random restart from best
            a = list(best_a)
            # Perturb multiple elements
            for _ in range(3):
                idx = int(np.random.randint(0, n))
                a[idx] = int(np.random.randint(1, max_val + 1))
            # Make valid
            while len(set(a)) < n or min(a) < 1 or max(a) > max_val:
                idx = int(np.random.randint(0, n))
                a[idx] = int(np.random.randint(1, max_val + 1))
            s, p, sc = _evaluate_np(a)
            if sc < best_score:
                best_score = sc
                best_a = list(a)
    
    return best_a, best_score

def _structured_candidates(n=20, max_val=10000):
    """Generate several structured candidate sets."""
    candidates = []
    
    # Geometric progressions
    for ratio in [2, 3, 4, 5]:
        for start in [1, 2, 3, 4, 5, 10, 20]:
            vals = [start * (ratio ** i) for i in range(n)]
            if all(v <= max_val for v in vals) and len(set(vals)) == n:
                candidates.append(vals)
    
    # Near-geometric with small perturbations
    base = 2
    ratio = 2
    vals = [base * (ratio ** i) for i in range(n)]
    if all(v <= max_val for v in vals):
        candidates.append(vals)
    
    # Clustered (arithmetic progression)
    for start in [1, 10, 50, 100]:
        for diff in [1, 2, 3, 5, 10]:
            vals = [start + i * diff for i in range(n)]
            if all(1 <= v <= max_val for v in vals) and len(set(vals)) == n:
                candidates.append(vals)
    
    # Powers of 2
    vals = [2**i for i in range(n)]
    if all(v <= max_val for v in vals):
        candidates.append(vals)
    
    # Mixed: powers of 2 times a constant
    for c in [1, 2, 3, 4, 5]:
        vals = [c * (2**i) for i in range(n)]
        if all(v <= max_val for v in vals) and len(set(vals)) == n:
            candidates.append(vals)
    
    return candidates

def construct():
    # Deadline guard
    if time.time() - _START_TIME > 55.0:
        return list(range(1, 21))
    
    n = 20
    max_val = 10000
    
    # Phase 1: Generate structured candidates
    best_a = list(range(1, 21))
    best_score = _evaluate_np(best_a)[2]
    
    # Try structured candidates
    candidates = _structured_candidates(n, max_val)
    for cand in candidates:
        if time.time() - _START_TIME > 55.0:
            break
        s, p, sc = _evaluate_np(cand)
        if sc < best_score:
            best_score = sc
            best_a = list(cand)
    
    # Phase 2: AP Union generation
    if time.time() - _START_TIME < 50.0:
        ap_a, ap_score = _generate_ap_union(n, max_val, seed=42)
        if ap_a is not None and ap_score < best_score:
            best_score = ap_score
            best_a = list(ap_a)
    
    # Phase 3: Local search
    if time.time() - _START_TIME < 50.0:
        remaining = min(35.0, 55.0 - (time.time() - _START_TIME))
        best_a, best_score = _local_search(best_a, n, max_val, remaining)
    
    # Validate
    if not _is_valid(best_a, n, max_val):
        best_a = list(range(1, 21))
    
    return best_a
