import math
import time
import random

def _score(U):
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for u in U:
        for v in U:
            sumset.add(u + v)
            diffset.add(u - v)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _score_geometric(base, n):
    """Compute score for U = {0, 1, floor(base^1), floor(base^2), ..., floor(base^(n-1))}"""
    if n <= 0:
        return 0
    if n == 1:
        return 1.0
    
    # Build the set
    U = [0]
    for i in range(1, n):
        U.append(int(round(base ** i)))
    U = sorted(set(U))
    
    if not U or 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    for u in U:
        for v in U:
            sumset.add(u + v)
            diffset.add(u - v)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _ternary_search_1d(f, lo, hi, iters=60):
    """Ternary search for maximum of unimodal function f on [lo, hi]."""
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = f(m1)
        f2 = f(m2)
        if f1 < f2:
            lo = m1
        else:
            hi = m2
    return f((lo + hi) / 2.0)

def _optimize_geometric(deadline):
    """Find best geometric set by ternary search over base for various n."""
    best_score = 0
    best_U = [0, 1, 3]
    
    for n in range(2, 15):
        if time.time() > deadline:
            break
        
        # Define objective: score as function of base
        def obj(base):
            return _score_geometric(base, n)
        
        # Base range: from just above 1 to a reasonable upper bound
        # For larger n, base should be smaller to keep max_u manageable
        lo = 1.01
        hi = min(10.0, 100.0 ** (1.0 / max(1, n - 1)))  # Keep max_u <= 100 roughly
        
        # Ensure hi > lo
        if hi <= lo:
            hi = lo * 1.5
        
        best_for_n = _ternary_search_1d(obj, lo, hi, iters=40)
        
        if best_for_n > best_score:
            best_score = best_for_n
            # Reconstruct the set at optimal base
            # Find the base that gave this score
            # We need to search more carefully
            # Do a finer search to find the exact base
            def obj2(base):
                return _score_geometric(base, n)
            
            # Binary search style: try several points
            best_base = lo
            best_s = 0
            for frac in range(0, 101):
                b = lo + frac * (hi - lo) / 100.0
                s = obj2(b)
                if s > best_s:
                    best_s = s
                    best_base = b
            
            U = [0]
            for i in range(1, n):
                U.append(int(round(best_base ** i)))
            U = sorted(set(U))
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = U
    
    return best_U, best_score

def _local_search(U, max_iterations=500, deadline=None):
    U = sorted(set(U))
    best_score = _score(U)
    best_U = U
    current_U = U
    current_score = best_score
    
    for i in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        max_u = max(current_U)
        
        if random.random() < 0.5:
            new_val = random.randint(1, max_u * 2)
        else:
            new_val = random.randint(1, max_u + 10)
        
        candidate = sorted(set(current_U + [new_val]))
        s = _score(candidate)
        if s > current_score:
            current_U = candidate
            current_score = s
            if s > best_score:
                best_score = s
                best_U = candidate
        
        if len(current_U) > 1:
            idx = random.randint(1, len(current_U) - 1)
            candidate = sorted(set(current_U[:idx] + current_U[idx+1:]))
            s = _score(candidate)
            if s > current_score:
                current_U = candidate
                current_score = s
                if s > best_score:
                    best_score = s
                    best_U = candidate
        else:
            candidate = [0, 1]
            s = _score(candidate)
            if s > current_score:
                current_U = candidate
                current_score = s
                if s > best_score:
                    best_score = s
                    best_U = candidate
        
        if len(current_U) > 1:
            idx = random.randint(1, len(current_U) - 1)
            shift = random.randint(-5, 5)
            if shift == 0:
                continue
            new_val = current_U[idx] + shift
            if new_val >= 0:
                candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
                if len(candidate) < len(current_U):
                    pass
                s = _score(candidate)
                if s > current_score:
                    current_U = candidate
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_U = candidate
                        
    return best_U

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin
    
    # Phase 1: Ternary search over geometric progressions
    geo_U, geo_score = _optimize_geometric(deadline)
    
    best_U = geo_U
    best_score = geo_score
    
    # Phase 2: Local search refinement from the geometric seed
    if time.time() < deadline:
        refined = _local_search(best_U, max_iterations=300, deadline=deadline)
        s_r = _score(refined)
        if s_r > best_score:
            best_score = s_r
            best_U = refined
    
    # Phase 3: Try a few random starts with local search
    for _ in range(5):
        if time.time() > deadline:
            break
        n = random.randint(3, 10)
        U = [0]
        max_val = random.randint(10, 200)
        for i in range(n - 1):
            U.append(random.randint(1, max_val))
        U = sorted(set(U))
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = U
        
        refined = _local_search(U, max_iterations=200, deadline=deadline)
        s_r = _score(refined)
        if s_r > best_score:
            best_score = s_r
            best_U = refined
    
    return best_U
