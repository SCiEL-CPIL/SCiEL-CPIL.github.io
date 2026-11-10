"""Rewritten program: polynomial-sequence search for sum-product extremal sets."""
import time
import random
from functools import lru_cache

# EVOLVE-BLOCK-START
import numpy as np

def _poly_eval(coeffs, xs):
    """Evaluate polynomial with coefficients (highest degree first) at xs."""
    # coeffs[0]*x^d + coeffs[1]*x^(d-1) + ... + coeffs[d]
    result = np.zeros_like(xs, dtype=np.int64)
    for c in coeffs:
        result = result * xs + c
    return result

def _make_set_from_poly(coeffs, n=20, max_val=10000):
    """Generate sorted distinct positive integers from polynomial at 0..n-1."""
    xs = np.arange(n, dtype=np.int64)
    vals = _poly_eval(coeffs, xs)
    vals = vals.tolist()
    if any(v <= 0 or v > max_val for v in vals):
        return None
    if len(set(vals)) != n:
        return None
    return sorted(vals)

def _score(A):
    """Compute max(|A+A|, |A*A|)."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _score_and_components(A):
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return len(sums), len(prods), max(len(sums), len(prods))

def _local_search(A, max_iter=2000, deadline=None):
    """Local search: swap elements or perturb to reduce score."""
    current = sorted(A)
    current_score = _score(current)
    best = current[:]
    best_score = current_score
    n = len(current)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        # Random perturbation: swap two elements or replace one
        op = random.random()
        cand = current[:]
        if op < 0.5:
            # Swap two elements
            i, j = random.sample(range(n), 2)
            cand[i], cand[j] = cand[j], cand[i]
        elif op < 0.8:
            # Replace one element with nearby value
            i = random.randrange(n)
            delta = random.randint(-5, 5)
            cand[i] = current[i] + delta
        else:
            # Replace with random value
            i = random.randrange(n)
            cand[i] = random.randint(1, 10000)
        
        cand.sort()
        if len(set(cand)) != n or any(v < 1 or v > 10000 for v in cand):
            continue
        
        s = _score(cand)
        if s <= current_score:
            current = cand
            current_score = s
            if s < best_score:
                best = cand[:]
                best_score = s
    return best, best_score

def _search_polynomial(deadline):
    """Search over low-degree polynomials to find good sets."""
    best_set = None
    best_score = float('inf')
    
    # Try different degree polynomials
    for degree in [1, 2, 3]:
        for trial in range(5000):
            if time.time() > deadline:
                break
            # Generate random coefficients ensuring positive values in [1, 10000]
            if degree == 1:
                # Linear: a*x + b, need a>0, b>=1, a*19+b <= 10000
                a = random.randint(1, 500)
                b = random.randint(1, 10000 - a * 19)
                if b < 1:
                    continue
                coeffs = [a, b]
            elif degree == 2:
                # Quadratic: a*x^2 + b*x + c
                a = random.randint(1, 30)
                b = random.randint(-100, 100)
                c = random.randint(1, 500)
                coeffs = [a, b, c]
            else:
                # Cubic: a*x^3 + b*x^2 + c*x + d
                a = random.randint(1, 5)
                b = random.randint(-20, 20)
                c = random.randint(-50, 50)
                d = random.randint(1, 200)
                coeffs = [a, b, c, d]
            
            A = _make_set_from_poly(coeffs)
            if A is None:
                continue
            s = _score(A)
            if s < best_score:
                best_score = s
                best_set = A[:]
                # Local search from here
                if time.time() < deadline - 5:
                    ls_set, ls_score = _local_search(A, max_iter=500, deadline=deadline - 2)
                    if ls_score < best_score:
                        best_score = ls_score
                        best_set = ls_set
    
    return best_set, best_score

def _search_structured(deadline):
    """Try structured families: arithmetic progressions, geometric-like, mixed."""
    best_set = None
    best_score = float('inf')
    
    # Arithmetic progressions
    for start in range(1, 50):
        for step in range(1, 500):
            if start + 19 * step > 10000:
                break
            A = [start + i * step for i in range(20)]
            s = _score(A)
            if s < best_score:
                best_score = s
                best_set = A[:]
    
    # Quadratic sequences (parabolic)
    for a in range(1, 50):
        for b in range(-20, 20):
            for c in range(1, 200):
                A = [a * i * i + b * i + c for i in range(20)]
                if any(v < 1 or v > 10000 for v in A) or len(set(A)) != 20:
                    continue
                A.sort()
                s = _score(A)
                if s < best_score:
                    best_score = s
                    best_set = A[:]
                if time.time() > deadline:
                    return best_set, best_score
    
    # Exponential-like (powers of small base with perturbations)
    for base in range(2, 10):
        for start in range(1, 5):
            A = [start * (base ** i) for i in range(20)]
            if max(A) > 10000:
                continue
            s = _score(A)
            if s < best_score:
                best_score = s
                best_set = A[:]
    
    # Mixed: small AP + large AP
    for split in range(5, 16):
        for s1 in range(1, 20):
            for d1 in range(1, 50):
                for s2 in range(100, 500):
                    for d2 in range(1, 500):
                        A1 = [s1 + i * d1 for i in range(split)]
                        A2 = [s2 + i * d2 for i in range(20 - split)]
                        A = sorted(A1 + A2)
                        if len(set(A)) != 20 or max(A) > 10000 or min(A) < 1:
                            continue
                        s = _score(A)
                        if s < best_score:
                            best_score = s
                            best_set = A[:]
                        if time.time() > deadline:
                            return best_set, best_score
    
    return best_set, best_score

def _refine_with_local_search(A, deadline):
    """Aggressive local search from a starting set."""
    current = sorted(A)
    current_score = _score(current)
    best = current[:]
    best_score = current_score
    n = len(current)
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        for _ in range(200):
            if time.time() > deadline:
                break
            op = random.random()
            cand = current[:]
            if op < 0.4:
                i, j = random.sample(range(n), 2)
                cand[i], cand[j] = cand[j], cand[i]
            elif op < 0.7:
                i = random.randrange(n)
                delta = random.randint(-10, 10)
                cand[i] = current[i] + delta
            else:
                i = random.randrange(n)
                # Perturb by replacing with value near another element
                j = random.randrange(n)
                delta = random.randint(-5, 5)
                cand[i] = current[j] + delta
            
            cand.sort()
            if len(set(cand)) != n or any(v < 1 or v > 10000 for v in cand):
                no_improve += 1
                continue
            
            s = _score(cand)
            if s < current_score:
                current = cand
                current_score = s
                improved = True
                if s < best_score:
                    best = cand[:]
                    best_score = s
            else:
                no_improve += 1
        if not improved:
            no_improve += 100
        else:
            no_improve = 0
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    random.seed(42)
    
    # Phase 1: Structured search (fast, deterministic)
    if time.time() < deadline - 30:
        s_set, s_score = _search_structured(deadline - 30)
    else:
        s_set, s_score = None, float('inf')
    
    # Phase 2: Polynomial search
    if time.time() < deadline - 20:
        p_set, p_score = _search_polynomial(deadline - 20)
    else:
        p_set, p_score = None, float('inf')
    
    # Determine best starting point
    best_set = None
    best_score = float('inf')
    for candidate, score in [(s_set, s_score), (p_set, p_score)]:
        if candidate is not None and score < best_score:
            best_score = score
            best_set = candidate
    
    if best_set is None:
        # Fallback: linear sequence
        best_set = list(range(1, 21))
        best_score = _score(best_set)
    
    # Phase 3: Local search refinement
    if time.time() < deadline - 5:
        refined, refined_score = _refine_with_local_search(best_set, deadline - 2)
        if refined_score < best_score:
            best_set = refined
            best_score = refined_score
    
    return best_set
# EVOLVE-BLOCK-END
