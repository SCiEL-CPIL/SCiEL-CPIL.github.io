"""Optimized construction for sumset/difference-set exponent lower bound."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _time_left(deadline):
    return deadline - time.time()

def _score_set(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or len(U) < 2:
        return 0.0
    U = sorted(set(U))
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    n = len(U)
    # Use numpy for speed
    arr = np.array(U, dtype=np.int64)
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    if sum_count == 0:
        return 0.0
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _score_set_fast(U):
    """Fast score using numpy, assumes U is sorted list of unique nonneg ints with 0."""
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    if sum_count == 0:
        return 0.0
    ratio = diff_count / sum_count
    denom = math.log(2 * max_u + 1)
    return 1.0 + math.log(ratio) / denom

def _singer_golomb(p):
    """Singer construction: Golomb ruler from finite field GF(p).
    Returns a set of size p with max element ~p^2."""
    if p < 3:
        return [0, 1]
    # Singer's construction using primitive root of GF(p)
    # For prime p, use the multiplicative group
    # Find primitive root
    def primitive_root(p):
        if p == 2:
            return 1
        for g in range(2, p):
            order = p - 1
            ok = True
            for q in range(2, int(math.isqrt(order)) + 1):
                if order % q == 0:
                    if pow(g, order // q, p) == 1:
                        ok = False
                        break
            if ok:
                return g
        return None
    
    g = primitive_root(p)
    if g is None:
        return [0, 1, 3]
    
    # Singer construction: elements are {0} union {g^i mod p : i=1..p-1} mapped
    # Actually the standard Singer Golomb ruler:
    # Take primitive element g in GF(p), the set {g^i mod p : i = 1, ..., p-1}
    # But we need it as integers. The classical construction:
    # U = {0} ∪ {g^i mod p : i = 1,...,p-1} doesn't directly work.
    
    # Better: Welch construction
    return _welch_golomb(p)

def _welch_golomb(p):
    """Welch construction for Golomb rulers of size p (prime)."""
    if p < 2:
        return [0, 1]
    # Find primitive root
    def primitive_root(p):
        if p == 2:
            return 1
        for g in range(2, p):
            order = p - 1
            ok = True
            for q in range(2, int(math.isqrt(order)) + 1):
                if order % q == 0:
                    if pow(g, order // q, p) == 1:
                        ok = False
                        break
            if ok:
                return g
        return None
    
    g = primitive_root(p)
    if g is None:
        return [0, 1, 3]
    
    # Welch: U = {0} ∪ {g^i mod p * p + (g^i mod p) : ...}
    # Actually simpler Welch: for prime p, U = {0, 1, ..., p-1} transformed
    # Standard Welch Golomb ruler of order p:
    # r_i = i * (i+1) // 2 mod p, then map to integers
    # Let me use a simpler known construction
    
    # Simple approach: use the set {i^2 mod p : i=0..p-1} won't work
    # Let me just use a greedy Golomb construction
    
    return _greedy_golomb(p)

def _greedy_golomb(n, deadline=None):
    """Greedy construction of Golomb ruler with n marks."""
    if deadline is None:
        deadline = time.time() + 10
    
    U = [0]
    for i in range(1, n):
        if time.time() > deadline - 5:
            break
        best_pos = None
        best_max_diff = -1
        # Try positions
        max_u = U[-1] if U else 0
        # Try from max_u+1 upward
        for pos in range(1, max_u + i + 50):
            if pos in U:
                continue
            # Check Golomb property: all pairwise differences unique
            new_diffs = set()
            ok = True
            for u in U:
                d = pos - u
                if d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            # Also need existing diffs not to collide
            existing_diffs = set()
            for a in range(len(U)):
                for b in range(a+1, len(U)):
                    existing_diffs.add(U[b] - U[a])
            if new_diffs & existing_diffs:
                continue
            # Valid Golomb position
            if best_pos is None or pos < best_pos:
                best_pos = pos
                break
        if best_pos is not None:
            U.append(best_pos)
        else:
            # Fallback: just add max_u + 1
            U.append(max_u + 1)
    return U

def _construct_sidon_sequence(n, deadline=None):
    """Construct a Sidon sequence (B2 sequence) of length n.
    Sidon: all pairwise sums are distinct (for i<=j).
    This minimizes |U+U| which helps our ratio."""
    if deadline is None:
        deadline = time.time() + 10
    
    U = [0, 1]
    for i in range(2, n):
        if time.time() > deadline - 5:
            break
        best_pos = None
        max_u = U[-1]
        for pos in range(2, max_u + i + 30):
            if pos in U:
                continue
            # Check Sidon: all sums a+b (a<=b) must be unique
            sums = set()
            ok = True
            for a in U:
                for b in U:
                    if a > b:
                        continue
                    s = a + b
                    if s in sums:
                        ok = False
                        break
                    sums.add(s)
                if not ok:
                    break
            if not ok:
                continue
            # Add pos and check
            new_sums = set()
            for a in U + [pos]:
                for b in U + [pos]:
                    if a > b:
                        continue
                    s = a + b
                    if s in new_sums:
                        ok = False
                        break
                    new_sums.add(s)
            if ok:
                best_pos = pos
                break
        if best_pos is not None:
            U.append(best_pos)
        else:
            U.append(max_u + 1)
    return U

def _local_search(U, deadline, max_iter=50000, rng_seed=42):
    """
    Local search with fast numpy scoring.
    Moves: replace an element, add an element, remove an element,
    or adjust gaps between consecutive elements.
    """
    U = sorted(set(U))
    if len(U) < 2:
        return U
    
    best_U = U[:]
    best_score = _score_set_fast(best_U)
    
    start_time = time.time()
    rng = random.Random(rng_seed)
    
    n = len(best_U)
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        
        move_type = rng.random()
        
        if move_type < 0.35:
            # Replace an element
            idx = rng.randint(1, len(best_U) - 1)
            old_val = best_U[idx]
            
            # Generate candidates
            candidates = set()
            for delta in range(-20, 21):
                v = old_val + delta
                if v > 0:
                    candidates.add(v)
            # Structured candidates
            for other in best_U:
                if other != old_val:
                    for delta in range(-3, 4):
                        v = other + delta
                        if v > 0 and v != old_val:
                            candidates.add(v)
            # Random
            max_u = max(best_U)
            for _ in range(15):
                v = rng.randint(1, max_u + 30)
                candidates.add(v)
            
            improved = False
            for v in sorted(candidates):
                if v in best_U:
                    continue
                new_U = best_U[:]
                new_U[idx] = v
                new_U = sorted(set(new_U))
                if 0 not in new_U:
                    continue
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    no_improve = 0
                    break
            
            if not improved:
                no_improve += 1
                
        elif move_type < 0.55:
            # Add an element
            max_u = best_U[-1]
            candidates = set()
            for u in best_U:
                for delta in range(-5, 6):
                    v = u + delta
                    if v > 0:
                        candidates.add(v)
            for i in range(len(best_U) - 1):
                gap = best_U[i+1] - best_U[i]
                if gap > 2:
                    mid = best_U[i] + gap // 2
                    for delta in range(-3, 4):
                        v = mid + delta
                        if 0 < v < best_U[i+1]:
                            candidates.add(v)
            for delta in range(1, 30):
                candidates.add(max_u + delta)
            for _ in range(30):
                v = rng.randint(1, max_u + 20)
                candidates.add(v)
            
            improved = False
            for v in sorted(candidates):
                if v in best_U:
                    continue
                new_U = sorted(best_U + [v])
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    no_improve = 0
                    break
            
            if not improved:
                no_improve += 1
                
        elif move_type < 0.70:
            # Remove an element
            if len(best_U) <= 2:
                no_improve += 1
                continue
            idx = rng.randint(1, len(best_U) - 1)
            new_U = best_U[:idx] + best_U[idx+1:]
            s = _score_set_fast(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                no_improve = 0
            else:
                no_improve += 1
                
        elif move_type < 0.85:
            # Gap adjustment: modify a gap between consecutive elements
            if len(best_U) < 3:
                no_improve += 1
                continue
            i = rng.randint(0, len(best_U) - 2)
            gap = best_U[i+1] - best_U[i]
            if gap <= 1:
                no_improve += 1
                continue
            # Try changing the gap
            for new_gap in range(1, min(gap + 5, gap + 15)):
                if new_gap == gap:
                    continue
                new_U = best_U[:]
                # Shift all elements after i
                delta = new_gap - gap
                for j in range(i + 1, len(new_U)):
                    new_U[j] += delta
                    if new_U[j] < 0:
                        break
                else:
                    if 0 not in new_U or len(set(new_U)) != len(new_U):
                        continue
                    s = _score_set_fast(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U
                        no_improve = 0
                        break
            else:
                no_improve += 1
                
        else:
            # Random large perturbation
            idx = rng.randint(1, len(best_U) - 1)
            for _ in range(20):
                delta = rng.randint(-50, 50)
                new_val = best_U[idx] + delta
                if new_val < 0:
                    continue
                new_U = best_U[:]
                new_U[idx] = new_val
                new_U = sorted(set(new_U))
                if 0 not in new_U:
                    continue
                s = _score_set_fast(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    no_improve = 0
                    break
            else:
                no_improve += 1
        
        # Occasional restart-like perturbation if stuck
        if no_improve > 200 and it % 100 == 0:
            # Try a fresh random element
            idx = rng.randint(1, len(best_U) - 1)
            new_val = rng.randint(1, max(best_U) + 100)
            new_U = best_U[:]
            new_U[idx] = new_val
            new_U = sorted(set(new_U))
            if 0 in new_U:
                s = _score_set_fast(new_U)
                if s > best_score - 0.01:  # Allow slight degradation for exploration
                    best_U = new_U
                    best_score = s
                    no_improve = 0
    
    return best_U

def _multi_start(deadline):
    """Try multiple different starting configurations."""
    starts = []
    
    # Golomb-like: superincreasing
    U = [0, 1]
    for i in range(1, 10):
        U.append(U[-1] * 2 + 1)
    starts.append(U)
    
    # Sidon-like: quadratic
    U = [i*i for i in range(12)]
    starts.append(U)
    
    # Fibonacci
    U = [0, 1]
    for i in range(1, 12):
        U.append(U[-1] + U[-2])
    starts.append(U)
    
    # Prime gaps
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    U = [0]
    for p in primes:
        U.append(U[-1] + p)
    starts.append(U)
    
    # Mixed radix
    U = [0, 1, 3, 10, 36, 130, 486, 1800, 6561]
    starts.append(U)
    
    # Sparse linear
    U = [0]
    for i in range(1, 10):
        U.append(i * 7)
    starts.append(U)
    
    # Cubes
    U = [i**3 for i in range(8)]
    starts.append(U)
    
    best_U = None
    best_score = -1
    
    for U in starts:
        if time.time() > deadline - 60:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    return best_U if best_U else [0, 1, 3]

def _greedy_construct(deadline):
    """Greedy construction starting from good seeds."""
    # Try multiple seeds and extend greedily
    seeds = [
        [0, 1, 3],
        [0, 1, 4, 10],
        [0, 1, 3, 7, 12],
        [0, 2, 5, 11],
    ]
    
    best_U = None
    best_score = -1
    
    for seed in seeds:
        if time.time() > deadline - 40:
            break
        U = seed[:]
        s = _score_set(U)
        # Try adding elements greedily
        for _ in range(15):
            if time.time() > deadline - 35:
                break
            max_u = U[-1]
            improved = False
            # Try positions
            for pos in range(1, max_u + 20):
                if pos in U:
                    continue
                new_U = sorted(U + [pos])
                ns = _score_set(new_U)
                if ns > s + 1e-12:
                    s = ns
                    U = new_U
                    improved = True
                    break
            if not improved:
                break
        if s > best_score:
            best_score = s
            best_U = U
    
    return best_U if best_U else [0, 1, 3]

def construct():
    deadline = time.time() + 130  # Safety margin from 150s
    
    # Multi-start to find a good initial set
    U = _multi_start(deadline)
    
    # Greedy extension
    if time.time() < deadline - 50:
        U2 = _greedy_construct(deadline)
        if _score_set(U2) > _score_set(U):
            U = U2
    
    # Local search - main workhorse
    if time.time() < deadline - 40:
        U = _local_search(U, deadline, max_iter=40000, rng_seed=42)
    
    # Second local search with different seed
    if time.time() < deadline - 25:
        U = _local_search(U, deadline, max_iter=30000, rng_seed=999)
    
    # Third pass with different seed
    if time.time() < deadline - 15:
        U = _local_search(U, deadline, max_iter=20000, rng_seed=12345)
    
    # Final polish: tight local search
    if time.time() < deadline - 10:
        U = _local_search(U, deadline, max_iter=15000, rng_seed=777)
    
    # Ensure 0 is in the set and all are nonnegative
    U = sorted(set([x for x in U if x >= 0]))
    if 0 not in U:
        U = [0] + U
    
    return U

# EVOLVE-BLOCK-END
