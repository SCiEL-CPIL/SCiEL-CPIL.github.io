"""Optimized search for finite set U maximizing sumset/difference-set exponent."""
import time
import random
from math import log
import numpy as np

# EVOLVE-BLOCK-START

def _score(U):
    """Fast score computation using numpy vectorized unique on flattened arrays."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    a = np.array(U, dtype=np.int64)
    # Use flattened outer operations to reduce memory overhead vs 2D arrays
    sums = np.unique(a[:, None] + a[None, :])
    n_sums = sums.size
    diffs = np.unique(a[:, None] - a[None, :])
    n_diffs = diffs.size
    
    ratio = n_diffs / n_sums
    if ratio <= 1:
        return -1
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def _has_distinct_diffs(U):
    """Check if all pairwise differences are distinct (maximizes |U-U|)."""
    n = len(U)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _score_geometric(r, n):
    """Score a geometric set {0, r, r^2, ..., r^(n-1)}."""
    if n < 2:
        return -1
    U = [0]
    val = 1
    for i in range(1, n):
        val *= r
        U.append(val)
    return _score(U)

def _continuous_geometric_search(deadline):
    """Search over geometric ratio r and size n to find good geometric sets."""
    best_score = -1
    best_U = [0, 1]
    for n in range(3, 14):
        if time.time() > deadline:
            break
        for r in range(2, 30):
            if time.time() > deadline:
                break
            U = [0]
            val = 1
            for i in range(1, n):
                val *= r
                U.append(val)
            s = _score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    return best_U, best_score

def _generate_sidon_like(n, max_val, seed=42):
    """Generate a set with distinct pairwise differences using greedy."""
    rng = random.Random(seed)
    U = [0]
    remaining = list(range(1, max_val + 1))
    rng.shuffle(remaining)
    for _ in range(n - 1):
        best = None
        best_score = -1
        for v in remaining[:min(50, len(remaining))]:
            candidate = U + [v]
            candidate.sort()
            if _has_distinct_diffs(candidate):
                # Prefer smaller values to keep max(U) small
                score = -v
                if score > best_score:
                    best_score = score
                    best = v
        if best is not None:
            U.append(best)
            remaining.remove(best)
            U.sort()
        else:
            break
    return sorted(U) if len(U) >= 2 else [0, 1]

def _generate_random_distinct(n, max_val, seed=42):
    """Generate random set, then try to fix collisions."""
    rng = random.Random(seed)
    U = [0] + sorted(rng.sample(range(1, max_val + 1), min(n - 1, max_val)))
    U.sort()
    # Try to fix collisions by shifting
    for _ in range(200):
        if _has_distinct_diffs(U):
            break
        # Find a collision and fix it
        diffs = {}
        for i in range(len(U)):
            for j in range(i + 1, len(U)):
                d = U[j] - U[i]
                if d in diffs:
                    # Move one of the elements
                    idx = rng.choice([i, j])
                    old = U[idx]
                    new = old + rng.choice([-3, -2, -1, 1, 2, 3])
                    if new > 0 and new < max_val + 10 and new not in U:
                        U[idx] = new
                        U.sort()
                else:
                    diffs[d] = (i, j)
    return U

def _simulated_annealing(U, deadline, rng, T0=5.0, cooling=0.995):
    """Simulated annealing to optimize the set."""
    U = sorted(U)
    current_score = _score(U)
    if current_score < 0:
        return U, current_score
    
    best_U = U[:]
    best_score = current_score
    T = T0
    
    while time.time() < deadline and T > 0.01:
        # Choose a random move
        move_type = rng.random()
        n = len(U)
        
        if move_type < 0.5 and n > 1:
            # Move one element
            i = rng.randint(1, n - 1)
            delta = rng.randint(-5, 5)
            if delta == 0:
                continue
            new_val = U[i] + delta
            if new_val <= 0 or new_val in U:
                continue
            new_U = U[:]
            new_U[i] = new_val
            new_U.sort()
        elif move_type < 0.7 and n > 2:
            # Swap two elements
            i, j = rng.sample(range(1, n), 2)
            new_U = U[:]
            new_U[i], new_U[j] = new_U[j], new_U[i]
            new_U.sort()
        elif move_type < 0.85 and n < 12:
            # Add an element
            max_u = U[-1]
            new_val = rng.randint(1, max_u + 20)
            if new_val not in U:
                new_U = sorted(U + [new_val])
            else:
                continue
        elif move_type < 0.95 and n > 3:
            # Remove an element
            i = rng.randint(1, n - 1)
            new_U = [U[j] for j in range(n) if j != i]
        else:
            # Random restart with small perturbation
            new_U = U[:]
            for _ in range(rng.randint(1, 3)):
                i = rng.randint(1, len(new_U) - 1)
                delta = rng.randint(-10, 10)
                if delta != 0:
                    new_val = new_U[i] + delta
                    if new_val > 0 and new_val not in new_U:
                        new_U[i] = new_val
                        new_U.sort()
        
        new_score = _score(new_U)
        if new_score < 0:
            new_score = -1
        
        # Accept if better or with probability exp((new - old)/T)
        if new_score >= current_score:
            U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]
        else:
            prob = exp_val = np.exp((new_score - current_score) / T) if T > 0.01 else 0
            if rng.random() < prob:
                U = new_U
                current_score = new_score
        
        T *= cooling
    
    return best_U, best_score

def _local_search(U, deadline, rng):
    """Greedy local search with multiple passes."""
    U = sorted(U)
    current_score = _score(U)
    if current_score < 0:
        return U, current_score
    
    best_U = U[:]
    best_score = current_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(U)
        
        # Try moving each element
        for i in range(1, n):
            for delta in range(-15, 16):
                if delta == 0:
                    continue
                new_val = U[i] + delta
                if new_val <= 0 or new_val in U:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U.sort()
                new_score = _score(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        # Try adding an element
        if not improved and n < 12:
            max_u = U[-1]
            for new_val in range(1, max_u + 30):
                if new_val not in U:
                    new_U = sorted(U + [new_val])
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        
        # Try removing an element
        if not improved and n > 3:
            for i in range(1, n):
                new_U = [U[j] for j in range(n) if j != i]
                new_score = _score(new_U)
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    improved = True
                    break
        
        # Try pairwise swaps
        if not improved:
            for i in range(1, n):
                for j in range(i + 1, n):
                    new_U = U[:]
                    new_U[i], new_U[j] = new_U[j], new_U[i]
                    new_U.sort()
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_U = U[:]
    
    return best_U, best_score

def _generate_parametric_candidates(deadline):
    """Generate candidates from parametric families that structurally maximize ratio."""
    candidates = []
    
    # Geometric progressions with various ratios
    for n in range(3, 13):
        if time.time() > deadline:
            break
        for r in range(2, 25):
            U = [0]
            val = 1
            for i in range(1, n):
                val *= r
                U.append(val)
            if max(U) < 10000:
                candidates.append(U)
    
    # Arithmetic progressions (these have bad ratio but good for baseline)
    for n in range(3, 12):
        if time.time() > deadline:
            break
        for d in range(1, 8):
            U = [0] + [d * i for i in range(1, n)]
            candidates.append(U)
    
    # Mixed: start geometric, then add arithmetic tail
    for base_n in range(3, 8):
        if time.time() > deadline:
            break
        for r in range(2, 8):
            U = [0]
            val = 1
            for i in range(1, base_n):
                val *= r
                U.append(val)
            # Add a few more elements with smaller gaps
            for extra in range(2, 6):
                new_val = U[-1] + extra
                if new_val not in U:
                    U.append(new_val)
            if max(U) < 5000:
                candidates.append(U)
    
    # Sparse random with large gaps (mimics Sidon-like structure)
    rng = random.Random(777)
    for _ in range(30):
        if time.time() > deadline:
            break
        n = rng.randint(4, 12)
        gap = rng.randint(3, 15)
        U = [0]
        cur = 0
        for i in range(n - 1):
            cur += gap + rng.randint(-1, 3)
            if cur <= 0:
                cur = 1
            U.append(cur)
        candidates.append(U)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    rng = random.Random(12345)
    
    best_set = [0, 1, 4, 9, 11]
    best_score = _score(best_set)
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Known good sets
    known = [
        [0, 1, 4, 9, 11],
        [0, 1, 4, 9, 11, 13],
        [0, 1, 4, 9, 11, 13, 15],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 3, 7, 12, 18],
        [0, 1, 4, 7, 10, 13, 16],
        [0, 1, 5, 9, 13, 17, 21],
        [0, 2, 3, 7, 12, 18, 25],
        [0, 1, 4, 8, 12, 16, 20],
        [0, 1, 3, 4, 9, 11],
        [0, 1, 4, 10, 12, 17, 23],
        # Add geometric candidates as known good starting points
        [0, 1, 2, 4, 8],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 9, 27],
        [0, 1, 3, 9, 27, 81],
        [0, 1, 4, 16, 64],
        [0, 1, 5, 25, 125],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 3, 9, 27, 81, 243],
    ]
    for k in known:
        candidates.append(k)
        
    # Generate parametric candidates
    param_candidates = _generate_parametric_candidates(deadline - 100)
    candidates.extend(param_candidates)
    
    # Add random sets to diversify
    for _ in range(40):
        if time.time() > deadline - 80:
            break
        n = rng.randint(3, 15)
        max_val = rng.randint(20, 200)
        U = [0] + sorted(rng.sample(range(1, max_val + 1), min(n - 1, max_val)))
        candidates.append(U)

    # Evaluate all candidates
    scored_candidates = []
    for c in candidates:
        if time.time() > deadline - 70:
            break
        s = _score(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
        if s > 0:
            scored_candidates.append((s, c))
            
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Phase 2: Simulated annealing from best candidates
    top_candidates = [c for _, c in scored_candidates[:min(20, len(scored_candidates))]]
    if best_set not in top_candidates:
        top_candidates.append(best_set)
        
    # Run SA on top candidates with adaptive time allocation
    sa_start = time.time()
    sa_end = deadline - 30
    n_sa = min(8, len(top_candidates))
    sa_per_candidate = (sa_end - sa_start) / n_sa if n_sa > 0 else 0
    
    for i, init in enumerate(top_candidates[:n_sa]):
        if time.time() > sa_end:
            break
        sa_deadline = min(sa_end, time.time() + sa_per_candidate * 1.5)
        result, score = _simulated_annealing(init, sa_deadline, rng, T0=5.0, cooling=0.995)
        if score > best_score:
            best_score = score
            best_set = result[:]
            
    # Phase 3: Local search polish on best
    if time.time() < deadline - 5:
        result, score = _local_search(best_set, deadline - 3, rng)
        if score > best_score:
            best_score = score
            best_set = result[:]
    
    # Phase 4: High-precision final polish with tighter moves
    if time.time() < deadline - 1:
        U = sorted(best_set)
        current_score = _score(U)
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            n = len(U)
            # Try fine-grained moves: delta in {-1, 1} first, then {-2, 2}, etc.
            for radius in [1, 2, 3, 5, 8, 13]:
                for i in range(1, n):
                    for delta in [-radius, radius]:
                        new_val = U[i] + delta
                        if new_val <= 0 or new_val in U:
                            continue
                        new_U = U[:]
                        new_U[i] = new_val
                        new_U.sort()
                        new_score = _score(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # Try adding/removing elements
            if not improved and n < 14:
                max_u = U[-1]
                for new_val in range(1, min(max_u + 15, 200)):
                    if new_val not in U:
                        new_U = sorted(U + [new_val])
                        new_score = _score(new_U)
                        if new_score > current_score:
                            U = new_U
                            current_score = new_score
                            improved = True
                            break
            if not improved and n > 3:
                for i in range(1, n):
                    new_U = [U[j] for j in range(n) if j != i]
                    new_score = _score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        if current_score > best_score:
            best_score = current_score
            best_set = U[:]
            
    # Final verification
    final_score = _score(best_set)
    if final_score < 0:
        best_set = [0, 1, 4, 9, 11]
        
    return best_set

# EVOLVE-BLOCK-END
