"""
Combined approach: Structural insight (greedy max-diff/min-sum construction with sum-free
binary/carry-avoidance bias) + Refinement machinery (fast numpy scoring, continuous
geometric search, dynamic candidate generation, simulated annealing, and local search).
"""
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

def _score_fast(U):
    """Even faster score for tight loops - avoids numpy overhead for small sets."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1

    # For small sets, pure Python sets can be faster than numpy
    if n <= 20:
        sums = set()
        diffs = set()
        for i in range(n):
            ui = U[i]
            for j in range(i, n):
                sums.add(ui + U[j])
            for j in range(i + 1, n):
                d = U[j] - ui
                diffs.add(d)
                diffs.add(-d)
        diffs.add(0)
        n_sums = len(sums)
        n_diffs = len(diffs)
    else:
        a = np.array(U, dtype=np.int64)
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

def _generate_recurrence_set(coeffs, n_terms, a0=0, a1=1):
    """
    Generate a set from a linear recurrence with characteristic roots of modulus > 1.
    coeffs = [c1, c2, ..., ck] defines a_{n+k} = c1*a_{n+k-1} + ... + ck*a_n
    For modulus > 1, terms grow exponentially, creating large gaps that favor |U-U|/|U+U|.
    """
    if n_terms < 2:
        return [0]
    
    # Build initial sequence
    seq = []
    if len(coeffs) == 0:
        # Geometric: a_{n+1} = a0 (just constant growth from a1)
        seq.append(a0)
        val = a1
        for _ in range(n_terms - 1):
            seq.append(val)
            val = val * max(2, abs(coeffs[0]) if coeffs else 2)
    else:
        k = len(coeffs)
        # Initialize with a0, a1, and derived initial terms
        seq.append(a0)
        seq.append(a1)
        for i in range(2, k):
            # Use simple growth for initial terms
            seq.append(a1 * (i + 1))
        
        # Generate remaining terms via recurrence
        for _ in range(n_terms - k):
            if len(seq) < k:
                break
            next_val = 0
            for j, c in enumerate(coeffs):
                next_val += c * seq[-(j + 1)]
            seq.append(next_val)
    
    # Convert to nonnegative integers, ensure 0 is present, sort
    U = set()
    for v in seq:
        if v >= 0:
            U.add(int(v))
    if 0 not in U:
        U.add(0)
    
    return sorted(U)

def _optimize_recurrence_params(deadline):
    """
    Search over recurrence coefficients to maximize the sumset/difference-set exponent.
    Focuses on 2nd and 3rd order recurrences with integer coefficients.
    """
    best_score = -1
    best_U = [0, 1]
    
    # 2nd order: a_{n+2} = c1*a_{n+1} + c2*a_n
    # Characteristic equation: x^2 - c1*x - c2 = 0
    # For modulus > 1, need |x_i| > 1 for both roots
    for c1 in range(1, 12):
        for c2 in range(-3, 6):
            if time.time() > deadline:
                break
            for n_terms in range(4, 14):
                if time.time() > deadline:
                    break
                # Check if roots have modulus > 1
                # Discriminant: c1^2 + 4*c2
                disc = c1 * c1 + 4 * c2
                if disc < 0:
                    # Complex roots: modulus = sqrt(|c2|)
                    if abs(c2) <= 1:
                        continue
                else:
                    # Real roots
                    import math
                    sqrt_disc = math.sqrt(disc)
                    r1 = (c1 + sqrt_disc) / 2
                    r2 = (c1 - sqrt_disc) / 2
                    if abs(r1) <= 1 or abs(r2) <= 1:
                        continue
                
                U = _generate_recurrence_set([c1, c2], n_terms)
                if len(U) < 3:
                    continue
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    # 3rd order: a_{n+3} = c1*a_{n+2} + c2*a_{n+1} + c3*a_n
    for c1 in range(1, 8):
        for c2 in range(-2, 5):
            for c3 in range(-2, 4):
                if time.time() > deadline:
                    break
                for n_terms in range(4, 12):
                    if time.time() > deadline:
                        break
                    U = _generate_recurrence_set([c1, c2, c3], n_terms)
                    if len(U) < 3:
                        continue
                    s = _score(U)
                    if s > best_score:
                        best_score = s
                        best_U = U[:]
    
    return best_U, best_score

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

def _greedy_construct_sum_free(target_n, max_val, seed=42):
    """
    Greedily construct a set by selecting elements that maximize the increase
    in |U-U| while minimizing the increase in |U+U|.
    Uses a bias toward values that avoid carries in binary addition (sum-free structure).
    """
    rng = random.Random(seed)
    U = [0]
    U_set = set(U)
    current_sums = {0}
    current_diffs = {0}
    
    # Precompute candidate pool: prefer numbers with sparse binary representations
    # to avoid carries, which keeps |U+U| small
    candidates = []
    for v in range(1, max_val + 1):
        # Score candidate by binary sparsity (fewer 1-bits = less carry risk)
        sparsity = bin(v).count('1')
        # Also prefer larger gaps from existing elements for diff diversity
        candidates.append((sparsity, v))
    
    # Sort by sparsity (ascending), then by value (ascending)
    candidates.sort()
    
    for _ in range(target_n - 1):
        best_candidate = None
        best_delta_score = -float('inf')
        
        # Try top candidates (by sparsity) first
        for sparsity, v in candidates[:min(200, len(candidates))]:
            if v in U_set:
                continue
            
            # Calculate increase in sums and diffs
            new_sums = set()
            new_diffs = set()
            for u in U:
                s = u + v
                if s not in current_sums:
                    new_sums.add(s)
                d = u - v
                if d not in current_diffs:
                    new_diffs.add(d)
                d2 = v - u
                if d2 not in current_diffs:
                    new_diffs.add(d2)
            
            delta_sums = len(new_sums)
            delta_diffs = len(new_diffs)
            
            # We want to maximize delta_diffs / (delta_sums + 1)
            # The +1 avoids division by zero and reflects the baseline
            if delta_sums == 0:
                delta_score = delta_diffs * 1000  # Huge bonus for no new sums
            else:
                delta_score = delta_diffs / (delta_sums + 1)
            
            # Bonus for larger values (increases D = 2*max+1 in denominator of log)
            # but not too large to keep the set compact
            value_bonus = min(v / max_val, 1.0) * 0.1
            
            total_score = delta_score + value_bonus
            
            if total_score > best_delta_score:
                best_delta_score = total_score
                best_candidate = v
        
        if best_candidate is None:
            break
        
        U.append(best_candidate)
        U_set.add(best_candidate)
        U.sort()
        
        # Update sums and diffs incrementally
        for u in U[:-1]:
            current_sums.add(u + best_candidate)
            current_diffs.add(u - best_candidate)
            current_diffs.add(best_candidate - u)
    
    return U

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
    
    # Greedy sum-free constructions with different seeds and parameters
    for seed in range(10):
        if time.time() > deadline:
            break
        for target_n in [5, 7, 10, 13]:
            for max_val in [100, 300, 800]:
                U = _greedy_construct_sum_free(target_n, max_val, seed=seed + 1000)
                if len(U) >= 3:
                    candidates.append(U)
    
    # Recurrence-based candidates: linear recurrences with modulus > 1
    rec_deadline = min(deadline, time.time() + 30)  # Cap at 30s for recurrence search
    rec_U, rec_score = _optimize_recurrence_params(rec_deadline)
    if rec_score > 0:
        candidates.append(rec_U)
    
    # Also generate several specific recurrence families
    # Fibonacci-like: a_{n+2} = a_{n+1} + a_n
    for n_terms in range(5, 15):
        if time.time() > deadline:
            break
        U = _generate_recurrence_set([1, 1], n_terms)
        if len(U) >= 3 and max(U) < 100000:
            candidates.append(U)
    
    # Lucas-like: a_{n+2} = a_{n+1} + a_n with different initial
    for n_terms in range(5, 15):
        if time.time() > deadline:
            break
        U = _generate_recurrence_set([1, 1], n_terms, a0=1, a1=2)
        if len(U) >= 3 and max(U) < 100000:
            candidates.append(U)
    
    # Higher growth: a_{n+2} = 2*a_{n+1} + a_n
    for n_terms in range(4, 12):
        if time.time() > deadline:
            break
        U = _generate_recurrence_set([2, 1], n_terms)
        if len(U) >= 3 and max(U) < 100000:
            candidates.append(U)
    
    # a_{n+2} = 3*a_{n+1} + a_n
    for n_terms in range(4, 10):
        if time.time() > deadline:
            break
        U = _generate_recurrence_set([3, 1], n_terms)
        if len(U) >= 3 and max(U) < 100000:
            candidates.append(U)
    
    # 3rd order: a_{n+3} = a_{n+2} + a_{n+1} + a_n (Tribonacci)
    for n_terms in range(5, 12):
        if time.time() > deadline:
            break
        U = _generate_recurrence_set([1, 1, 1], n_terms)
        if len(U) >= 3 and max(U) < 100000:
            candidates.append(U)
    
    # a_{n+3} = 2*a_{n+2} + a_{n+1} + a_n
    for n_terms in range(4, 10):
        if time.time() > deadline:
            break
        U = _generate_recurrence_set([2, 1, 1], n_terms)
        if len(U) >= 3 and max(U) < 100000:
            candidates.append(U)
    
    return candidates

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
            prob = np.exp((new_score - current_score) / T) if T > 0.01 else 0
            if rng.random() < prob:
                U = new_U
                current_score = new_score
        
        T *= cooling
    
    return best_U, best_score

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
        # Add recurrence-based known good sets
        [0, 1, 1, 2, 3, 5, 8, 13],  # Fibonacci
        [0, 1, 2, 3, 5, 8, 13, 21],  # Fibonacci shifted
        [0, 1, 3, 4, 7, 11, 18],     # a_{n+2} = 2*a_{n+1} + a_n
        [0, 1, 3, 7, 17, 41],        # Higher growth recurrence
    ]
    for k in known:
        # Deduplicate
        k_sorted = sorted(set(k))
        if 0 in k_sorted and len(k_sorted) >= 2:
            candidates.append(k_sorted)
        
    # Generate parametric candidates (includes greedy sum-free and recurrence-based)
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
