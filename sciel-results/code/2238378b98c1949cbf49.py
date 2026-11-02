"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft
from sympy import isprime

def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set using FFT
    total_len = 4 * max_u + 1
    fft_len = 1
    while fft_len < total_len:
        fft_len *= 2
    
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    freq = fft(indicator)
    
    # Sumset
    sum_conv = ifft(freq * freq).real
    sumset_size = int(np.sum(sum_conv[:2*max_u+1] > 1e-9))
    
    # Difference set via autocorrelation
    autocorr = ifft(freq * np.conj(freq)).real
    diffset_size = 1
    for k in range(1, max_u + 1):
        if autocorr[k] > 1e-9:
            diffset_size += 1
        if autocorr[fft_len - k] > 1e-9:
            diffset_size += 1
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    if n <= 20:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    else:
        return _score_set_fft(U)

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using greedy construction."""
    if n <= 1:
        return [0]
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    }
    if n in known:
        return known[n]
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _quadratic_residues_mod_p(p, include_zero=True):
    """Get quadratic residues modulo p (as integers 0..p-1)."""
    residues = set()
    for x in range(p):
        residues.add((x * x) % p)
    if not include_zero:
        residues.discard(0)
    return sorted(residues)

def _multiplicative_subgroup_mod_p(p, generator, size):
    """Get a multiplicative subgroup of Z_p^* of given size."""
    if size == 1:
        return [1]
    # Find a generator of the subgroup of order 'size'
    # The multiplicative group has order p-1
    # A subgroup of order 'size' exists if size divides p-1
    if (p - 1) % size != 0:
        return []
    # g^((p-1)/size) generates the subgroup of order size
    g = pow(generator, (p - 1) // size, p)
    subgroup = set()
    x = 1
    for i in range(size):
        subgroup.add(x)
        x = (x * g) % p
    return sorted(subgroup)

def _finite_field_set(p, method='quadratic', subgroup_size=None, scale=1):
    """Construct a set U from finite field structure.
    
    method: 'quadratic' for quadratic residues, 'subgroup' for multiplicative subgroup
    Returns sorted list of nonnegative integers containing 0.
    """
    if method == 'quadratic':
        residues = _quadratic_residues_mod_p(p, include_zero=True)
        U = [0] + [r * scale for r in residues if r > 0]
    elif method == 'subgroup':
        if subgroup_size is None:
            subgroup_size = min(10, (p-1)//2)
        # Find a primitive root
        gen = None
        for g in range(2, p):
            if all(pow(g, (p-1)//q, p) != 1 for q in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]):
                gen = g
                break
        if gen is None:
            # Fallback: use 2
            gen = 2
        subgroup = _multiplicative_subgroup_mod_p(p, gen, subgroup_size)
        U = [0] + [s * scale for s in subgroup if s > 0]
    else:
        U = [0]
    
    return sorted(set(U))

def _local_search(U, max_iters=5000, deadline=None):
    """Improve set U via local search (swap elements)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 300
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try swapping one element
        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            candidates = set()
            for offset in range(-50, 51):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            for _ in range(20):
                candidates.add(random.randint(1, upper_bound))
            
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < 400:
                for _ in range(40):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        break
            
            if not improved:
                # Try removing an element
                if len(U) > 2:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
                            break
        
        if not improved:
            # Random perturbation
            if random.random() < 0.2:
                new_U = U[:]
                for _ in range(min(6, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
            
            if not improved:
                break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    candidates = []
    random.seed(42)
    
    # === Phase 1: Finite field constructions ===
    # Quadratic residues in various primes
    primes_to_try = []
    p = 5
    while len(primes_to_try) < 40 and p < 500:
        if isprime(p):
            primes_to_try.append(p)
        p += 1
    
    for p in primes_to_try:
        if time.time() > deadline - 50:
            break
        # Quadratic residues
        U = _finite_field_set(p, method='quadratic', scale=1)
        if len(U) >= 2:
            candidates.append(U)
        
        # Also try with different scales
        for scale in [2, 3]:
            U2 = _finite_field_set(p, method='quadratic', scale=scale)
            if len(U2) >= 2 and max(U2) < 20000:
                candidates.append(U2)
    
    # Multiplicative subgroups
    for p in primes_to_try[:20]:
        if time.time() > deadline - 50:
            break
        for sub_size in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
            if (p - 1) % sub_size == 0:
                U = _finite_field_set(p, method='subgroup', subgroup_size=sub_size, scale=1)
                if len(U) >= 2:
                    candidates.append(U)
                # Scaled versions
                for scale in [2, 3, 5]:
                    U2 = _finite_field_set(p, method='subgroup', subgroup_size=sub_size, scale=scale)
                    if len(U2) >= 2 and max(U2) < 20000:
                        candidates.append(U2)
    
    # === Phase 2: Traditional constructions ===
    # Golomb rulers
    for n in range(3, 20):
        if time.time() > deadline - 50:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Random sets
    for trial in range(30):
        if time.time() > deadline - 50:
            break
        n = random.randint(3, 50)
        max_val = random.randint(20, 1000)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Arithmetic progressions
    for base in [2, 3, 5, 7, 11]:
        if time.time() > deadline - 50:
            break
        U = [0]
        for i in range(1, 20):
            U.append(i * base)
        candidates.append(U)
    
    # Powers
    for p in [2, 3]:
        if time.time() > deadline - 50:
            break
        U = [0]
        for i in range(1, 12):
            U.append(p ** i)
            if U[-1] > 10000:
                break
        candidates.append(U)
    
    # === Phase 3: Evaluate all candidates ===
    candidate_scores = []
    for U in candidates:
        if time.time() > deadline - 25:
            break
        s = _score_set(U)
        candidate_scores.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Sort by score descending
    candidate_scores.sort(key=lambda x: -x[0])
    
    # === Phase 4: Local search on top candidates ===
    top_k = min(15, len(candidate_scores))
    for i in range(top_k):
        if time.time() > deadline - 8:
            break
        s_init, U_init = candidate_scores[i]
        if s_init < best_score - 0.05:
            continue
        improved_U, improved_score = _local_search(U_init, max_iters=3000, deadline=deadline - 8)
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # === Phase 5: Final polish on best solution ===
    if time.time() < deadline - 3:
        best_U, best_score = _local_search(best_U, max_iters=5000, deadline=deadline - 3)
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
