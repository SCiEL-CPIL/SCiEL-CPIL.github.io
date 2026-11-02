"""Model-and-optimize approach for sumset/difference-set exponent lower bound."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset and difference set efficiently using sets
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set_fast(U):
    """Faster scoring using numpy where beneficial for moderate sizes."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    arr = np.array(sorted(U_set), dtype=np.int64)
    
    # Sumset via broadcasting for small n
    if n <= 150:
        sums = arr[:, None] + arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = arr[:, None] - arr[None, :]
        diffset_size = len(np.unique(diffs))
    else:
        # Fall back to set-based for large n
        sumset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
        sumset_size = len(sumset)
        
        diffset = set()
        for a in U_set:
            for b in U_set:
                diffset.add(a - b)
        diffset_size = len(diffset)
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _autocorrelation_energy(U, max_val=None):
    """
    Compute autocorrelation energy using FFT.
    For a binary sequence x supported on U, the autocorrelation at lag k is
    sum_i x[i] * x[i+k]. The 'collision energy' is the sum of squares of
    autocorrelation values excluding the main peak (lag 0).
    Lower energy means fewer sum collisions.
    """
    if max_val is None:
        max_val = max(U) if U else 0
    
    # Create binary sequence
    N = 2 * max_val + 1
    # Pad to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < N:
        fft_size *= 2
    
    x = np.zeros(fft_size, dtype=np.float64)
    for u in U:
        x[u] = 1.0
    
    # Autocorrelation via FFT
    X = np.fft.fft(x)
    acf = np.fft.ifft(X * np.conj(X)).real
    
    # The main peak is at index 0 (lag 0)
    # Energy excluding main peak
    main_peak = acf[0]
    energy = np.sum(acf[1:] ** 2)
    
    return energy, main_peak

def _spectral_objective(U, max_val=None):
    """
    Spectral objective: minimize autocorrelation sidelobe energy while
    maintaining good difference-set sparsity.
    """
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return float('inf')
    max_u = max(U_set)
    if max_u == 0:
        return 0.0
    
    energy, main_peak = _autocorrelation_energy(U_set, max_val=max_u)
    
    # The sumset size is approximately n^2 / (1 + energy/main_peak^2)
    # We want to minimize energy relative to the main peak
    if main_peak == 0:
        return float('inf')
    
    # Normalized energy (sidelobe to main peak ratio)
    norm_energy = energy / (main_peak ** 2)
    
    return norm_energy

def _spectral_local_search(U, max_iters=5000, deadline=None, max_size=80, max_val_limit=500):
    """
    Local search guided by spectral objective.
    Tries to minimize autocorrelation sidelobe energy while keeping the set sparse.
    """
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = min(max_u + 50, max_val_limit)
    
    scorer = _score_set_fast
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        U_work = U[:]
        random.shuffle(U_work)
        
        # Try modifying one element
        for i in range(len(U_work)):
            old_val = U_work[i]
            candidates = []
            # Try values near current
            for delta in [-3, -2, -1, 1, 2, 3, 5, 8, 13, 21]:
                nv = old_val + delta
                if 0 <= nv <= upper_bound and nv != old_val:
                    candidates.append(nv)
            # Try random values
            for _ in range(2):
                nv = random.randint(0, upper_bound)
                if nv != old_val:
                    candidates.append(nv)
            
            for new_val in candidates:
                new_U = U_work[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                new_U_sorted = sorted(new_U_set)
                if new_U_sorted[0] != 0:
                    continue
                s = scorer(new_U_sorted)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U_sorted
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < max_size:
                # Use spectral guidance to choose where to add
                # Try positions that minimize autocorrelation energy
                tried_positions = set(U)
                best_add_val = None
                best_add_score = best_score
                
                # Sample a few random positions
                for _ in range(20):
                    new_val = random.randint(1, upper_bound)
                    if new_val in tried_positions:
                        continue
                    tried_positions.add(new_val)
                    new_U = sorted(U + [new_val])
                    s = scorer(new_U)
                    if s > best_add_score:
                        best_add_score = s
                        best_add_val = new_val
                
                if best_add_val is not None:
                    best_score = best_add_score
                    best_U = sorted(U + [best_add_val])
                    improved = True
        
        if not improved:
            # Try removing an element
            if len(U) > 3:
                removable = [x for x in U if x != 0]
                random.shuffle(removable)
                for val in removable[:5]:
                    new_U = [x for x in U if x != val]
                    s = scorer(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
        
        if not improved:
            # Spectral-guided mutation: try to reduce autocorrelation energy
            if random.random() < 0.15:
                mutated = U[:]
                # Add 1-2 random elements
                for _ in range(random.randint(1, 2)):
                    nv = random.randint(1, upper_bound)
                    if nv not in mutated:
                        mutated.append(nv)
                # Remove 0-1 random elements (not 0)
                removable = [x for x in mutated if x != 0]
                if removable and random.random() < 0.5:
                    idx = random.randint(0, len(removable) - 1)
                    val = removable.pop(idx)
                    mutated.remove(val)
                if 0 in mutated:
                    mutated = sorted(set(mutated))
                    s = scorer(mutated)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = mutated
                        improved = True
        
        if not improved and it > 50:
            # Try expanding the search space
            upper_bound = min(upper_bound + 30, max_val_limit)
            U = best_U[:]
    
    return sorted(set(best_U)), best_score

def _construct_golomb(n):
    """Construct a Golomb ruler of order n."""
    if n <= 1:
        return [0]
    
    # Known optimal Golomb rulers
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
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
        13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
        14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 84],
    }
    if n in known:
        return known[n]
    
    # Greedy construction
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

def _construct_sidon_sequence(n, max_val=None):
    """Construct a Sidon (B_2) sequence of length n."""
    if max_val is None:
        max_val = n * n * 4
    
    seq = [0]
    used_sums = {0}
    
    for i in range(1, n):
        pos = 1
        while pos <= max_val:
            valid = True
            for j, a in enumerate(seq):
                s = a + pos
                if s in used_sums:
                    valid = False
                    break
            if valid:
                # Check all pairs including new element
                all_vals = seq + [pos]
                sums = set()
                for x in range(len(all_vals)):
                    for y in range(x, len(all_vals)):
                        s = all_vals[x] + all_vals[y]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if valid:
                    seq.append(pos)
                    break
            pos += 1
        if len(seq) < i + 1:
            break
    
    return seq

def _local_search(U, max_iters=3000, deadline=None, max_size=50):
    """Improve set U via local search with add/remove/modify operations."""
    U = sorted(set(U))
    best_score = _score_set_fast(U)
    best_U = U[:]
    max_u = max(U) if U else 0
    upper_bound = max_u + 100
    
    # Use fast scorer for sets up to 150 elements
    use_fast = len(U) <= 150
    scorer = _score_set_fast if use_fast else _score_set
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        improved = False
        U_work = U[:]
        random.shuffle(U_work)
        
        # Try modifying one element
        for i in range(len(U_work)):
            old_val = U_work[i]
            # Try a few candidate replacements
            candidates = []
            # Try values near current
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 10, 20]:
                nv = old_val + delta
                if 0 <= nv <= upper_bound and nv != old_val:
                    candidates.append(nv)
            # Try random values
            for _ in range(3):
                nv = random.randint(0, upper_bound)
                if nv != old_val:
                    candidates.append(nv)
            
            for new_val in candidates:
                new_U = U_work[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                new_U_sorted = sorted(new_U_set)
                if new_U_sorted[0] != 0:
                    continue
                s = scorer(new_U_sorted)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U_sorted
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < max_size:
                for new_val in range(0, upper_bound + 1):
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = scorer(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = sorted(new_U)
                        improved = True
                        break
        
        if not improved:
            # Try removing an element (if size > 2)
            if len(U) > 2:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i+1:]
                    s = scorer(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
        
        if not improved:
            # Random restart: mutate the whole set
            if random.random() < 0.1:
                mutated = U[:]
                # Add 1-3 random elements
                for _ in range(random.randint(1, 3)):
                    nv = random.randint(1, upper_bound)
                    if nv not in mutated:
                        mutated.append(nv)
                # Remove 0-2 random elements (not 0)
                removable = [x for x in mutated if x != 0]
                for _ in range(random.randint(0, min(2, len(removable)))):
                    if removable:
                        idx = random.randint(0, len(removable) - 1)
                        val = removable.pop(idx)
                        mutated.remove(val)
                if 0 in mutated:
                    mutated = sorted(set(mutated))
                    s = scorer(mutated)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = mutated
                        improved = True
        
        if not improved and it > 100:
            # Try expanding the search space
            upper_bound += 50
            U = best_U[:]
    
    return sorted(set(best_U)), best_score

def _optimize_parameters(start_time, deadline):
    """Optimize construction parameters directly against the objective."""
    best_score = -1.0
    best_U = [0]
    
    # Try various construction strategies with optimized parameters
    
    # Strategy 1: Golomb rulers of various sizes
    for n in range(3, 25):
        if time.time() > deadline:
            break
        gr = _construct_golomb(n)
        s = _score_set(gr)
        if s > best_score:
            best_score = s
            best_U = gr[:]
    
    # Strategy 2: Sidon sequences
    for n in range(3, 20):
        if time.time() > deadline:
            break
        ss = _construct_sidon_sequence(n, max_val=n * n * 5)
        if len(ss) >= n:
            s = _score_set(ss)
            if s > best_score:
                best_score = s
                best_U = ss[:]
    
    # Strategy 3: Parametric family U = {0, a, 2a, ..., (n-1)a} with perturbations
    for n in range(3, 15):
        if time.time() > deadline:
            break
        for a in range(1, 10):
            base = list(range(n)) * a  # {0, a, 2a, ..., (n-1)a}
            # Add small perturbations
            for pert in [0, 1, -1, 2]:
                if pert == 0:
                    candidate = base[:]
                else:
                    candidate = []
                    for i, v in enumerate(base):
                        if i == 0:
                            candidate.append(0)
                        elif i == n - 1:
                            candidate.append(v + pert)
                        else:
                            candidate.append(v + (pert if i % 2 == 0 else -pert))
                    candidate = sorted(set(x for x in candidate if x >= 0))
                    if 0 not in candidate:
                        continue
                if len(candidate) < 2:
                    continue
                s = _score_set(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
    
    # Strategy 4: Power-of-prime constructions
    for p in [2, 3, 5, 7]:
        if time.time() > deadline:
            break
        for n in range(3, 8):
            candidate = [0]
            val = 1
            for i in range(n - 1):
                val *= p
                candidate.append(val)
            s = _score_set(candidate)
            if s > best_score:
                best_score = s
                best_U = candidate[:]
    
    # Strategy 5: Quadratic-like constructions
    for n in range(3, 12):
        if time.time() > deadline:
            break
        for c in range(1, 6):
            candidate = sorted(set(i * i * c for i in range(n)))
            if 0 not in candidate:
                candidate = [0] + [x for x in candidate if x > 0]
            if len(candidate) >= 2:
                s = _score_set(candidate)
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
    
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set_fast(best_U)
    
    # Phase 1: Parameter optimization
    opt_U, opt_score = _optimize_parameters(start_time, deadline)
    if opt_score > best_score:
        best_score = opt_score
        best_U = opt_U[:]
    
    # Phase 2: Spectral-guided local search
    if time.time() < deadline - 30:
        improved_U, improved_score = _spectral_local_search(
            best_U, max_iters=6000, deadline=deadline - 20, max_size=70, max_val_limit=400
        )
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Phase 3: Standard local search refinement
    if time.time() < deadline - 10:
        improved_U, improved_score = _local_search(
            best_U, max_iters=4000, deadline=deadline - 5, max_size=60
        )
        if improved_score > best_score:
            best_score = improved_score
            best_U = improved_U
    
    # Phase 4: Try local search on other good candidates
    if time.time() < deadline:
        # Generate some diverse candidates
        candidates = []
        
        for n in [5, 8, 12, 15, 18, 20]:
            if time.time() > deadline:
                break
            gr = _construct_golomb(n)
            candidates.append(gr)
        
        # Add quadratic and cubic sequences
        for c in [1, 2, 3]:
            if time.time() > deadline:
                break
            for n in [8, 12, 16]:
                candidate = sorted(set(i * i * c for i in range(n)))
                if len(candidate) >= 2 and 0 in candidate:
                    candidates.append(candidate)
        
        random.seed(123)
        for _ in range(10):
            if time.time() > deadline:
                break
            n = random.randint(4, 20)
            max_val = random.randint(20, 150)
            U = [0] + random.sample(range(1, max_val + 1), n - 1)
            U = sorted(U)
            candidates.append(U)
        
        for U in candidates:
            if time.time() > deadline:
                break
            s = _score_set_fast(U)
            if s > best_score - 0.05:  # Only if close to best
                improved_U, improved_score = _local_search(U, max_iters=2000, deadline=deadline, max_size=50)
                if improved_score > best_score:
                    best_score = improved_score
                    best_U = improved_U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END
