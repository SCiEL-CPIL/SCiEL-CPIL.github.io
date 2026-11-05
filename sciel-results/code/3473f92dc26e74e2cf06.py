"""Combined approach: lattice-based construction with structured local search."""
# EVOLVE-BLOCK-START
import time
import random

def _compute_sizes(A):
    """Compute |A+A| and |A*A| with equal summands/factors allowed."""
    n = len(A)
    sum_set = set()
    prod_set = set()
    for i in range(n):
        for j in range(i, n):
            sum_set.add(A[i] + A[j])
            prod_set.add(A[i] * A[j])
    return len(sum_set), len(prod_set)

def _score(A):
    """Return max(|A+A|, |A*A|)."""
    s, p = _compute_sizes(A)
    return max(s, p)

def _lattice_candidates(n=20, max_val=10000):
    """Generate diverse lattice-based candidate sets."""
    candidates = []
    
    # 1. Dense 2D grids: i in 0..i_max, j in 0..j_max
    for i_max in range(0, 14):
        for j_max in range(0, 10):
            points = [(i, j) for i in range(i_max + 1) for j in range(j_max + 1)]
            if len(points) >= n:
                points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
                subset = points_sorted[:n]
                vals = [(2 ** i) * (3 ** j) for i, j in subset]
                vals = sorted(set(v for v in vals if 1 <= v <= max_val))
                if len(vals) == n:
                    candidates.append(vals)
    
    # 2. Lines in the lattice: i + 2j = c
    for c in range(0, 25):
        points = []
        for j in range(0, 12):
            i = c - 2 * j
            if 0 <= i <= 14:
                points.append((i, j))
        if len(points) >= n:
            points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
            subset = points_sorted[:n]
            vals = [(2 ** i) * (3 ** j) for i, j in subset]
            vals = sorted(set(v for v in vals if 1 <= v <= max_val))
            if len(vals) == n:
                candidates.append(vals)
    
    # 3. Lines: 2i + j = c
    for c in range(0, 30):
        points = []
        for i in range(0, 14):
            j = c - 2 * i
            if 0 <= j <= 12:
                points.append((i, j))
        if len(points) >= n:
            points_sorted = sorted(points, key=lambda p: (2 ** p[0]) * (3 ** p[1]))
            subset = points_sorted[:n]
            vals = [(2 ** i) * (3 ** j) for i, j in subset]
            vals = sorted(set(v for v in vals if 1 <= v <= max_val))
            if len(vals) == n:
                candidates.append(vals)
    
    # 4. Geometric progressions
    for base in [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5]:
        vals = []
        v = 1
        for _ in range(n):
            vals.append(int(round(v)))
            v *= base
        vals = sorted(set(x for x in vals if 1 <= x <= max_val))
        while len(vals) < n:
            vals.append(vals[-1] + 1 if vals[-1] < max_val else max_val)
            vals = sorted(set(vals))
        if len(vals) == n and all(1 <= x <= max_val for x in vals):
            candidates.append(vals)
    
    # 5. Hand-crafted
    hand = [
        list(range(1, 21)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 64, 72, 81, 96],
    ]
    candidates.extend(hand)
    
    # Deduplicate
    seen = set()
    unique = []
    for cand in candidates:
        key = tuple(cand)
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    
    return unique

def _local_search(A, max_time=30.0, start_time=None):
    """Structured local search with multiplicative and additive neighbors."""
    if start_time is None:
        start_time = time.time()
    
    A = sorted(set(A))
    n = len(A)
    best_score = _score(A)
    best_A = list(A)
    
    deadline = start_time + max_time
    no_improve = 0
    max_no_improve = 30
    
    def get_neighbors(x, current_set):
        """Generate candidate replacements for element x."""
        cands = set()
        
        # Multiplicative neighbors (exploit lattice structure)
        for factor in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
            nv = int(round(x * factor))
            if 1 <= nv <= 10000:
                cands.add(nv)
        
        # Additive neighbors
        for d in [-5, -3, -2, -1, 1, 2, 3, 5, 10, 20, 50, 100]:
            nv = x + d
            if 1 <= nv <= 10000:
                cands.add(nv)
        
        # Lattice-inspired: multiply/divide by 2, 3, 5
        for pm in [2, 3, 5]:
            nv = x * pm
            if 1 <= nv <= 10000:
                cands.add(nv)
            if x % pm == 0:
                nv = x // pm
                if 1 <= nv <= 10000:
                    cands.add(nv)
        
        # Random
        cands.add(random.randint(1, 10000))
        cands.add(random.randint(1, 10000))
        
        # Filter out existing elements
        cands = [c for c in cands if c not in current_set and 1 <= c <= 10000]
        return cands
    
    while time.time() < deadline - 2.0:
        improved = False
        random.shuffle(A)
        
        for i in range(n):
            if time.time() >= deadline - 2.0:
                break
            
            neighbors = get_neighbors(A[i], set(A))
            if not neighbors:
                continue
            
            for nv in random.sample(neighbors, min(5, len(neighbors))):
                if time.time() >= deadline - 2.0:
                    break
                
                A_new = list(A)
                A_new[i] = nv
                A_new = sorted(set(A_new))
                
                if len(A_new) != n or any(v > 10000 for v in A_new):
                    continue
                
                s = _score(A_new)
                if s < best_score:
                    A = A_new
                    best_score = s
                    best_A = list(A)
                    improved = True
                    no_improve = 0
                    break
                elif s == best_score and random.random() < 0.1:
                    # Accept equal-score moves occasionally to escape local minima
                    A = A_new
                    improved = True
                    no_improve = 0
                    break
        
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        else:
            no_improve = 0
    
    return best_A, best_score

def _multi_start_local_search(candidates, max_time=60.0, start_time=None):
    """Run local search from multiple starting points."""
    if start_time is None:
        start_time = time.time()
    
    deadline = start_time + max_time
    best_A = None
    best_score = float('inf')
    
    # Evaluate all candidates first
    for cand in candidates:
        if time.time() >= deadline - 10.0:
            break
        s = _score(cand)
        if s < best_score:
            best_score = s
            best_A = list(cand)
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_score = _score(best_A)
    
    # Run local search from the best candidate
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        best_A, best_score = _local_search(best_A, max_time=min(remaining, 30.0), start_time=start_time)
    
    # If time remains, try a few more candidates
    remaining = deadline - time.time() - 3.0
    if remaining > 10.0:
        # Sort candidates by initial score and try the next few
        scored_cands = []
        for cand in candidates:
            if cand == best_A:
                continue
            s = _score(cand)
            scored_cands.append((s, cand))
        
        scored_cands.sort(key=lambda x: x[0])
        
        for s, cand in scored_cands[:3]:
            if time.time() >= deadline - 5.0:
                break
            if s >= best_score:
                break
            
            trial_A, trial_score = _local_search(cand, max_time=min(remaining / 3, 15.0), start_time=start_time)
            if trial_score < best_score:
                best_score = trial_score
                best_A = trial_A
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Generate diverse lattice-based candidates
    candidates = _lattice_candidates(n=20, max_val=10000)
    
    # Add some random candidates for diversity
    for _ in range(5):
        candidates.append(sorted(random.sample(range(1, 10001), 20)))
    
    # Run multi-start local search
    best_A, best_score = _multi_start_local_search(
        candidates, 
        max_time=80.0, 
        start_time=start_time
    )
    
    # Final safety check
    if time.time() > deadline - 2.0:
        # Ensure we return a valid set
        best_A = sorted(set(best_A))
        if len(best_A) != 20:
            best_A = list(range(1, 21))
    
    return best_A
# EVOLVE-BLOCK-END
