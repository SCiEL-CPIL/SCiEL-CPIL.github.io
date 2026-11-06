import time
import numpy as np

BASE = 51975  # 3^3 * 5^2 * 7 * 11

_DIVISORS = None

def _get_divisors():
    global _DIVISORS
    if _DIVISORS is None:
        divs = set()
        for a in range(4):
            p3 = 3 ** a
            for b in range(3):
                p15 = p3 * (5 ** b)
                for c in range(2):
                    p105 = p15 * (7 ** c)
                    for d in range(2):
                        v = p105 * (11 ** d)
                        if v > 1:
                            divs.add(v)
        _DIVISORS = sorted(divs)
    return _DIVISORS

def _coverage_vector(m, r):
    """Return numpy bool array of length BASE where position i is covered."""
    return (np.arange(BASE) % m) == r

def _uncovered_count(coverage):
    """Count uncovered positions given a boolean coverage array."""
    return int(np.sum(~coverage))

def _greedy_init(deadline):
    """Greedy set cover initialization using numpy."""
    divisors = _get_divisors()
    uncovered = np.ones(BASE, dtype=bool)
    classes = []
    remaining = list(divisors)
    
    # Precompute coverage vectors for efficiency
    cov_cache = {}
    for m in divisors:
        for r in range(m):
            cov_cache[(m, r)] = _coverage_vector(m, r)
    
    while remaining:
        if time.time() > deadline:
            break
        if not np.any(uncovered):
            break
        
        best_m = None
        best_r = None
        max_gain = -1
        
        for m in remaining:
            # Try each residue for this modulus
            for r in range(m):
                cov = cov_cache[(m, r)]
                gain = int(np.sum(uncovered & cov))
                if gain > max_gain:
                    max_gain = gain
                    best_m = m
                    best_r = r
        
        if max_gain <= 0:
            break
        
        classes.append([best_m, best_r])
        remaining.remove(best_m)
        uncovered &= ~cov_cache[(best_m, best_r)]
    
    return classes, uncovered, cov_cache

def _local_search_swap(classes, uncovered, cov_cache, deadline):
    """Local search: try swapping residues of existing classes."""
    n = len(classes)
    if n == 0:
        return classes, uncovered
    
    current_uncovered = uncovered.copy()
    current_score = _uncovered_count(current_uncovered)
    best_classes = [c[:] for c in classes]
    best_uncovered = current_uncovered.copy()
    best_score = current_score
    
    max_passes = 50
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            m_i, r_i = classes[i]
            
            # Compute coverage without class i
            # Rebuild coverage from all other classes
            cov_without_i = np.zeros(BASE, dtype=bool)
            for j in range(n):
                if j != i:
                    cov_without_i |= cov_cache[(classes[j][0], classes[j][1])]
            
            # Uncovered without class i
            unc_without_i = ~cov_without_i
            
            # Find best residue for m_i
            best_r = -1
            best_gain = -1
            for r in range(m_i):
                cov_r = cov_cache[(m_i, r)]
                gain = int(np.sum(unc_without_i & cov_r))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r != r_i and best_gain > 0:
                new_cov_i = cov_cache[(m_i, best_r)]
                new_uncovered = unc_without_i & ~new_cov_i
                new_score = _uncovered_count(new_uncovered)
                
                if new_score < current_score:
                    classes[i] = [m_i, best_r]
                    current_uncovered = new_uncovered
                    current_score = new_score
                    improved = True
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in classes]
                        best_uncovered = current_uncovered.copy()
        
        if not improved:
            break
    
    return best_classes, best_uncovered

def _try_add_moduli(classes, uncovered, cov_cache, deadline):
    """Try adding unused moduli if they improve coverage."""
    used_moduli = set(m for m, r in classes)
    all_divisors = _get_divisors()
    current_uncovered = uncovered.copy()
    current_score = _uncovered_count(current_uncovered)
    best_classes = [c[:] for c in classes]
    best_uncovered = current_uncovered.copy()
    best_score = current_score
    
    for m in all_divisors:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        best_r = -1
        best_gain = -1
        for r in range(m):
            cov_r = cov_cache[(m, r)]
            gain = int(np.sum(current_uncovered & cov_r))
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_gain > 0:
            new_cov = cov_cache[(m, best_r)]
            new_uncovered = current_uncovered & ~new_cov
            new_score = _uncovered_count(new_uncovered)
            
            if new_score < current_score:
                classes.append([m, best_r])
                current_uncovered = new_uncovered
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_classes = [c[:] for c in classes]
                    best_uncovered = current_uncovered.copy()
    
    return best_classes, best_uncovered

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Phase 1: Greedy initialization
    classes, uncovered, cov_cache = _greedy_init(deadline)
    
    # Phase 2: Local search
    if time.time() < deadline:
        classes, uncovered = _local_search_swap(classes, uncovered, cov_cache, deadline)
    
    # Phase 3: Try adding moduli
    if time.time() < deadline:
        classes, uncovered = _try_add_moduli(classes, uncovered, cov_cache, deadline)
    
    # Phase 4: Another local search
    if time.time() < deadline:
        classes, uncovered = _local_search_swap(classes, uncovered, cov_cache, deadline)
    
    # Phase 5: Final add + search
    if time.time() < deadline:
        classes, uncovered = _try_add_moduli(classes, uncovered, cov_cache, deadline)
        classes, uncovered = _local_search_swap(classes, uncovered, cov_cache, deadline)
    
    return classes
