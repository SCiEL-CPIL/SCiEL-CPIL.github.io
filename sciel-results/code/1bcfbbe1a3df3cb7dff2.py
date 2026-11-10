"""Doubly-covered prefix via conflict-graph repair and iterative deepening."""
import time
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count(residues, L_MAX):
    """Build count array for x in [1, L_MAX]."""
    count = [0] * (L_MAX + 1)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
    return count

def _find_prefix_L(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    L = 0
    for x in range(1, L_MAX + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count(residues, L_MAX)
    return _find_prefix_L(count, L_MAX)

def _conflict_graph_repair(residues, count, current_L, L_MAX, deadline):
    """
    Conflict-graph-based repair: find a set of non-conflicting residue changes
    that fix the first uncovered integer without breaking any covered integer.
    
    Returns (new_residues, new_count, new_L) or None if no repair found.
    """
    if current_L >= L_MAX:
        return None
    
    gap = current_L + 1
    
    # Find all primes whose residue change could help cover 'gap'
    # A prime i can help if we change its residue to gap % p
    helpful_primes = []
    for i in range(_NPRIMES):
        p = _PRIMES[i]
        desired_r = gap % p
        if residues[i] == desired_r:
            continue  # Already covers gap
        helpful_primes.append((i, p, desired_r))
    
    if not helpful_primes:
        return None
    
    # Build conflict graph: two prime changes conflict if applying both
    # would reduce any previously-covered integer's count below 2
    # For efficiency, we use a greedy independent set approach
    
    # First, identify which primes are "safe" to change individually
    # A prime i is safe if changing it doesn't break any x in [1, current_L]
    safe_primes = []
    for i, p, desired_r in helpful_primes:
        old_r = residues[i]
        old_start = p if old_r == 0 else old_r
        # Check if removing old coverage breaks anything in [1, current_L]
        safe = True
        for x in range(old_start, current_L + 1, p):
            if count[x] < 3:
                safe = False
                break
        if safe:
            safe_primes.append((i, p, desired_r))
    
    if not safe_primes:
        return None
    
    # Greedy: pick the prime that gives the most additional coverage
    # beyond just fixing the gap
    best_idx = None
    best_gain = -1
    for idx, (i, p, desired_r) in enumerate(safe_primes):
        old_r = residues[i]
        old_start = p if old_r == 0 else old_r
        new_start = p if desired_r == 0 else desired_r
        
        gain = 1  # At least fixes the gap
        # Count additional positions in (current_L, L_MAX] that get covered
        for x in range(new_start, L_MAX + 1, p):
            if x > current_L and count[x] <= 1:
                gain += 1
        
        if gain > best_gain:
            best_gain = gain
            best_idx = idx
    
    i, p, desired_r = safe_primes[best_idx]
    old_r = residues[i]
    old_start = p if old_r == 0 else old_r
    new_start = p if desired_r == 0 else desired_r
    
    # Apply the change
    for x in range(old_start, L_MAX + 1, p):
        count[x] -= 1
    for x in range(new_start, L_MAX + 1, p):
        count[x] += 1
    
    residues[i] = desired_r
    
    # Extend L
    new_L = current_L
    while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
        new_L += 1
    
    return residues, count, new_L

def _iterative_deepening_search(residues, deadline, L_MAX=5000):
    """
    Iterative deepening: repeatedly find the first uncovered integer,
    then use conflict-graph repair to fix it. Continue until no repair
    is possible or deadline is reached.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    stall_count = 0
    max_stall = 30
    
    while time.time() < deadline:
        if current_L >= L_MAX:
            break
        
        if count[current_L + 1] >= 2:
            # Extend L
            while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
                current_L += 1
            stall_count = 0
            continue
        
        # Try conflict-graph repair
        result = _conflict_graph_repair(residues, count, current_L, L_MAX, deadline)
        if result is not None:
            residues, count, current_L = result
            stall_count = 0
        else:
            # No single-prime repair works; try random perturbation
            stall_count += 1
            if stall_count > max_stall:
                break
            
            # Try a random prime change that might help
            improved = False
            for attempt in range(10):
                i = random.randint(0, _NPRIMES - 1)
                p = _PRIMES[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                
                old_start = p if old_r == 0 else old_r
                new_start = p if new_r == 0 else new_r
                
                # Check safety: don't break any covered integer
                safe = True
                for x in range(old_start, current_L + 1, p):
                    if count[x] < 3:
                        safe = False
                        break
                
                if not safe:
                    continue
                
                # Apply tentatively
                for x in range(old_start, L_MAX + 1, p):
                    count[x] -= 1
                for x in range(new_start, L_MAX + 1, p):
                    count[x] += 1
                
                new_L = current_L
                while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                    new_L += 1
                
                if new_L > current_L:
                    residues[i] = new_r
                    current_L = new_L
                    improved = True
                    stall_count = 0
                    break
                else:
                    # Revert
                    for x in range(new_start, L_MAX + 1, p):
                        count[x] -= 1
                    for x in range(old_start, L_MAX + 1, p):
                        count[x] += 1
            
            if not improved:
                if stall_count > max_stall:
                    break
    
    return residues, current_L

def _initial_greedy(deadline):
    """
    Initial greedy construction: process primes from small to large,
    choosing residues to maximize coverage of the current gap region.
    """
    residues = [0] * _NPRIMES
    L_MAX = 5000
    count = [0] * (L_MAX + 1)
    
    prime_order = list(range(_NPRIMES))  # Already in increasing prime order
    
    current_L = 0
    for idx in prime_order:
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        gap = current_L + 1
        
        # Try residues that cover the gap and nearby positions
        candidates = set()
        candidates.add(gap % p)
        for dx in range(-3, 4):
            x = gap + dx
            if x >= 1:
                candidates.add(x % p)
        
        best_r = 0
        best_score = -1
        for r in candidates:
            if r >= p:
                continue
            start = p if r == 0 else r
            gain = 0
            for x in range(start, L_MAX + 1, p):
                if count[x] <= 1:
                    gain += 1
            if gap % p == r:
                gain += 50
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        for x in range(start, L_MAX + 1, p):
            count[x] += 1
        
        while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
            current_L += 1
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_residues = None
    best_score = 0
    
    # Multi-restart with different strategies
    for restart in range(4):
        if time.time() > deadline - 20:
            break
        
        random.seed(42 + restart * 1000)
        
        if restart == 0:
            # Standard greedy + iterative deepening
            residues, current_L = _initial_greedy(deadline - 30)
            residues, current_L = _iterative_deepening_search(residues, deadline - 10, L_MAX=5000)
        elif restart == 1:
            # Greedy with larger primes first + iterative deepening
            residues = [0] * _NPRIMES
            L_MAX = 5000
            count = [0] * (L_MAX + 1)
            prime_order = sorted(range(_NPRIMES), key=lambda i: -_PRIMES[i])
            current_L = 0
            for idx in prime_order:
                if time.time() > deadline - 30:
                    break
                p = _PRIMES[idx]
                gap = current_L + 1
                candidates = set()
                candidates.add(gap % p)
                for dx in range(-5, 6):
                    x = gap + dx
                    if x >= 1:
                        candidates.add(x % p)
                best_r = 0
                best_score_local = -1
                for r in candidates:
                    if r >= p:
                        continue
                    start = p if r == 0 else r
                    gain = 0
                    for x in range(start, L_MAX + 1, p):
                        if count[x] <= 1:
                            gain += 1
                    if gap % p == r:
                        gain += 30
                    if gain > best_score_local:
                        best_score_local = gain
                        best_r = r
                residues[idx] = best_r
                start = p if best_r == 0 else best_r
                for x in range(start, L_MAX + 1, p):
                    count[x] += 1
                while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
                    current_L += 1
            residues, current_L = _iterative_deepening_search(residues, deadline - 10, L_MAX=5000)
        elif restart == 2:
            # Random init + iterative deepening
            residues = [random.randint(0, _PRIMES[i] - 1) for i in range(_NPRIMES)]
            residues, current_L = _iterative_deepening_search(residues, deadline - 10, L_MAX=5000)
        else:
            # Hybrid: greedy for small primes, random for large, then iterative deepening
            residues = [0] * _NPRIMES
            L_MAX = 5000
            count = [0] * (L_MAX + 1)
            current_L = 0
            for idx in range(_NPRIMES):
                if time.time() > deadline - 30:
                    break
                p = _PRIMES[idx]
                if p <= 100:
                    # Greedy for small primes
                    gap = current_L + 1
                    candidates = set()
                    candidates.add(gap % p)
                    for dx in range(-2, 3):
                        x = gap + dx
                        if x >= 1:
                            candidates.add(x % p)
                    best_r = 0
                    best_score_local = -1
                    for r in candidates:
                        if r >= p:
                            continue
                        start = p if r == 0 else r
                        gain = 0
                        for x in range(start, L_MAX + 1, p):
                            if count[x] <= 1:
                                gain += 1
                        if gap % p == r:
                            gain += 40
                        if gain > best_score_local:
                            best_score_local = gain
                            best_r = r
                else:
                    # Random for large primes
                    best_r = random.randint(0, p - 1)
                
                residues[idx] = best_r
                start = p if best_r == 0 else best_r
                for x in range(start, L_MAX + 1, p):
                    count[x] += 1
                while current_L + 1 <= L_MAX and count[current_L + 1] >= 2:
                    current_L += 1
            residues, current_L = _iterative_deepening_search(residues, deadline - 10, L_MAX=5000)
        
        # Compute actual score
        if time.time() < deadline - 5:
            actual_score = _score(residues)
            if actual_score > best_score:
                best_score = actual_score
                best_residues = list(residues)
    
    if best_residues is None:
        # Fallback: simple seed
        best_residues = [(i + 1) % p for i, p in enumerate(_PRIMES)]
    
    return best_residues
