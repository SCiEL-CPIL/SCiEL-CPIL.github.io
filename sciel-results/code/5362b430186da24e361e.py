"""Optimized doubly-covered prefix: NumPy-accelerated greedy + local search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)
_PRIMES_ARR = np.array(_PRIMES, dtype=np.int64)

def _build_count_np(residues, L_MAX):
    """Build count array for x in [1, L_MAX] using NumPy."""
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    # Create boolean array: True where count[x] >= 2
    good = count[1:] >= 2
    if not good.any():
        return 0
    # Find first False
    false_idx = np.where(~good)[0]
    if len(false_idx) == 0:
        return L_MAX
    return int(false_idx[0])

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count_np(residues, L_MAX)
    return _find_prefix_L_np(count, L_MAX)

def _greedy_construct(deadline):
    """
    Greedy construction using NumPy for speed.
    Process primes in increasing order, choosing residue that maximizes double coverage.
    """
    residues = [0] * _NPRIMES
    L_TARGET = 5000
    count = np.zeros(L_TARGET + 1, dtype=np.int32)
    current_L = 0
    
    for idx in range(_NPRIMES):
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        
        # Try all residues for small primes, subset for large
        if p <= 50:
            candidates = list(range(p))
        else:
            gap = current_L + 1
            candidates = [gap % p, 0]
            # Add residues that cover positions with count == 1
            for dx in range(0, min(20, L_TARGET - gap + 1)):
                x = gap + dx
                if x <= L_TARGET and count[x] == 1:
                    candidates.append(x % p)
            # Deduplicate
            candidates = list(set(candidates))
        
        best_r = 0
        best_score = -1
        
        for r in candidates:
            if r >= p:
                continue
            # Calculate score: number of positions that reach count >= 2
            start = p if r == 0 else r
            positions = np.arange(start, L_TARGET + 1, p)
            # Count how many have count == 1 (will become 2) or count >= 2 (stay >= 2)
            score = np.sum(count[positions] == 1) * 10 + np.sum(count[positions] >= 2)
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = best_r
        
        # Update count
        start = p if best_r == 0 else best_r
        count[start::p] += 1
        
        # Update current_L
        while current_L + 1 <= L_TARGET and count[current_L + 1] >= 2:
            current_L += 1
    
    return residues, current_L

def _local_search_1flip(residues, deadline, L_MAX=8000):
    """
    Local search with single flips using NumPy.
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = _find_prefix_L_np(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 30
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
            
        if current_L >= L_MAX:
            break
            
        gap = current_L + 1
        best_move = None
        best_new_L = current_L
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            current_r = residues[i]
            
            # Generate candidates
            candidates = [gap % p, 0]
            if p <= 30:
                candidates.extend(random.sample(range(p), min(5, p)))
            # Add residues covering count==1 positions near gap
            for dx in range(0, min(10, L_MAX - gap + 1)):
                x = gap + dx
                if x <= L_MAX and count[x] == 1:
                    candidates.append(x % p)
            candidates = list(set(candidates))
            
            for new_r in candidates:
                if new_r == current_r or new_r >= p:
                    continue
                
                # Simulate flip
                old_start = p if current_r == 0 else current_r
                new_start = p if new_r == 0 else new_r
                
                old_pos = np.arange(old_start, L_MAX + 1, p)
                new_pos = np.arange(new_start, L_MAX + 1, p)
                
                count[old_pos] -= 1
                count[new_pos] += 1
                
                # Find new L
                broke = False
                new_L = 0
                # Check if prefix is broken
                if current_L > 0:
                    prefix_good = count[1:current_L+1] >= 2
                    if not prefix_good.all():
                        broke = True
                        false_idx = np.where(~prefix_good)[0]
                        new_L = int(false_idx[0])
                
                if not broke:
                    new_L = current_L
                    # Try to extend
                    while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                        new_L += 1
                
                # Undo
                count[new_pos] -= 1
                count[old_pos] += 1
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_move = (i, new_r, new_L)
        
        if best_move is not None:
            i, new_r, new_L = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            old_pos = np.arange(old_start, L_MAX + 1, p)
            new_pos = np.arange(new_start, L_MAX + 1, p)
            
            count[old_pos] -= 1
            count[new_pos] += 1
            
            residues[i] = new_r
            current_L = new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def _local_search_2flip(residues, deadline, L_MAX=5000):
    """
    Local search with 2-flips using NumPy.
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = _find_prefix_L_np(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 10
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
            
        if current_L >= L_MAX:
            break
            
        gap = current_L + 1
        
        # Try pairs of small primes
        small_primes_idx = [i for i in range(_NPRIMES) if _PRIMES[i] < 50]
        
        best_move = None
        best_new_L = current_L
        
        for i1 in range(len(small_primes_idx)):
            idx1 = small_primes_idx[i1]
            p1 = _PRIMES[idx1]
            r1_old = residues[idx1]
            
            cand_r1 = [gap % p1, 0]
            
            for r1_new in cand_r1:
                if r1_new == r1_old or r1_new >= p1:
                    continue
                
                old_start1 = p1 if r1_old == 0 else r1_old
                new_start1 = p1 if r1_new == 0 else r1_new
                
                old_pos1 = np.arange(old_start1, L_MAX + 1, p1)
                new_pos1 = np.arange(new_start1, L_MAX + 1, p1)
                
                count[old_pos1] -= 1
                count[new_pos1] += 1
                
                for i2 in range(i1 + 1, len(small_primes_idx)):
                    idx2 = small_primes_idx[i2]
                    p2 = _PRIMES[idx2]
                    r2_old = residues[idx2]
                    
                    cand_r2 = [gap % p2, 0]
                    
                    for r2_new in cand_r2:
                        if r2_new == r2_old or r2_new >= p2:
                            continue
                        
                        old_start2 = p2 if r2_old == 0 else r2_old
                        new_start2 = p2 if r2_new == 0 else r2_new
                        
                        old_pos2 = np.arange(old_start2, L_MAX + 1, p2)
                        new_pos2 = np.arange(new_start2, L_MAX + 1, p2)
                        
                        count[old_pos2] -= 1
                        count[new_pos2] += 1
                        
                        # Evaluate
                        broke = False
                        new_L = 0
                        if current_L > 0:
                            prefix_good = count[1:current_L+1] >= 2
                            if not prefix_good.all():
                                broke = True
                                false_idx = np.where(~prefix_good)[0]
                                new_L = int(false_idx[0])
                        
                        if not broke:
                            new_L = current_L
                            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                                new_L += 1
                        
                        # Undo flip 2
                        count[new_pos2] -= 1
                        count[old_pos2] += 1
                        
                        if new_L > best_new_L:
                            best_new_L = new_L
                            best_move = (idx1, r1_new, idx2, r2_new, new_L)
                
                # Undo flip 1
                count[new_pos1] -= 1
                count[old_pos1] += 1
        
        if best_move is not None:
            idx1, r1_new, idx2, r2_new, new_L = best_move
            p1 = _PRIMES[idx1]
            r1_old = residues[idx1]
            p2 = _PRIMES[idx2]
            r2_old = residues[idx2]
            
            old_start1 = p1 if r1_old == 0 else r1_old
            new_start1 = p1 if r1_new == 0 else r1_new
            old_start2 = p2 if r2_old == 0 else r2_old
            new_start2 = p2 if r2_new == 0 else r2_new
            
            old_pos1 = np.arange(old_start1, L_MAX + 1, p1)
            new_pos1 = np.arange(new_start1, L_MAX + 1, p1)
            old_pos2 = np.arange(old_start2, L_MAX + 1, p2)
            new_pos2 = np.arange(new_start2, L_MAX + 1, p2)
            
            count[old_pos1] -= 1
            count[new_pos1] += 1
            count[old_pos2] -= 1
            count[new_pos2] += 1
            
            residues[idx1] = r1_new
            residues[idx2] = r2_new
            current_L = new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Phase 1: Greedy Construction
    residues, L = _greedy_construct(deadline - 80)
    best_residues = list(residues)
    best_L = L
    
    # Phase 2: 1-flip Local Search (main workhorse)
    if time.time() < deadline - 40:
        residues, L = _local_search_1flip(residues, deadline - 40, L_MAX=8000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 3: 2-flip Local Search (escape local minima)
    if time.time() < deadline - 15:
        residues, L = _local_search_2flip(residues, deadline - 15, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Phase 4: Final 1-flip polish
    if time.time() < deadline - 5:
        residues, L = _local_search_1flip(residues, deadline - 5, L_MAX=8000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)
    
    # Final verification with full score
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L
    
    return best_residues

# EVOLVE-BLOCK-END
