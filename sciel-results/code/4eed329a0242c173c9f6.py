"""Doubly-covered prefix via conflict-graph batch repair (structural rewrite)."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = 0
    return [i for i in range(2, n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)
_P = np.array(_PRIMES, dtype=np.int64)
_P_FLOAT = _P.astype(np.float64)

def _build_count_np(residues, L_MAX):
    """Build count array using numpy vectorization."""
    residues = np.array(residues, dtype=np.int64)
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i in range(_NPRIMES):
        p = _PRIMES[i]
        r = residues[i]
        start = p if r == 0 else r
        if start <= L_MAX:
            positions = np.arange(start, L_MAX + 1, p)
            count[positions] += 1
    return count

def _find_prefix_L(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    # count[0] is unused; check 1..L_MAX
    below = np.where(count[1:L_MAX + 1] < 2)[0]
    if len(below) == 0:
        return L_MAX
    return int(below[0])

def _score(residues, L_MAX=100000):
    count = _build_count_np(residues, L_MAX)
    return _find_prefix_L(count, L_MAX)

def _greedy_init(deadline):
    """Greedy initialization: assign residues to maximize early double coverage."""
    residues = [0] * _NPRIMES
    L_MAX = 5000
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    
    # Process primes in increasing order
    for idx in range(_NPRIMES):
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_score = -1
        
        # For each candidate residue, compute marginal benefit
        # Benefit = number of positions in [1, 2000] that go from 0 to 1 or 1 to 2
        for r in range(p):
            start = p if r == 0 else r
            if start > L_MAX:
                continue
            positions = np.arange(start, min(2000, L_MAX) + 1, p)
            current_vals = count[positions]
            # Positions going from 0->1: benefit 1
            # Positions going from 1->2: benefit 1 (now doubly covered)
            # Positions going from 2->3: benefit 0
            gain = np.sum((current_vals < 2))
            # Small penalty for over-covering
            penalty = 0.05 * np.sum((current_vals >= 2))
            score = gain - penalty
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        if start <= L_MAX:
            positions = np.arange(start, L_MAX + 1, p)
            count[positions] += 1
    
    return residues

def _conflict_graph_batch_search(residues, deadline, L_MAX=8000):
    """
    Conflict-graph based batch repair:
    - Identify the bottleneck (first position with count < 2)
    - Build a set of candidate moves (prime, new_residue) that cover the bottleneck
    - A move is "safe" if it doesn't reduce any previously-covered position below 2
    - Use greedy set-cover-like selection to apply multiple safe moves at once
    """
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 50
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
        
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        # Extend if possible
        while gap <= L_MAX and count[gap] >= 2:
            current_L = gap
            gap += 1
        if gap != current_L + 1:
            improved = True
            stall_count = 0
            continue
        
        # Need to cover gap. Find all primes that can be reassigned to cover gap
        # and are safe (don't break existing coverage)
        candidates = []  # (prime_idx, new_residue, affected_positions_set)
        
        for i in range(_NPRIMES):
            p = _PRIMES[i]
            current_r = residues[i]
            desired_r = gap % p
            
            if current_r == desired_r:
                continue  # Already covers gap
            
            # Check safety: removing current_r's contribution shouldn't break [1, current_L]
            old_start = p if current_r == 0 else current_r
            new_start = p if desired_r == 0 else desired_r
            
            # Positions covered by old residue in [1, current_L]
            if old_start <= current_L:
                old_pos = np.arange(old_start, current_L + 1, p)
                # Check if any of these positions would drop below 2
                if np.any(count[old_pos] <= 2):
                    continue  # Not safe
            
            # This move is safe. Add to candidates.
            # Also check: does new residue help beyond gap?
            if new_start <= L_MAX:
                new_pos = np.arange(new_start, L_MAX + 1, p)
            else:
                new_pos = np.array([], dtype=np.int64)
            
            # Score: how many new positions get covered (count goes 0->1 or 1->2)
            if len(new_pos) > 0:
                new_vals = count[new_pos]
                benefit = np.sum(new_vals < 2)
            else:
                benefit = 0
            
            candidates.append((i, desired_r, benefit))
        
        if not candidates:
            # No single safe move found. Try 2-move combinations among small primes
            found = False
            small_indices = [i for i in range(_NPRIMES) if _PRIMES[i] <= 200]
            
            for i1 in range(len(small_indices)):
                if time.time() > deadline:
                    break
                i = small_indices[i1]
                p = _PRIMES[i]
                current_r1 = residues[i]
                desired_r1 = gap % p
                if current_r1 == desired_r1:
                    continue
                
                old_start1 = p if current_r1 == 0 else current_r1
                new_start1 = p if desired_r1 == 0 else desired_r1
                
                # Check safety of move 1
                safe1 = True
                if old_start1 <= current_L:
                    old_pos1 = np.arange(old_start1, current_L + 1, p)
                    if np.any(count[old_pos1] <= 2):
                        safe1 = False
                
                if not safe1:
                    continue
                
                # Apply move 1 tentatively
                if old_start1 <= L_MAX:
                    old_p1 = np.arange(old_start1, L_MAX + 1, p)
                    count[old_p1] -= 1
                if new_start1 <= L_MAX:
                    new_p1 = np.arange(new_start1, L_MAX + 1, p)
                    count[new_p1] += 1
                
                for i2 in range(i1 + 1, len(small_indices)):
                    if time.time() > deadline:
                        break
                    j = small_indices[i2]
                    q = _PRIMES[j]
                    current_r2 = residues[j]
                    desired_r2 = gap % q
                    if current_r2 == desired_r2:
                        continue
                    
                    old_start2 = q if current_r2 == 0 else current_r2
                    new_start2 = q if desired_r2 == 0 else desired_r2
                    
                    # Check safety of move 2 (given move 1 applied)
                    safe2 = True
                    if old_start2 <= current_L:
                        old_pos2 = np.arange(old_start2, current_L + 1, q)
                        if np.any(count[old_pos2] <= 2):
                            safe2 = False
                    
                    if not safe2:
                        continue
                    
                    # Apply move 2 tentatively
                    if old_start2 <= L_MAX:
                        old_p2 = np.arange(old_start2, L_MAX + 1, q)
                        count[old_p2] -= 1
                    if new_start2 <= L_MAX:
                        new_p2 = np.arange(new_start2, L_MAX + 1, q)
                        count[new_p2] += 1
                    
                    # Check new L
                    new_L = current_L
                    while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                        new_L += 1
                    
                    # Undo move 2
                    if new_start2 <= L_MAX:
                        count[new_p2] -= 1
                    if old_start2 <= L_MAX:
                        count[old_p2] += 1
                    
                    gain = new_L - current_L
                    if gain > 0:
                        # Apply both moves permanently
                        if old_start1 <= L_MAX:
                            count[old_p1] -= 1
                        if new_start1 <= L_MAX:
                            count[new_p1] += 1
                        if old_start2 <= L_MAX:
                            count[old_p2] -= 1
                        if new_start2 <= L_MAX:
                            count[new_p2] += 1
                        residues[i] = desired_r1
                        residues[j] = desired_r2
                        current_L = new_L
                        improved = True
                        stall_count = 0
                        found = True
                        break
                else:
                    if not found:
                        # Undo move 1
                        if new_start1 <= L_MAX:
                            count[new_p1] -= 1
                        if old_start1 <= L_MAX:
                            count[old_p1] += 1
                    continue
                break
            
            if found:
                continue
        
        # Greedy set-cover: apply multiple safe candidates that together extend L
        # Sort by benefit descending
        candidates.sort(key=lambda x: -x[2])
        
        # Try to apply the best candidate
        best_applied = False
        for (i, new_r, benefit) in candidates[:20]:  # Try top 20
            p = _PRIMES[i]
            old_r = residues[i]
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            # Re-verify safety (count may have changed if we applied others)
            if old_start <= current_L:
                old_pos = np.arange(old_start, current_L + 1, p)
                if np.any(count[old_pos] <= 2):
                    continue
            
            # Apply
            if old_start <= L_MAX:
                old_p = np.arange(old_start, L_MAX + 1, p)
                count[old_p] -= 1
            if new_start <= L_MAX:
                new_p = np.arange(new_start, L_MAX + 1, p)
                count[new_p] += 1
            
            residues[i] = new_r
            
            # Extend L
            new_L = current_L
            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                new_L += 1
            
            if new_L > current_L:
                current_L = new_L
                improved = True
                stall_count = 0
                best_applied = True
                break
        
        if best_applied:
            continue
    
    return residues, current_L

def _extended_polish(residues, deadline, L_MAX=15000):
    """Polish with larger range to extend further."""
    residues = list(residues)
    count = _build_count_np(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)
    
    improved = True
    stall_count = 0
    max_stall = 30
    
    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break
        
        gap = current_L + 1
        if gap > L_MAX:
            break
        
        # Extend if possible
        while gap <= L_MAX and count[gap] >= 2:
            current_L = gap
            gap += 1
        if gap != current_L + 1:
            improved = True
            stall_count = 0
            continue
        
        # Try reassignments of small-to-medium primes
        best_move = None
        best_gain = 0
        
        for i in range(_NPRIMES):
            if time.time() > deadline:
                break
            p = _PRIMES[i]
            if p > 300:
                continue
            
            current_r = residues[i]
            desired_r = gap % p
            if current_r == desired_r:
                continue
            
            old_start = p if current_r == 0 else current_r
            new_start = p if desired_r == 0 else desired_r
            
            # Safety check
            if old_start <= current_L:
                old_pos = np.arange(old_start, current_L + 1, p)
                if np.any(count[old_pos] <= 2):
                    continue
            
            # Apply tentatively
            if old_start <= L_MAX:
                old_p = np.arange(old_start, L_MAX + 1, p)
                count[old_p] -= 1
            if new_start <= L_MAX:
                new_p = np.arange(new_start, L_MAX + 1, p)
                count[new_p] += 1
            
            new_L = current_L
            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                new_L += 1
            
            # Undo
            if new_start <= L_MAX:
                count[new_p] -= 1
            if old_start <= L_MAX:
                count[old_p] += 1
            
            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_move = (i, desired_r, new_L)
        
        if best_move is not None:
            i, new_r, new_L = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r
            
            if old_start <= L_MAX:
                old_p = np.arange(old_start, L_MAX + 1, p)
                count[old_p] -= 1
            if new_start <= L_MAX:
                new_p = np.arange(new_start, L_MAX + 1, p)
                count[new_p] += 1
            
            residues[i] = new_r
            current_L = new_L
            improved = True
            stall_count = 0
    
    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Phase 1: Greedy initialization
    residues = _greedy_init(deadline - 60)
    best_residues = list(residues)
    
    # Phase 2: Conflict-graph batch search
    if time.time() < deadline - 30:
        residues, L = _conflict_graph_batch_search(residues, deadline - 30, L_MAX=8000)
        best_residues = list(residues)
    
    # Phase 3: Extended polish
    if time.time() < deadline - 10:
        residues, L = _extended_polish(best_residues, deadline - 10, L_MAX=15000)
        best_residues = list(residues)
    
    # Final verification (capped at 10^5)
    final_L = _score(best_residues, L_MAX=100000)
    
    # Ensure residues are proper Python ints
    result = [int(r) for r in best_residues]
    return result

# EVOLVE-BLOCK-END
