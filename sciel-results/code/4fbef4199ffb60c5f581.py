"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def _primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _build_counts_np(residues, primes, max_L):
    """Build count array using numpy for speed."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        start = r if r >= 1 else r + p
        if start > max_L:
            continue
        idxs = np.arange(start, max_L + 1, p)
        counts[idxs] += 1
    return counts

def _find_prefix_length(counts, max_L):
    """Find prefix length from counts array."""
    L = 0
    for x in range(1, max_L + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _apply_move(counts, residues, primes, idx, new_r, max_L):
    """Apply a residue change and update counts. Returns new_score delta info."""
    p = primes[idx]
    old_r = residues[idx]
    
    # Remove old coverage
    start_old = old_r if old_r >= 1 else old_r + p
    if start_old <= max_L:
        idxs_old = np.arange(start_old, max_L + 1, p)
        counts[idxs_old] -= 1
    
    # Add new coverage
    start_new = new_r if new_r >= 1 else new_r + p
    if start_new <= max_L:
        idxs_new = np.arange(start_new, max_L + 1, p)
        counts[idxs_new] += 1
    
    residues[idx] = new_r

def _undo_move(counts, residues, primes, idx, old_r, max_L):
    """Undo a residue change."""
    p = primes[idx]
    new_r = residues[idx]
    
    # Remove new coverage
    start_new = new_r if new_r >= 1 else new_r + p
    if start_new <= max_L:
        idxs_new = np.arange(start_new, max_L + 1, p)
        counts[idxs_new] -= 1
    
    # Add back old coverage
    start_old = old_r if old_r >= 1 else old_r + p
    if start_old <= max_L:
        idxs_old = np.arange(start_old, max_L + 1, p)
        counts[idxs_old] += 1
    
    residues[idx] = old_r

def _greedy_init(primes, max_L):
    """Greedy initialization: choose residue for each prime to maximize coverage of under-covered positions."""
    n_primes = len(primes)
    residues = [0] * n_primes
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    for idx in range(n_primes):
        p = primes[idx]
        best_r = 0
        best_metric = -1
        
        if p <= max_L:
            for r in range(p):
                start_x = r if r >= 1 else r + p
                if start_x > max_L:
                    gain = 0
                else:
                    idxs = np.arange(start_x, max_L + 1, p)
                    gain = np.sum(counts[idxs] < 2)
                if gain > best_metric:
                    best_metric = gain
                    best_r = r
        else:
            best_r = 0
            best_metric = -1
            for r in range(1, min(p, max_L + 1)):
                gain = 1 if counts[r] < 2 else 0
                if gain > best_metric:
                    best_metric = gain
                    best_r = r
            if best_metric <= 0:
                best_r = 0
        
        residues[idx] = best_r
        start_x = best_r if best_r >= 1 else best_r + p
        if start_x <= max_L:
            idxs = np.arange(start_x, max_L + 1, p)
            counts[idxs] += 1
    
    return residues, counts

def _simulated_annealing(residues, counts, primes, max_L, deadline, iterations_budget):
    """Run simulated annealing to improve the solution."""
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    T_initial = 5.0
    T_final = 0.05
    cooling_rate = 0.9995
    T = T_initial
    
    iteration = 0
    while time.time() < deadline - 15.0 and iteration < iterations_budget:
        iteration += 1
        
        # Choose a prime to modify - bias toward small primes (more impact)
        if random.random() < 0.7:
            # Weighted toward small primes
            weights = [1.0 / primes[i] for i in range(n_primes)]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            idx = 0
            for i in range(n_primes):
                cum += weights[i]
                if r <= cum:
                    idx = i
                    break
        else:
            idx = random.randint(0, n_primes - 1)
        
        p = primes[idx]
        old_r = residues[idx]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Apply move
        _apply_move(counts, residues, primes, idx, new_r, max_L)
        new_score = _find_prefix_length(counts, max_L)
        
        # Acceptance
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
                best_counts = counts.copy()
        else:
            delta = new_score - current_score
            if delta >= 0 or (T > T_final and random.random() < math.exp(delta / T)):
                current_score = new_score
            else:
                _undo_move(counts, residues, primes, idx, old_r, max_L)
        
        T *= cooling_rate
        if T < T_final:
            T = T_final
    
    return best_residues, best_counts, best_score

def _local_search(residues, counts, primes, max_L, deadline):
    """Targeted local search to fix gaps with smarter candidate selection."""
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    max_rounds = 200
    for round_num in range(max_rounds):
        if time.time() > deadline - 10.0:
            break
        
        current_score = _find_prefix_length(counts, max_L)
        
        if current_score >= max_L:
            break
        
        gap_pos = current_score + 1
        
        # Find primes covering gap_pos
        covering = []
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            if gap_pos % p == r % p:
                covering.append(i)
        
        if len(covering) >= 2:
            # Already covered, move to next gap
            continue
        
        # Try to change a non-covering prime to cover gap_pos
        candidates = []
        for i in range(n_primes):
            if i in covering:
                continue
            p = primes[i]
            old_r = residues[i]
            new_r = gap_pos % p
            if new_r == old_r:
                continue
            
            # Calculate loss: how many positions in [1, current_score] would drop below 2
            start_old = old_r if old_r >= 1 else old_r + p
            loss = 0
            if start_old <= min(current_score, max_L):
                idxs = np.arange(start_old, min(current_score, max_L) + 1, p)
                loss = np.sum(counts[idxs] == 2)
            
            # Calculate gain: how many under-covered positions in [1, current_score] would be fixed
            start_new = new_r if new_r >= 1 else new_r + p
            gain = 0
            if start_new <= min(current_score, max_L):
                idxs_new = np.arange(start_new, min(current_score, max_L) + 1, p)
                gain = np.sum(counts[idxs_new] < 2)
            
            # Also consider future coverage: positions in (current_score, max_L] that would be covered
            future_gain = 0
            if start_new <= max_L:
                future_idxs = np.arange(max(start_new, current_score + 1), max_L + 1, p)
                if len(future_idxs) > 0:
                    future_gain = np.sum(counts[future_idxs] < 2)
            
            # Score: prefer low loss, high gain + future_gain
            # Weight: loss is bad, gain and future_gain are good
            score_candidate = -loss * 10 + gain + future_gain * 0.5
            candidates.append((score_candidate, i, new_r, loss, gain))
        
        if candidates:
            # Sort by candidate score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Try top candidates
            improved_this_round = False
            for score_c, i, new_r, loss, gain in candidates[:5]:
                if time.time() > deadline - 10.0:
                    break
                
                old_r = residues[i]
                _apply_move(counts, residues, primes, i, new_r, max_L)
                new_score = _find_prefix_length(counts, max_L)
                
                if new_score > current_score:
                    current_score = new_score
                    improved_this_round = True
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues[:]
                        best_counts = counts.copy()
                    break
                else:
                    _undo_move(counts, residues, primes, i, old_r, max_L)
            
            if not improved_this_round:
                # Try random moves to escape local optimum
                for _ in range(10):
                    if time.time() > deadline - 10.0:
                        break
                    i = random.randint(0, n_primes - 1)
                    old_r = residues[i]
                    p = primes[i]
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        continue
                    _apply_move(counts, residues, primes, i, new_r, max_L)
                    new_score = _find_prefix_length(counts, max_L)
                    if new_score > current_score:
                        current_score = new_score
                        improved_this_round = True
                        if current_score > best_score:
                            best_score = current_score
                            best_residues = residues[:]
                            best_counts = counts.copy()
                        break
                    else:
                        _undo_move(counts, residues, primes, i, old_r, max_L)
    
    return best_residues, best_counts, best_score

def _pairwise_local_search(residues, counts, primes, max_L, deadline):
    """Try targeted pairwise changes to improve coverage."""
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    max_rounds = 100
    for round_num in range(max_rounds):
        if time.time() > deadline - 10.0:
            break
        
        current_score = _find_prefix_length(counts, max_L)
        if current_score >= max_L:
            break
        
        # Find the first gap
        gap_pos = current_score + 1
        
        # Find primes that could help cover gap_pos
        helpful_primes = []
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            if gap_pos % p != r % p:
                # This prime doesn't currently cover gap_pos
                # Check if it covers many under-covered positions
                start = r if r >= 1 else r + p
                if start <= current_score:
                    idxs = np.arange(start, current_score + 1, p)
                    under_covered = np.sum(counts[idxs] < 2)
                    if under_covered > 0:
                        helpful_primes.append((under_covered, i))
        
        # Try changing one helpful prime to cover gap_pos, and another to compensate
        found_improvement = False
        for under_cov, i in helpful_primes[:10]:
            if time.time() > deadline - 10.0:
                break
            
            p_i = primes[i]
            old_ri = residues[i]
            new_ri = gap_pos % p_i
            
            if new_ri == old_ri:
                continue
            
            # Try all j to find a good compensation
            best_j = None
            best_new_score = current_score
            
            for j in range(n_primes):
                if j == i:
                    continue
                p_j = primes[j]
                old_rj = residues[j]
                
                # Try a few random new values for j
                for trial in range(3):
                    new_rj = random.randint(0, p_j - 1)
                    if new_rj == old_rj:
                        continue
                    
                    # Apply both moves
                    _apply_move(counts, residues, primes, i, new_ri, max_L)
                    _apply_move(counts, residues, primes, j, new_rj, max_L)
                    
                    new_score = _find_prefix_length(counts, max_L)
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_j = (j, new_rj)
                    
                    # Undo
                    _undo_move(counts, residues, primes, j, new_rj, max_L)
                    _undo_move(counts, residues, primes, i, new_ri, max_L)
            
            if best_j is not None and best_new_score > current_score:
                j, new_rj = best_j
                _apply_move(counts, residues, primes, i, new_ri, max_L)
                _apply_move(counts, residues, primes, j, new_rj, max_L)
                current_score = best_new_score
                found_improvement = True
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
                    best_counts = counts.copy()
                break
        
        if not found_improvement:
            # Try random single moves
            for _ in range(20):
                if time.time() > deadline - 10.0:
                    break
                i = random.randint(0, n_primes - 1)
                old_r = residues[i]
                p = primes[i]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                _apply_move(counts, residues, primes, i, new_r, max_L)
                new_score = _find_prefix_length(counts, max_L)
                if new_score > current_score:
                    current_score = new_score
                    found_improvement = True
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = residues[:]
                        best_counts = counts.copy()
                    break
                else:
                    _undo_move(counts, residues, primes, i, old_r, max_L)
    
    return best_residues, best_counts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    WORK_L = 3000  # Increased working length for better exploration
    
    best_overall_residues = None
    best_overall_score = 0
    
    # Multiple restarts with adaptive time allocation
    num_restarts = 6
    time_budget = deadline - start_time - 25.0
    time_per_restart = time_budget / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 25.0:
            break
        
        # Greedy initialization
        residues, counts = _greedy_init(primes, WORK_L)
        score = _find_prefix_length(counts, WORK_L)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_residues = residues[:]
        
        # Simulated annealing - allocate 50% of restart time
        sa_deadline = min(deadline - 25.0, start_time + time_per_restart * 0.5)
        iterations = int(300000 * (time_per_restart * 0.5) / 10.0)
        residues, counts, score = _simulated_annealing(residues, counts, primes, WORK_L, sa_deadline, iterations)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_residues = residues[:]
        
        # Local search - allocate 30% of restart time
        ls_deadline = min(deadline - 20.0, start_time + time_per_restart * 0.8)
        residues, counts, score = _local_search(residues, counts, primes, WORK_L, ls_deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_residues = residues[:]
        
        # Pairwise local search - allocate 20% of restart time
        pl_deadline = min(deadline - 15.0, start_time + time_per_restart * 0.98)
        residues, counts, score = _pairwise_local_search(residues, counts, primes, WORK_L, pl_deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_residues = residues[:]
    
    # Final intensive local search on best solution
    if best_overall_residues is not None and time.time() < deadline - 10.0:
        counts = _build_counts_np(best_overall_residues, primes, WORK_L)
        final_deadline = deadline - 5.0
        best_overall_residues, _, _ = _local_search(best_overall_residues, counts, primes, WORK_L, final_deadline)
        
        # Another round of pairwise search
        if time.time() < deadline - 5.0:
            counts = _build_counts_np(best_overall_residues, primes, WORK_L)
            best_overall_residues, _, _ = _pairwise_local_search(best_overall_residues, counts, primes, WORK_L, deadline - 2.0)
    
    if best_overall_residues is None:
        # Fallback
        best_overall_residues, _ = _greedy_init(primes, WORK_L)
    
    # Ensure all residues are Python ints
    return [int(r) for r in best_overall_residues]
# EVOLVE-BLOCK-END
