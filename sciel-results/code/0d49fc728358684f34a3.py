import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 150s limit
    
    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_np = np.array(primes, dtype=np.int64)
    
    # The range of integers to check.
    MAX_CHECK = 5000
    x_values = np.arange(1, MAX_CHECK + 1)
    
    # Precompute x mod p for all x and p.
    x_mod_p = x_values[:, None] % primes_np[None, :]
    
    # Helper: Compute the current score (L)
    def compute_score(residues):
        match = (x_mod_p == residues[None, :])
        counts = match.sum(axis=1)
        invalid = counts < 2
        if not np.any(invalid):
            return MAX_CHECK
        else:
            first_invalid_idx = np.argmax(invalid)
            return int(first_invalid_idx)
    
    # Helper: Try to extend coverage by finding best single move at the gap
    def try_extend(residues, current_score):
        if current_score >= MAX_CHECK:
            return residues, current_score
        
        gap_idx = current_score
        match_at_gap = (x_mod_p[gap_idx] == residues)
        candidates = np.where(~match_at_gap)[0]
        
        if len(candidates) == 0:
            return residues, current_score
        
        M = (x_mod_p == residues[None, :])
        old_counts = M.sum(axis=1)
        
        C = candidates
        R = x_mod_p[gap_idx, C]
        
        x_mod_p_cands = x_mod_p[:, C]
        new_matches_cands = (x_mod_p_cands == R[None, :])
        M_cands = M[:, C]
        
        delta_counts = new_matches_cands.astype(np.int8) - M_cands.astype(np.int8)
        new_counts = old_counts[:, None].astype(np.int8) + delta_counts
        
        invalid = new_counts < 2
        has_gap = invalid.any(axis=0)
        first_gap_idx = np.argmax(invalid, axis=0)
        scores = np.where(has_gap, first_gap_idx, MAX_CHECK)
        
        best_idx_in_scores = np.argmax(scores)
        best_candidate_idx = C[best_idx_in_scores]
        best_new_score = scores[best_idx_in_scores]
        
        if best_new_score > current_score:
            new_residues = residues.copy()
            new_residues[best_candidate_idx] = R[best_idx_in_scores]
            return new_residues, best_new_score
        
        # If no improvement, try a move that keeps score same but helps at gap
        if best_new_score == current_score:
            # Check if this move actually increases count at gap
            old_count_at_gap = old_counts[gap_idx]
            if old_count_at_gap < 2:
                new_residues = residues.copy()
                new_residues[best_candidate_idx] = R[best_idx_in_scores]
                return new_residues, current_score
        
        return residues, current_score
    
    # Helper: Two-move local search - try pairs of residue changes
    def two_move_search(residues, current_score, max_moves=50):
        if current_score >= MAX_CHECK:
            return residues, current_score
        
        best_res = residues.copy()
        best_sc = current_score
        
        gap_idx = current_score
        match_at_gap = (x_mod_p[gap_idx] == residues)
        candidates = np.where(~match_at_gap)[0]
        
        if len(candidates) == 0:
            return best_res, best_sc
        
        # Limit candidates for two-move search to keep it fast
        if len(candidates) > 30:
            np.random.shuffle(candidates)
            candidates = candidates[:30]
        
        M = (x_mod_p == residues[None, :])
        old_counts = M.sum(axis=1)
        
        moves_made = 0
        while moves_made < max_moves and time.time() < deadline - 2.0:
            moved = False
            gap_idx = best_sc
            match_at_gap = (x_mod_p[gap_idx] == best_res)
            candidates = np.where(~match_at_gap)[0]
            
            if len(candidates) == 0:
                break
            
            if len(candidates) > 30:
                np.random.shuffle(candidates)
                candidates = candidates[:30]
            
            M = (x_mod_p == best_res[None, :])
            old_counts = M.sum(axis=1)
            
            C = candidates
            R = x_mod_p[gap_idx, C]
            
            # Try all pairs of candidates
            best_pair_score = best_sc
            best_pair = None
            
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    # Simulate changing both C[i] and C[j]
                    new_res = best_res.copy()
                    new_res[C[i]] = R[i]
                    new_res[C[j]] = R[j]
                    
                    match = (x_mod_p == new_res[None, :])
                    counts = match.sum(axis=1)
                    invalid = counts < 2
                    if not np.any(invalid):
                        pair_score = MAX_CHECK
                    else:
                        pair_score = int(np.argmax(invalid))
                    
                    if pair_score > best_pair_score:
                        best_pair_score = pair_score
                        best_pair = (C[i], C[j], R[i], R[j])
            
            if best_pair is not None and best_pair_score > best_sc:
                ci, cj, ri, rj = best_pair
                best_res[ci] = ri
                best_res[cj] = rj
                best_sc = best_pair_score
                moved = True
                moves_made += 1
            
            if not moved:
                # Try single moves as fallback
                new_res, new_sc = try_extend(best_res, best_sc)
                if new_sc > best_sc:
                    best_res = new_res
                    best_sc = new_sc
                    moves_made += 1
                else:
                    break
        
        return best_res, best_sc
    
    # Strategy: Multiple restarts with different initializations
    np.random.seed(42)
    
    best_residues = None
    best_score = 0
    
    # Time budget per restart
    restart_time_budget = 15.0
    
    while time.time() < deadline - 3.0:
        # Different initialization strategies
        init_strategy = np.random.randint(0, 4)
        
        if init_strategy == 0:
            # All ones
            current_residues = np.ones(n_primes, dtype=np.int64)
        elif init_strategy == 1:
            # Random residues
            current_residues = np.array([np.random.randint(0, p) for p in primes], dtype=np.int64)
        elif init_strategy == 2:
            # Start with r_p = 0 for small primes, 1 for large
            current_residues = np.where(primes_np < 50, 0, 1).astype(np.int64)
        else:
            # Greedy initialization: for each x from 1, assign residues to cover it
            current_residues = np.zeros(n_primes, dtype=np.int64)
            for x in range(1, min(50, MAX_CHECK + 1)):
                # Find primes that don't match x
                matches = (x_values[x-1][None, :] == current_residues[None, :]) if x <= MAX_CHECK else None
                # Simply set some residues to x % p
                for pi in np.random.permutation(n_primes)[:3]:
                    current_residues[pi] = x % primes[pi]
        
        current_score = compute_score(current_residues)
        
        # Phase 1: Greedy single-move extension
        restart_deadline = time.time() + restart_time_budget * 0.4
        while time.time() < restart_deadline and current_score < MAX_CHECK:
            new_res, new_sc = try_extend(current_residues, current_score)
            if new_sc > current_score:
                current_residues = new_res
                current_score = new_sc
            else:
                break
        
        # Phase 2: Two-move local search
        restart_deadline2 = time.time() + restart_time_budget * 0.5
        if time.time() < restart_deadline2 and current_score < MAX_CHECK:
            current_residues, current_score = two_move_search(current_residues, current_score, max_moves=100)
        
        # Phase 3: More greedy extension after two-move
        restart_deadline3 = time.time() + restart_time_budget * 0.1
        while time.time() < restart_deadline3 and current_score < MAX_CHECK:
            new_res, new_sc = try_extend(current_residues, current_score)
            if new_sc > current_score:
                current_residues = new_res
                current_score = new_sc
            else:
                break
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_residues = current_residues.copy()
        
        if best_score >= MAX_CHECK:
            break
    
    # Final: Try to extend the best solution further with remaining time
    if best_residues is not None and time.time() < deadline - 1.0:
        final_res = best_residues.copy()
        final_sc = best_score
        while time.time() < deadline - 0.5 and final_sc < MAX_CHECK:
            new_res, new_sc = try_extend(final_res, final_sc)
            if new_sc > final_sc:
                final_res = new_res
                final_sc = new_sc
            else:
                # Try two-move search one more time
                final_res, final_sc = two_move_search(final_res, final_sc, max_moves=50)
                if final_sc <= best_score:
                    break
                if final_sc > best_score:
                    best_score = final_sc
                    best_residues = final_res.copy()
                else:
                    break
    
    if best_residues is None:
        best_residues = np.ones(n_primes, dtype=np.int64)
    
    return [int(r) for r in best_residues]

if __name__ == "__main__":
    res = construct()
    print(len(res))
