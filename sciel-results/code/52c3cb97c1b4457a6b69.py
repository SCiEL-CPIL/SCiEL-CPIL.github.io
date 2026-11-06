"""
Strategy: Multi-phase coverage construction with redundancy-aware repair.
Phase 1: Build high-density coverage using small primes strategically.
Phase 2: Repair gaps with lookahead to avoid creating new failures.
Phase 3: Polish with local search for maximum prefix extension.
"""
import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s kill
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    max_check = 50000  # Monitor up to 50000 for potential high scores
    
    # residues[i] is the residue for primes[i]
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # counts[x] = number of primes p such that x == residues[idx] (mod p)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    def update_counts(idx, old_r, new_r, delta):
        """
        Update counts array.
        delta = +1 means adding the residue, -1 means removing.
        """
        p = primes[idx]
        r = new_r if delta == 1 else old_r
        
        if r == 0:
            start = p
        else:
            start = r
            
        if start <= max_check:
            counts[start::p] += delta

    def get_prefix_score():
        """Find the largest L such that counts[1..L] >= 2."""
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        return below_two[0]

    def is_valid_change(idx, old_r, new_r, current_score):
        """Check if changing residue[idx] from old_r to new_r breaks the prefix."""
        p = primes[idx]
        if old_r == new_r:
            return False
        
        # Check if removing coverage from old_r breaks any prefix point
        if old_r == 0:
            start_pos = p
        else:
            start_pos = old_r
            
        if start_pos <= current_score:
            positions = np.arange(start_pos, current_score + 1, p)
            if np.any(counts[positions] == 2):
                return False
        
        return True

    # --- Phase 1: Initial Dense Coverage ---
    # Strategy: Use small primes to create high coverage density.
    # Assign residues to spread coverage evenly.
    
    # Initialize with a pattern that creates good coverage
    for i in range(n_primes):
        p = primes[i]
        # Use a hash-like function to spread residues
        residues[i] = (i * 7 + 3) % p
    
    # Apply initial counts
    for i in range(n_primes):
        update_counts(i, 0, residues[i], 1)
    
    best_score = get_prefix_score()
    best_residues = residues.copy()
    
    # --- Phase 2: Iterative Repair with Lookahead ---
    # Fix gaps one by one, but with lookahead to avoid creating new gaps
    
    current_score = best_score
    max_repair_steps = 20000
    step = 0
    
    while time.time() < deadline - 20.0 and step < max_repair_steps:
        step += 1
        
        current_score = get_prefix_score()
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        if current_score >= max_check:
            break
            
        x_target = current_score + 1
        
        # Find the best prime to cover x_target
        # Score candidates by: (1) doesn't break prefix, (2) minimizes future risk
        
        best_candidate = -1
        best_score_val = -1
        
        # Try primes in order of increasing p (smaller primes affect more positions)
        # But prioritize those that don't break the prefix
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            if old_r == new_r:
                continue
            
            if not is_valid_change(i, old_r, new_r, current_score):
                continue
            
            # Score this candidate: prefer primes that create less "fragility"
            # Fragility = number of positions in [current_score+1, current_score+100] 
            # that would have count exactly 2 after this change
            # (These are positions that could easily become gaps)
            
            # For simplicity, just pick the first valid one, but prefer larger p
            # (larger p affects fewer positions, so less risk)
            score_val = p  # Larger p is better (less impact)
            
            if score_val > best_score_val:
                best_score_val = score_val
                best_candidate = i
        
        if best_candidate >= 0:
            i = best_candidate
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            update_counts(i, old_r, new_r, -1)
            residues[i] = new_r
            update_counts(i, new_r, new_r, 1)
        else:
            # Stuck! No single move can fix the gap.
            # Try a "rebalance" move: change a prime to reduce fragility
            # Find positions in the prefix with count == 2 (fragile)
            # and try to add coverage to them using a prime that's not critical elsewhere
            
            fragile_positions = np.where(counts[1:current_score+1] == 2)[0] + 1
            if len(fragile_positions) == 0:
                break
            
            # Pick a random fragile position and try to add coverage to it
            target_pos = fragile_positions[np.random.randint(len(fragile_positions))]
            
            found_rebalance = False
            for i in range(n_primes - 1, -1, -1):  # Large primes first
                p = primes[i]
                old_r = residues[i]
                new_r = target_pos % p
                
                if old_r == new_r:
                    continue
                
                if not is_valid_change(i, old_r, new_r, current_score):
                    continue
                
                # Apply the rebalance
                update_counts(i, old_r, new_r, -1)
                residues[i] = new_r
                update_counts(i, new_r, new_r, 1)
                found_rebalance = True
                break
            
            if not found_rebalance:
                # Try a more aggressive move: temporarily break the prefix
                # to escape the local minimum
                # Pick a prime with high redundancy and change it randomly
                # Then try to repair the damage
                
                # Find primes that are "safe" to move (their removal doesn't break prefix)
                safe_primes = []
                for i in range(n_primes):
                    p = primes[i]
                    old_r = residues[i]
                    if old_r == 0:
                        start_pos = p
                    else:
                        start_pos = old_r
                    
                    if start_pos > current_score:
                        safe_primes.append(i)
                    else:
                        positions = np.arange(start_pos, current_score + 1, p)
                        if not np.any(counts[positions] == 2):
                            safe_primes.append(i)
                
                if len(safe_primes) == 0:
                    break
                
                # Pick a random safe prime and move it
                i = safe_primes[np.random.randint(len(safe_primes))]
                p = primes[i]
                old_r = residues[i]
                new_r = np.random.randint(p)
                
                if new_r == old_r:
                    new_r = (new_r + 1) % p
                
                update_counts(i, old_r, new_r, -1)
                residues[i] = new_r
                update_counts(i, new_r, new_r, 1)
    
    # --- Phase 3: Final Polish ---
    # Try to extend the prefix further with more aggressive moves
    
    current_score = get_prefix_score()
    if current_score > best_score:
        best_score = current_score
        best_residues = residues.copy()
    
    # Try a few more iterations with different strategies
    polish_steps = 5000
    for step in range(polish_steps):
        if time.time() > deadline - 5.0:
            break
        
        current_score = get_prefix_score()
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
        
        if current_score >= max_check:
            break
        
        x_target = current_score + 1
        
        # Try all primes, pick the one that gives the best "lookahead" score
        best_candidate = -1
        best_lookahead = -1
        
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            if old_r == new_r:
                continue
            
            if not is_valid_change(i, old_r, new_r, current_score):
                continue
            
            # Lookahead: simulate the change and check how many new gaps appear
            # in the next 50 positions
            update_counts(i, old_r, new_r, -1)
            residues[i] = new_r
            update_counts(i, new_r, new_r, 1)
            
            # Count new gaps in [current_score+1, current_score+50]
            new_gaps = np.sum(counts[current_score+1:min(current_score+51, max_check+1)] < 2)
            lookahead = -new_gaps + p  # Prefer fewer new gaps, and larger p
            
            # Revert
            update_counts(i, new_r, new_r, -1)
            residues[i] = old_r
            update_counts(i, old_r, old_r, 1)
            
            if lookahead > best_lookahead:
                best_lookahead = lookahead
                best_candidate = i
        
        if best_candidate >= 0:
            i = best_candidate
            p = primes[i]
            old_r = residues[i]
            new_r = x_target % p
            
            update_counts(i, old_r, new_r, -1)
            residues[i] = new_r
            update_counts(i, new_r, new_r, 1)
        else:
            break
    
    # Final check
    final_score = get_prefix_score()
    if final_score > best_score:
        best_score = final_score
        best_residues = residues.copy()

    return [int(r) for r in best_residues]
