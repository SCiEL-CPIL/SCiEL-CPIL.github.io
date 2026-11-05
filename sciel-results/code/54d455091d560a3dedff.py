import time
import numpy as np

def _get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _compute_counts(residues, primes, max_L):
    """Compute coverage counts for 1..max_L given residues and primes."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        pos = np.arange(start, max_L + 1, p)
        if len(pos) > 0:
            counts[pos] += 1
    return counts

def _find_L(counts, max_L):
    """Find largest L such that all positions 1..L have count >= 2."""
    below = np.where(counts[1:] < 2)[0]
    if len(below) == 0:
        return max_L
    return below[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Forward greedy construction
    # For each prime in order, choose the residue that maximizes the doubly-covered prefix
    # We use a moderate max_L for evaluation during construction, then refine
    max_L_eval = 5000
    
    residues = [0] * n_primes
    current_counts = np.zeros(max_L_eval + 1, dtype=np.int32)
    
    for i in range(n_primes):
        if (time.time() - start_time) > deadline - 20:
            break
        p = primes[i]
        best_r = 0
        best_L = 0
        
        # Try all residues for small primes, sample for large ones
        if p <= 50:
            candidates = range(p)
        else:
            # Sample residues: focus on residues that cover early integers
            # Residue r covers r, r+p, r+2p, ...
            # For early coverage, we want r to be small (covers 1, 2, 3, ...)
            # or r such that r+k*p hits uncovered positions
            candidates = list(range(min(p, 100)))
            # Also add some random samples
            for _ in range(50):
                candidates.append(int(np.random.randint(0, p)))
        
        for r in candidates:
            # Compute new counts if we set residue[i] = r
            # Only need to add this prime's contribution
            if r == 0:
                start = p
            else:
                start = r
            if start > max_L_eval:
                new_counts = current_counts.copy()
            else:
                new_counts = current_counts.copy()
                pos = np.arange(start, max_L_eval + 1, p)
                if len(pos) > 0:
                    new_counts[pos] += 1
            
            L = _find_L(new_counts, max_L_eval)
            if L > best_L:
                best_L = L
                best_r = r
        
        residues[i] = best_r
        # Update current_counts
        r = best_r
        if r == 0:
            start = p
        else:
            start = r
        if start <= max_L_eval:
            pos = np.arange(start, max_L_eval + 1, p)
            if len(pos) > 0:
                current_counts[pos] += 1
    
    # Now refine: backward pass, re-optimizing each residue given all others
    # Use a larger max_L for final evaluation
    max_L_final = 100000
    
    # Compute full counts for the current solution
    full_counts = _compute_counts(residues, primes, max_L_final)
    best_L = _find_L(full_counts, max_L_final)
    
    # Backward refinement: go from last prime to first, re-optimize each
    for i in range(n_primes - 1, -1, -1):
        if (time.time() - start_time) > deadline - 10:
            break
        p = primes[i]
        current_r = residues[i]
        
        # Remove current prime's contribution
        if current_r == 0:
            start_old = p
        else:
            start_old = current_r
        if start_old <= max_L_final:
            pos_old = np.arange(start_old, max_L_final + 1, p)
            if len(pos_old) > 0:
                full_counts[pos_old] -= 1
        
        # Try candidate residues
        if p <= 100:
            candidates = range(p)
        else:
            # Focus on residues that might help extend coverage
            # Try small residues (cover early integers) and residues near gaps
            candidates = list(range(min(p, 200)))
            for _ in range(100):
                candidates.append(int(np.random.randint(0, p)))
        
        best_r = current_r
        best_new_L = best_L
        
        for r in candidates:
            if r == current_r:
                continue
            # Add this prime's contribution
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_L_final:
                pos = np.arange(start, max_L_final + 1, p)
                if len(pos) > 0:
                    full_counts[pos] += 1
            
            L = _find_L(full_counts, max_L_final)
            
            # Remove this prime's contribution
            if start <= max_L_final:
                pos = np.arange(start, max_L_final + 1, p)
                if len(pos) > 0:
                    full_counts[pos] -= 1
            
            if L > best_new_L:
                best_new_L = L
                best_r = r
        
        # Apply best residue
        if best_r != current_r:
            residues[i] = best_r
            # Update full_counts
            if best_r == 0:
                start = p
            else:
                start = best_r
            if start <= max_L_final:
                pos = np.arange(start, max_L_final + 1, p)
                if len(pos) > 0:
                    full_counts[pos] += 1
            best_L = best_new_L
    
    # Forward refinement pass
    for i in range(n_primes):
        if (time.time() - start_time) > deadline - 5:
            break
        p = primes[i]
        current_r = residues[i]
        
        # Remove current prime's contribution
        if current_r == 0:
            start_old = p
        else:
            start_old = current_r
        if start_old <= max_L_final:
            pos_old = np.arange(start_old, max_L_final + 1, p)
            if len(pos_old) > 0:
                full_counts[pos_old] -= 1
        
        # Try candidate residues
        if p <= 100:
            candidates = range(p)
        else:
            candidates = list(range(min(p, 200)))
            for _ in range(100):
                candidates.append(int(np.random.randint(0, p)))
        
        best_r = current_r
        best_new_L = best_L
        
        for r in candidates:
            if r == current_r:
                continue
            if r == 0:
                start = p
            else:
                start = r
            if start <= max_L_final:
                pos = np.arange(start, max_L_final + 1, p)
                if len(pos) > 0:
                    full_counts[pos] += 1
            
            L = _find_L(full_counts, max_L_final)
            
            if start <= max_L_final:
                pos = np.arange(start, max_L_final + 1, p)
                if len(pos) > 0:
                    full_counts[pos] -= 1
            
            if L > best_new_L:
                best_new_L = L
                best_r = r
        
        if best_r != current_r:
            residues[i] = best_r
            if best_r == 0:
                start = p
            else:
                start = best_r
            if start <= max_L_final:
                pos = np.arange(start, max_L_final + 1, p)
                if len(pos) > 0:
                    full_counts[pos] += 1
            best_L = best_new_L
    
    return residues
