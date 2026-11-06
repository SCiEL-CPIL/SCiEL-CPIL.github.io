"""Simulated annealing with multi-start for doubly-covered prefix optimization."""
import time
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    n_primes = len(primes)  # 168
    
    # Use a larger range to allow L to grow
    MAX_L = 20000
    
    def compute_L(coverage):
        """Find the largest L such that all 1..L have coverage >= 2."""
        # Find first index >= 1 where coverage < 2
        gaps = np.where(coverage[1:] < 2)[0]
        if len(gaps) == 0:
            return MAX_L
        return gaps[0]  # gaps[0] is 0-indexed relative to coverage[1:], so +1 gives actual index, but we want L = first_gap - 1... wait
        # Actually: coverage[1:] < 2 finds positions in [1, MAX_L] with cov < 2
        # If gaps[0] = k, then position k+1 has cov < 2, so L = k
        # Wait no: coverage[1:] is positions 1,2,3,...,MAX_L
        # np.where returns 0-indexed indices into this array
        # So gaps[0] = 0 means position 1 has cov < 2, L = 0
        # gaps[0] = 5 means position 6 has cov < 2, L = 5
        return gaps[0] if len(gaps) > 0 else MAX_L

    def apply_congruence(coverage, idx, r, sign):
        """Add or remove a congruence. sign=+1 to add, -1 to remove."""
        p = primes[idx]
        start = r if r > 0 else p
        coverage[start::p] += sign

    def greedy_init():
        """Create initial residues using greedy gap-filling."""
        coverage = np.zeros(MAX_L + 1, dtype=np.int32)
        residues = np.zeros(n_primes, dtype=np.int32)
        
        for idx, p in enumerate(primes):
            # Find current gap
            gaps = np.where(coverage[1:] < 2)[0]
            if len(gaps) == 0:
                r = np.random.randint(0, p)
            else:
                gap_pos = gaps[0] + 1  # actual position
                r = gap_pos % p
            
            residues[idx] = r
            apply_congruence(coverage, idx, r, +1)
        
        return residues, coverage

    def simulated_annealing(residues, coverage, max_iters, t0, t1, alpha):
        """Run simulated annealing starting from given state."""
        current_L = compute_L(coverage)
        best_L = current_L
        best_residues = residues.copy()
        
        # Temperature schedule
        t = t0
        iters = 0
        while iters < max_iters and t > t1:
            iters += 1
            
            # Pick a random prime to change
            idx = np.random.randint(0, n_primes)
            p = primes[idx]
            old_r = residues[idx]
            
            # Pick a new random residue
            new_r = np.random.randint(0, p)
            if new_r == old_r:
                t *= alpha
                continue
            
            # Apply change
            apply_congruence(coverage, idx, old_r, -1)
            apply_congruence(coverage, idx, new_r, +1)
            
            new_L = compute_L(coverage)
            delta = new_L - current_L
            
            if delta > 0 or np.random.random() < np.exp(delta / max(t, 1e-10)):
                # Accept
                current_L = new_L
                residues[idx] = new_r
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                # Revert
                apply_congruence(coverage, idx, new_r, -1)
                apply_congruence(coverage, idx, old_r, +1)
            
            t *= alpha
        
        return best_L, best_residues, coverage

    # Multi-start simulated annealing
    best_overall_L = 0
    best_overall_residues = None
    
    # First, get a greedy baseline
    residues, coverage = greedy_init()
    base_L = compute_L(coverage)
    best_overall_L = base_L
    best_overall_residues = residues.copy()
    
    # Multiple SA restarts with different initial temperatures
    np.random.seed(42)
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # For each restart, use a slightly perturbed version of the best solution
        if restart == 0:
            start_residues = best_overall_residues.copy()
            start_coverage = coverage.copy()
        else:
            # Perturb the best solution
            start_residues = best_overall_residues.copy()
            start_coverage = np.zeros(MAX_L + 1, dtype=np.int32)
            for idx in range(n_primes):
                apply_congruence(start_coverage, idx, start_residues[idx], +1)
            
            # Make some random changes
            num_changes = np.random.randint(5, 20)
            for _ in range(num_changes):
                idx = np.random.randint(0, n_primes)
                p = primes[idx]
                new_r = np.random.randint(0, p)
                old_r = start_residues[idx]
                if new_r != old_r:
                    apply_congruence(start_coverage, idx, old_r, -1)
                    apply_congruence(start_coverage, idx, new_r, +1)
                    start_residues[idx] = new_r
        
        # Run SA with parameters tuned for this restart
        t0 = 50.0 / (restart + 1)
        t1 = 0.01
        alpha = 0.999
        max_iters = min(50000, int((deadline - 20 - time.time()) * 1000 / 0.001))
        
        if max_iters < 1000:
            break
        
        sa_L, sa_residues, sa_coverage = simulated_annealing(
            start_residues, start_coverage, max_iters, t0, t1, alpha
        )
        
        if sa_L > best_overall_L:
            best_overall_L = sa_L
            best_overall_residues = sa_residues.copy()
            coverage = sa_coverage.copy()
    
    # Final local search polish
    if time.time() < deadline - 10:
        current_L = compute_L(coverage)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            gaps = np.where(coverage[1:] < 2)[0]
            if len(gaps) == 0:
                break
            
            gap_pos = gaps[0] + 1
            
            for idx in range(n_primes):
                if time.time() > deadline - 5:
                    break
                
                p = primes[idx]
                target_r = gap_pos % p
                old_r = residues[idx]
                
                if old_r == target_r:
                    continue
                
                # Try the change
                apply_congruence(coverage, idx, old_r, -1)
                apply_congruence(coverage, idx, target_r, +1)
                
                new_L = compute_L(coverage)
                if new_L > current_L:
                    current_L = new_L
                    residues[idx] = target_r
                    improved = True
                else:
                    apply_congruence(coverage, idx, target_r, -1)
                    apply_congruence(coverage, idx, old_r, +1)
                
                if improved:
                    break
    
    return residues.tolist()
# EVOLVE-BLOCK-END
