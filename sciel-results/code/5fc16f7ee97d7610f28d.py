import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i in range(2, limit+1) if sieve[i]]

    primes = get_primes(1000)
    n = len(primes)  # 168

    # --- Fast numpy-based scoring ---
    def _build_count_array(residues, primes, max_L):
        """Build coverage count array for positions 1..max_L."""
        counts = np.zeros(max_L + 1, dtype=np.int32)
        for i in range(len(primes)):
            r = residues[i]
            p = primes[i]
            r_eff = p if r == 0 else r
            if r_eff <= max_L:
                positions = np.arange(r_eff, max_L + 1, p)
                counts[positions] += 1
        return counts

    def _find_L(counts, max_L):
        """Find largest L such that all positions 1..L have count >= 2."""
        deficient = np.where(counts[1:max_L+1] < 2)[0]
        if len(deficient) == 0:
            return max_L
        return int(deficient[0])

    def _score(residues, primes, max_L=20000):
        """Compute the largest L such that every integer 1..L is covered at least twice."""
        counts = _build_count_array(residues, primes, max_L)
        return _find_L(counts, max_L)

    def _score_full(residues, primes, max_L=100000):
        """Full range score for final verification."""
        counts = _build_count_array(residues, primes, max_L)
        return _find_L(counts, max_L)

    # --- Greedy forward construction with lookahead ---
    def _greedy_construct(primes, max_L_check=10000):
        """
        Forward greedy: process primes in order, for each prime choose the residue
        that maximizes coverage of currently-deficient positions with lookahead.
        """
        n = len(primes)
        residues = [0] * n
        counts = np.zeros(max_L_check + 1, dtype=np.int32)

        for idx in range(n):
            p = primes[idx]

            # Focus on positions 1..min(max_L_check, 1000) for speed
            search_range = min(max_L_check, 1000)
            
            # Try all residues and pick the one that maximizes the minimum count
            # in the early range, with tie-breaking by total coverage
            best_r = 0
            best_key = (-1, -1)
            
            for r in range(p):
                r_eff = p if r == 0 else r
                if r_eff > max_L_check:
                    continue
                    
                # Compute how many deficient positions this residue would cover
                # in the early range
                if r_eff <= search_range:
                    positions_in_range = np.arange(r_eff, search_range + 1, p)
                    # Count how many of these have count < 2
                    deficient_covered = np.sum(counts[positions_in_range] < 2)
                else:
                    deficient_covered = 0
                
                # Total positions covered in full range
                total_covered = (max_L_check - r_eff) // p + 1 if r_eff <= max_L_check else 0
                
                key = (deficient_covered, total_covered)
                if key > best_key:
                    best_key = key
                    best_r = r

            residues[idx] = best_r

            # Update counts
            r = best_r
            r_eff = p if r == 0 else r
            if r_eff <= max_L_check:
                positions = np.arange(r_eff, max_L_check + 1, p)
                counts[positions] += 1

        return residues

    # --- Local search with pair swaps ---
    def _local_search(residues, primes, max_L_check, deadline, max_iter=50000):
        """
        Local search: try changing one or two residues at a time to improve score.
        """
        n = len(primes)
        best_res = residues[:]
        counts = np.zeros(max_L_check + 1, dtype=np.int32)
        for i in range(n):
            r = best_res[i]
            p = primes[i]
            r_eff = p if r == 0 else r
            if r_eff <= max_L_check:
                positions = np.arange(r_eff, max_L_check + 1, p)
                counts[positions] += 1
        
        current_L = _find_L(counts, max_L_check)
        best_L = current_L
        best_res_copy = best_res[:]
        random.seed(42)

        iter_count = 0
        while time.time() < deadline and iter_count < max_iter:
            iter_count += 1
            
            # Decide whether to do single or double move
            if random.random() < 0.7:
                # Single move
                idx = random.randint(0, n - 1)
                p = primes[idx]
                old_r = best_res[idx]
                new_r = random.randint(0, p - 1)
                
                if new_r == old_r:
                    continue

                old_r_eff = p if old_r == 0 else old_r
                new_r_eff = p if new_r == 0 else new_r

                # Update counts
                if old_r_eff <= max_L_check:
                    old_pos = np.arange(old_r_eff, max_L_check + 1, p)
                    counts[old_pos] -= 1
                
                if new_r_eff <= max_L_check:
                    new_pos = np.arange(new_r_eff, max_L_check + 1, p)
                    counts[new_pos] += 1

                new_L = _find_L(counts, max_L_check)

                if new_L > current_L:
                    current_L = new_L
                    best_res[idx] = new_r
                    if current_L > best_L:
                        best_L = current_L
                        best_res_copy = best_res[:]
                else:
                    # Revert
                    if new_r_eff <= max_L_check:
                        counts[new_pos] -= 1
                    if old_r_eff <= max_L_check:
                        counts[old_pos] += 1
            else:
                # Double move
                idx1 = random.randint(0, n - 1)
                idx2 = random.randint(0, n - 1)
                if idx1 == idx2:
                    continue
                    
                p1 = primes[idx1]
                p2 = primes[idx2]
                old_r1 = best_res[idx1]
                old_r2 = best_res[idx2]
                new_r1 = random.randint(0, p1 - 1)
                new_r2 = random.randint(0, p2 - 1)
                
                if new_r1 == old_r1 and new_r2 == old_r2:
                    continue

                # Update counts for both
                old_r1_eff = p1 if old_r1 == 0 else old_r1
                old_r2_eff = p2 if old_r2 == 0 else old_r2
                new_r1_eff = p1 if new_r1 == 0 else new_r1
                new_r2_eff = p2 if new_r2 == 0 else new_r2

                if old_r1_eff <= max_L_check:
                    old_pos1 = np.arange(old_r1_eff, max_L_check + 1, p1)
                    counts[old_pos1] -= 1
                if old_r2_eff <= max_L_check:
                    old_pos2 = np.arange(old_r2_eff, max_L_check + 1, p2)
                    counts[old_pos2] -= 1
                
                if new_r1_eff <= max_L_check:
                    new_pos1 = np.arange(new_r1_eff, max_L_check + 1, p1)
                    counts[new_pos1] += 1
                if new_r2_eff <= max_L_check:
                    new_pos2 = np.arange(new_r2_eff, max_L_check + 1, p2)
                    counts[new_pos2] += 1

                new_L = _find_L(counts, max_L_check)

                if new_L > current_L:
                    current_L = new_L
                    best_res[idx1] = new_r1
                    best_res[idx2] = new_r2
                    if current_L > best_L:
                        best_L = current_L
                        best_res_copy = best_res[:]
                else:
                    # Revert
                    if new_r1_eff <= max_L_check:
                        counts[new_pos1] -= 1
                    if new_r2_eff <= max_L_check:
                        counts[new_pos2] -= 1
                    if old_r1_eff <= max_L_check:
                        counts[old_pos1] += 1
                    if old_r2_eff <= max_L_check:
                        counts[old_pos2] += 1

        return best_res_copy, best_L

    # --- Simulated Annealing with adaptive temperature ---
    def _simulated_annealing(residues, primes, max_L_check, deadline):
        """
        Simulated Annealing: allows worsening moves to escape local minima.
        """
        n = len(primes)
        best_res = residues[:]
        
        # Initialize counts
        counts = np.zeros(max_L_check + 1, dtype=np.int32)
        for i in range(n):
            r = best_res[i]
            p = primes[i]
            r_eff = p if r == 0 else r
            if r_eff <= max_L_check:
                positions = np.arange(r_eff, max_L_check + 1, p)
                counts[positions] += 1
        
        current_L = _find_L(counts, max_L_check)
        best_L = current_L
        best_res_copy = best_res[:]
        random.seed(42)

        # SA Parameters
        T = 100.0
        cooling_rate = 0.998
        min_T = 0.1
        no_improve_count = 0
        max_no_improve = 2000

        while time.time() < deadline and T > min_T:
            idx = random.randint(0, n - 1)
            p = primes[idx]
            old_r = best_res[idx]
            new_r = random.randint(0, p - 1)
            
            if new_r == old_r:
                continue

            old_r_eff = p if old_r == 0 else old_r
            new_r_eff = p if new_r == 0 else new_r

            # Update counts
            if old_r_eff <= max_L_check:
                old_pos = np.arange(old_r_eff, max_L_check + 1, p)
                counts[old_pos] -= 1
            
            if new_r_eff <= max_L_check:
                new_pos = np.arange(new_r_eff, max_L_check + 1, p)
                counts[new_pos] += 1

            new_L = _find_L(counts, max_L_check)

            # Acceptance
            if new_L >= current_L:
                accept = True
            else:
                delta = new_L - current_L
                prob = np.exp(delta / T)
                accept = (random.random() < prob)

            if accept:
                current_L = new_L
                best_res[idx] = new_r
                if current_L > best_L:
                    best_L = current_L
                    best_res_copy = best_res[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Revert
                if new_r_eff <= max_L_check:
                    counts[new_pos] -= 1
                if old_r_eff <= max_L_check:
                    counts[old_pos] += 1

            T *= cooling_rate

            # Reset temperature if stuck
            if no_improve_count > max_no_improve:
                T = max(T, 50.0)
                no_improve_count = 0

        return best_res_copy, best_L

    # --- Main algorithm ---
    # Step 1: Greedy forward construction
    residues = _greedy_construct(primes, max_L_check=10000)
    current_L = _score(residues, primes, 20000)

    # Step 2: Local search
    residues, current_L = _local_search(residues, primes, 20000, deadline, max_iter=30000)

    # Step 3: Simulated Annealing
    residues, current_L = _simulated_annealing(residues, primes, 20000, deadline)

    # Step 4: Final local search pass
    residues, current_L = _local_search(residues, primes, 20000, deadline, max_iter=20000)

    # Step 5: Final verification with full range
    final_L = _score_full(residues, primes, 100000)

    # Ensure pure Python ints
    return [int(r) for r in residues]
