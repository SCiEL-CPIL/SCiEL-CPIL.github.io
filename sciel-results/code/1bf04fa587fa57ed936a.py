import time
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
        if max_L <= 0:
            return 0
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

    # --- Simulated Annealing ---
    def _simulated_annealing(residues, primes, deadline, max_L_check=20000):
        """
        Simulated Annealing: perturb residues, accept improvements and some
        worsenings with a temperature schedule.
        """
        import random
        random.seed(12345)

        best_res = residues[:]
        best_L = _score(best_res, primes, max_L_check)
        current_res = residues[:]
        current_L = best_L

        # Temperature schedule
        T_start = 100.0
        T_end = 0.1
        num_steps = 50000

        step = 0
        while time.time() < deadline and step < num_steps:
            step += 1

            # Temperature decay
            T = T_start * (T_end / T_start) ** (step / num_steps)

            # Pick a random index to perturb
            idx = random.randint(0, n - 1)
            p = primes[idx]
            old_r = current_res[idx]

            # Try a random new value
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue

            current_res[idx] = new_r
            new_L = _score(current_res, primes, max_L_check)

            delta = new_L - current_L

            # Accept if improvement, or with probability exp(-delta/T) for worsenings
            if delta >= 0:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_res = current_res[:]
            else:
                # Accept worsening with probability
                if random.random() < np.exp(-delta / T):
                    current_L = new_L
                else:
                    # Revert
                    current_res[idx] = old_r

        return best_res, best_L

    # --- Greedy forward construction (improved) ---
    def _greedy_construct(primes, max_L_check=10000):
        """
        Forward greedy: process primes in order, for each prime choose the residue
        that maximizes coverage of currently-deficient positions.
        """
        n = len(primes)
        residues = [0] * n
        counts = np.zeros(max_L_check + 1, dtype=np.int32)

        for idx in range(n):
            p = primes[idx]

            # Find deficient positions in early range
            search_range = min(max_L_check, 1000)
            deficient_positions = []
            for pos in range(1, search_range + 1):
                if counts[pos] < 2:
                    deficient_positions.append(pos)
                    if len(deficient_positions) >= 200:
                        break

            if not deficient_positions:
                # No deficient positions in early range
                best_r = 0
                best_count = -1
                for r in range(p):
                    r_eff = p if r == 0 else r
                    if r_eff > max_L_check:
                        continue
                    num_positions = (max_L_check - r_eff) // p + 1
                    if num_positions > best_count:
                        best_count = num_positions
                        best_r = r
            else:
                # Weighted histogram: positions with count 0 get weight 3, count 1 get weight 1
                residue_weighted = np.zeros(p, dtype=np.int32)
                for pos in deficient_positions:
                    r_needed = pos % p
                    weight = 3 if counts[pos] == 0 else 1
                    residue_weighted[r_needed] += weight
                best_r = int(np.argmax(residue_weighted))

            residues[idx] = best_r

            # Update counts
            r = best_r
            r_eff = p if r == 0 else r
            if r_eff <= max_L_check:
                positions = np.arange(r_eff, max_L_check + 1, p)
                counts[positions] += 1

        return residues

    # --- Local search with random perturbations ---
    def _local_search(residues, primes, max_L_check, deadline):
        """
        Local search: randomly perturb residues and accept improvements.
        """
        import random
        random.seed(42)

        best_res = residues[:]
        best_L = _score(best_res, primes, max_L_check)
        no_improve_count = 0
        max_no_improve = 50

        while time.time() < deadline:
            improved = False
            indices = list(range(n))
            random.shuffle(indices)

            for i in indices:
                if time.time() > deadline:
                    break
                p = primes[i]
                old_r = best_res[i]

                # Try a few random new values
                for _ in range(5):
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        continue
                    best_res[i] = new_r
                    new_L = _score(best_res, primes, max_L_check)
                    if new_L > best_L:
                        best_L = new_L
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        best_res[i] = old_r

            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
            else:
                no_improve_count = 0

        return best_res, best_L

    # --- Main algorithm ---
    # Step 1: Greedy forward construction
    residues = _greedy_construct(primes, max_L_check=10000)
    current_L = _score(residues, primes, 20000)

    # Step 2: Simulated Annealing (new structural approach)
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        sa_deadline = time.time() + min(remaining_time - 10, 80)
        residues, current_L = _simulated_annealing(residues, primes, sa_deadline, max_L_check=20000)

    # Step 3: Local search with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        ls_deadline = time.time() + min(remaining_time - 5, 30)
        residues, current_L = _local_search(residues, primes, 20000, ls_deadline)

    # Step 4: Final verification with full range
    final_L = _score_full(residues, primes, 100000)

    # Ensure pure Python ints
    return [int(r) for r in residues]
