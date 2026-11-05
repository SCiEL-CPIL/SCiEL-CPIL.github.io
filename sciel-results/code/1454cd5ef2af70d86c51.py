import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    # State
    residues = np.zeros(n_primes, dtype=np.int32)
    counts = np.zeros(MAX_L + 1, dtype=np.int8)

    def build_counts():
        counts[:] = 0
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                counts[p::p] += 1
            else:
                counts[r::p] += 1

    def find_L(current_L=0):
        if current_L >= MAX_L:
            return MAX_L
        if current_L > 0:
            prefix = counts[1:current_L + 1]
            if np.min(prefix) < 2:
                bad = np.where(prefix < 2)[0]
                return bad[0]
        if current_L < MAX_L:
            rest = counts[current_L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            return current_L + bad_idx[0]
        return current_L

    def apply_change(i, new_r):
        p = primes[i]
        old_r = residues[i]
        if old_r == new_r:
            return
        if old_r == 0:
            counts[p::p] -= 1
        else:
            counts[old_r::p] -= 1
        residues[i] = new_r
        if new_r == 0:
            counts[p::p] += 1
        else:
            counts[new_r::p] += 1

    def revert_change(i, old_r):
        p = primes[i]
        new_r = residues[i]
        if old_r == new_r:
            return
        if new_r == 0:
            counts[p::p] -= 1
        else:
            counts[new_r::p] -= 1
        residues[i] = old_r
        if old_r == 0:
            counts[p::p] += 1
        else:
            counts[old_r::p] += 1

    # --- Phase 1: Greedy Initialization ---
    build_counts()
    current_L = find_L(0)
    best_L = current_L
    best_residues = residues.copy()

    # --- Phase 2: Targeted Greedy Extension ---
    # Directly try to extend L by finding primes that can safely cover L+1
    extension_rounds = 0
    while current_L < MAX_L and time.time() < deadline - 30.0:
        x = current_L + 1
        candidates = []
        for i, p in enumerate(primes):
            target_r = x % p
            if residues[i] == target_r:
                continue
            old_r = residues[i]
            # Check if removing old_r coverage breaks any y in [1, x-1]
            safe = True
            if old_r == 0:
                # Coverage positions: p, 2p, 3p, ...
                # Check positions in [1, x-1] that are multiples of p
                if p < x:
                    # positions p, 2p, ..., floor((x-1)/p)*p
                    pos = p
                    while pos < x:
                        if counts[pos] <= 2:
                            safe = False
                            break
                        pos += p
            else:
                # Coverage positions: old_r, old_r+p, old_r+2p, ...
                pos = old_r
                while pos < x:
                    if pos >= 1 and counts[pos] <= 2:
                        safe = False
                        break
                    pos += p
            if safe:
                candidates.append(i)

        if candidates:
            # Pick the candidate that gives the best extension
            best_i = None
            best_new_L = current_L
            for i in candidates:
                p = primes[i]
                target_r = x % p
                old_r = residues[i]
                apply_change(i, target_r)
                new_L = find_L(current_L)
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_i = i
                    best_target_r = target_r
                    best_old_r = old_r
                revert_change(i, old_r)
            
            if best_i is not None:
                apply_change(best_i, best_target_r)
                current_L = best_new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                extension_rounds += 1
        else:
            break

    # --- Phase 3: Simulated Annealing for refinement ---
    T = 5.0
    cooling_rate = 0.99995
    min_T = 0.05
    restart_count = 0
    max_restarts = 5

    while time.time() < deadline - 10.0:
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = np.random.randint(p)

        if new_r == old_r:
            continue

        apply_change(i, new_r)
        new_L = find_L(current_L)

        delta = new_L - current_L

        if delta > 0:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Accept with probability based on delta
            if delta > -500:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_L = new_L
                else:
                    revert_change(i, old_r)
            else:
                revert_change(i, old_r)

        T *= cooling_rate
        if T < min_T:
            T = min_T
            # Periodic restart to escape local minima
            restart_count += 1
            if restart_count % 10000 == 0 and restart_count <= max_restarts * 10000:
                if time.time() < deadline - 15.0:
                    T = 3.0
                    # Small perturbation: change a few random residues
                    for _ in range(3):
                        j = np.random.randint(n_primes)
                        old_j = residues[j]
                        new_j = np.random.randint(primes[j])
                        if new_j != old_j:
                            apply_change(j, new_j)
                            new_L_check = find_L(current_L)
                            if new_L_check < current_L - 200:
                                revert_change(j, old_j)
                            else:
                                current_L = new_L_check

    # --- Phase 4: Final Polish - targeted local search near frontier ---
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        # Try to extend by checking each prime's residue more carefully
        for _ in range(100):
            if time.time() > polish_deadline:
                break
            progress = False
            for i in range(n_primes):
                if time.time() > polish_deadline:
                    break
                p = primes[i]
                old_r = residues[i]
                # Try nearby residues and target residues near frontier
                offsets = [-3, -2, -1, 1, 2, 3, -5, 5, -10, 10, -20, 20]
                for offset in offsets:
                    new_r = (old_r + offset) % p
                    if new_r == old_r:
                        continue
                    apply_change(i, new_r)
                    new_L = find_L(current_L)
                    if new_L > current_L:
                        current_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best_residues = residues.copy()
                        progress = True
                        break
                    else:
                        revert_change(i, old_r)
                
                # Also try residues that cover positions near current_L
                for target_offset in range(1, 30):
                    if time.time() > polish_deadline:
                        break
                    target = current_L + target_offset
                    if target > MAX_L:
                        break
                    target_r = target % p
                    if target_r == old_r:
                        continue
                    apply_change(i, target_r)
                    new_L = find_L(current_L)
                    if new_L > current_L:
                        current_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best_residues = residues.copy()
                        progress = True
                        break
                    else:
                        revert_change(i, old_r)
            
            if not progress:
                break

    # Final evaluation to verify
    final_counts = np.zeros(MAX_L + 1, dtype=np.int8)
    for i, p in enumerate(primes):
        r = best_residues[i]
        if r == 0:
            final_counts[p::p] += 1
        else:
            final_counts[r::p] += 1

    # Find final L
    if best_L > 0 and np.min(final_counts[1:best_L + 1]) < 2:
        bad = np.where(final_counts[1:best_L + 1] < 2)[0]
        final_L = bad[0]
    else:
        final_L = best_L
        if final_L < MAX_L:
            rest = final_counts[final_L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) > 0:
                final_L = final_L + bad_idx[0]

    return best_residues.tolist()
