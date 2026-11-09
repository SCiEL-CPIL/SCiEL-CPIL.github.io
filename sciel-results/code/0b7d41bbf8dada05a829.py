import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    prime_indices = {p: i for i, p in enumerate(primes)}

    # Precompute reciprocal costs as Fractions
    costs = {p: Fraction(1, p) for p in primes}

    def calculate_coverage(config, limit=5000):
        """
        Returns the largest L such that every integer 1..L is covered.
        config: list of [p, r]
        """
        covered = [False] * (limit + 1)
        for p, r in config:
            start = r if r >= 1 else p
            if start > limit:
                continue
            for n in range(start, limit + 1, p):
                covered[n] = True

        L = 0
        for i in range(1, limit + 1):
            if not covered[i]:
                L = i - 1
                break
        else:
            L = limit
        return L

    def calculate_coverage_numpy(config, limit=5000):
        """
        Vectorized-ish coverage calculation using numpy for speed.
        """
        covered = np.zeros(limit + 1, dtype=bool)
        for p, r in config:
            start = r if r >= 1 else p
            if start > limit:
                continue
            # Mark positions start, start+p, start+2p, ...
            # Using slicing
            covered[start:limit+1:p] = True

        # Find first False in covered[1:]
        # If all True, L = limit
        # Otherwise L = index of first False - 1
        if not covered[1:].any():
            return 0
        # Find first gap
        gaps = np.where(~covered[1:])[0]
        if len(gaps) == 0:
            return limit
        else:
            return gaps[0]

    def random_config():
        """
        Generate a random valid configuration.
        """
        shuffled = primes[:]
        random.shuffle(shuffled)
        current_sum = Fraction(0, 1)
        config = []
        for p in shuffled:
            cost = costs[p]
            if current_sum + cost <= 1:
                current_sum += cost
                config.append([p, random.randint(0, p - 1)])
        return config, current_sum

    # --- Main Search ---

    best_config = []
    best_score = 0

    # 1. Initialize with a few random configs
    for _ in range(10):
        config, _ = random_config()
        score = calculate_coverage_numpy(config, limit=3000)
        if score > best_score:
            best_score = score
            best_config = [c[:] for c in config]

    # 2. Local Search
    # We will maintain a current state and try moves.
    # Moves:
    # A. Change residue of a prime in config
    # B. Swap a prime in config with a prime not in config (budget check)
    # C. Add a prime (budget check)
    # D. Remove a prime

    current_config = [c[:] for c in best_config]
    current_sum = sum(costs[p] for p, r in current_config)
    current_score = best_score
    in_config = set(p for p, r in current_config)
    not_in_config = [p for p in primes if p not in in_config]

    # Pre-compute something? Maybe just use lists.
    # To speed up, we can keep not_in_config as a list and update it.

    # Helper to update in_config/not_in_config
    def add_prime_to_config(p):
        in_config.add(p)
        if p in not_in_config:
            not_in_config.remove(p)

    def remove_prime_from_config(p):
        in_config.remove(p)
        not_in_config.append(p)

    # Simulated Annealing / Hill Climbing loop
    # We use a simple hill climbing with random restarts if stuck?
    # Or just keep going.
    # Since time is limited, we just do hill climbing on the best found so far.
    # If we get stuck, we might try a random perturbation.

    # Let's implement a simple hill climbing with a "stuck" counter.
    stuck_count = 0
    max_stuck = 500

    while time.time() < deadline:
        move_type = random.randint(0, 3)

        if move_type == 0:
            # Change residue
            if not current_config:
                continue
            idx = random.randint(0, len(current_config) - 1)
            p, old_r = current_config[idx]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            current_config[idx][1] = new_r
            new_score = calculate_coverage_numpy(current_config, limit=3000)
            if new_score > current_score:
                current_score = new_score
                stuck_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_config = [c[:] for c in current_config]
            else:
                current_config[idx][1] = old_r
                stuck_count += 1

        elif move_type == 1:
            # Swap prime
            if not current_config or not not_in_config:
                continue
            idx = random.randint(0, len(current_config) - 1)
            old_p, old_r = current_config[idx]
            # Pick a random candidate from not_in_config
            # To avoid O(n) removal, we can pick a random index
            cand_idx = random.randint(0, len(not_in_config) - 1)
            new_p = not_in_config[cand_idx]

            old_cost = costs[old_p]
            new_cost = costs[new_p]
            if current_sum - old_cost + new_cost <= 1:
                # Perform swap
                current_config[idx] = [new_p, random.randint(0, new_p - 1)]
                # Update sets
                in_config.remove(old_p)
                not_in_config[cand_idx] = old_p  # Replace new_p with old_p in list
                in_config.add(new_p)
                current_sum = current_sum - old_cost + new_cost

                new_score = calculate_coverage_numpy(current_config, limit=3000)
                if new_score > current_score:
                    current_score = new_score
                    stuck_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_config = [c[:] for c in current_config]
                else:
                    # Revert
                    current_config[idx] = [old_p, old_r]
                    in_config.remove(new_p)
                    not_in_config[cand_idx] = new_p
                    in_config.add(old_p)
                    current_sum = current_sum - new_cost + old_cost
                    stuck_count += 1

        elif move_type == 2:
            # Add prime
            if not not_in_config:
                continue
            cand_idx = random.randint(0, len(not_in_config) - 1)
            new_p = not_in_config[cand_idx]
            new_cost = costs[new_p]
            if current_sum + new_cost <= 1:
                new_r = random.randint(0, new_p - 1)
                current_config.append([new_p, new_r])
                in_config.add(new_p)
                not_in_config.pop(cand_idx)
                current_sum += new_cost

                new_score = calculate_coverage_numpy(current_config, limit=3000)
                if new_score > current_score:
                    current_score = new_score
                    stuck_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_config = [c[:] for c in current_config]
                else:
                    # Revert
                    current_config.pop()
                    in_config.remove(new_p)
                    not_in_config.append(new_p)
                    current_sum -= new_cost
                    stuck_count += 1

        else:
            # Remove prime
            if len(current_config) <= 1:
                continue
            idx = random.randint(0, len(current_config) - 1)
            p, r = current_config[idx]
            cost = costs[p]
            current_config.pop(idx)
            in_config.remove(p)
            not_in_config.append(p)
            current_sum -= cost

            new_score = calculate_coverage_numpy(current_config, limit=3000)
            if new_score > current_score:
                current_score = new_score
                stuck_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_config = [c[:] for c in current_config]
            else:
                # Revert
                current_config.insert(idx, [p, r])
                in_config.add(p)
                not_in_config.pop(not_in_config.index(p)) # Remove the one we added
                current_sum += cost
                stuck_count += 1

        # If stuck for a while, try a random restart from best_config
        if stuck_count > max_stuck:
            current_config = [c[:] for c in best_config]
            current_sum = sum(costs[p] for p, r in current_config)
            current_score = best_score
            in_config = set(p for p, r in current_config)
            not_in_config = [p for p in primes if p not in in_config]
            stuck_count = 0
            # Maybe perturb best_config slightly?
            # For now, just restart search from best.

    return best_config
