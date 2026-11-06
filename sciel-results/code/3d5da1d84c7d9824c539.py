import math
import random
import time

def construct():
    deadline = time.time() + 140.0
    N = 511
    C = 512

    # Sieve for primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # Weights: w[a] = 1/(512 - a)
    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (C - a)

    # Precompute prime factors for each number
    pf = [[] for _ in range(N + 1)]
    for p in primes:
        for multiple in range(p, N + 1, p):
            pf[multiple].append(p)

    # Build conflict masks using bitsets over integers 1..511
    # conflict_mask[a] has bit b set iff gcd(a, b) > 1 (i.e., they share a prime factor)
    # Using Python int as a 512-bit bitset
    conflict_mask = [0] * (N + 1)
    for a in range(1, N + 1):
        mask = 0
        for b in range(1, N + 1):
            if a != b and pf[a] and pf[b]:
                # Check if they share any prime factor
                if any(p in pf[b] for p in pf[a]):
                    mask |= (1 << b)
        conflict_mask[a] = mask

    # Also track which primes divide which numbers for the matching perspective
    # prime_to_numbers[p] = set of numbers divisible by p
    prime_to_numbers = {p: [] for p in primes}
    for a in range(2, N + 1):
        for p in pf[a]:
            prime_to_numbers[p].append(a)

    # Greedy initialization: weight-to-conflict-ratio heuristic
    # For each candidate a, compute score = w[a] / (1 + number of high-weight conflicts)
    # This is a structural change: instead of just picking highest weight,
    # we penalize numbers that conflict with many other high-weight numbers
    S = set()
    S.add(1)  # 1 is coprime to everything and has weight 1/511

    # Compute conflict counts weighted by the other number's weight
    conflict_weight = [0.0] * (N + 1)
    for a in range(2, N + 1):
        cw = 0.0
        for b in range(1, N + 1):
            if a != b and (conflict_mask[a] >> b) & 1:
                cw += w[b]
        conflict_weight[a] = cw

    # Greedy: iteratively pick the number with highest w[a] / (1 + alpha * conflict_weight[a])
    alpha = 0.5
    remaining = list(range(2, N + 1))
    while remaining:
        # Find best candidate by ratio
        best_a = None
        best_ratio = -1
        for a in remaining:
            # Check if a is coprime to all in S
            if all(not ((conflict_mask[a] >> b) & 1) for b in S):
                ratio = w[a] / (1.0 + alpha * conflict_weight[a])
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_a = a
        if best_a is None:
            break
        S.add(best_a)
        remaining.remove(best_a)

    current_score = sum(w[a] for a in S)

    # Local search with simulated annealing
    rng = random.Random(123)
    T = 0.3
    T_min = 1e-8
    cooling = 0.9995

    all_nums = list(range(1, N + 1))

    def is_valid_add(a, S_set):
        """Check if a can be added to S_set (coprime to all elements)."""
        for b in S_set:
            if (conflict_mask[a] >> b) & 1:
                return False
        return True

    best_S = S.copy()
    best_score = current_score

    while time.time() < deadline - 5.0:
        if T < T_min:
            break

        move_type = rng.random()
        new_S = S.copy()
        delta = 0.0

        if move_type < 0.4:
            # Swap: Remove one, Add one
            if S:
                x = rng.choice(list(S))
                new_S.discard(x)
                # Find best candidate to add
                best_add = None
                best_add_w = -1
                for a in all_nums:
                    if a in new_S:
                        continue
                    if is_valid_add(a, new_S):
                        if w[a] > best_add_w:
                            best_add_w = w[a]
                            best_add = a
                if best_add is not None:
                    new_S.add(best_add)
                    delta = w[best_add] - w[x]
                else:
                    continue
        elif move_type < 0.6:
            # Merge: Remove two, Add one
            if len(S) >= 2:
                x, y = rng.sample(list(S), 2)
                new_S.discard(x)
                new_S.discard(y)
                best_add = None
                best_add_w = -1
                for a in all_nums:
                    if a in new_S:
                        continue
                    if is_valid_add(a, new_S):
                        if w[a] > best_add_w:
                            best_add_w = w[a]
                            best_add = a
                if best_add is not None:
                    new_S.add(best_add)
                    delta = w[best_add] - w[x] - w[y]
                else:
                    continue
        elif move_type < 0.8:
            # Split: Remove one, Add two
            if S:
                x = rng.choice(list(S))
                new_S.discard(x)
                # Add best single, then best single again
                best_add1 = None
                best_add1_w = -1
                for a in all_nums:
                    if a in new_S:
                        continue
                    if is_valid_add(a, new_S):
                        if w[a] > best_add1_w:
                            best_add1_w = w[a]
                            best_add1 = a

                if best_add1 is not None:
                    temp_S = new_S.copy()
                    temp_S.add(best_add1)
                    best_add2 = None
                    best_add2_w = -1
                    for a in all_nums:
                        if a in temp_S:
                            continue
                        if is_valid_add(a, temp_S):
                            if w[a] > best_add2_w:
                                best_add2_w = w[a]
                                best_add2 = a
                    if best_add2 is not None:
                        new_S.add(best_add1)
                        new_S.add(best_add2)
                        delta = w[best_add1] + w[best_add2] - w[x]
                    else:
                        new_S.add(best_add1)
                        delta = w[best_add1] - w[x]
                else:
                    continue
        else:
            # Random Perturbation
            if S:
                x = rng.choice(list(S))
                new_S.discard(x)
                candidates = [a for a in all_nums if a not in new_S and is_valid_add(a, new_S)]
                if candidates:
                    a = rng.choice(candidates)
                    new_S.add(a)
                    delta = w[a] - w[x]
                else:
                    continue

        # Acceptance
        if delta > 0 or rng.random() < math.exp(delta / T):
            S = new_S
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_S = S.copy()

        T *= cooling

    return list(best_S)
