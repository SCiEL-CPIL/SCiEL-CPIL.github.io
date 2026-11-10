import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    # Precompute all triples
    all_triples = []
    triple_masks = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                all_triples.append((a, b, c))
                triple_masks.append((1 << a) | (1 << b) | (1 << c))

    n_triples = len(all_triples)  # 560

    # Precompute compatibility as bitsets
    compat = [0] * n_triples
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            inter = mi & mj
            cnt = inter.bit_count()
            if cnt <= 1:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)

    # Precompute popcounts
    popcounts_global = np.zeros(1 << 16, dtype=np.uint8)
    for m in range(1, 1 << 16):
        popcounts_global[m] = popcounts_global[m >> 1] + (m & 1)

    def compute_alpha_fast(selected_indices):
        """Compute independence number using precomputed popcounts."""
        if not selected_indices:
            return 16

        triple_set = set()
        for idx in selected_indices:
            triple_set.add(triple_masks[idx])

        bad = np.zeros(1 << 16, dtype=np.uint8)
        for t in triple_set:
            remaining = ((1 << 16) - 1) ^ t
            sub = remaining
            while True:
                bad[t | sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & remaining

        valid = bad == 0
        if not valid.any():
            return 0
        return int(popcounts_global[valid].max())

    # Greedy construction with randomization
    def greedy_construct(seed):
        rng = random.Random(seed)
        available = (1 << n_triples) - 1
        selected = set()

        while available:
            bits = bin(available).count('1')
            if bits == 0:
                break
            idx_in_bits = rng.randrange(bits)
            tmp = available
            for _ in range(idx_in_bits):
                tmp &= tmp - 1
            bit = tmp & (-tmp)
            i = bit.bit_length() - 1

            selected.add(i)
            available &= compat[i]
            available &= ~(1 << i)

        return selected

    # Local search with multiple move types
    def local_search(system, max_time, rng):
        current = set(system)
        current_alpha = compute_alpha_fast(current)
        best = set(current)
        best_alpha = current_alpha

        T = 1.5
        T_min = 0.05
        cooling = 0.999

        while time.time() < max_time:
            if T < T_min:
                T = T_min

            move_type = rng.random()

            if move_type < 0.8 and len(current) > 0:
                # Swap: remove i, add j
                i = rng.choice(list(current))
                S = current - {i}

                candidates = (1 << n_triples) - 1
                for k in current:
                    candidates &= ~(1 << k)
                for k in S:
                    candidates &= compat[k]

                if candidates == 0:
                    continue

                bits = bin(candidates).count('1')
                idx_in_bits = rng.randrange(bits)
                tmp = candidates
                for _ in range(idx_in_bits):
                    tmp &= tmp - 1
                bit = tmp & (-tmp)
                j = bit.bit_length() - 1

                new_system = S | {j}
                new_alpha = compute_alpha_fast(new_system)
            else:
                if rng.random() < 0.5:
                    # Add
                    candidates = (1 << n_triples) - 1
                    for k in current:
                        candidates &= ~(1 << k)
                        candidates &= compat[k]

                    if candidates == 0:
                        continue

                    bits = bin(candidates).count('1')
                    idx_in_bits = rng.randrange(bits)
                    tmp = candidates
                    for _ in range(idx_in_bits):
                        tmp &= tmp - 1
                    bit = tmp & (-tmp)
                    j = bit.bit_length() - 1

                    new_system = current | {j}
                    new_alpha = compute_alpha_fast(new_system)
                else:
                    # Remove
                    if not current:
                        continue
                    i = rng.choice(list(current))
                    new_system = current - {i}
                    new_alpha = compute_alpha_fast(new_system)

            delta = new_alpha - current_alpha

            if delta <= 0:
                if delta < 0 or rng.random() < 0.5:
                    current = new_system
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best = set(current)
            else:
                if rng.random() < (2.71828 ** (-delta / T)):
                    current = new_system
                    current_alpha = new_alpha

            T *= cooling

        return best, best_alpha

    # Multi-restart strategy
    rng = random.Random(42)
    best_system = None
    best_alpha = 16

    # Phase 1: Generate diverse initial solutions
    num_initial = 12
    for i in range(num_initial):
        if time.time() > deadline - 25:
            break

        seed = i * 1000 + 42
        initial = greedy_construct(seed)
        alpha = compute_alpha_fast(initial)

        if best_system is None or alpha < best_alpha:
            best_system = initial
            best_alpha = alpha

    # Phase 2: Local search on best and nearby solutions
    remaining_time = deadline - time.time()
    num_ls_runs = min(8, max(1, int(remaining_time / 12)))

    for i in range(num_ls_runs):
        if time.time() > deadline - 3:
            break

        if i == 0 or rng.random() < 0.6:
            start_system = best_system
        else:
            start_system = greedy_construct(rng.randint(0, 10000))

        max_time = deadline - 3 - time.time()
        if max_time < 2:
            break

        ls_system, ls_alpha = local_search(start_system, time.time() + max_time, rng)

        if ls_alpha < best_alpha:
            best_alpha = ls_alpha
            best_system = ls_system

    return [list(all_triples[i]) for i in sorted(best_system)]
