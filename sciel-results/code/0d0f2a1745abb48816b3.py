"""Improved program for sumset/difference-set exponent optimization."""
import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    best_score = 0.0
    best_set = [0, 1, 3]

    def compute_score(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)

        if n < 15:
            sumset = set()
            diffset = set()
            for i in range(n):
                a = U[i]
                for j in range(n):
                    b = U[j]
                    sumset.add(a + b)
                    diffset.add(a - b)
            n_sum = len(sumset)
            n_diff = len(diffset)
        else:
            # FFT-based for larger sets
            max_sum = 2 * max_u
            len_sum = 2 * max_u + 1
            fft_size = 1
            while fft_size < 2 * len_sum:
                fft_size *= 2

            f_sum = np.zeros(fft_size, dtype=np.float64)
            for u in U:
                f_sum[u] = 1.0
            F_sum = np.fft.fft(f_sum)
            conv_sum = np.fft.ifft(F_sum * F_sum).real
            n_sum = int(np.sum(conv_sum > 0.5))

            len_diff = 2 * max_u + 1
            fft_size_d = 1
            while fft_size_d < 2 * len_diff:
                fft_size_d *= 2

            f_diff = np.zeros(fft_size_d, dtype=np.float64)
            for u in U:
                f_diff[u] = 1.0
            f_rev = np.zeros(fft_size_d, dtype=np.float64)
            for u in U:
                f_rev[max_u - u] = 1.0

            F_diff = np.fft.fft(f_diff)
            F_rev = np.fft.fft(f_rev)
            conv_diff = np.fft.ifft(F_diff * F_rev).real
            n_diff = int(np.sum(conv_diff > 0.5))

        if n_sum == 0:
            return -1

        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    # Phase 1: Known Golomb rulers
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 25, 34],
        [0, 1, 4, 10, 18, 23, 25, 34, 42],
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52],
    ]

    for gr in golomb_rulers:
        s = compute_score(gr)
        if s > best_score:
            best_score = s
            best_set = list(gr)

    # Scaled variants
    for gr in golomb_rulers[:6]:
        for scale in [2, 3, 4, 5, 7]:
            scaled = [x * scale for x in gr]
            s = compute_score(scaled)
            if s > best_score:
                best_score = s
                best_set = list(scaled)

    # Phase 2: Spectral method for cyclic groups
    # For a set U in Z_N, |U+U| and |U-U| can be computed via DFT
    # The key insight: we want large |U-U| / |U+U| ratio
    # Difference set is typically larger than sumset for well-spread sets

    def cyclic_score(U, N):
        """Compute score for set U interpreted in cyclic group Z_N."""
        if not U:
            return -1
        n = len(U)
        if n < 2:
            return -1

        # Build indicator vector
        f = np.zeros(N, dtype=np.float64)
        for u in U:
            f[u % N] = 1.0

        # Sumset: |U+U| via convolution f * f
        F = np.fft.fft(f)
        conv_sum = np.fft.ifft(F * F).real
        n_sum = int(np.sum(conv_sum > 0.5))

        # Difference set: |U-U| via f * f_rev
        f_rev = np.zeros(N, dtype=np.float64)
        for u in U:
            f_rev[(-u) % N] = 1.0
        F_rev = np.fft.fft(f_rev)
        conv_diff = np.fft.ifft(F * F_rev).real
        n_diff = int(np.sum(conv_diff > 0.5))

        if n_sum == 0:
            return -1

        base = N  # effective "2*max_u+1" in cyclic setting
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    # Try cyclic constructions with various primes
    if time.time() < deadline - 20:
        primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                  151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

        for p in primes:
            if time.time() > deadline - 10:
                break

            # Generate random subsets of Z_p
            for trial in range(30):
                if time.time() > deadline - 10:
                    break
                random.seed(p * 1000 + trial)
                size = random.randint(5, min(30, p // 3))
                U_cyclic = random.sample(range(p), size)
                # Ensure 0 is in the set
                if 0 not in U_cyclic:
                    U_cyclic[0] = 0

                # Convert to linear set (take residues, sort)
                U_linear = sorted(U_cyclic)
                s = compute_score(U_linear)
                if s > best_score:
                    best_score = s
                    best_set = list(U_linear)

    # Phase 3: Structured constructions - quadratic residues, multiplicative subgroups
    if time.time() < deadline - 15:
        # Multiplicative subgroup construction
        for p in [101, 103, 107, 109, 113, 127, 131, 137]:
            if time.time() > deadline - 10:
                break
            # Find multiplicative order
            for g in range(2, p):
                if g ** (p - 1) == 1:
                    break

            # Generate subgroups of various sizes
            orders = [(p - 1) // k for k in range(2, 10) if (p - 1) % k == 0]
            orders = list(set(orders))

            for order in orders:
                if order < 4 or order > p // 2:
                    continue
                # Subgroup generated by g^((p-1)/order)
                gen = pow(g, (p - 1) // order, p)
                subgroup = [pow(gen, i, p) for i in range(order)]
                if 0 not in subgroup:
                    subgroup.append(0)
                U_lin = sorted(subgroup)
                s = compute_score(U_lin)
                if s > best_score:
                    best_score = s
                    best_set = list(U_lin)

        # Arithmetic progressions with gaps
        for base in range(2, 20):
            if time.time() > deadline - 10:
                break
            for start in range(1, 10):
                for length in range(5, 25):
                    if time.time() > deadline - 10:
                        break
                    U = [start + i * base for i in range(length)]
                    if 0 not in U:
                        U.append(0)
                    U = sorted(U)
                    s = compute_score(U)
                    if s > best_score:
                        best_score = s
                        best_set = list(U)

    # Phase 4: Local search with FFT acceleration
    if time.time() < deadline - 10:
        current = list(best_set)
        current_score = best_score
        no_improve = 0
        max_no_improve = 500

        while no_improve < max_no_improve and time.time() < deadline - 5:
            improved = False

            # Try adding elements
            if len(current) < 50:
                max_cur = max(current)
                candidates = set()
                for el in current:
                    for delta in range(-5, 6):
                        new_el = el + delta
                        if new_el > 0 and new_el not in current:
                            candidates.add(new_el)
                for new_el in range(max_cur + 1, max_cur + 30):
                    candidates.add(new_el)
                for mult in [2, 3]:
                    if max_cur * mult < 5000:
                        candidates.add(max_cur * mult)

                for new_el in sorted(candidates):
                    cand = sorted(current + [new_el])
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        improved = True
                        break

            # Try replacing elements
            if len(current) >= 3 and not improved:
                max_cur = max(current)
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    old_el = current[i]
                    for new_el in range(max(1, old_el - 8), min(max_cur + 15, old_el + 9)):
                        if new_el == old_el or new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand or len(cand) != len(current):
                            continue
                        s = compute_score(cand)
                        if s > current_score + 1e-15:
                            current = list(cand)
                            current_score = s
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                            improved = True
                            break
                    if improved:
                        break

            # Try removing elements
            if len(current) > 5 and not improved:
                for i in range(len(current) - 1, 0, -1):
                    cand = current[:i] + current[i + 1:]
                    s = compute_score(cand)
                    if s > current_score + 1e-15:
                        current = list(cand)
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                        improved = True
                        break

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

    # Phase 5: Random restarts with diverse initializations
    if time.time() < deadline - 5:
        random.seed(999)
        for restart in range(80):
            if time.time() > deadline - 5:
                break

            rtype = random.randint(0, 4)

            if rtype == 0:
                # Geometric sequence
                size = random.randint(5, 15)
                ratio = random.uniform(1.5, 2.5)
                U = [0]
                val = 1
                for _ in range(size - 1):
                    U.append(int(val))
                    val *= ratio
            elif rtype == 1:
                # Golomb-ruler-like greedy
                size = random.randint(4, 12)
                max_u = random.randint(20, 200)
                U = [0, 1]
                used_diffs = {1}
                for _ in range(size - 2):
                    best_next = -1
                    for candidate in range(2, max_u + 1):
                        valid = True
                        for existing in U:
                            d = candidate - existing
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            best_next = candidate
                            break
                    if best_next == -1:
                        break
                    U.append(best_next)
                    for existing in U[:-1]:
                        used_diffs.add(U[-1] - existing)
            elif rtype == 2:
                # Random sparse
                size = random.randint(6, 20)
                max_u = random.randint(30, 300)
                U = [0] + sorted(random.sample(range(1, max_u + 1), min(size - 1, max_u)))
            elif rtype == 3:
                # Powers
                size = random.randint(4, 10)
                base = random.choice([2, 3, 4, 5])
                U = [base ** i for i in range(size)]
            else:
                # Mixed: start from best, perturb
                U = list(best_set)
                if len(U) > 3:
                    # Remove a random element
                    idx = random.randint(1, len(U) - 1)
                    U = [x for i, x in enumerate(U) if i != idx]
                    # Add a new one
                    max_cur = max(U)
                    new_el = random.randint(max_cur - 5, max_cur + 20)
                    if new_el > 0 and new_el not in U:
                        U.append(new_el)
                U = sorted(U)

            if len(U) >= 3 and 0 in U:
                U = sorted(set(U))
                s = compute_score(U)
                if s > best_score:
                    best_score = s
                    best_set = list(U)
                elif s > best_score - 0.02:
                    # Local polish
                    cur = list(U)
                    cur_s = s
                    for _ in range(30):
                        if time.time() > deadline - 5:
                            break
                        max_cur = max(cur)
                        cands = set()
                        for el in cur:
                            for d in range(-3, 4):
                                ne = el + d
                                if ne > 0 and ne not in cur:
                                    cands.add(ne)
                        for ne in range(max_cur + 1, max_cur + 10):
                            cands.add(ne)

                        imp = False
                        for ne in sorted(cands):
                            cand2 = sorted(cur + [ne])
                            s2 = compute_score(cand2)
                            if s2 > cur_s + 1e-15:
                                cur = list(cand2)
                                cur_s = s2
                                if s2 > best_score:
                                    best_score = s2
                                    best_set = list(cand2)
                                imp = True
                                break
                        if not imp:
                            break

    return list(best_set)
