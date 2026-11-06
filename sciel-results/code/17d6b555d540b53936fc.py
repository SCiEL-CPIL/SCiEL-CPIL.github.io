import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    def greedy_construct():
        """Greedy: include x if it doesn't create duplicate products."""
        values = []
        products = set()
        for x in range(2, 129):
            new_tails = [x]
            p = x
            for a in reversed(values):
                p *= a
                new_tails.append(p)
            if not products.intersection(new_tails):
                values.append(x)
                products.update(new_tails)
        return values

    initial_seq = greedy_construct()
    best_seq = initial_seq[:]
    best_len = len(initial_seq)

    # Precompute prime factorizations for all numbers 2-128
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    prime_factors = {x: factorize(x) for x in range(2, 129)}
    # Precompute product vectors (exponent vectors) for quick comparison
    def vec_add(v1, v2):
        result = dict(v1)
        for k, v in v2.items():
            result[k] = result.get(k, 0) + v
        return result

    def vec_equal(v1, v2):
        if len(v1) != len(v2):
            return False
        for k in v1:
            if v1.get(k, 0) != v2.get(k, 0):
                return False
        return True

    # Fast product set computation using exponent vectors
    def compute_product_vectors(seq):
        """Return set of exponent vectors for all consecutive block products."""
        n = len(seq)
        if n == 0:
            return set()
        prods = set()
        for i in range(n):
            v = {}
            for j in range(i, n):
                v = vec_add(v, prime_factors[seq[j]])
                prods.add(frozenset(v.items()))
        return prods

    def count_conflicts_fast(seq):
        """Count conflicts using exponent vectors."""
        if len(seq) < 2:
            return 0
        prods = compute_product_vectors(seq)
        # Count duplicates by rebuilding
        n = len(seq)
        all_prods = []
        for i in range(n):
            v = {}
            for j in range(i, n):
                v = vec_add(v, prime_factors[seq[j]])
                all_prods.append(frozenset(v.items()))
        c = Counter(all_prods)
        conflicts = 0
        for v in c.values():
            if v > 1:
                conflicts += v * (v - 1) // 2
        return conflicts

    def get_score(seq):
        """Score: length if valid, else length - penalty * conflicts."""
        if not seq:
            return 0
        conf = count_conflicts_fast(seq)
        if conf == 0:
            return len(seq)
        return len(seq) - 50 * conf

    # Fast incremental check using exponent vectors
    def check_insert_fast(seq, pos, x):
        """Check if inserting x at pos creates any new product conflicts using vectors."""
        n = len(seq)
        if n == 0:
            return True
        # Compute existing product vectors
        existing_prods = compute_product_vectors(seq)
        # New products involving x
        new_prods = set()
        # Left extensions
        v = dict(prime_factors[x])
        new_prods.add(frozenset(v.items()))
        for i in range(pos - 1, -1, -1):
            v = vec_add(v, prime_factors[seq[i]])
            new_prods.add(frozenset(v.items()))
        # Right extensions
        v = dict(prime_factors[x])
        for j in range(pos, n):
            v = vec_add(v, prime_factors[seq[j]])
            new_prods.add(frozenset(v.items()))
        # Check overlap
        if new_prods & existing_prods:
            return False
        return True

    # Simulated annealing with improved schedule
    current_seq = initial_seq[:]
    current_score = get_score(current_seq)
    current_conflicts = count_conflicts_fast(current_seq)

    T = 15.0
    cool_rate = 0.9995
    no_improve = 0
    step = 0

    while time.time() < deadline - 3:
        step += 1
        r = random.random()

        if r < 0.35:
            # Try to add an element
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if not available:
                continue
            x = random.choice(available)
            pos = 0
            for i, v in enumerate(current_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            if not check_insert_fast(current_seq, pos, x):
                continue
            new_seq = current_seq[:pos] + [x] + current_seq[pos:]
        elif r < 0.60:
            # Try to remove an element
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            new_seq = current_seq[:idx] + current_seq[idx+1:]
        elif r < 0.80:
            # Try a swap: remove one element, add another
            if len(current_seq) < 2:
                continue
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if not available:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            x = random.choice(available)
            temp = current_seq[:idx] + current_seq[idx+1:]
            pos = 0
            for i, v in enumerate(temp):
                if v < x:
                    pos = i + 1
                else:
                    break
            new_seq = temp[:pos] + [x] + temp[pos:]
        else:
            # Multi-move: remove 2, add 2 (larger perturbation)
            if len(current_seq) < 4:
                continue
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if len(available) < 2:
                continue
            idx1, idx2 = random.sample(range(len(current_seq)), 2)
            x1, x2 = random.sample(available, 2)
            temp = current_seq[:]
            for idx in sorted([idx1, idx2], reverse=True):
                temp = temp[:idx] + temp[idx+1:]
            for x in sorted([x1, x2]):
                pos = 0
                for i, v in enumerate(temp):
                    if v < x:
                        pos = i + 1
                    else:
                        break
                temp = temp[:pos] + [x] + temp[pos:]
            new_seq = temp

        new_score = get_score(new_seq)
        delta = new_score - current_score

        # Accept if better or with probability exp(-delta/T) for worse
        if delta >= 0 or (T > 0.01 and random.random() < np.exp(delta / T)):
            current_seq = new_seq
            current_score = new_score
            current_conflicts = count_conflicts_fast(current_seq)

            if current_conflicts == 0:
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    no_improve = 0
                    T = max(T, 8.0)
            else:
                no_improve += 1

        # Cooling
        T *= cool_rate
        if T < 0.1:
            T = 0.1

        # Restart if stuck
        if no_improve > 600:
            no_improve = 0
            T = max(T, 8.0)
            if current_seq:
                num_perturb = min(5, len(current_seq))
                for _ in range(num_perturb):
                    if random.random() < 0.40:
                        idx = random.randint(0, len(current_seq) - 1)
                        current_seq = current_seq[:idx] + current_seq[idx+1:]
                    else:
                        in_seq = set(current_seq)
                        available = [x for x in range(2, 129) if x not in in_seq]
                        if available:
                            x = random.choice(available)
                            pos = 0
                            for i, v in enumerate(current_seq):
                                if v < x:
                                    pos = i + 1
                                else:
                                    break
                            current_seq = current_seq[:pos] + [x] + current_seq[pos:]
                current_score = get_score(current_seq)
                current_conflicts = count_conflicts_fast(current_seq)

    # Final polish: aggressive multi-pass insertion with multiple orderings
    def prime_factor_count(x):
        return sum(prime_factors[x].values())

    def omega(x):
        """Number of distinct prime factors."""
        return len(prime_factors[x])

    # Pass 1: ascending order
    in_best = set(best_seq)
    changed = True
    while changed:
        changed = False
        for x in range(2, 129):
            if x not in in_best:
                pos = 0
                for i, v in enumerate(best_seq):
                    if v < x:
                        pos = i + 1
                    else:
                        break
                if check_insert_fast(best_seq, pos, x):
                    best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                    in_best.add(x)
                    changed = True
                    best_len = len(best_seq)
                    break

    # Pass 2: descending order
    in_best = set(best_seq)
    changed = True
    while changed:
        changed = False
        for x in range(128, 1, -1):
            if x not in in_best:
                pos = 0
                for i, v in enumerate(best_seq):
                    if v < x:
                        pos = i + 1
                    else:
                        break
                if check_insert_fast(best_seq, pos, x):
                    best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                    in_best.add(x)
                    changed = True
                    best_len = len(best_seq)
                    break

    # Pass 3: sorted by fewest prime factors
    in_best = set(best_seq)
    changed = True
    while changed:
        changed = False
        candidates = [x for x in range(2, 129) if x not in in_best]
        candidates.sort(key=lambda x: (prime_factor_count(x), x))
        for x in candidates:
            pos = 0
            for i, v in enumerate(best_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            if check_insert_fast(best_seq, pos, x):
                best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                in_best.add(x)
                changed = True
                best_len = len(best_seq)
                break

    # Pass 4: sorted by fewest distinct prime factors
    in_best = set(best_seq)
    changed = True
    while changed:
        changed = False
        candidates = [x for x in range(2, 129) if x not in in_best]
        candidates.sort(key=lambda x: (omega(x), x))
        for x in candidates:
            pos = 0
            for i, v in enumerate(best_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            if check_insert_fast(best_seq, pos, x):
                best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                in_best.add(x)
                changed = True
                best_len = len(best_seq)
                break

    # Pass 5: random order (multiple shuffles)
    in_best = set(best_seq)
    changed = True
    for _ in range(5):
        if not changed:
            break
        changed = False
        candidates = [x for x in range(2, 129) if x not in in_best]
        random.shuffle(candidates)
        for x in candidates:
            pos = 0
            for i, v in enumerate(best_seq):
                if v < x:
                    pos = i + 1
                else:
                    break
            if check_insert_fast(best_seq, pos, x):
                best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                in_best.add(x)
                changed = True
                best_len = len(best_seq)
                break

    # Pass 6: remove-and-readd strategy
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        in_best = set(best_seq)
        removed_and_readded = False
        for idx in range(len(best_seq)):
            if time.time() > deadline - 5:
                break
            removed_elem = best_seq[idx]
            temp_seq = best_seq[:idx] + best_seq[idx+1:]
            in_temp = set(temp_seq)
            candidates = [x for x in range(2, 129) if x not in in_temp]
            pos = 0
            for i, v in enumerate(temp_seq):
                if v < removed_elem:
                    pos = i + 1
                else:
                    break
            if check_insert_fast(temp_seq, pos, removed_elem):
                temp_seq2 = temp_seq[:pos] + [removed_elem] + temp_seq[pos:]
                in_temp2 = set(temp_seq2)
                for x in candidates:
                    if x != removed_elem:
                        pos2 = 0
                        for i, v in enumerate(temp_seq2):
                            if v < x:
                                pos2 = i + 1
                            else:
                                break
                        if check_insert_fast(temp_seq2, pos2, x):
                            new_seq = temp_seq2[:pos2] + [x] + temp_seq2[pos2:]
                            if count_conflicts_fast(new_seq) == 0 and len(new_seq) > len(best_seq):
                                best_seq = new_seq
                                best_len = len(best_seq)
                                removed_and_readded = True
                                break
                if removed_and_readded:
                    break
        if not removed_and_readded:
            break

    return best_seq
