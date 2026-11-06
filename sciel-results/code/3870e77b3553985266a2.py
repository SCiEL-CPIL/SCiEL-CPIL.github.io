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

    # Cache for product sets to avoid recomputation
    _prod_cache = {}

    def compute_all_products(seq):
        """Return list of all consecutive block products."""
        key = tuple(seq)
        if key in _prod_cache:
            return _prod_cache[key]
        n = len(seq)
        if n == 0:
            return []
        prods = []
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        _prod_cache[key] = prods
        return prods

    def get_product_set(seq):
        """Return set of all consecutive block products."""
        key = ('set', tuple(seq))
        if key in _prod_cache:
            return _prod_cache[key]
        prods = compute_all_products(seq)
        prod_set = set(prods)
        _prod_cache[key] = prod_set
        return prod_set

    def count_conflicts(seq):
        """Count number of duplicate product pairs."""
        if len(seq) < 2:
            return 0
        prods = compute_all_products(seq)
        c = Counter(prods)
        conflicts = 0
        for v in c.values():
            if v > 1:
                conflicts += v * (v - 1) // 2
        return conflicts

    def get_score(seq):
        """Score: length if valid, else length - penalty * conflicts."""
        if not seq:
            return 0
        conf = count_conflicts(seq)
        if conf == 0:
            return len(seq)
        return len(seq) - 50 * conf

    # --- Fast incremental conflict check for a single insertion/removal ---
    def check_insert(seq, pos, x):
        """Check if inserting x at pos creates any new product conflicts.
        Returns True if valid (no new conflicts)."""
        n = len(seq)
        if n == 0:
            return True
        # Use cached product set
        existing_prods = get_product_set(seq)
        # New products involving x
        new_prods = set()
        # Left extensions: x * seq[pos-1] * ... * seq[i]
        p = x
        new_prods.add(p)
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            new_prods.add(p)
        # Right extensions: x * seq[pos] * ... * seq[j]
        p = x
        for j in range(pos, n):
            p *= seq[j]
            new_prods.add(p)
        # Check overlap
        if new_prods & existing_prods:
            return False
        return True

    def check_remove(seq, idx):
        """Check if removing seq[idx] keeps all remaining products distinct.
        Returns True if valid."""
        new_seq = seq[:idx] + seq[idx+1:]
        return count_conflicts(new_seq) == 0

    # Simulated annealing
    current_seq = initial_seq[:]
    current_score = get_score(current_seq)
    current_conflicts = count_conflicts(current_seq)

    T = 10.0
    cool_rate = 0.9998
    no_improve = 0
    step = 0

    while time.time() < deadline - 3:
        step += 1
        r = random.random()

        if r < 0.40:
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
            # Fast check: only verify new products don't conflict
            if not check_insert(current_seq, pos, x):
                continue
            new_seq = current_seq[:pos] + [x] + current_seq[pos:]
        elif r < 0.65:
            # Try to remove an element
            if not current_seq:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            new_seq = current_seq[:idx] + current_seq[idx+1:]
        elif r < 0.85:
            # Try a swap: remove one element, add another
            if len(current_seq) < 2:
                continue
            in_seq = set(current_seq)
            available = [x for x in range(2, 129) if x not in in_seq]
            if not available:
                continue
            idx = random.randint(0, len(current_seq) - 1)
            x = random.choice(available)
            # Remove idx, then insert x in sorted position
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
            # Remove both
            temp = current_seq[:]
            for idx in sorted([idx1, idx2], reverse=True):
                temp = temp[:idx] + temp[idx+1:]
            # Insert both in sorted order
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
            current_conflicts = count_conflicts(current_seq)

            if current_conflicts == 0:
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    no_improve = 0
                    T = max(T, 5.0)
            else:
                no_improve += 1

        # Cooling
        T *= cool_rate
        if T < 0.1:
            T = 0.1

        # Restart if stuck
        if no_improve > 800:
            no_improve = 0
            T = max(T, 6.0)
            # Perturb current state more aggressively
            if current_seq:
                num_perturb = min(4, len(current_seq))
                for _ in range(num_perturb):
                    if random.random() < 0.35:
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
                current_conflicts = count_conflicts(current_seq)

    # Final polish: aggressive multi-pass insertion
    def prime_factor_count(x):
        count = 0
        d = 2
        n = x
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

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
                if check_insert(best_seq, pos, x):
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
                if check_insert(best_seq, pos, x):
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
            if check_insert(best_seq, pos, x):
                best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                in_best.add(x)
                changed = True
                best_len = len(best_seq)
                break

    # Pass 4: random order
    in_best = set(best_seq)
    changed = True
    while changed:
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
            if check_insert(best_seq, pos, x):
                best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                in_best.add(x)
                changed = True
                best_len = len(best_seq)
                break

    # Pass 5: remove-and-readd strategy - try removing each element and see if we can add 2 back
    for _ in range(3):
        in_best = set(best_seq)
        removed_and_readded = False
        for idx in range(len(best_seq)):
            if time.time() > deadline - 5:
                break
            removed_elem = best_seq[idx]
            temp_seq = best_seq[:idx] + best_seq[idx+1:]
            # Try to add back removed_elem and one more
            in_temp = set(temp_seq)
            candidates = [x for x in range(2, 129) if x not in in_temp]
            # Try adding removed_elem first, then another
            pos = 0
            for i, v in enumerate(temp_seq):
                if v < removed_elem:
                    pos = i + 1
                else:
                    break
            if check_insert(temp_seq, pos, removed_elem):
                temp_seq2 = temp_seq[:pos] + [removed_elem] + temp_seq[pos:]
                in_temp2 = set(temp_seq2)
                # Now try to add another element
                for x in candidates:
                    if x != removed_elem:
                        pos2 = 0
                        for i, v in enumerate(temp_seq2):
                            if v < x:
                                pos2 = i + 1
                            else:
                                break
                        if check_insert(temp_seq2, pos2, x):
                            new_seq = temp_seq2[:pos2] + [x] + temp_seq2[pos2:]
                            if count_conflicts(new_seq) == 0 and len(new_seq) > len(best_seq):
                                best_seq = new_seq
                                best_len = len(best_seq)
                                removed_and_readded = True
                                break
                if removed_and_readded:
                    break
        if not removed_and_readded:
            break

    return best_seq
