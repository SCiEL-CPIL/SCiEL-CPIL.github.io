import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Precompute: for each x in [2,128], store it
    ALL_VALS = list(range(2, 129))
    N_TOTAL = len(ALL_VALS)

    def greedy_construct():
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

    # Product cache keyed by frozenset of included values
    _prod_cache = {}

    def get_all_products(seq):
        key = tuple(seq)
        if key in _prod_cache:
            return _prod_cache[key]
        n = len(seq)
        prods = []
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.append(p)
        _prod_cache[key] = prods
        return prods

    def get_product_set(seq):
        key = ('set', tuple(seq))
        if key in _prod_cache:
            return _prod_cache[key]
        prods = get_all_products(seq)
        ps = set(prods)
        _prod_cache[key] = ps
        return ps

    def count_conflicts(seq):
        if len(seq) < 2:
            return 0
        prods = get_all_products(seq)
        c = Counter(prods)
        conflicts = 0
        for v in c.values():
            if v > 1:
                conflicts += v * (v - 1) // 2
        return conflicts

    def smooth_score(seq):
        """Smooth surrogate: length minus soft penalty for conflicts.
        Uses log-sum-exp style soft count of duplicates."""
        if not seq:
            return 0.0
        prods = get_all_products(seq)
        c = Counter(prods)
        n_prods = len(prods)
        n_unique = len(c)
        # Number of "excess" products
        excess = n_prods - n_unique
        # Soft penalty: for each product appearing k times, penalty ~ k-1
        # But smoother: use sum of (k-1)^2 for stronger gradient
        penalty = 0.0
        for v in c.values():
            if v > 1:
                penalty += (v - 1) ** 2
        return len(seq) - 10.0 * penalty

    def exact_score(seq):
        """Exact: length if valid, negative penalty if not."""
        if not seq:
            return 0
        conf = count_conflicts(seq)
        if conf == 0:
            return len(seq)
        return len(seq) - 100 * conf

    def check_insert(seq, pos, x):
        """Check if inserting x at pos creates any new product conflicts."""
        n = len(seq)
        if n == 0:
            return True
        existing_prods = get_product_set(seq)
        new_prods = set()
        p = x
        new_prods.add(p)
        for i in range(pos - 1, -1, -1):
            p *= seq[i]
            new_prods.add(p)
        p = x
        for j in range(pos, n):
            p *= seq[j]
            new_prods.add(p)
        if new_prods & existing_prods:
            return False
        return True

    def find_pos(seq, x):
        """Find insertion position for x in sorted seq."""
        pos = 0
        for i, v in enumerate(seq):
            if v < x:
                pos = i + 1
            else:
                break
        return pos

    # --- Skip-set representation ---
    # State: set of skipped values. Included = ALL_VALS - skipped.
    def seq_from_skipped(skipped_set):
        return [x for x in ALL_VALS if x not in skipped_set]

    def skipped_from_seq(seq):
        s = set(seq)
        return set(ALL_VALS) - s

    # Initial skip set from greedy
    initial_skipped = skipped_from_seq(initial_seq)

    # --- Hill climbing on skip set with restarts ---
    best_skipped = initial_skipped.copy()
    best_exact = exact_score(initial_seq)
    best_smooth = smooth_score(initial_seq)

    def hill_climb_skipped(start_skipped, time_limit=2.0):
        """Hill climb on the skip set. Moves: add/remove a single element from skipped."""
        t_start = time.time()
        cur_skipped = start_skipped.copy()
        cur_seq = seq_from_skipped(cur_skipped)
        cur_exact = exact_score(cur_seq)
        cur_smooth = smooth_score(cur_seq)
        local_best_skipped = cur_skipped.copy()
        local_best_exact = cur_exact
        local_best_smooth = cur_smooth

        no_improve = 0
        max_no_improve = 200

        while time.time() < t_start + time_limit and time.time() < deadline - 3:
            # Choose a move
            r = random.random()
            if r < 0.6:
                # Remove an element from skipped (i.e., include it)
                if not cur_skipped:
                    continue
                x = random.choice(list(cur_skipped))
                new_skipped = cur_skipped - {x}
                new_seq = seq_from_skipped(new_skipped)
                # Fast check: does adding x create conflicts?
                pos = find_pos(cur_seq, x)
                if not check_insert(cur_seq, pos, x):
                    no_improve += 1
                    if no_improve > max_no_improve:
                        break
                    continue
                new_exact = exact_score(new_seq)
                new_smooth = smooth_score(new_seq)
            else:
                # Add an element to skipped (i.e., exclude it)
                in_seq = set(cur_seq)
                available = [x for x in ALL_VALS if x not in in_seq]
                if not available:
                    continue
                x = random.choice(available)
                new_skipped = cur_skipped | {x}
                new_seq = seq_from_skipped(new_skipped)
                new_exact = exact_score(new_seq)
                new_smooth = smooth_score(new_seq)

            # Acceptance: use smooth score for guidance
            delta_smooth = new_smooth - cur_smooth
            delta_exact = new_exact - cur_exact

            # Accept if exact improves, or smooth improves with some noise
            accept = False
            if delta_exact > 0:
                accept = True
            elif delta_exact == 0 and delta_smooth > 0:
                accept = True
            elif delta_smooth > 0 and random.random() < 0.3:
                accept = True
            elif delta_smooth < 0 and random.random() < 0.05:
                accept = True  # small chance of accepting worse for exploration

            if accept:
                cur_skipped = new_skipped
                cur_seq = new_seq
                cur_exact = new_exact
                cur_smooth = new_smooth
                no_improve = 0

                if cur_exact > local_best_exact:
                    local_best_exact = cur_exact
                    local_best_smooth = cur_smooth
                    local_best_skipped = cur_skipped.copy()
                elif cur_exact == local_best_exact and cur_smooth > local_best_smooth:
                    local_best_smooth = cur_smooth
                    local_best_skipped = cur_skipped.copy()
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

        return local_best_skipped, local_best_exact, local_best_smooth

    # Multiple restarts of hill climbing
    num_restarts = 0
    while time.time() < deadline - 8:
        # Perturb the best known skip set
        if num_restarts == 0:
            start_skipped = best_skipped.copy()
        else:
            start_skipped = best_skipped.copy()
            # Random perturbation: flip 1-4 elements
            n_perturb = random.randint(1, 4)
            all_vals_set = set(ALL_VALS)
            for _ in range(n_perturb):
                if random.random() < 0.5 and start_skipped:
                    # Remove a random element from skipped
                    x = random.choice(list(start_skipped))
                    start_skipped.discard(x)
                else:
                    # Add a random element to skipped
                    in_seq = set(seq_from_skipped(start_skipped))
                    available = [x for x in ALL_VALS if x not in in_seq]
                    if available:
                        x = random.choice(available)
                        start_skipped.add(x)

        time_limit = min(2.5, max(0.5, deadline - 8 - time.time()))
        if time_limit < 0.3:
            break

        new_skipped, new_exact, new_smooth = hill_climb_skipped(start_skipped, time_limit)

        if new_exact > best_exact:
            best_exact = new_exact
            best_smooth = new_smooth
            best_skipped = new_skipped
            best_seq = seq_from_skipped(best_skipped)
            best_len = len(best_seq)
            num_restarts = 0
        else:
            num_restarts += 1

        # Also try a fresh random restart occasionally
        if num_restarts % 5 == 0 and time.time() < deadline - 10:
            # Random initial skip set
            random_skipped = set()
            for x in ALL_VALS:
                if random.random() < 0.3:
                    random_skipped.add(x)
            random_seq = seq_from_skipped(random_skipped)
            random_exact = exact_score(random_seq)
            if random_exact > best_exact - 5:  # Not too far behind
                time_limit = min(1.5, max(0.3, deadline - 8 - time.time()))
                if time_limit > 0.2:
                    new_skipped, new_exact, new_smooth = hill_climb_skipped(random_skipped, time_limit)
                    if new_exact > best_exact:
                        best_exact = new_exact
                        best_smooth = new_smooth
                        best_skipped = new_skipped
                        best_seq = seq_from_skipped(best_skipped)
                        best_len = len(best_seq)

    # --- Simulated annealing on the skip set for fine-tuning ---
    if time.time() < deadline - 5:
        cur_skipped = best_skipped.copy()
        cur_seq = seq_from_skipped(cur_skipped)
        cur_exact = exact_score(cur_seq)
        cur_smooth = smooth_score(cur_seq)

        T = 3.0
        cool_rate = 0.999
        no_improve = 0

        while time.time() < deadline - 4:
            r = random.random()
            if r < 0.55:
                # Remove from skipped (include)
                if not cur_skipped:
                    continue
                x = random.choice(list(cur_skipped))
                new_skipped = cur_skipped - {x}
                new_seq = seq_from_skipped(new_skipped)
                pos = find_pos(cur_seq, x)
                if not check_insert(cur_seq, pos, x):
                    continue
                new_exact = exact_score(new_seq)
                new_smooth = smooth_score(new_seq)
            else:
                # Add to skipped (exclude)
                in_seq = set(cur_seq)
                available = [x for x in ALL_VALS if x not in in_seq]
                if not available:
                    continue
                x = random.choice(available)
                new_skipped = cur_skipped | {x}
                new_seq = seq_from_skipped(new_skipped)
                new_exact = exact_score(new_seq)
                new_smooth = smooth_score(new_seq)

            delta_smooth = new_smooth - cur_smooth
            delta_exact = new_exact - cur_exact

            accept = False
            if delta_exact > 0:
                accept = True
            elif delta_exact == 0 and delta_smooth >= 0:
                accept = True
            elif T > 0.05 and random.random() < np.exp(delta_smooth / T):
                accept = True

            if accept:
                cur_skipped = new_skipped
                cur_seq = new_seq
                cur_exact = new_exact
                cur_smooth = new_smooth
                no_improve = 0

                if cur_exact > best_exact:
                    best_exact = cur_exact
                    best_seq = cur_seq
                    best_len = len(best_seq)
                    best_skipped = cur_skipped.copy()
            else:
                no_improve += 1

            T *= cool_rate
            if T < 0.05:
                T = 0.05

            if no_improve > 500:
                no_improve = 0
                T = max(T, 2.0)

    # --- Final polish: multi-pass insertion on best_seq ---
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

    # Pass 1: ascending
    in_best = set(best_seq)
    changed = True
    while changed:
        changed = False
        for x in range(2, 129):
            if x not in in_best:
                pos = find_pos(best_seq, x)
                if check_insert(best_seq, pos, x):
                    best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                    in_best.add(x)
                    changed = True
                    best_len = len(best_seq)
                    break

    # Pass 2: descending
    in_best = set(best_seq)
    changed = True
    while changed:
        changed = False
        for x in range(128, 1, -1):
            if x not in in_best:
                pos = find_pos(best_seq, x)
                if check_insert(best_seq, pos, x):
                    best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                    in_best.add(x)
                    changed = True
                    best_len = len(best_seq)
                    break

    # Pass 3: fewest prime factors first
    in_best = set(best_seq)
    changed = True
    while changed:
        changed = False
        candidates = [x for x in range(2, 129) if x not in in_best]
        candidates.sort(key=lambda x: (prime_factor_count(x), x))
        for x in candidates:
            pos = find_pos(best_seq, x)
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
            pos = find_pos(best_seq, x)
            if check_insert(best_seq, pos, x):
                best_seq = best_seq[:pos] + [x] + best_seq[pos:]
                in_best.add(x)
                changed = True
                best_len = len(best_seq)
                break

    # Pass 5: remove-and-readd
    for _ in range(3):
        in_best = set(best_seq)
        improved = False
        for idx in range(len(best_seq)):
            if time.time() > deadline - 2:
                break
            removed_elem = best_seq[idx]
            temp_seq = best_seq[:idx] + best_seq[idx+1:]
            in_temp = set(temp_seq)
            candidates = [x for x in range(2, 129) if x not in in_temp]
            pos = find_pos(temp_seq, removed_elem)
            if check_insert(temp_seq, pos, removed_elem):
                temp_seq2 = temp_seq[:pos] + [removed_elem] + temp_seq[pos:]
                in_temp2 = set(temp_seq2)
                for x in candidates:
                    if x != removed_elem:
                        pos2 = find_pos(temp_seq2, x)
                        if check_insert(temp_seq2, pos2, x):
                            new_seq = temp_seq2[:pos2] + [x] + temp_seq2[pos2:]
                            if count_conflicts(new_seq) == 0 and len(new_seq) > len(best_seq):
                                best_seq = new_seq
                                best_len = len(best_seq)
                                improved = True
                                break
                if improved:
                    break
        if not improved:
            break

    return best_seq
