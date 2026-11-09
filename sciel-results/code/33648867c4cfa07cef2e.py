import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]

    def check_valid(seq):
        """All consecutive block products (incl. singletons) distinct.
        Exact integer products uniquely identify a block's prime factorization."""
        n = len(seq)
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def try_insert(seq, c):
        n = len(seq)
        valid_positions = []
        for i in range(n + 1):
            if i > 0 and seq[i - 1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            valid_positions.append(i)
        if not valid_positions:
            return None
        mid = n / 2.0
        valid_positions.sort(key=lambda i: abs(i - mid))
        for i in valid_positions:
            new_seq = seq[:i] + [c] + seq[i:]
            if check_valid(new_seq):
                return new_seq
        return None

    def local_search(seq, time_budget=0.6):
        current = seq[:]
        used = set(current)
        remaining = [x for x in all_numbers if x not in used]
        t0 = time.time()
        while time.time() - t0 < time_budget and time.time() < deadline:
            improved = False
            # insertion pass
            if remaining:
                random.shuffle(remaining)
                for c in remaining:
                    ns = try_insert(current, c)
                    if ns is not None:
                        current = ns
                        used = set(current)
                        remaining = [x for x in all_numbers if x not in used]
                        improved = True
                        break
            if improved:
                continue
            # swap pass (cheap random attempts)
            for _ in range(80):
                if time.time() - t0 > time_budget or time.time() > deadline:
                    break
                if not remaining or not current:
                    break
                i = random.randrange(len(current))
                c = random.choice(remaining)
                base = [x for x in current if x != current[i]]
                ins = try_insert(base, c)
                if ins is not None and len(ins) >= len(current):
                    current = ins
                    used = set(current)
                    remaining = [x for x in all_numbers if x not in used]
                    improved = True
                    break
            if not improved:
                break
        return current

    def build_initial(seed_val, time_budget=2.0):
        rng = random.Random(seed_val)
        bi_deadline = time.time() + time_budget
        seq = primes[:]
        used = set(seq)
        remaining = [x for x in all_numbers if x not in used]
        rng.shuffle(remaining)
        progress = True
        while progress and time.time() < bi_deadline and time.time() < deadline:
            progress = False
            for c in remaining[:]:
                new_seq = try_insert(seq, c)
                if new_seq is not None:
                    seq = new_seq
                    remaining.remove(c)
                    progress = True
                    break
        return seq

    def csp_search(timeout=12):
        """Chronological backtracking building left-to-right.
        Tracks all block products as exact ints; prunes on collision."""
        search_start = time.time()
        best = [0]
        best_seq = [[]]

        def search(seq, seen, last):
            if time.time() - search_start > timeout:
                return
            if len(seq) > best[0]:
                best[0] = len(seq)
                best_seq[0] = seq[:]
            ub = len(seq) + (128 - last)
            if ub <= best[0]:
                return
            for c in range(last + 1, 129):
                if time.time() - search_start > timeout:
                    return
                # new block products: c times each suffix of seq (incl. empty)
                new_prods = set()
                p = c
                ok = True
                if p in seen:
                    ok = False
                else:
                    new_prods.add(p)
                for j in range(len(seq) - 1, -1, -1):
                    p *= seq[j]
                    if p in seen or p in new_prods:
                        ok = False
                        break
                    new_prods.add(p)
                if ok:
                    search(seq + [c], seen | new_prods, c)

        search([], set(), 1)
        return best_seq[0]

    def remove_one_add_two(seq, time_budget=4.0):
        """Escape move: drop one element, then add two (net +1)."""
        current = seq[:]
        used = set(current)
        remaining = [x for x in all_numbers if x not in used]
        t0 = time.time()
        while time.time() - t0 < time_budget and time.time() < deadline:
            improved = False
            for _ in range(120):
                if time.time() - t0 > time_budget or time.time() > deadline:
                    break
                if not current:
                    break
                i = random.randrange(len(current))
                base = [x for x in current if x != current[i]]
                used_b = set(base)
                rem_b = [x for x in all_numbers if x not in used_b]
                if len(rem_b) < 2:
                    break
                c1, c2 = random.sample(rem_b, 2)
                ins1 = try_insert(base, c1)
                if ins1 is None:
                    continue
                used1 = set(ins1)
                rem1 = [x for x in all_numbers if x not in used1]
                if not rem1:
                    continue
                c2b = random.choice(rem1)
                ins2 = try_insert(ins1, c2b)
                if ins2 is not None and len(ins2) > len(current):
                    current = ins2
                    used = set(current)
                    remaining = [x for x in all_numbers if x not in used]
                    improved = True
                    break
            if not improved:
                break
        return current

    best_seq = primes[:]
    best_len = len(best_seq)

    # Phase 1: greedy + local search from several seeds
    restart_deadline = start_time + 28
    for seed in range(60):
        if time.time() > restart_deadline:
            break
        seq = build_initial(seed * 12345 + 7, time_budget=2.0)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
        improved = local_search(seq, time_budget=0.6)
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved[:]

    # Phase 2: CSP backtracking search
    if time.time() < deadline - 10:
        csp_result = csp_search(timeout=min(12, deadline - time.time() - 10))
        if len(csp_result) > best_len:
            best_len = len(csp_result)
            best_seq = csp_result[:]

    # Phase 3: final polish (local search + escape moves)
    if time.time() < deadline:
        remaining_time = deadline - time.time()
        final = local_search(best_seq, time_budget=min(6.0, remaining_time - 2))
        if len(final) > best_len:
            best_len = len(final)
            best_seq = final
    if time.time() < deadline - 1:
        esc = remove_one_add_two(best_seq, time_budget=min(4.0, deadline - time.time() - 1))
        if len(esc) > best_len:
            best_len = len(esc)
            best_seq = esc
    if time.time() < deadline:
        final2 = local_search(best_seq, time_budget=min(2.0, deadline - time.time()))
        if len(final2) > best_len:
            best_seq = final2

    return best_seq
