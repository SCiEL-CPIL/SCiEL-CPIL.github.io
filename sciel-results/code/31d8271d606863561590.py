"""
Strategy: Combine multi-start greedy with incremental O(n) conflict checks
and a targeted local-search polish. The key insight from the improved island
is that incremental validity checking during insertion is far cheaper than
full re-validation, enabling many more diverse starting points and repair
passes within the time budget.
"""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    N = 128
    all_numbers = list(range(2, 129))
    n_all = len(all_numbers)  # 127

    # ---------- fast prime test & factorization ----------
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

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

    factor_cache = {x: factorize(x) for x in range(2, 129)}

    # ---------- incremental validity check ----------
    # When inserting element x at position pos into a sorted sequence,
    # only blocks that include x can create new collisions.
    # These are blocks [i..j] where i <= pos <= j.
    # We maintain the set of all existing block products.

    def compute_all_products(seq):
        """Return set of all consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def is_valid(seq):
        """Full validity check."""
        if len(seq) == 0:
            return True
        n = len(seq)
        prods = set()
        expected = n * (n + 1) // 2
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return len(prods) == expected

    def try_insert(seq, prods, x):
        """
        Try to insert x into sorted seq, maintaining product set.
        Returns (new_seq, new_prods, True) if valid, else (None, None, False).
        Only checks blocks involving x for efficiency.
        """
        # Find insertion position
        pos = 0
        while pos < len(seq) and seq[pos] < x:
            pos += 1
        if pos < len(seq) and seq[pos] == x:
            return None, None, False

        new_seq = seq[:pos] + [x] + seq[pos:]

        # Check all blocks that include x (i.e., start <= pos < end)
        # Block [i..j] includes x iff i <= pos <= j
        n = len(new_seq)
        new_prods = set(prods)

        for i in range(pos + 1):  # start from i=pos down to 0
            p = 1
            for j in range(i, n):
                p *= new_seq[j]
                if j >= pos:  # this block includes x
                    if p in new_prods:
                        return None, None, False
                    new_prods.add(p)
                else:
                    # block doesn't include x, already in prods
                    pass

        # Also need to check: blocks that include x might collide with
        # blocks that DON'T include x. We already added them to new_prods
        # which starts as a copy of prods, so the check above handles it.
        # But we also need to make sure blocks NOT including x don't
        # collide with each other (they don't, since seq was valid).

        return new_seq, new_prods, True

    # ---------- greedy with a given insertion order ----------
    def greedy_with_order(order, deadline):
        """
        Start empty, insert elements in given order.
        order: list of elements to try inserting (in that order).
        Returns (seq, prods, length).
        """
        seq = []
        prods = set()
        remaining = set(all_numbers)

        for x in order:
            if time.time() > deadline - 3:
                break
            if x not in remaining:
                continue
            new_seq, new_prods, ok = try_insert(seq, prods, x)
            if ok:
                seq = new_seq
                prods = new_prods
                remaining.discard(x)

        return seq, prods, len(seq)

    # ---------- generate diverse insertion orders ----------
    def make_orders(deadline):
        """Generate many diverse insertion orders."""
        orders = []
        random.seed(42)

        # 1. Ascending
        orders.append(('asc', all_numbers[:]))
        # 2. Descending
        orders.append(('desc', all_numbers[::-1]))
        # 3. Primes first, then composites ascending
        primes = [x for x in all_numbers if is_prime(x)]
        composites = [x for x in all_numbers if not is_prime(x)]
        orders.append(('primes_asc', primes + composites))
        orders.append(('primes_desc', primes + composites[::-1]))
        # 4. Composites first
        orders.append(('comp_asc', composites + primes))
        # 5. Square-free composites first
        sq_free = [x for x in composites if all(v == 1 for v in factor_cache[x].values())]
        non_sq = [x for x in composites if x not in sq_free]
        orders.append(('sqfree', sq_free + primes + non_sq))
        orders.append(('sqfree_desc', sq_free[::-1] + primes + non_sq[::-1]))
        # 6. Small numbers first
        small = all_numbers[:30]
        large = all_numbers[30:]
        orders.append(('small_first', small + large))
        orders.append(('large_first', large + small))
        # 7. Random shuffles (many)
        for i in range(200):
            shuffled = all_numbers[:]
            random.shuffle(shuffled)
            orders.append((f'rand_{i}', shuffled))
        # 8. Middle-out
        mid = len(all_numbers) // 2
        middle_out = []
        for d in range(mid + 1):
            if mid - d >= 0:
                middle_out.append(all_numbers[mid - d])
            if d > 0 and mid + d < len(all_numbers):
                middle_out.append(all_numbers[mid + d])
        orders.append(('mid_out', middle_out))
        # 9. Prime powers first
        prime_powers = []
        for x in all_numbers:
            f = factor_cache[x]
            if len(f) == 1:
                prime_powers.append(x)
        rest = [x for x in all_numbers if x not in prime_powers]
        orders.append(('pp_first', prime_powers + rest))
        # 10. High prime factor count first
        def pf_count(x):
            return sum(factor_cache[x].values())
        by_pf = sorted(all_numbers, key=pf_count, reverse=True)
        orders.append(('pf_desc', by_pf))
        by_pf_asc = sorted(all_numbers, key=pf_count)
        orders.append(('pf_asc', by_pf_asc))

        return orders

    # ---------- local search: try to add missing elements ----------
    def local_add_pass(seq, prods, deadline):
        """Try to add missing elements in various orders."""
        seq = seq[:]
        prods = set(prods)
        seq_set = set(seq)
        missing = [x for x in all_numbers if x not in seq_set]

        # Try multiple orders for missing elements
        random.seed(123)
        attempts = 0
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            attempts += 1
            # Try ascending, descending, random
            for order_name in ['asc', 'desc', 'rand']:
                if time.time() > deadline - 5:
                    break
                if order_name == 'asc':
                    order = sorted(missing)
                elif order_name == 'desc':
                    order = sorted(missing, reverse=True)
                else:
                    order = missing[:]
                    random.shuffle(order)

                for x in order:
                    if time.time() > deadline - 5:
                        break
                    if x in seq_set:
                        continue
                    new_seq, new_prods, ok = try_insert(seq, prods, x)
                    if ok:
                        seq = new_seq
                        prods = new_prods
                        seq_set.add(x)
                        missing.remove(x)
                        improved = True
                        if not missing:
                            break

        return seq, prods

    # ---------- local search: swap to improve ----------
    def local_swap_pass(seq, prods, deadline):
        """Try removing one element and adding two different ones."""
        seq = seq[:]
        prods = set(prods)
        seq_set = set(seq)
        n = len(seq)

        # For each element, try removing it and adding back a different one
        for idx in range(n):
            if time.time() > deadline - 5:
                break
            removed = seq[idx]
            without = seq[:idx] + seq[idx + 1:]
            # Recompute products for 'without'
            without_prods = compute_all_products(without)
            without_set = set(without)
            missing = [x for x in all_numbers if x not in without_set and x != removed]

            # Try adding single elements
            for x in missing:
                if time.time() > deadline - 5:
                    break
                new_seq, new_prods, ok = try_insert(without, without_prods, x)
                if ok and len(new_seq) > n:
                    return new_seq, new_prods

        return seq, prods

    # ---------- main search ----------
    best_seq = []
    best_len = 0

    # Phase 1: Multi-start greedy
    orders = make_orders(DEADLINE)
    for name, order in orders:
        if time.time() > DEADLINE - 10:
            break
        seq, prods, length = greedy_with_order(order, DEADLINE)
        if length > best_len:
            best_len = length
            best_seq = seq[:]

    # Phase 2: Local add pass on best solution
    if time.time() < DEADLINE - 10:
        best_prods = compute_all_products(best_seq)
        best_seq, best_prods = local_add_pass(best_seq, best_prods, DEADLINE)
        best_len = len(best_seq)

    # Phase 3: Try local add on top few solutions
    if time.time() < DEADLINE - 15:
        # Re-run top strategies and try to improve
        top_orders = ['asc', 'desc', 'primes_asc', 'small_first', 'rand_0', 'rand_50', 'rand_100']
        for name in top_orders:
            if time.time() > DEADLINE - 8:
                break
            order = None
            for nm, o in orders:
                if nm == name:
                    order = o
                    break
            if order is None:
                continue
            seq, prods, length = greedy_with_order(order, DEADLINE)
            if length >= best_len - 2:
                seq, prods = local_add_pass(seq, prods, DEADLINE)
                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = seq[:]

    # Phase 4: Final polish - try to add any remaining elements
    if time.time() < DEADLINE - 3:
        best_prods = compute_all_products(best_seq)
        best_seq, best_prods = local_add_pass(best_seq, best_prods, DEADLINE)
        best_len = len(best_seq)

    # Final validation
    if not is_valid(best_seq):
        # Fallback: simple greedy ascending
        seq = []
        prods = set()
        for x in all_numbers:
            new_seq, new_prods, ok = try_insert(seq, prods, x)
            if ok:
                seq = new_seq
                prods = new_prods
        best_seq = seq
        best_len = len(seq)

    return best_seq
