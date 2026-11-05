import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55

    all_numbers = list(range(2, 129))

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def get_prime_factorization(n):
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

    def is_valid(s):
        """Check if all consecutive block products are distinct using prefix products."""
        n = len(s)
        if n == 0:
            return True
        pref = [1] * (n + 1)
        for i in range(n):
            pref[i + 1] = pref[i] * s[i]
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                p = pref[j] // pref[i]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def is_valid_fast(s, existing_prods=None, existing_pref=None):
        """
        Faster validity check. If existing_prods and existing_pref are given,
        we only need to check new blocks involving the newly added element.
        """
        n = len(s)
        if n == 0:
            return True
        if existing_pref is None:
            existing_pref = [1] * (n + 1)
            for i in range(n):
                existing_pref[i + 1] = existing_pref[i] * s[i]
        if existing_prods is None:
            existing_prods = set()
            for i in range(n):
                for j in range(i + 1, n + 1):
                    existing_prods.add(existing_pref[j] // existing_pref[i])
        # Check all blocks - this is the full check but with precomputed prefix
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                p = existing_pref[j] // existing_pref[i]
                if p in prods:
                    return False
                prods.add(p)
        return True

    # Beam search approach
    # State: (sequence, prefix_products, set_of_products)
    # We build sequences left to right, deciding at each step which number to add next

    def beam_search(initial_seq, beam_width=200, max_time=45.0):
        """
        Beam search over subsequences of all_numbers.
        At each step, we consider adding any unused number that is greater than the last element.
        """
        initial_seq = sorted(initial_seq)
        n = len(initial_seq)
        if n == 0:
            return []

        # Precompute prefix products and product set for initial sequence
        pref = [1] * (n + 1)
        for i in range(n):
            pref[i + 1] = pref[i] * initial_seq[i]
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                prods.add(pref[j] // pref[i])

        # Beam: list of (sequence, prefix, prods_set)
        beam = [(initial_seq[:], pref[:], frozenset(prods))]
        best_seq = initial_seq[:]
        best_len = n

        # Numbers greater than the last element of initial_seq
        min_val = initial_seq[-1] + 1
        candidates = [x for x in all_numbers if x >= min_val]

        for step in range(128 - min_val + 1):
            if time.time() > deadline - max_time:
                break

            new_beam = []
            seen_states = set()

            for seq, pref_list, prod_set in beam:
                last = seq[-1]
                # Try adding each candidate > last
                for c in candidates:
                    if c <= last:
                        continue

                    # Check if adding c creates duplicate products
                    # New products are: c, and seq[i]*...*seq[-1]*c for i in range(len(seq))
                    # Actually, new blocks ending at the new position:
                    # Block [i..n] for i in 0..n-1 where n is new length
                    # Product = (pref[n] // pref[i]) where pref[n] = pref_old[-1] * c

                    new_pref_last = pref_list[-1] * c
                    valid = True
                    for i in range(len(seq)):
                        p = new_pref_last // pref_list[i]
                        if p in prod_set:
                            valid = False
                            break
                        # Also check against other new products
                        # We'll collect all new products and check for duplicates among them
                        # Actually, let's just collect and check
                    if not valid:
                        continue

                    # Collect all new products and check for duplicates among them
                    new_prods = []
                    for i in range(len(seq)):
                        p = new_pref_last // pref_list[i]
                        new_prods.append(p)
                    # Check for duplicates among new_prods and against prod_set
                    new_set = set(new_prods)
                    if len(new_set) != len(new_prods):
                        continue
                    if new_set & prod_set:
                        continue

                    new_seq = seq + [c]
                    new_pref = pref_list + [new_pref_last]
                    new_prod_set = prod_set | new_set

                    # State key for deduplication
                    state_key = tuple(new_seq)
                    if state_key in seen_states:
                        continue
                    seen_states.add(state_key)

                    new_beam.append((new_seq, new_pref, new_prod_set))

                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq[:]

            if not new_beam:
                break

            # Sort by length descending, then by some heuristic
            new_beam.sort(key=lambda x: (len(x[0]), -x[0][-1]), reverse=True)

            # Keep top beam_width
            if len(new_beam) > beam_width:
                new_beam = new_beam[:beam_width]

            beam = new_beam

            if not beam:
                break

        return best_seq

    # Strategy 1: Start with primes
    primes = [x for x in all_numbers if is_prime(x)]
    result1 = beam_search(primes, beam_width=300, max_time=20.0)

    # Strategy 2: Start with small numbers
    small = list(range(2, 30))
    result2 = beam_search(small, beam_width=300, max_time=15.0)

    # Strategy 3: Start with empty, let beam search find its way
    # Actually, starting empty is too slow. Let's start with a few primes.
    few_primes = primes[:10]
    result3 = beam_search(few_primes, beam_width=200, max_time=15.0)

    # Pick the best
    best_seq = max([result1, result2, result3], key=len)

    # Local improvement: try inserting missing numbers
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            current_set = set(best_seq)
            missing = [x for x in all_numbers if x not in current_set]

            for add_val in missing:
                if time.time() > deadline - 8:
                    break
                pos = 0
                while pos < len(best_seq) and best_seq[pos] < add_val:
                    pos += 1
                candidate = best_seq[:pos] + [add_val] + best_seq[pos:]
                if is_valid(candidate) and len(candidate) > len(best_seq):
                    best_seq = candidate
                    improved = True
                    break

    return best_seq
