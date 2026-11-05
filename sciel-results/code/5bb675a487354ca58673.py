"""Stronger program: iterative improvement via swaps and insertions."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 88.0  # safety margin before 90s kill

    def get_all_products(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def is_valid(seq):
        """Check if sequence is strictly increasing and all block products are distinct."""
        if len(seq) == 0:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i+1]:
                return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        products = get_all_products(seq)
        return len(products) == len(seq) * (len(seq) + 1) // 2

    def greedy_construct():
        """Standard greedy: try numbers 2..128 in order."""
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    best_seq = greedy_construct()
    best_len = len(best_seq)

    if not is_valid(best_seq):
        best_seq = list(range(2, 129))
        best_len = len(best_seq)

    # Try different greedy orderings
    def count_distinct_prime_factors(n):
        count = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                count += 1
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            count += 1
        return count

    def is_prime_power(n):
        for p in [2, 3, 5, 7, 11, 13]:
            q = n
            while q % p == 0:
                q //= p
            if q == 1:
                return True
        return False

    candidates = list(range(2, 129))

    def greedy_with_order(order):
        values, products = [], set()
        for x in order:
            if x < 2 or x > 128:
                continue
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        values.sort()
        return values

    # Try ordering by number of distinct prime factors (ascending)
    order1 = sorted(candidates, key=lambda x: count_distinct_prime_factors(x))
    if time.time() - start_time < DEADLINE:
        seq1 = greedy_with_order(order1)
        if is_valid(seq1) and len(seq1) > best_len:
            best_seq = seq1
            best_len = len(seq1)

    # Try ordering prioritizing prime powers
    order3 = sorted(candidates, key=lambda x: (not is_prime_power(x), x))
    if time.time() - start_time < DEADLINE:
        seq3 = greedy_with_order(order3)
        if is_valid(seq3) and len(seq3) > best_len:
            best_seq = seq3
            best_len = len(seq3)

    # Local search: try swaps and insertions
    used = set(best_seq)
    remaining = [x for x in range(2, 129) if x not in used]

    def try_swap(seq, i, j, new_val):
        """Try replacing seq[j] with new_val, check if still valid."""
        if new_val in seq:
            return None
        new_seq = seq[:]
        new_seq[j] = new_val
        new_seq.sort()
        if not is_valid(new_seq):
            return None
        return new_seq

    def try_insert(seq, new_elem, pos):
        """Try inserting new_elem at position pos in seq."""
        if new_elem in seq:
            return None
        new_seq = seq[:pos] + [new_elem] + seq[pos:]
        if not is_valid(new_seq):
            return None
        return new_seq

    # Iterative improvement: try to add one more element at a time
    max_iterations = 50
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            break

        improved = False

        # First, try simple insertions
        for new_elem in remaining[:]:
            if time.time() - start_time > DEADLINE:
                break
            for pos in range(len(best_seq) + 1):
                result = try_insert(best_seq, new_elem, pos)
                if result is not None:
                    best_seq = result
                    best_len = len(result)
                    used.add(new_elem)
                    remaining.remove(new_elem)
                    improved = True
                    break
            if improved:
                break

        if improved:
            continue

        # If no simple insertion works, try swapping an element to free up space
        # Strategy: for each element in the sequence, try replacing it with a
        # different number from remaining, then try to insert
        swapped = False
        for j in range(len(best_seq)):
            if time.time() - start_time > DEADLINE:
                break
            orig_val = best_seq[j]
            for new_elem in remaining[:]:
                if time.time() - start_time > DEADLINE:
                    break
                # Try replacing best_seq[j] with new_elem
                test_seq = best_seq[:]
                test_seq[j] = new_elem
                test_seq.sort()
                if not is_valid(test_seq):
                    continue

                # Now try to insert orig_val somewhere
                for pos in range(len(test_seq) + 1):
                    result = try_insert(test_seq, orig_val, pos)
                    if result is not None:
                        best_seq = result
                        best_len = len(result)
                        # Update used/remaining
                        used.discard(new_elem)
                        used.add(orig_val)
                        if new_elem not in remaining:
                            remaining.append(new_elem)
                        if orig_val in remaining:
                            remaining.remove(orig_val)
                        swapped = True
                        break
                if swapped:
                    break
            if swapped:
                break

        if not swapped and not improved:
            # Try more aggressive: remove one element, then try to add two
            for remove_idx in range(len(best_seq)):
                if time.time() - start_time > DEADLINE:
                    break
                removed_val = best_seq[remove_idx]
                test_seq = best_seq[:remove_idx] + best_seq[remove_idx+1:]
                used.discard(removed_val)
                if removed_val not in remaining:
                    remaining.append(removed_val)

                # Try to insert two elements
                found_two = False
                for e1 in remaining[:]:
                    if time.time() - start_time > DEADLINE:
                        break
                    for pos1 in range(len(test_seq) + 1):
                        seq1 = try_insert(test_seq, e1, pos1)
                        if seq1 is None:
                            continue
                        for e2 in remaining[:]:
                            if e2 == e1:
                                continue
                            if time.time() - start_time > DEADLINE:
                                break
                            for pos2 in range(len(seq1) + 1):
                                seq2 = try_insert(seq1, e2, pos2)
                                if seq2 is not None and len(seq2) > best_len:
                                    best_seq = seq2
                                    best_len = len(seq2)
                                    used.add(e1)
                                    used.add(e2)
                                    if e1 in remaining:
                                        remaining.remove(e1)
                                    if e2 in remaining:
                                        remaining.remove(e2)
                                    found_two = True
                                    break
                            if found_two:
                                break
                        if found_two:
                            break
                    if found_two:
                        break
                if found_two:
                    break

                # Restore state
                used.add(removed_val)
                if removed_val in remaining:
                    remaining.remove(removed_val)

                if not found_two:
                    continue

            if not swapped and not improved:
                break

    # Final validation
    if not is_valid(best_seq):
        best_seq = greedy_construct()

    return best_seq
