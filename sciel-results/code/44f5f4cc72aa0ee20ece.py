import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    def compute_tails(seq, x):
        """Compute all consecutive block products ending at x."""
        tails = [x]
        prod = x
        for a in reversed(seq):
            prod *= a
            tails.append(prod)
        return tails

    def check_valid(products, tails):
        """Check if tails have no collision with existing products."""
        for p in tails:
            if p in products:
                return False
        return True

    def greedy_construct(start=2, end=129):
        """Standard greedy: pick smallest valid x."""
        values = []
        products = set()
        for x in range(start, end):
            tails = compute_tails(values, x)
            if check_valid(products, tails):
                values.append(x)
                products.update(tails)
        return values, products

    def full_product_set(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def try_local_search(seq, products, time_limit):
        """
        Swap-and-insert local search:
        For each position i, try removing seq[i], then greedily re-inserting
        the displaced value and new candidates to see if we gain length.
        """
        best_seq = list(seq)
        best_len = len(seq)
        best_products = set(products)
        n = len(seq)

        for i in range(n):
            if time.time() > time_limit:
                break

            # Remove element at position i
            removed = seq[i]
            new_seq = seq[:i] + seq[i+1:]
            new_products = full_product_set(new_seq)

            # Try to re-insert the removed value at various positions
            # and also try inserting new values
            improved = False

            # Strategy 1: Try inserting removed value back at different positions
            for insert_pos in range(len(new_seq) + 1):
                if insert_pos > 0 and new_seq[insert_pos - 1] >= removed:
                    continue
                if insert_pos < len(new_seq) and new_seq[insert_pos] <= removed:
                    continue

                # Build candidate sequence with removed value at insert_pos
                cand_seq = new_seq[:insert_pos] + [removed] + new_seq[insert_pos:]

                # Now try to extend greedily from this candidate
                cand_products = full_product_set(cand_seq)
                cand_vals = list(cand_seq)

                # Continue greedy from where we left off
                last_val = cand_vals[-1] if cand_vals else 1
                for x in range(last_val + 1, 129):
                    if time.time() > time_limit:
                        break
                    tails = compute_tails(cand_vals, x)
                    if check_valid(cand_products, tails):
                        cand_vals.append(x)
                        cand_products.update(tails)

                if len(cand_vals) > best_len:
                    best_len = len(cand_vals)
                    best_seq = cand_vals
                    best_products = cand_products
                    improved = True
                    break

            # Strategy 2: Without re-inserting removed, try to fill the gap
            # with a different value that allows more extensions
            for new_val in range(2, 129):
                if new_val == removed:
                    continue
                # Find valid insertion position
                valid_pos = -1
                for pos in range(len(new_seq) + 1):
                    if pos > 0 and new_seq[pos - 1] >= new_val:
                        continue
                    if pos < len(new_seq) and new_seq[pos] <= new_val:
                        continue
                    valid_pos = pos
                    break

                if valid_pos == -1:
                    continue

                cand_seq = new_seq[:valid_pos] + [new_val] + new_seq[valid_pos:]
                cand_products = full_product_set(cand_seq)
                cand_vals = list(cand_seq)

                # Continue greedy
                last_val = cand_vals[-1] if cand_vals else 1
                for x in range(last_val + 1, 129):
                    if time.time() > time_limit:
                        break
                    tails = compute_tails(cand_vals, x)
                    if check_valid(cand_products, tails):
                        cand_vals.append(x)
                        cand_products.update(tails)

                if len(cand_vals) > best_len:
                    best_len = len(cand_vals)
                    best_seq = cand_vals
                    best_products = cand_products
                    improved = True
                    break

            if improved:
                # Restart local search with the improved sequence
                return try_local_search(best_seq, best_products, time_limit)

        return best_seq

    def beam_search(seed_seq, time_limit, beam_width=50):
        """Beam search starting from seed sequence."""
        best_seq = list(seed_seq)
        best_len = len(seed_seq)

        # Initialize beams with the seed and some variations
        beams = []
        seed_products = full_product_set(seed_seq)
        beams.append((list(seed_seq), seed_products))

        # Also add some greedy variations
        for skip in range(1, min(5, len(seed_seq))):
            if time.time() > time_limit:
                break
            mod_seq = seed_seq[:skip] + seed_seq[skip+1:]
            if len(mod_seq) < 2:
                continue
            mod_products = full_product_set(mod_seq)
            beams.append((mod_seq, mod_products))

        while time.time() < time_limit:
            new_beams = []

            for seq, prods in beams:
                if not seq:
                    last_val = 1
                else:
                    last_val = seq[-1]

                # Try candidates greater than last_val
                for c in range(last_val + 1, min(last_val + 30 + 1, 129)):
                    tails = compute_tails(seq, c)
                    if check_valid(prods, tails):
                        new_seq = seq + [c]
                        new_prods = prods.union(tails)
                        new_beams.append((new_seq, new_prods))

                        if len(new_seq) > best_len:
                            best_len = len(new_seq)
                            best_seq = new_seq

            if not new_beams:
                break

            # Prune beams: sort by length desc, then last value asc
            def beam_key(b):
                s, _ = b
                return (-len(s), s[-1] if s else 0)

            new_beams.sort(key=beam_key)
            beams = new_beams[:beam_width]

        return best_seq

    # Phase 1: Greedy baseline
    greedy_seq, greedy_prods = greedy_construct()
    best_seq = greedy_seq
    best_len = len(greedy_seq)

    # Phase 2: Local search on greedy solution
    if time.time() < deadline - 5:
        local_seq = try_local_search(greedy_seq, greedy_prods, deadline - 10)
        if len(local_seq) > best_len:
            best_seq = local_seq
            best_len = len(local_seq)

    # Phase 3: Beam search with remaining time
    if time.time() < deadline - 2:
        beam_seq = beam_search(best_seq, deadline - 1, beam_width=80)
        if len(beam_seq) > best_len:
            best_seq = beam_seq
            best_len = len(beam_seq)

    return best_seq
# EVOLVE-BLOCK-END
