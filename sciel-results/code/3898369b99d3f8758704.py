"""Combined: multi-strategy greedy + insertion-based search.
Starts from strong bases (primes, odds) and inserts remaining candidates
at any valid position to maximize sequence length while keeping
strict increase and distinct consecutive-block products."""
import time
import random

def construct():
    deadline = time.time() + 50.0  # safety margin well under the 90s kill
    random.seed(42)

    candidates = list(range(2, 129))

    def _all_block_products(seq):
        """Return the set of all nonempty consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
        return prods

    def _is_valid(seq):
        """Check strict increase and distinct block products."""
        n = len(seq)
        for i in range(1, n):
            if seq[i] <= seq[i-1]:
                return False
        expected = n * (n + 1) // 2
        return len(_all_block_products(seq)) == expected

    def _try_insert(seq, c):
        """Try inserting c at every valid position; return best new seq or None.
        Prefers positions that keep the sequence 'spread out'."""
        n = len(seq)
        best_new = None
        best_score = -1
        for i in range(n + 1):
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            new_seq = seq[:i] + [c] + seq[i:]
            # Quick check: count distinct products
            prods = set()
            valid = True
            for a in range(len(new_seq)):
                p = 1
                for b in range(a, len(new_seq)):
                    p *= new_seq[b]
                    if p in prods:
                        valid = False
                        break
                    prods.add(p)
                if not valid:
                    break
            if valid:
                # Score: prefer insertion that maximizes min gap to neighbors
                gap = float('inf')
                if i > 0:
                    gap = min(gap, c - seq[i-1])
                if i < n:
                    gap = min(gap, seq[i] - c)
                if gap > best_score:
                    best_score = gap
                    best_new = new_seq
        return best_new

    def _greedy_insertion(base_seq, remaining, time_limit):
        """Repeatedly insert the best candidate at the best position."""
        current = list(base_seq)
        used = set(current)
        rem = list(remaining)
        progress = True
        while progress and time.time() < time_limit:
            progress = False
            best_new = None
            best_c = None
            best_pos = -1
            for c in rem:
                if time.time() > time_limit:
                    break
                new_seq = _try_insert(current, c)
                if new_seq is not None:
                    # Prefer longer sequences, then more 'spread'
                    best_new = new_seq
                    best_c = c
                    progress = True
                    break  # Insert one at a time, restart loop
            if best_c is not None:
                current = best_new
                used.add(best_c)
                rem.remove(best_c)
        return current

    def _simple_greedy(cands):
        """Fast append-only greedy."""
        values, products = [], set()
        for x in cands:
            if time.time() > deadline:
                break
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    def _is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [x for x in candidates if _is_prime(x)]
    odds = list(range(3, 128, 2))
    odds_composite = [x for x in odds if x not in primes]

    best = []
    best_len = 0

    # Strategy 1: Simple greedy on odds ascending (fast baseline)
    cand = _simple_greedy(odds)
    if len(cand) > best_len and _is_valid(cand):
        best = cand
        best_len = len(cand)

    # Strategy 2: Simple greedy on full range
    cand = _simple_greedy(candidates)
    if len(cand) > best_len and _is_valid(cand):
        best = cand
        best_len = len(cand)

    # Strategy 3: Insertion search starting from primes
    if time.time() < deadline - 15:
        remaining = [x for x in candidates if x not in set(primes)]
        random.shuffle(remaining)
        cand = _greedy_insertion(primes, remaining, deadline - 10)
        if len(cand) > best_len and _is_valid(cand):
            best = cand
            best_len = len(cand)

    # Strategy 4: Insertion search starting from best so far, add remaining
    if time.time() < deadline - 10 and len(best) < 20:
        used = set(best)
        remaining = [x for x in candidates if x not in used]
        random.shuffle(remaining)
        cand = _greedy_insertion(best, remaining, deadline - 5)
        if len(cand) > best_len and _is_valid(cand):
            best = cand
            best_len = len(cand)

    # Strategy 5: Multiple random restarts of insertion from small bases
    if time.time() < deadline - 8:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            # Start from a random subset of primes
            base_size = min(8, len(primes))
            base = random.sample(primes, base_size)
            base.sort()
            if not _is_valid(base):
                continue
            remaining = [x for x in candidates if x not in set(base)]
            random.shuffle(remaining)
            cand = _greedy_insertion(base, remaining, deadline - 5)
            if len(cand) > best_len and _is_valid(cand):
                best = cand
                best_len = len(cand)

    # Final validation
    if not _is_valid(best):
        # Fallback to simple greedy
        best = _simple_greedy(odds)

    return best
