import time
import math
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

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
    composites = [x for x in all_numbers if not is_prime(x)]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # Precompute prime exponent vectors for each number
    exp_vecs = {}
    for n in all_numbers:
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(primes):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)

    # For efficient checking: we need to track all suffix products
    # A suffix product is the product of elements from position i to the end
    # When we append a new element x, new suffix products are:
    #   x, x*last_suffix, x*last_two_suffixes, ...
    # So we need to track all current suffix products as exponent vectors

    def check_sequence_valid(seq):
        """Check if a sequence has all distinct consecutive block products."""
        n = len(seq)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            vec = [0] * num_primes
            for j in range(i, n):
                ej = exp_vecs[seq[j]]
                for k in range(num_primes):
                    vec[k] += ej[k]
                tv = tuple(vec)
                if tv in prods:
                    return False
                prods.add(tv)
        return True

    def get_suffix_products(seq):
        """Get all suffix product exponent vectors for a sequence."""
        n = len(seq)
        suffixes = []
        vec = [0] * num_primes
        for i in range(n - 1, -1, -1):
            ej = exp_vecs[seq[i]]
            for k in range(num_primes):
                vec[k] += ej[k]
            suffixes.append(tuple(vec))
        suffixes.reverse()  # suffixes[i] = product of seq[i:]
        return suffixes

    def try_append(seq, c, suffix_products=None):
        """Try to append c to seq. Returns (new_seq, new_suffixes) or None."""
        if suffix_products is None:
            suffix_products = get_suffix_products(seq)
        
        new_suffixes = [tuple(exp_vecs[c])]
        for sp in suffix_products:
            new_vec = list(sp)
            for k in range(num_primes):
                new_vec[k] += exp_vecs[c][k]
            new_suffixes.append(tuple(new_vec))
        
        # Check for conflicts: new suffixes must not conflict with each other or existing ones
        existing_set = set(suffix_products)
        for ns in new_suffixes:
            if ns in existing_set:
                return None
            if ns in new_suffixes[new_suffixes.index(ns)+1:] if new_suffixes.index(ns) < len(new_suffixes)-1 else True:
                # More careful check
                pass
        
        # Better: check all pairs
        all_prods = set()
        for sp in suffix_products:
            all_prods.add(sp)
        for ns in new_suffixes:
            if ns in all_prods:
                return None
            all_prods.add(ns)
        
        new_seq = seq + [c]
        return (new_seq, new_suffixes)

    def try_insert_at(seq, pos, c):
        """Try to insert c at position pos in seq. Returns (new_seq, new_suffixes) or None."""
        if pos < 0 or pos > len(seq):
            return None
        if pos > 0 and seq[pos-1] >= c:
            return None
        if pos < len(seq) and c >= seq[pos]:
            return None
        
        new_seq = seq[:pos] + [c] + seq[pos:]
        if not check_sequence_valid(new_seq):
            return None
        return (new_seq, get_suffix_products(new_seq))

    def greedy_build(start_seq, remaining, rng, time_limit=None):
        """Greedily add elements from remaining to start_seq."""
        seq = start_seq[:]
        suffixes = get_suffix_products(seq)
        used = set(seq)
        rem = [x for x in remaining if x not in used]
        rng.shuffle(rem)
        
        if time_limit is None:
            time_limit = deadline
        
        progress = True
        while progress and time.time() < time_limit:
            progress = False
            for c in rem[:]:
                if time.time() > time_limit:
                    break
                result = try_append(seq, c, suffixes)
                if result is not None:
                    seq, suffixes = result
                    rem.remove(c)
                    progress = True
                    break
        return seq

    def local_search(seq, max_iter=100, time_limit=None):
        """Improve sequence through local search with insertions and swaps."""
        if time_limit is None:
            time_limit = deadline
        
        current = seq[:]
        used = set(current)
        remaining = [x for x in all_numbers if x not in used]
        
        for iteration in range(max_iter):
            if time.time() > time_limit:
                break
            improved = False
            
            # Try insertions in random order
            rng_local = random.Random(hash((iteration, len(current))) % 100000)
            remaining_copy = remaining[:]
            rng_local.shuffle(remaining_copy)
            for c in remaining_copy:
                if time.time() > time_limit:
                    break
                # Try all valid positions
                best_pos = None
                for pos in range(len(current) + 1):
                    if pos > 0 and current[pos-1] >= c:
                        continue
                    if pos < len(current) and c >= current[pos]:
                        continue
                    new_seq = current[:pos] + [c] + current[pos:]
                    if check_sequence_valid(new_seq):
                        best_pos = pos
                        break
                if best_pos is not None:
                    current = current[:best_pos] + [c] + current[best_pos:]
                    used = set(current)
                    remaining = [x for x in all_numbers if x not in used]
                    improved = True
                    break
            
            # Try swaps: remove one element, add another
            if not improved and time.time() < time_limit:
                for i in range(len(current)):
                    if time.time() > time_limit:
                        break
                    removed = current[i]
                    base = current[:i] + current[i+1:]
                    for c in remaining:
                        if time.time() > time_limit:
                            break
                        # Try inserting c into base
                        found = False
                        for pos in range(len(base) + 1):
                            if pos > 0 and base[pos-1] >= c:
                                continue
                            if pos < len(base) and c >= base[pos]:
                                continue
                            new_seq = base[:pos] + [c] + base[pos:]
                            if check_sequence_valid(new_seq):
                                current = new_seq
                                used = set(current)
                                remaining = [x for x in all_numbers if x not in used]
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current

    def systematic_greedy():
        """Systematic greedy: try to add each number in increasing order."""
        seq = []
        for c in all_numbers:
            if time.time() > deadline:
                break
            # Try to append
            result = try_append(seq, c)
            if result is not None:
                seq = result[0]
                continue
            # Try inserting at various positions
            for pos in range(len(seq) + 1):
                if pos > 0 and seq[pos-1] >= c:
                    continue
                if pos < len(seq) and c >= seq[pos]:
                    continue
                new_seq = seq[:pos] + [c] + seq[pos:]
                if check_sequence_valid(new_seq):
                    seq = new_seq
                    break
        return seq

    def branch_and_bound(seq, start_idx, best_len):
        """Branch and bound for longest valid subsequence."""
        if time.time() > deadline:
            return seq
        
        if len(seq) > best_len:
            best_len = len(seq)
        
        # Upper bound
        remaining_count = len(all_numbers) - start_idx
        if len(seq) + remaining_count <= best_len:
            return seq
        
        for i in range(start_idx, len(all_numbers)):
            if time.time() > deadline:
                break
            c = all_numbers[i]
            new_seq = seq + [c]
            if check_sequence_valid(new_seq):
                result = branch_and_bound(new_seq, i + 1, best_len)
                if len(result) > len(seq):
                    seq = result
        return seq

    # Strategy: Multiple approaches, keep best

    best_seq = []

    # Approach 1: Start with all primes, add composites greedily
    seq1 = primes[:]
    if check_sequence_valid(seq1):
        used = set(seq1)
        remaining = [x for x in composites if x not in used]
        rng1 = random.Random(42)
        seq1 = greedy_build(seq1, remaining, rng1, time_limit=deadline - 30)
        seq1 = local_search(seq1, max_iter=200, time_limit=deadline - 15)
        if len(seq1) > len(best_seq):
            best_seq = seq1

    # Approach 2: Systematic greedy from scratch
    if time.time() < deadline - 20:
        seq2 = systematic_greedy()
        seq2 = local_search(seq2, max_iter=100, time_limit=deadline - 10)
        if len(seq2) > len(best_seq):
            best_seq = seq2

    # Approach 3: Randomized restarts with different prime subsets
    for restart in range(10):
        if time.time() > deadline - 5:
            break
        rng = random.Random(1000 + restart)
        # Random subset of primes
        k = rng.randint(15, len(primes))
        prime_subset = sorted(rng.sample(primes, k))
        if not check_sequence_valid(prime_subset):
            prime_subset = primes[:]
        
        used = set(prime_subset)
        remaining = [x for x in composites if x not in used]
        seq3 = greedy_build(prime_subset, remaining, rng, time_limit=deadline - 10 - restart)
        seq3 = local_search(seq3, max_iter=50, time_limit=deadline - 5)
        if len(seq3) > len(best_seq):
            best_seq = seq3

    # Approach 4: If we have time, try branch and bound on a smaller problem
    if time.time() < deadline - 10:
        # B&B on first 40 numbers
        small_numbers = all_numbers[:40]
        seq4 = []
        def bb_small(seq, start_idx):
            if time.time() > deadline - 5:
                return seq
            if len(seq) > len(seq4):
                seq4 = seq[:]
            remaining_count = len(small_numbers) - start_idx
            if len(seq) + remaining_count <= len(seq4):
                return seq
            for i in range(start_idx, len(small_numbers)):
                if time.time() > deadline - 5:
                    break
                c = small_numbers[i]
                new_seq = seq + [c]
                if check_sequence_valid(new_seq):
                    bb_small(new_seq, i + 1)
            return seq
        bb_small([], 0)
        if len(seq4) > len(best_seq):
            best_seq = seq4

    # Final polish: try to add more elements to best_seq
    if time.time() < deadline - 2:
        used = set(best_seq)
        remaining = [x for x in all_numbers if x not in used]
        rng_final = random.Random(999)
        rng_final.shuffle(remaining)
        for c in remaining:
            if time.time() > deadline - 1:
                break
            for pos in range(len(best_seq) + 1):
                if pos > 0 and best_seq[pos-1] >= c:
                    continue
                if pos < len(best_seq) and c >= best_seq[pos]:
                    continue
                new_seq = best_seq[:pos] + [c] + best_seq[pos:]
                if check_sequence_valid(new_seq):
                    best_seq = new_seq
                    break

    return best_seq
