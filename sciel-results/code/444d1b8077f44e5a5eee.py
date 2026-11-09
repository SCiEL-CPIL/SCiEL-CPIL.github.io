import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 210
    max_val = 10210

    # Sieve for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}

    # Precompute parity masks using sieve
    masks = [0] * (max_val + 1)
    for p in primes:
        p_bit = 1 << prime_index[p]
        for multiple in range(p, max_val + 1, p):
            cnt = 0
            t = multiple
            while t % p == 0:
                cnt += 1
                t //= p
            if cnt % 2 == 1:
                masks[multiple] ^= p_bit

    target = masks[210]

    def verify_solution(sol):
        prod = 210
        for x in sol:
            prod *= x
        s = math.isqrt(prod)
        return s * s == prod

    def reduce_solution(sol_set, available_range):
        """Try to reduce solution size using local search."""
        if len(sol_set) <= 100:
            return sol_set

        sol_list = list(sol_set)
        current_size = len(sol_list)
        improved = True

        while improved and len(sol_list) > 100:
            improved = False
            sol_mask_map = {}
            for e in sol_list:
                sol_mask_map[masks[e]] = e

            # Try to find kernel vectors: subsets that XOR to 0
            # If we find {a, b, c} in sol with mask[a]^mask[b]^mask[c]=0, remove all three
            # Or {a, b} with mask[a]=mask[b], remove both

            # First pass: remove duplicate masks
            new_sol = set()
            seen_masks = {}
            for e in sol_list:
                m = masks[e]
                if m == 0:
                    continue  # doesn't affect XOR
                if m in seen_masks:
                    # Remove the earlier one (or keep smaller)
                    pass
                else:
                    seen_masks[m] = e
                    new_sol.add(e)
            if len(new_sol) < len(sol_list):
                sol_list = list(new_sol)
                improved = True
                continue

            # Try triple removal: find a,b,c in sol with mask[a]^mask[b]^mask[c]=0
            # This means mask[a]^mask[b] = mask[c]
            found_triple = False
            for i in range(len(sol_list)):
                if found_triple:
                    break
                for j in range(i + 1, len(sol_list)):
                    if found_triple:
                        break
                    xor_ab = masks[sol_list[i]] ^ masks[sol_list[j]]
                    if xor_ab in sol_mask_map:
                        k = sol_mask_map[xor_ab]
                        if k != sol_list[i] and k != sol_list[j]:
                            sol_list.remove(sol_list[i])
                            sol_list.remove(sol_list[j])
                            sol_list.remove(k)
                            found_triple = True
                            improved = True
                            break
                    # Also check if xor_ab is 0 (pair cancellation)
                    if xor_ab == 0:
                        sol_list.remove(sol_list[i])
                        sol_list.remove(sol_list[j])
                        found_triple = True
                        improved = True
                        break

        return set(sol_list)

    def find_solution_for_M(M):
        """Find subset of {211,...,M} with XOR=target, size<=100."""
        # Build basis with history
        basis_vec = {}
        basis_hist = {}

        for x in range(211, M + 1):
            v = masks[x]
            h = 1 << (x - 211)
            tv = v
            th = h
            while tv:
                p = tv.bit_length() - 1
                if p in basis_vec:
                    tv ^= basis_vec[p]
                    th ^= basis_hist[p]
                else:
                    basis_vec[p] = tv
                    basis_hist[p] = th
                    break

        # Check if target is in span
        tv = target
        th = 0
        while tv:
            p = tv.bit_length() - 1
            if p in basis_vec:
                tv ^= basis_vec[p]
                th ^= basis_hist[p]
            else:
                return None

        if tv != 0:
            return None

        # Extract solution
        sol_set = set()
        hh = th
        while hh:
            lsb = hh & -hh
            idx = lsb.bit_length() - 1
            sol_set.add(211 + idx)
            hh ^= lsb

        if len(sol_set) <= 100:
            sol = list(sol_set)
            if verify_solution(sol):
                return sol

        # Try to reduce
        reduced = reduce_solution(sol_set, range(211, M + 1))
        if len(reduced) <= 100:
            sol = list(reduced)
            if verify_solution(sol):
                return sol

        return None

    # Main search
    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break

        sol = find_solution_for_M(M)
        if sol is not None:
            return [int(x) for x in sol]

    return [840]
