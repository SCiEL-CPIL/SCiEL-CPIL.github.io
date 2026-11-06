"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Sieve primes up to 10000
    N = 10000
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0] = is_prime[1] = 0
    for x in range(2, int(N ** 0.5) + 1):
        if is_prime[x]:
            is_prime[x * x::x] = bytearray(len(is_prime[x * x::x]))
    
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return primes
    
    # Build prime_index for O(1) lookup
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # DP approach:
    # dp[i][g] = max length of valid sequence ending at prime i with last gap exactly g
    # g ranges from 1 to N (max possible gap)
    # But we only need gaps that actually occur between pairs of primes
    
    # Alternative: dp[i] = dict mapping last_gap -> max_length
    # For each prime i, for each previous prime j < i:
    #   gap = primes[i] - primes[j]
    #   For each (last_gap_j, len_j) in dp[j]:
    #     if gap >= last_gap_j:
    #       dp[i][gap] = max(dp[i].get(gap, 0), len_j + 1)
    #   Also: dp[i][gap] = max(dp[i].get(gap, 0), 1)  # sequence of just [primes[i]]
    # But this is O(n^2 * avg_dict_size) which could be slow
    
    # Better: dp[i] = list of (last_gap, max_length) Pareto-optimal pairs
    # A pair (g1, l1) dominates (g2, l2) if g1 <= g2 and l1 >= l2
    # We keep only non-dominated pairs
    
    # Let's try a different DP:
    # dp[i] = max length of valid sequence ending at primes[i]
    # To compute dp[i], we need: for each j < i, if there exists a valid sequence ending at j
    # with last gap <= primes[i] - primes[j], then we can extend it.
    # But we don't track the last gap of the best sequence ending at j.
    
    # Key insight: for a fixed ending prime i, the best predecessor j should minimize
    # primes[i] - primes[j] subject to the constraint that the sequence ending at j has
    # last gap <= primes[i] - primes[j].
    
    # Let's use: dp[i] = max over j < i of (1 + max_len_ending_at_j_with_last_gap_le_gap)
    # where gap = primes[i] - primes[j]
    
    # We can precompute for each prime j, a data structure that given a threshold g,
    # returns the max length of a valid sequence ending at j with last gap <= g.
    
    # For each j, let's maintain a sorted list of (last_gap, max_length) Pareto-optimal pairs.
    # Then query is binary search.
    
    # But building this naively is O(n^2 * log(n)) which is about 1.2M * 20 = 24M ops. Should be OK.
    
    # Actually, let's think about it differently.
    # For each pair (j, i) with j < i, gap = primes[i] - primes[j].
    # We want to find the max length of a valid sequence ending at j with last gap <= gap.
    # Then dp[i] = max over all j < i of (1 + that value), and also dp[i] >= 1.
    
    # To make this efficient, for each j, we can precompute an array best_le[j][g] = max length
    # of valid sequence ending at j with last gap <= g, for g = 0, 1, ..., N.
    # But that's n * N = 1260 * 10000 = 12.6M entries, each an int. That's about 100MB. Might be too much.
    
    # Alternative: for each j, store a list of (gap, max_len) sorted by gap, and for each gap value,
    # store the max length for last_gap <= that gap. Then binary search.
    
    # Let me try a simpler approach first: since n is only 1260, O(n^2) is about 1.6M, which is fast.
    # For each i, iterate over all j < i, and for each j, we need to know the max length of a
    # valid sequence ending at j with last gap <= primes[i] - primes[j].
    
    # Let's maintain for each j a dictionary: dp_dict[j] = {last_gap: max_length}
    # And also a prefix-max structure: for each j, sort the keys and compute prefix max.
    
    # Actually, let's just do O(n^2) with a simple approach:
    # dp[i] = 1 (base case)
    # For each i from 1 to n-1:
    #   For each j from 0 to i-1:
    #     gap = primes[i] - primes[j]
    #     # Find max length of sequence ending at j with last gap <= gap
    #     # We can check all entries in dp_dict[j]
    #     best_prev = 0
    #     for (lg, ll) in dp_dict[j]:
    #       if lg <= gap:
    #         best_prev = max(best_prev, ll)
    #     if best_prev > 0:
    #       dp[i] = max(dp[i], best_prev + 1)
    #     # Also, we can start a new sequence at i with gap from j
    #     # The sequence [primes[j], primes[i]] has last gap = gap
    #     # So dp_dict[i][gap] = max(dp_dict[i].get(gap, 0), best_prev + 1) if best_prev > 0
    #     # Or if best_prev == 0, then [primes[i]] alone has length 1, but we need to track
    #     # the "last gap" for extension. For a single element, there's no last gap.
    #     # Let's define: for a sequence of length 1, last_gap = 0 (so it can be extended by any gap >= 0)
    #     # Actually, that's a good convention.
    
    # Let me redefine:
    # dp_dict[i] = dict: last_gap -> max_length, where last_gap is the gap between the last two elements
    # For a sequence of length 1, we use last_gap = 0.
    # For a sequence [a, b] with b - a = g, last_gap = g.
    
    # dp_dict[i][0] = 1  (sequence [primes[i]] alone)
    # For j < i:
    #   gap = primes[i] - primes[j]
    #   For each (lg, ll) in dp_dict[j]:
    #     if lg <= gap:
    #       new_len = ll + 1
    #       dp_dict[i][gap] = max(dp_dict[i].get(gap, 0), new_len)
    
    # Then dp[i] = max(dp_dict[i].values())
    
    # The total work is sum over i of (i * avg_dict_size[j]). 
    # avg_dict_size might be small since we only add entries when we find valid extensions.
    
    # Let's implement this.
    
    dp_dict = [dict() for _ in range(n)]
    dp = [1] * n  # dp[i] = max length ending at i
    
    for i in range(n):
        dp_dict[i][0] = 1  # sequence of just primes[i]
        for j in range(i):
            gap = primes[i] - primes[j]
            # For each (lg, ll) in dp_dict[j], if lg <= gap, extend
            for lg, ll in dp_dict[j].items():
                if lg <= gap:
                    new_len = ll + 1
                    if new_len > dp_dict[i].get(gap, 0):
                        dp_dict[i][gap] = new_len
                    if new_len > dp[i]:
                        dp[i] = new_len
        
        if time.time() - t0 > DEADLINE:
            # Return best so far
            best_i = max(range(i + 1), key=lambda x: dp[x])
            # Reconstruct
            return _reconstruct(primes, dp_dict, dp, best_i, prime_index)
    
    # Find best
    best_i = max(range(n), key=lambda x: dp[x])
    
    # Reconstruct the sequence
    seq = _reconstruct(primes, dp_dict, dp, best_i, prime_index)
    
    return seq


def _reconstruct(primes, dp_dict, dp, best_i, prime_index):
    """Reconstruct the sequence ending at best_i."""
    n = len(primes)
    seq = [primes[best_i]]
    curr_i = best_i
    curr_len = dp[curr_i]
    
    while curr_len > 1:
        # Find predecessor j and the gap that was used
        # We need to find j < curr_i and a gap g such that:
        #   primes[curr_i] - primes[j] == g
        #   dp_dict[j] has an entry (lg, curr_len - 1) with lg <= g
        #   And the entry for g in dp_dict[curr_i] has value curr_len
        #   Actually, we need to find the specific transition.
        
        target_len = curr_len - 1
        # The last gap in the sequence ending at curr_i with length curr_len
        # is some value g such that dp_dict[curr_i][g] >= curr_len
        # But there might be multiple such g. We need to find one that is consistent
        # with a predecessor.
        
        # Let's find all g such that dp_dict[curr_i].get(g, 0) >= curr_len
        candidate_gaps = [g for g, l in dp_dict[curr_i].items() if l >= curr_len]
        
        found = False
        for g in candidate_gaps:
            prev_prime_val = primes[curr_i] - g
            if prev_prime_val not in prime_index:
                continue
            prev_j = prime_index[prev_prime_val]
            # Check if there's a valid sequence ending at prev_j with last gap <= g and length target_len
            # We need dp_dict[prev_j] to have some entry (lg, ll) with lg <= g and ll == target_len
            # Actually, we need ll >= target_len, but since we're reconstructing the exact path,
            # we need ll == target_len or we can take a longer one... 
            # Hmm, this is tricky. Let me think.
            
            # Actually, the dp_dict stores the MAX length for each gap value. So if dp_dict[prev_j][lg] = L,
            # that means there's a sequence of length L ending at prev_j with last gap lg.
            # When we extended it to curr_i with gap g >= lg, we got length L+1.
            # So for the reconstruction, we need: there exists lg in dp_dict[prev_j] such that
            #   lg <= g and dp_dict[prev_j][lg] == target_len
            # And then the predecessor is prev_j with last gap lg.
            
            valid = False
            for lg, ll in dp_dict[prev_j].items():
                if lg <= g and ll == target_len:
                    valid = True
                    break
            
            if valid:
                seq.append(primes[prev_j])
                curr_i = prev_j
                curr_len = target_len
                found = True
                break
        
        if not found:
            # Fallback: just break
            break
    
    seq.reverse()
    return seq
# EVOLVE-BLOCK-END
