"""Rewritten program: DP over primes with (len, last_gap, last_prime) state, then bounded local search."""
# EVOLVE-BLOCK-START
import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, 101):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)

    def is_valid(seq):
        if len(seq) < 2:
            return True
        prev_gap = 0
        for i in range(1, len(seq)):
            gap = seq[i] - seq[i - 1]
            if gap < prev_gap:
                return False
            prev_gap = gap
        return True

    # DP: dp[i] = (length, last_gap) for the best sequence ending at prime i
    # We iterate i from 0..n-1. For each i, look at all j < i where primes[i] - primes[j] >= last_gap_j.
    # We want to maximize length. On tie, prefer smaller last_gap (tighter, more extensible).
    # On further tie, prefer larger j (later predecessor) - actually let's think:
    # If two predecessors j1 < j2 both give the same length and the same new last_gap, 
    # then the sequence ending at i has the same last gap either way. The difference is in
    # the earlier part of the sequence. For future extensibility, what matters is the last gap,
    # which is the same. So it doesn't matter which predecessor we pick.
    # But wait: the "last_gap" stored in dp[j] is the gap before j, not the gap into j.
    # Let me re-think the state.
    
    # State: dp[i] = (best_len, best_prev_gap) where best_prev_gap is the gap between
    # dp[i]'s predecessor and prime[i]. When we extend from j to i, the new gap is
    # primes[i] - primes[j], and we need primes[i] - primes[j] >= dp[j].prev_gap.
    # The new state's prev_gap becomes primes[i] - primes[j].
    
    # So: for each i, for each j < i:
    #   gap = primes[i] - primes[j]
    #   if gap >= dp[j].prev_gap:
    #       candidate_len = dp[j].len + 1
    #       candidate_gap = gap
    #       Update dp[i] if candidate_len > dp[i].len, or
    #       candidate_len == dp[i].len and candidate_gap < dp[i].prev_gap
    
    dp_len = [1] * n
    dp_gap = [0] * n  # prev_gap for the sequence ending at i
    dp_prev = [-1] * n
    
    for i in range(n):
        best_len = 1
        best_gap = 0
        best_j = -1
        for j in range(i):
            gap = primes[i] - primes[j]
            if gap >= dp_gap[j]:
                new_len = dp_len[j] + 1
                new_gap = gap
                if new_len > best_len or (new_len == best_len and new_gap < best_gap):
                    best_len = new_len
                    best_gap = new_gap
                    best_j = j
        dp_len[i] = best_len
        dp_gap[i] = best_gap
        dp_prev[i] = best_j
        if time.time() > deadline:
            break
    
    # Find best endpoint
    max_len = 0
    end_idx = 0
    for i in range(n):
        if dp_len[i] > max_len or (dp_len[i] == max_len and dp_gap[i] < dp_gap[end_idx]):
            max_len = dp_len[i]
            end_idx = i
    
    # Backtrack
    seq = []
    curr = end_idx
    while curr != -1:
        seq.append(primes[curr])
        curr = dp_prev[curr]
    seq.reverse()
    
    best_seq = seq[:]
    current_seq = seq[:]
    
    # Local search to potentially improve
    # Try inserting primes at various positions
    in_seq = set(current_seq)
    
    def try_insert_at(seq, new_p, pos):
        # pos is where to insert (0..len(seq))
        # Check increasing order
        if pos > 0 and new_p <= seq[pos - 1]:
            return None
        if pos < len(seq) and new_p >= seq[pos]:
            return None
        new_seq = seq[:pos] + [new_p] + seq[pos:]
        if is_valid(new_seq):
            return new_seq
        return None
    
    no_improve = 0
    max_no_improve = 5000
    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random prime not in current_seq
        p_idx = random.randint(0, n - 1)
        p = primes[p_idx]
        if p in in_seq:
            continue
        
        # Find possible insertion positions by value
        pos = bisect.bisect_left(current_seq, p)
        found = False
        for candidate_pos in [pos, pos - 1]:
            if candidate_pos < 0 or candidate_pos > len(current_seq):
                continue
            new_seq = try_insert_at(current_seq, p, candidate_pos)
            if new_seq is not None:
                current_seq = new_seq
                in_seq = set(current_seq)
                no_improve = 0
                if len(current_seq) > len(best_seq):
                    best_seq = current_seq[:]
                found = True
                break
        if not found:
            no_improve += 1
    
    # Final validation
    if not is_valid(best_seq):
        return [2, 3]
    
    return best_seq
# EVOLVE-BLOCK-END
