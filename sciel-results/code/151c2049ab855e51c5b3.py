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

    def is_valid_fast(seq):
        """Check non-decreasing gaps in O(len) with early exit."""
        for i in range(1, len(seq) - 1):
            if seq[i + 1] - seq[i] < seq[i] - seq[i - 1]:
                return False
        return True

    # DP with refined tie-breaking:
    # Primary: maximize length
    # Secondary: minimize gap into i (more extensible)
    # Tertiary: minimize dp_gap[j] of predecessor (prefix also more extensible)
    dp_len = [1] * n
    dp_gap = [0] * n  # gap between predecessor and prime[i]
    dp_prev = [-1] * n

    for i in range(n):
        best_len = 1
        best_gap = 0
        best_j = -1
        best_pred_gap = 0
        for j in range(i):
            gap = primes[i] - primes[j]
            if gap >= dp_gap[j]:
                new_len = dp_len[j] + 1
                new_gap = gap
                if (new_len > best_len or
                    (new_len == best_len and new_gap < best_gap) or
                    (new_len == best_len and new_gap == best_gap and dp_gap[j] < best_pred_gap)):
                    best_len = new_len
                    best_gap = new_gap
                    best_j = j
                    best_pred_gap = dp_gap[j]
        dp_len[i] = best_len
        dp_gap[i] = best_gap
        dp_prev[i] = best_j
        if time.time() > deadline:
            break

    # Find best endpoint: maximize length, then minimize last gap
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
    best_len_found = len(best_seq)

    # Local search: systematic + stochastic
    in_seq = set(current_seq)

    def try_insert_at(seq, new_p, pos):
        if pos > 0 and new_p <= seq[pos - 1]:
            return None
        if pos < len(seq) and new_p >= seq[pos]:
            return None
        new_seq = seq[:pos] + [new_p] + seq[pos:]
        if is_valid_fast(new_seq):
            return new_seq
        return None

    def try_remove_at(seq, pos):
        """Remove element at pos and check validity."""
        if pos < 0 or pos >= len(seq):
            return None
        new_seq = seq[:pos] + seq[pos + 1:]
        if is_valid_fast(new_seq):
            return new_seq
        return None

    def try_swap_remove_insert(seq, remove_pos, new_p, insert_pos):
        """Remove element at remove_pos, insert new_p at insert_pos (adjusted)."""
        # First remove
        removed = seq[remove_pos]
        seq_without = seq[:remove_pos] + seq[remove_pos + 1:]
        # Adjust insert_pos if needed
        if insert_pos > remove_pos:
            insert_pos -= 1
        # Now insert new_p into seq_without
        if insert_pos < 0 or insert_pos > len(seq_without):
            return None
        if insert_pos > 0 and new_p <= seq_without[insert_pos - 1]:
            return None
        if insert_pos < len(seq_without) and new_p >= seq_without[insert_pos]:
            return None
        new_seq = seq_without[:insert_pos] + [new_p] + seq_without[insert_pos:]
        if is_valid_fast(new_seq):
            return new_seq
        return None

    no_improve = 0
    max_no_improve = 10000
    phase = 0  # 0: systematic insert, 1: random insert, 2: swap moves

    while time.time() < deadline - 2.0:
        if phase == 0:
            # Systematic: try each out-of-sequence prime at its natural position
            improved = False
            for p_idx in range(n):
                p = primes[p_idx]
                if p in in_seq:
                    continue
                pos = bisect.bisect_left(current_seq, p)
                for candidate_pos in [pos, pos - 1]:
                    if candidate_pos < 0 or candidate_pos > len(current_seq):
                        continue
                    new_seq = try_insert_at(current_seq, p, candidate_pos)
                    if new_seq is not None:
                        current_seq = new_seq
                        in_seq = set(current_seq)
                        no_improve = 0
                        if len(current_seq) > best_len_found:
                            best_seq = current_seq[:]
                            best_len_found = len(best_seq)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                phase = 1
                no_improve = 0
                continue
        elif phase == 1:
            # Random insertion
            p_idx = random.randint(0, n - 1)
            p = primes[p_idx]
            if p in in_seq:
                no_improve += 1
                continue
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
                    if len(current_seq) > best_len_found:
                        best_seq = current_seq[:]
                        best_len_found = len(best_seq)
                    found = True
                    break
            if not found:
                no_improve += 1
                if no_improve > 5000:
                    phase = 2
                    no_improve = 0
        else:
            # Phase 2: swap-based moves (remove one, insert another)
            if random.random() < 0.3:
                # Try removing a random element and inserting a different prime
                if len(current_seq) <= 1:
                    no_improve += 1
                    continue
                rem_pos = random.randint(0, len(current_seq) - 1)
                # Try inserting a random prime at a random valid position
                p_idx = random.randint(0, n - 1)
                new_p = primes[p_idx]
                if new_p in in_seq and new_p != current_seq[rem_pos]:
                    no_improve += 1
                    continue
                # Try several insertion positions
                found = False
                for _ in range(3):
                    ins_pos = random.randint(0, len(current_seq))
                    new_seq = try_swap_remove_insert(current_seq, rem_pos, new_p, ins_pos)
                    if new_seq is not None and len(new_seq) >= len(current_seq):
                        current_seq = new_seq
                        in_seq = set(current_seq)
                        no_improve = 0
                        if len(current_seq) > best_len_found:
                            best_seq = current_seq[:]
                            best_len_found = len(best_seq)
                        found = True
                        break
                if not found:
                    no_improve += 1
            else:
                # Try removing an element that might be blocking
                if len(current_seq) > 2:
                    rem_pos = random.randint(1, len(current_seq) - 2)  # avoid ends
                    new_seq = try_remove_at(current_seq, rem_pos)
                    if new_seq is not None:
                        # Now try to insert a prime at the freed-up position
                        p_idx = random.randint(0, n - 1)
                        new_p = primes[p_idx]
                        if new_p not in in_seq:
                            pos = bisect.bisect_left(new_seq, new_p)
                            for candidate_pos in [pos, pos - 1]:
                                if candidate_pos < 0 or candidate_pos > len(new_seq):
                                    continue
                                final_seq = try_insert_at(new_seq, new_p, candidate_pos)
                                if final_seq is not None and len(final_seq) > len(current_seq):
                                    current_seq = final_seq
                                    in_seq = set(current_seq)
                                    no_improve = 0
                                    if len(current_seq) > best_len_found:
                                        best_seq = current_seq[:]
                                        best_len_found = len(best_seq)
                                    break
                            else:
                                # Just accept the removal if it enables future growth
                                # (don't actually accept shorter sequences as best)
                                pass
                        no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            if no_improve > 8000:
                break

    # Final validation
    if not is_valid(best_seq):
        return [2, 3]

    return best_seq
# EVOLVE-BLOCK-END
