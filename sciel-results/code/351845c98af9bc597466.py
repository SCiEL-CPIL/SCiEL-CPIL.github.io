import time
from math import isqrt
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    # DP to get initial best sequence
    # dp[i] stores list of (gap, length, prev_idx, prev_idx_in_dp)
    # sorted by gap, pruned to max length for increasing gaps
    dp = [[] for _ in range(n)]
    
    def get_max_len(j, g):
        if not dp[j]:
            return 0, -1
        lo, hi = 0, len(dp[j]) - 1
        res_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if dp[j][mid][0] <= g:
                res_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if res_idx == -1:
            return 0, -1
        return dp[j][res_idx][1], res_idx

    for i in range(n):
        if time.time() > deadline:
            break
        candidates = [(0, 1, -1, -1)]
        for j in range(i):
            g = primes[i] - primes[j]
            prev_len, prev_idx = get_max_len(j, g)
            if prev_len > 0:
                candidates.append((g, prev_len + 1, j, prev_idx))
        
        candidates.sort(key=lambda x: x[0])
        pruned = []
        max_len_so_far = 0
        for c in candidates:
            g, L, j, idx = c
            if L > max_len_so_far:
                pruned.append(c)
                max_len_so_far = L
            elif L == max_len_so_far and g < pruned[-1][0]:
                pruned[-1] = c
        dp[i] = pruned

    best_len = 0
    best_i = -1
    for i in range(n):
        if dp[i]:
            L = dp[i][-1][1]
            if L > best_len:
                best_len = L
                best_i = i

    if best_i == -1:
        return [primes[0]] if n > 0 else []

    # Reconstruct initial sequence
    seq = []
    curr_i = best_i
    while curr_i != -1:
        seq.append(curr_i)
        state = dp[curr_i][-1]
        curr_i = state[2]
    seq.reverse()
    
    # Local Search: Backward Push to minimize final gap and allow extension
    # seq is list of indices into primes
    # We want to maximize length.
    # Strategy:
    # 1. Try to extend.
    # 2. If fail, try to increase S[k-1], then S[k-2], etc., to reduce gaps.
    
    # Helper to find largest prime < upper_bound and > lower_bound
    # such that p <= (lower_bound + upper_bound) / 2 ?
    # No, the constraint is g_prev <= g_curr.
    # g_prev = p - prev_val
    # g_curr = next_val - p
    # p - prev_val <= next_val - p  => 2p <= prev_val + next_val
    # So p <= (prev_val + next_val) / 2.
    
    def try_increase(idx, seq):
        # Try to increase seq[idx]
        # seq[idx] is prime index.
        # prev_val = primes[seq[idx-1]] if idx > 0 else 0 (no constraint on first gap from left? No, just increasing)
        # Actually, for idx=0, we can increase it to any prime < seq[1].
        # But increasing seq[0] reduces g_0.
        # For idx > 0, we have constraint from left: g_{idx-1} <= g_idx.
        # g_{idx-1} = p - prev_val
        # g_idx = next_val - p
        # Constraint: p - prev_val <= next_val - p  => p <= (prev_val + next_val) / 2.
        
        curr_prime_val = primes[seq[idx]]
        next_prime_val = primes[seq[idx+1]]
        
        if idx == 0:
            prev_prime_val = 0 # No lower bound on gap, just p > 0? No, p must be prime.
            # Constraint: p < next_prime_val.
            # We want largest p < next_prime_val.
            # But we also want p to be as large as possible to minimize g_0.
            # So just largest prime < next_prime_val.
            # However, we also need p > 0.
            # And p must be > any previous? No, it's the first.
            # But wait, if we increase p, g_0 decreases.
            # Is there a lower bound on g_0? No.
            # So we want the largest prime < next_prime_val.
            # But we also need p to be a valid prime.
            # And we must maintain p < next_prime_val.
            # Also, we should probably keep p as large as possible.
            
            # Find largest prime < next_prime_val
            idx_next = seq[idx+1]
            # primes are sorted.
            # bisect_left for next_prime_val gives idx_next.
            # So largest prime < next_prime_val is primes[idx_next - 1].
            # But we need to ensure it's > 0.
            # And we need to ensure it's actually larger than current curr_prime_val?
            # If curr_prime_val is already the largest, we can't increase.
            
            candidate_idx = idx_next - 1
            if candidate_idx < 0:
                return False
            candidate_prime = primes[candidate_idx]
            if candidate_prime > curr_prime_val:
                # Check if this is valid.
                # For idx=0, no left constraint.
                seq[idx] = candidate_idx
                return True
            return False
            
        else:
            prev_prime_val = primes[seq[idx-1]]
            # Constraint: p <= (prev_prime_val + next_prime_val) / 2
            max_allowed_p = (prev_prime_val + next_prime_val) / 2.0
            
            # Find largest prime < next_prime_val and <= max_allowed_p
            # And > curr_prime_val
            
            # Upper bound for search: min(next_prime_val, max_allowed_p + 1)
            # We want largest prime < next_prime_val.
            # Let's find index of next_prime_val.
            idx_next = seq[idx+1]
            # Candidates are primes[0 : idx_next]
            # We want largest index j < idx_next such that primes[j] <= max_allowed_p
            # And primes[j] > curr_prime_val
            
            # Binary search for max_allowed_p in primes
            # bisect_right(max_allowed_p) gives first index > max_allowed_p.
            # So valid indices are < that.
            # But also must be < idx_next.
            
            # Actually, we can just iterate backwards from idx_next - 1?
            # N is small (1229). Iteration is fine.
            # But let's use bisect for elegance.
            
            # Find rightmost prime <= max_allowed_p
            # bisect_right returns insertion point to maintain order.
            # If we want largest prime <= X, we use bisect_right(X) - 1.
            # But we also need it to be < next_prime_val.
            # Since max_allowed_p < next_prime_val (usually), this is handled.
            # Actually (prev + next)/2 < next always if prev < next.
            
            target = int(max_allowed_p)
            # Find rightmost prime <= target
            # bisect_right(primes, target)
            pos = bisect.bisect_right(primes, target)
            # pos is index of first prime > target.
            # So pos - 1 is largest prime <= target.
            candidate_idx = pos - 1
            
            # Ensure candidate_idx is within bounds for this position
            # It must be < idx_next (strictly less than next prime)
            # And > idx_prev (strictly greater than prev prime)
            # And > current index (to ensure increase)
            
            if candidate_idx < seq[idx-1]:
                # The largest allowed prime is not even greater than the previous prime?
                # This implies no valid prime exists that satisfies the gap constraint and ordering.
                # Wait, if candidate_idx < seq[idx-1], then primes[candidate_idx] <= primes[seq[idx-1]].
                # We need p > prev_prime_val.
                # So no candidate.
                return False
                
            if candidate_idx >= idx_next:
                # This shouldn't happen if max_allowed_p < next_prime_val
                # But just in case, clamp to idx_next - 1
                candidate_idx = idx_next - 1
                # But we must check if primes[candidate_idx] <= max_allowed_p
                if primes[candidate_idx] > max_allowed_p:
                    return False

            # Now check if this candidate is an improvement
            if candidate_idx > seq[idx]:
                # We can increase seq[idx] to candidate_idx
                seq[idx] = candidate_idx
                return True
            else:
                return False

    # Main local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        L = len(seq)
        
        # 1. Try to extend
        # Last prime is primes[seq[-1]]
        # Last gap is primes[seq[-1]] - primes[seq[-2]] if L > 1 else 0
        if L == 1:
            last_gap = 0
            last_prime_val = primes[seq[0]]
            # Find smallest prime > last_prime_val
            # Any prime > last_prime_val works since gap >= 0
            idx_last = seq[0]
            # Next prime is primes[idx_last + 1] if exists
            if idx_last + 1 < n:
                seq.append(idx_last + 1)
                improved = True
                continue
        else:
            last_prime_val = primes[seq[-1]]
            prev_prime_val = primes[seq[-2]]
            last_gap = last_prime_val - prev_prime_val
            
            # Find smallest prime q > last_prime_val such that q - last_prime_val >= last_gap
            # q >= last_prime_val + last_gap
            target_q = last_prime_val + last_gap
            # Find first prime >= target_q
            pos = bisect.bisect_left(primes, target_q)
            if pos < n:
                # Check if this prime is > last_prime_val (it is, since target_q > last_prime_val)
                seq.append(pos)
                improved = True
                continue
        
        # 2. If cannot extend, try to "tighten" from the end
        # Try to increase S[k-1], then S[k-2], etc.
        # We do this in a single pass from end to start.
        # If we increase S[i], it might allow increasing S[i+1] (which was previously blocked)?
        # Actually, increasing S[i] reduces g_i.
        # This makes it easier to satisfy g_{i-1} <= g_i? No, g_{i-1} is fixed (from left).
        # It makes it easier to satisfy g_i <= g_{i+1}?
        # g_i is smaller. So g_i <= g_{i+1} is easier to satisfy.
        # So increasing S[i] relaxes the constraint on S[i+1].
        # So we should try to increase S[i] and then re-optimize S[i+1]?
        # Or just increase S[i] and then try to extend again?
        
        # Let's try to increase indices from L-2 down to 0.
        # If we increase S[L-2], g_{L-1} decreases.
        # This might allow S[L-1] to be increased?
        # S[L-1] is the last element. We can't increase it without breaking the "last" status?
        # No, S[L-1] is fixed as the end of the sequence.
        # But we can increase S[L-2] to reduce g_{L-1}.
        # Then we try to extend again.
        
        # What if increasing S[L-2] allows us to increase S[L-1]?
        # S[L-1] is the last prime. If we increase S[L-1], the sequence ends at a larger prime.
        # This might allow further extension.
        # But increasing S[L-1] increases g_{L-1} (if S[L-2] fixed).
        # We want to *decrease* g_{L-1} to allow extension.
        # So we should NOT increase S[L-1]. We should increase S[L-2].
        
        # So:
        # Try to increase S[L-2].
        # If successful, loop back to "Try to extend".
        # If not, try to increase S[L-3].
        # ...
        
        # But increasing S[L-3] reduces g_{L-2}.
        # This relaxes constraint on S[L-2].
        # So we might then be able to increase S[L-2].
        
        # So a single backward pass:
        # for i in range(L-2, -1, -1):
        #     if try_increase(i):
        #         improved = True
        #         break # Restart extension check
        
        # Let's implement this.
        for i in range(L-2, -1, -1):
            if try_increase(i, seq):
                improved = True
                break

    return [primes[i] for i in seq]
