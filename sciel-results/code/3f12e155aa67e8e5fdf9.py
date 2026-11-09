"""
Combined approach: CSP-style search with forward checking + beam search refinement +
conflict-aware local search. The core idea is to use the CSP solver for global exploration
and beam search for maintaining diverse partial sequences, then refine with targeted
local search that considers future constraint propagation.
"""
import time
import random
import sys
from collections import defaultdict

# Maximum possible sum: 256 elements * 256 value = 65536
_MAX_SUM = 65537
_SUM_BITS = 1 << 65536  # Python handles big integers natively

def _check_valid(seq):
    """Verify strictly increasing in [1,256] and all consecutive block sums distinct."""
    n = len(seq)
    if n == 0:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    # Use bitset for O(1) membership tests
    sum_bits = 0
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s >= _MAX_SUM:
                return False
            bit = 1 << s
            if sum_bits & bit:
                return False
            sum_bits |= bit
    return True

def _compute_new_sums(seq, v):
    """Compute new block sums when appending v to seq."""
    new_sums = [v]
    total = v
    for i in range(len(seq) - 1, -1, -1):
        total += seq[i]
        new_sums.append(total)
    return new_sums

def _compute_new_sum_bits(seq, v):
    """Compute new block sums as a bitmask when appending v to seq."""
    bits = 0
    total = v
    bits |= (1 << total)
    for i in range(len(seq) - 1, -1, -1):
        total += seq[i]
        bits |= (1 << total)
    return bits

def _construct_powers():
    """Direct construction: powers-of-2-like gaps guarantee unique block sums."""
    seq = []
    s = 0
    x = 1
    while x <= 256:
        seq.append(x)
        s += x
        x = s + 1
    return seq

def _construct_greedy_smallest():
    """Deterministic greedy smallest - fast baseline."""
    values, sums = [], set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _construct_greedy_largest():
    """Greedy largest - different bias, may find different optima."""
    values, sums = [], set()
    for x in range(256, 0, -1):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    values.sort()
    return values

def _get_all_sums(seq):
    """Compute all consecutive block sums of seq."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _solve_csp(deadline, seed=None):
    """
    CSP-style solver with forward checking using bitset sum representation.
    """
    if seed is not None:
        random.seed(seed)
    
    best_seq = []
    best_len = 0
    
    def dfs(seq, sum_bits, last_val):
        nonlocal best_seq, best_len
        if time.time() > deadline:
            return
        
        n = len(seq)
        if n > best_len:
            best_len = n
            best_seq = list(seq)
        
        # Branch-and-bound: max possible additional elements
        max_additional = 256 - last_val
        if n + max_additional <= best_len:
            return
        
        # Generate valid candidates with forward checking using bitsets
        candidates = []
        for v in range(last_val + 1, 257):
            new_bits = 0
            total = v
            new_bits |= (1 << total)
            for i in range(len(seq) - 1, -1, -1):
                total += seq[i]
                new_bits |= (1 << total)
            if not (sum_bits & new_bits):
                candidates.append((v, new_bits))
        
        if not candidates:
            return
        
        # Value ordering: mix of smallest (most room) and random for diversity
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > 8:
            ordered = candidates[:4] + random.sample(candidates[4:], min(4, len(candidates) - 4))
        else:
            ordered = candidates
        
        for v, new_bits in ordered:
            if time.time() > deadline:
                return
            dfs(seq + [v], sum_bits | new_bits, v)
    
    dfs([], 0, 0)
    return best_seq

def _beam_search(deadline, beam_width=12):
    """
    Beam search maintaining top-K partial sequences.
    Each beam entry prioritizes length first, then smaller last value for more room.
    """
    best_result = []
    
    # Initial beam with empty sequence
    beam = [(0, 0, [], set())]  # (neg_len, last_val, values, sums)
    
    while beam and time.time() < deadline - 2:
        new_beam = []
        for neg_len, last_val, values, sums in beam:
            # Try extending this sequence
            for x in range(last_val + 1, 257):
                if time.time() > deadline - 2:
                    break
                tails = _compute_new_sums(values, x)
                if not sums.intersection(tails):
                    new_values = values + [x]
                    new_sums = sums | set(tails)
                    new_neg_len = -(len(new_values))
                    new_beam.append((new_neg_len, x, new_values, new_sums))
            
            # Keep current state as candidate
            new_beam.append((neg_len, last_val, values, sums))
        
        if not new_beam:
            break
        
        # Sort and keep top beam_width
        new_beam.sort(key=lambda e: (e[0], e[1]))
        beam = new_beam[:beam_width]
        
        # Track best
        for entry in beam:
            if len(entry[2]) > len(best_result):
                best_result = entry[2]
    
    return best_result

def _solve_csp_multi_restart(deadline, n_restarts=6):
    """
    Multi-restart CSP solver with different random seeds.
    """
    best_seq = _construct_greedy_smallest()
    best_len = len(best_seq)
    
    remaining_time = deadline - time.time()
    if remaining_time <= 1:
        return best_seq
    
    per_restart = remaining_time / max(n_restarts, 1)
    
    for r in range(n_restarts):
        if time.time() > deadline - 2:
            break
        restart_deadline = min(deadline - 2, time.time() + per_restart)
        try:
            result = _solve_csp(restart_deadline, seed=r * 12345 + 678)
            if len(result) > best_len:
                best_seq = result
                best_len = len(result)
        except Exception:
            pass
    
    return best_seq

def _local_search_refine(seq, deadline):
    """
    Fast local search using bitset-based sum checking for O(1) membership tests.
    Tries greedy extension, then backtracks last few positions to find better continuations.
    """
    if not seq:
        return seq
    
    best = list(seq)
    best_len = len(best)
    
    def _try_extend(candidate, deadline):
        """Greedily extend candidate, returning the longest valid extension."""
        if not candidate:
            return candidate
        # Build initial sum bits
        sum_bits = 0
        n = len(candidate)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += candidate[j]
                sum_bits |= (1 << s)
        
        ext = list(candidate)
        while time.time() < deadline - 0.5:
            found = False
            for v in range(ext[-1] + 1, 257):
                # Compute new sums bits
                new_bits = 0
                total = v
                new_bits |= (1 << total)
                for i in range(len(ext) - 1, -1, -1):
                    total += ext[i]
                    new_bits |= (1 << total)
                if not (sum_bits & new_bits):
                    sum_bits |= new_bits
                    ext.append(v)
                    found = True
                    break
            if not found:
                break
        return ext
    
    # Phase 1: Greedy extend
    best = _try_extend(best, deadline)
    best_len = len(best)
    
    # Phase 2: Backtrack last 3 positions and try alternatives
    for pos in range(max(0, len(best) - 3), len(best)):
        if time.time() > deadline - 1:
            break
        prev_val = best[pos-1] if pos > 0 else 0
        for v in range(prev_val + 1, 257):
            if time.time() > deadline - 1:
                break
            candidate = best[:pos] + [v]
            # Quick validity check for the prefix
            if not _check_valid(candidate):
                continue
            extended = _try_extend(candidate, deadline)
            if len(extended) > best_len:
                best = extended
                best_len = len(best)
    
    return best

def _conflict_aware_extension(seq, deadline):
    """
    Conflict-aware extension using bitset sums: for positions near the end,
    try alternative values and measure how many future values they block.
    """
    if not seq:
        return seq
    
    best = list(seq)
    best_len = len(best)
    
    def _count_blocked_values(sum_bits, last_val):
        """Count how many values > last_val are blocked by existing sums."""
        blocked = 0
        for v in range(last_val + 1, 257):
            # Check if v alone conflicts
            if sum_bits & (1 << v):
                blocked += 1
        return blocked
    
    # For positions near the end, try alternatives
    for pos in range(max(0, len(best) - 4), len(best)):
        if time.time() > deadline - 2:
            break
        
        original_val = best[pos]
        prev_val = best[pos-1] if pos > 0 else 0
        
        for v in range(prev_val + 1, 257):
            if v == original_val:
                continue
            if time.time() > deadline - 2:
                break
            
            candidate = best[:pos] + [v]
            if not _check_valid(candidate):
                continue
            
            # Compute sum bits for this candidate
            sum_bits = 0
            n = len(candidate)
            for i in range(n):
                s = 0
                for j in range(i, n):
                    s += candidate[j]
                    sum_bits |= (1 << s)
            
            # Try to extend greedily
            ext = list(candidate)
            ext_bits = sum_bits
            while time.time() < deadline - 2:
                found = False
                for w in range(ext[-1] + 1, 257):
                    new_bits = 0
                    total = w
                    new_bits |= (1 << total)
                    for i in range(len(ext) - 1, -1, -1):
                        total += ext[i]
                        new_bits |= (1 << total)
                    if not (ext_bits & new_bits):
                        ext_bits |= new_bits
                        ext.append(w)
                        found = True
                        break
                if not found:
                    break
            
            if len(ext) > best_len:
                best = ext
                best_len = len(best)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Multiple baselines
    best = _construct_greedy_smallest()
    best_len = len(best)
    
    try:
        greedy_large = _construct_greedy_largest()
        if len(greedy_large) > best_len:
            best = greedy_large
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 15:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Multi-restart CSP solver
    try:
        csp_result = _solve_csp_multi_restart(deadline - 10, n_restarts=6)
        if len(csp_result) > best_len:
            best = csp_result
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 10:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Beam search for diverse partial sequences
    try:
        beam_result = _beam_search(deadline - 5, beam_width=12)
        if len(beam_result) > best_len:
            best = beam_result
            best_len = len(best)
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        if not _check_valid(best):
            best = _construct_powers()
        return best
    
    # Conflict-aware extension
    try:
        conflict_result = _conflict_aware_extension(best, deadline - 2)
        if len(conflict_result) > best_len:
            best = conflict_result
            best_len = len(best)
    except Exception:
        pass
    
    # Local search refinement on the best sequence found
    try:
        refined = _local_search_refine(best, deadline)
        if len(refined) > best_len:
            best = refined
            best_len = len(best)
    except Exception:
        pass
    
    # Final verification
    if not _check_valid(best):
        best = _construct_powers()
    
    return best
