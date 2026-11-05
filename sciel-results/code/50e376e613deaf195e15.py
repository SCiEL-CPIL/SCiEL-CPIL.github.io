"""
Beam search + local search for longest strictly increasing sequence in [1,256]
with all consecutive block sums distinct.

Structural approach:
- Beam search maintaining top-K partial sequences, scored by potential for extension
- Local search that removes elements to free up sum space
- Suffix-sum tracking for efficient conflict checking
"""
import time
import random
import heapq

def _compute_all_sums(seq):
    """Compute set of all non-empty consecutive block sums."""
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _suffix_sums(seq):
    """Return list of suffix sums ending at last element."""
    tails = []
    total = 0
    for a in reversed(seq):
        total += a
        tails.append(total)
    tails.reverse()  # now tails[0] = sum of whole seq, tails[-1] = last element
    return tails

def _can_append(x, seq, occupied_sums):
    """Check if x can be appended to seq without creating duplicate block sums."""
    if seq and x <= seq[-1]:
        return False, []
    # Compute new block sums ending at x
    new_sums = []
    total = x
    new_sums.append(total)
    for a in reversed(seq):
        total += a
        new_sums.append(total)
    # Check for conflicts
    for s in new_sums:
        if s in occupied_sums:
            return False, new_sums
    return True, new_sums

def _greedy_extend(seq, occupied_sums, candidates, rng, max_steps=200):
    """Greedily extend seq using given candidates."""
    seq = list(seq)
    occ = set(occupied_sums)
    for x in candidates:
        if not seq or x > seq[-1]:
            ok, new_sums = _can_append(x, seq, occ)
            if ok:
                seq.append(x)
                occ.update(new_sums)
    return seq, occ

def _score_sequence(seq, occupied_sums):
    """
    Score a sequence for beam search:
    - Primary: length
    - Secondary: how much 'space' remains for future extensions
    """
    if not seq:
        return (0, 0)
    length = len(seq)
    last = seq[-1]
    # Count how many values > last are available
    available_after = 256 - last
    # Penalize if occupied sums are clustered (less future flexibility)
    # Approximate: fewer occupied sums in high range = better
    high_sum_count = sum(1 for s in occupied_sums if s > 500)
    score = (length, available_after, -high_sum_count)
    return score

def _beam_search(time_limit, rng, beam_width=20):
    """
    Beam search: maintain top beam_width partial sequences.
    State: (seq, occupied_sums)
    """
    start_time = time.time()
    
    # Initialize beam with single elements
    beam = []
    for x in range(1, 257):
        seq = [x]
        sums = {x}
        beam.append((seq, sums))
    
    # Sort initial beam by score
    beam.sort(key=lambda item: _score_sequence(item[0], item[1]), reverse=True)
    beam = beam[:beam_width]
    
    best_seq = beam[0][0] if beam else []
    best_sums = beam[0][1] if beam else set()
    best_score = _score_sequence(best_seq, best_sums) if best_seq else (0, 0)
    
    for step in range(30):  # Max 30 iterations
        if time.time() > time_limit:
            break
        
        new_beam = []
        seen_prefixes = set()
        
        for seq, sums in beam:
            if time.time() > time_limit:
                break
            
            last = seq[-1] if seq else 0
            # Try to extend with values > last
            # Use a subset of candidates to keep it fast
            candidates = list(range(last + 1, 257))
            
            # Limit candidates per beam element for speed
            if len(candidates) > 50:
                # Take a random sample biased toward spread-out values
                if rng.random() < 0.5:
                    # Take evenly spaced
                    step_size = max(1, len(candidates) // 20)
                    candidates = candidates[::step_size]
                else:
                    candidates = rng.sample(candidates, min(20, len(candidates)))
            
            for x in candidates:
                ok, new_sums = _can_append(x, seq, sums)
                if ok:
                    new_seq = seq + [x]
                    new_occ = sums | set(new_sums)
                    prefix = tuple(new_seq)
                    if prefix not in seen_prefixes:
                        seen_prefixes.add(prefix)
                        new_beam.append((new_seq, new_occ))
        
        if not new_beam:
            break
        
        # Sort new beam by score
        new_beam.sort(key=lambda item: _score_sequence(item[0], item[1]), reverse=True)
        new_beam = new_beam[:beam_width]
        
        # Update best
        for seq, sums in new_beam:
            sc = _score_sequence(seq, sums)
            if sc > best_score:
                best_seq = seq
                best_sums = sums
                best_score = sc
        
        beam = new_beam
        if not beam:
            break
    
    return best_seq, best_sums

def _local_search_polish(seq, sums, time_limit, rng):
    """
    Local search: try removing elements and re-extending.
    """
    best_seq = list(seq)
    best_sums = set(sums)
    best_len = len(best_seq)
    start_time = time.time()
    
    while time.time() < time_limit:
        # Try removing a random element (not the first or last to keep valid)
        if len(best_seq) <= 2:
            break
        
        # Pick a random position to remove
        pos = rng.randint(1, len(best_seq) - 2)  # Don't remove first or last
        removed = best_seq[pos]
        truncated = best_seq[:pos] + best_seq[pos+1:]
        
        # Recompute sums for truncated
        tr_sums = _compute_all_sums(truncated)
        
        # Try to extend from truncated
        used = set(truncated)
        last_val = truncated[-1] if truncated else 0
        remaining = [x for x in range(last_val + 1, 257) if x not in used]
        
        if not remaining:
            continue
        
        # Try multiple random orderings
        for _ in range(10):
            if time.time() > time_limit:
                break
            shuffled = remaining[:]
            rng.shuffle(shuffled)
            new_seq, new_sums = _greedy_extend(truncated, tr_sums, shuffled, rng)
            if len(new_seq) > best_len:
                best_seq = new_seq
                best_sums = new_sums
                best_len = len(best_seq)
    
    # Also try simple append-based extension
    used = set(best_seq)
    last_val = best_seq[-1] if best_seq else 0
    remaining = [x for x in range(last_val + 1, 257) if x not in used]
    
    if remaining:
        for _ in range(20):
            if time.time() > time_limit:
                break
            shuffled = remaining[:]
            rng.shuffle(shuffled)
            new_seq, new_sums = _greedy_extend(best_seq, best_sums, shuffled, rng)
            if len(new_seq) > best_len:
                best_seq = new_seq
                best_sums = new_sums
                best_len = len(best_seq)
    
    return best_seq, best_sums, best_len

def _dfs_backtrack(seq, occupied_sums, candidates, depth, max_depth, time_limit, best_len_ref):
    """
    Backtracking DFS with memoization on suffix sum state.
    """
    if time.time() > time_limit:
        return list(seq), set(occupied_sums)
    
    if depth >= max_depth:
        return list(seq), set(occupied_sums)
    
    if not seq and len(seq) <= best_len_ref[0]:
        return list(seq), set(occupied_sums)
    
    last = seq[-1] if seq else 0
    best_local = list(seq)
    best_local_sums = set(occupied_sums)
    
    # Try candidates in order (ascending for efficiency)
    for i, x in enumerate(candidates):
        if time.time() > time_limit:
            break
        if x <= last:
            continue
        
        ok, new_sums = _can_append(x, seq, occupied_sums)
        if ok:
            new_occ = occupied_sums | set(new_sums)
            # Recurse
            result_seq, result_sums = _dfs_backtrack(
                seq + [x], new_occ, candidates[i+1:], depth+1, max_depth, time_limit, best_len_ref
            )
            if len(result_seq) > len(best_local):
                best_local = result_seq
                best_local_sums = result_sums
    
    return best_local, best_local_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe within 90s
    
    rng = random.Random(123)
    
    best_seq = []
    best_sums = set()
    best_len = 0
    
    # Phase 1: Simple greedy to get a baseline quickly
    def greedy_asc():
        seq = []
        sums = set()
        for x in range(1, 257):
            ok, new_sums = _can_append(x, seq, sums)
            if ok:
                seq.append(x)
                sums.update(new_sums)
        return seq, sums
    
    seq, sums = greedy_asc()
    if len(seq) > best_len:
        best_seq, best_sums, best_len = seq, sums, len(seq)
    
    # Phase 2: Beam search
    if time.time() < deadline - 10:
        beam_seq, beam_sums = _beam_search(deadline - 10, rng, beam_width=15)
        if len(beam_seq) > best_len:
            best_seq, best_sums, best_len = beam_seq, beam_sums, len(beam_seq)
    
    # Phase 3: DFS backtracking from good prefix
    if time.time() < deadline - 15 and best_len >= 5:
        # Use the best sequence's first few elements as prefix
        prefix_len = min(3, len(best_seq))
        prefix = best_seq[:prefix_len]
        prefix_sums = _compute_all_sums(prefix)
        
        # Candidates: values > prefix[-1]
        if prefix:
            candidates = list(range(prefix[-1] + 1, 257))
        else:
            candidates = list(range(1, 257))
        
        best_len_ref = [best_len]
        dfs_seq, dfs_sums = _dfs_backtrack(
            prefix, prefix_sums, candidates, 0, 15, deadline - 15, best_len_ref
        )
        if len(dfs_seq) > best_len:
            best_seq, best_sums, best_len = dfs_seq, dfs_sums, len(dfs_seq)
    
    # Phase 4: Local search polish
    if time.time() < deadline - 5:
        best_seq, best_sums, best_len = _local_search_polish(
            best_seq, best_sums, deadline - 3, rng
        )
    
    # Final validation
    def validate(seq):
        if not seq:
            return False
        # Strictly increasing
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        # All values in [1, 256]
        for x in seq:
            if x < 1 or x > 256:
                return False
        # All consecutive block sums distinct
        block_sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                if s in block_sums:
                    return False
                block_sums.add(s)
        return True
    
    if not validate(best_seq):
        # Fallback to powers of 2 (always valid)
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return best_seq

if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Sequence: {result}")

