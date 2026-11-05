"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def _is_valid(seq):
    """Check if all consecutive block sums in seq are distinct."""
    n = len(seq)
    if n == 0:
        return True
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _beam_search(beam_width=50, max_len=15):
    """
    Beam search to find a long valid sequence.
    State: tuple of the sequence.
    Score: Length (primary), then negative sum of elements (to prefer smaller numbers? 
           Actually, we just want to maximize length. If lengths are equal, 
           we might prefer sequences with smaller max element or smaller total sum 
           to leave room for more additions).
    """
    # Start with empty sequence
    # To break symmetry and help search, we can start with a few known good starts
    # or just empty.
    # Let's start with a few random small valid sequences to bootstrap.
    # Or just start with empty.
    
    # To make it faster, we pre-compute candidates
    candidates = list(range(1, 257))
    
    # Initial beam: empty sequence
    beam = [((), 0)] # (sequence, length)
    
    best_seq = ()
    best_len = 0
    
    for step in range(max_len):
        new_beam = []
        for seq, _ in beam:
            seq_set = set(seq)
            # Try adding each candidate
            # Optimization: Only try candidates that are not in seq
            # We can also prune: if we've already found a sequence of length L, 
            # we only care about extending to L+1.
            
            # To speed up, we can randomize candidates if beam is large
            # But let's try systematic first.
            for x in candidates:
                if x in seq_set:
                    continue
                
                # Construct new sequence
                # We need to insert x in the correct sorted position?
                # The problem says "Return a strictly increasing sequence".
                # So the sequence is sorted.
                # If we maintain sorted order, we can insert x.
                
                # Binary search for insertion point
                # Since seq is sorted, we can use bisect
                import bisect
                pos = bisect.bisect_left(seq, x)
                new_seq = seq[:pos] + (x,) + seq[pos:]
                
                # Check validity
                # We can optimize validity check by reusing sums?
                # For beam search, simple check is safer.
                if _is_valid(new_seq):
                    # Score: We want to keep diverse candidates.
                    # Primary: Length (same for all in this step)
                    # Secondary: We want to keep sequences that are "flexible".
                    # Heuristic: Smaller sum is better? Or smaller max?
                    # Let's sort by sum of elements. Smaller sum leaves more "room" in [1,256]?
                    # Actually, max value is fixed at 256.
                    # Let's sort by the sum of the sequence.
                    score = sum(new_seq)
                    new_beam.append((new_seq, score))
        
        if not new_beam:
            break
            
        # Sort new_beam by score (sum of elements)
        # We want to keep the "best" beam_width candidates.
        # "Best" is heuristic: smaller sum might allow adding more small numbers later?
        # Or maybe we should keep a mix?
        # Let's try keeping the ones with smallest sum first.
        new_beam.sort(key=lambda x: x[1])
        
        # Keep top beam_width
        beam = new_beam[:beam_width]
        
        # Update best
        # All sequences in beam have length step+1
        if step + 1 > best_len:
            best_len = step + 1
            # Pick the one with smallest sum from the beam? Or just the first one?
            # It doesn't matter for length, but let's pick the one with smallest sum as it might be "better"
            best_seq = beam[0][0]
            
    return list(best_seq)

def construct():
    """
    Combines multiple strategies to maximize sequence length.
    1. Beam Search
    2. Greedy (as fallback/initialization)
    """
    start_time = time.time()
    best = []
    
    # Strategy 1: Beam Search
    # Time limit: 30 seconds
    try:
        seq_beam = _beam_search(beam_width=100, max_len=12)
        if len(seq_beam) > len(best):
            best = seq_beam
    except Exception:
        pass
    
    if time.time() - start_time > 40:
        return best
        
    # Strategy 2: Greedy Forward (Fast)
    def _greedy_forward():
        values = []
        sums = set()
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

    seq_greedy = _greedy_forward()
    if len(seq_greedy) > len(best):
        best = seq_greedy
        
    if time.time() - start_time > 50:
        return best

    # Strategy 3: Random Greedy Restarts
    def _greedy_random(order_seed):
        rng = random.Random(order_seed)
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        S = []
        for x in candidates:
            candidate = sorted(S + [x])
            if _is_valid(candidate):
                S.append(x)
        return sorted(S)

    for seed in range(100):
        if time.time() - start_time > 55:
            break
        seq_rand = _greedy_random(seed)
        if len(seq_rand) > len(best):
            best = seq_rand

    return best
# EVOLVE-BLOCK-END
