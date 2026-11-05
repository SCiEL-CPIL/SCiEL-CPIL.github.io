import time
import random

def _check_extension(seq, sums, x):
    """
    Check if adding x to the end of seq is valid.
    Returns the new sums to add if valid, or None if invalid.
    """
    if not seq:
        if x in sums:
            return None
        return [x]
    
    tails = []
    total = x
    tails.append(total)
    for a in reversed(seq):
        total += a
        tails.append(total)
        
    for s in tails:
        if s in sums:
            return None
            
    return tails

def _greedy_randomized(k=1, seed=None, max_val=256):
    """
    Greedy construction with randomization.
    If k=1, it's standard greedy (smallest valid).
    If k>1, it picks randomly from the last k valid candidates seen.
    """
    if seed is not None:
        random.seed(seed)
        
    seq = []
    sums = set()
    candidates_buffer = []
    
    for x in range(1, max_val + 1):
        new_tails = _check_extension(seq, sums, x)
        
        if new_tails is not None:
            candidates_buffer.append((x, new_tails))
            
            if len(candidates_buffer) >= k:
                idx = random.randint(0, len(candidates_buffer) - 1)
                x_val, tails = candidates_buffer[idx]
                
                seq.append(x_val)
                sums.update(tails)
                
                candidates_buffer = []
                
    if candidates_buffer:
        idx = random.randint(0, len(candidates_buffer) - 1)
        x_val, tails = candidates_buffer[idx]
        seq.append(x_val)
        sums.update(tails)

    return seq

def _random_skip_greedy(seed=None, max_val=256, skip_prob=0.1):
    """
    Randomized greedy that sometimes skips a valid candidate to explore different paths.
    """
    if seed is not None:
        random.seed(seed)
        
    seq = []
    sums = set()
    
    for x in range(1, max_val + 1):
        new_tails = _check_extension(seq, sums, x)
        
        if new_tails is not None:
            # With some probability, skip this candidate to try later ones
            if random.random() > skip_prob:
                seq.append(x)
                sums.update(new_tails)
            else:
                # Skip, but we need to be careful: if we skip, we might not find a valid one later
                # For now, just skip. If no more valid ones, we stop.
                pass
                
    return seq

def _beam_search(seed=None, beam_width=10, max_val=256):
    """
    Beam search: keep top 'beam_width' partial sequences at each step.
    """
    if seed is not None:
        random.seed(seed)
        
    # Each beam is (seq, sums)
    beams = [([], set())]
    
    for step in range(max_val):
        new_beams = []
        for seq, sums in beams:
            # Try to extend this beam
            for x in range(1, max_val + 1):
                if seq and x <= seq[-1]:
                    continue
                new_tails = _check_extension(seq, sums, x)
                if new_tails is not None:
                    new_seq = seq + [x]
                    new_sums = sums.union(new_tails)
                    new_beams.append((new_seq, new_sums))
        
        if not new_beams:
            break
            
        # Keep top 'beam_width' beams based on some heuristic (e.g., length, then sum)
        # Sort by length descending, then by last element ascending (to leave room)
        new_beams.sort(key=lambda b: (-len(b[0]), b[0][-1] if b[0] else 0))
        beams = new_beams[:beam_width]
        
        # Check if any beam is complete (can't be extended)
        # If all beams are complete, we can stop
        # But for now, we just keep going until no new beams are generated
        
    # Return the best beam (longest)
    best = max(beams, key=lambda b: len(b[0]))
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_seq = []
    best_len = 0
    
    # 1. Deterministic Greedy (k=1)
    seq_greedy = _greedy_randomized(k=1, seed=0)
    if len(seq_greedy) > best_len:
        best_len = len(seq_greedy)
        best_seq = seq_greedy[:]
        
    # 2. Randomized Greedy with various k and seeds
    for k in [2, 3, 5, 10, 20]:
        for seed in range(20):
            if time.time() > deadline:
                break
            seq = _greedy_randomized(k=k, seed=seed)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
        if time.time() > deadline:
            break
            
    # 3. Random Skip Greedy
    for skip_prob in [0.05, 0.1, 0.2, 0.3]:
        for seed in range(10):
            if time.time() > deadline:
                break
            seq = _random_skip_greedy(seed=seed, skip_prob=skip_prob)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
                
        if time.time() > deadline:
            break
            
    # 4. Beam Search
    for bw in [5, 10, 20, 50]:
        if time.time() > deadline:
            break
        seq = _beam_search(seed=0, beam_width=bw)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
    return best_seq
