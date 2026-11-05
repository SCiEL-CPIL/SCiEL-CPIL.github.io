import time
import random

def _is_valid_candidate(candidate):
    sums = set()
    n = len(candidate)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += candidate[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _greedy_build(order):
    selected = []
    for x in order:
        if selected and x <= selected[-1]:
            continue
        candidate = selected + [x]
        if _is_valid_candidate(candidate):
            selected = candidate
    return selected

def _greedy_build_with_bias(order, bias='none'):
    """Greedy build with optional bias.
    bias='none': use order as-is
    bias='largest': prefer larger numbers earlier
    bias='smallest': prefer smaller numbers earlier
    """
    if bias == 'largest':
        order = sorted(order, reverse=True)
    elif bias == 'smallest':
        order = sorted(order)
    
    selected = []
    for x in order:
        if selected and x <= selected[-1]:
            continue
        candidate = selected + [x]
        if _is_valid_candidate(candidate):
            selected = candidate
    return selected

def _local_search_improve(seq, rng, deadline):
    """Try to improve a sequence by swapping, inserting, or replacing elements."""
    if not seq:
        return seq
    
    best = list(seq)
    n = len(best)
    
    # Try inserting a new element at various positions
    all_numbers = set(range(1, 257))
    used = set(best)
    available = list(all_numbers - used)
    
    if not available:
        return best
    
    # Try adding elements one at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for pos in range(len(best) + 1):
            # Find a value that can be inserted at pos
            prev_val = best[pos - 1] if pos > 0 else 0
            next_val = best[pos] if pos < len(best) else 257
            
            # Candidate values between prev_val and next_val
            candidates = [v for v in available if prev_val < v < next_val]
            rng.shuffle(candidates)
            
            for v in candidates:
                new_seq = best[:pos] + [v] + best[pos:]
                if _is_valid_candidate(new_seq):
                    best = new_seq
                    available.remove(v)
                    improved = True
                    break
            if improved:
                break
        
        # Also try replacing elements
        if not improved and time.time() < deadline:
            for i in range(len(best)):
                prev_val = best[i - 1] if i > 0 else 0
                next_val = best[i + 1] if i < len(best) - 1 else 257
                
                # Try replacing best[i] with a different value
                candidates = [v for v in available if prev_val < v < next_val]
                rng.shuffle(candidates)
                
                for v in candidates:
                    new_seq = best[:i] + [v] + best[i+1:]
                    if _is_valid_candidate(new_seq):
                        # Remove old value from best, add new
                        old_val = best[i]
                        best = new_seq
                        available.remove(old_val)
                        available.append(v)
                        improved = True
                        break
                if improved:
                    break
    
    return best

def _backtrack_fill(seq, deadline):
    """Try to extend the sequence using backtracking for the last few positions."""
    if not seq:
        return seq
    
    best = list(seq)
    last_val = best[-1] if best else 0
    
    # Try to add more elements after the last one using backtracking
    def try_add(start_val, depth, max_depth=3):
        if depth > max_depth or time.time() > deadline:
            return None
        
        best_result = None
        for v in range(start_val + 1, 257):
            candidate = best + [v]
            if _is_valid_candidate(candidate):
                # Try to extend further
                extended = try_add(v, depth + 1, max_depth)
                if extended is not None:
                    if best_result is None or len(extended) > len(best_result):
                        best_result = extended
                else:
                    if best_result is None or len(candidate) > len(best_result):
                        best_result = candidate
        
        return best_result
    
    result = try_add(last_val, 0, max_depth=2)
    if result is not None and len(result) > len(best):
        best = result
    
    return best

def construct():
    start_time = time.time()
    # Safety margin: 55s limit, 60s hard limit, 90s kill
    deadline = start_time + 55.0
    
    # 1. Deterministic Baselines
    best_seq = []
    
    # Ascending
    seq = _greedy_build(list(range(1, 257)))
    if len(seq) > len(best_seq):
        best_seq = seq
        
    # Descending
    seq = _greedy_build(list(range(256, 0, -1)))
    if len(seq) > len(best_seq):
        best_seq = seq
    
    # Largest-first bias
    seq = _greedy_build_with_bias(list(range(1, 257)), bias='largest')
    if len(seq) > len(best_seq):
        best_seq = seq
    
    # Smallest-first bias
    seq = _greedy_build_with_bias(list(range(1, 257)), bias='smallest')
    if len(seq) > len(best_seq):
        best_seq = seq

    # 2. Random Restarts with diverse strategies
    rng = random.Random(42)
    base_range = list(range(1, 257))
    
    while time.time() < deadline:
        # Choose a strategy
        strategy = rng.choice(['random', 'largest_bias', 'smallest_bias', 'mixed'])
        
        if strategy == 'random':
            order = list(base_range)
            rng.shuffle(order)
            seq = _greedy_build(order)
        elif strategy == 'largest_bias':
            # Mix of random and largest-first
            order = list(base_range)
            rng.shuffle(order)
            # Put some large numbers first
            large = sorted([x for x in order if x > 128], reverse=True)
            small = [x for x in order if x <= 128]
            order = large[:rng.randint(5, 15)] + small
            rng.shuffle(order[15:])
            seq = _greedy_build(order)
        elif strategy == 'smallest_bias':
            # Mix of random and smallest-first
            order = list(base_range)
            rng.shuffle(order)
            # Put some small numbers first
            small = sorted([x for x in order if x < 128])
            large = [x for x in order if x >= 128]
            order = small[:rng.randint(5, 15)] + large
            rng.shuffle(order[15:])
            seq = _greedy_build(order)
        else:  # mixed
            order = list(base_range)
            rng.shuffle(order)
            # Random interleaving of small and large
            small = sorted([x for x in order if x < 128])
            large = sorted([x for x in order if x >= 128], reverse=True)
            mixed = []
            si, li = 0, 0
            while si < len(small) or li < len(large):
                if rng.random() < 0.5 and si < len(small):
                    mixed.append(small[si])
                    si += 1
                elif li < len(large):
                    mixed.append(large[li])
                    li += 1
                elif si < len(small):
                    mixed.append(small[si])
                    si += 1
            seq = _greedy_build(mixed)
        
        if len(seq) > len(best_seq):
            best_seq = seq
    
    # 3. Local search improvement on the best sequence
    if time.time() < deadline - 5:
        best_seq = _local_search_improve(best_seq, rng, deadline - 5)
    
    # 4. Backtracking fill
    if time.time() < deadline - 2:
        best_seq = _backtrack_fill(best_seq, deadline - 2)
    
    return best_seq
# EVOLVE-BLOCK-END
