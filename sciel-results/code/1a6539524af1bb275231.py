"""Program for increasing sequences with unique consecutive-block sums."""
import random
import math
import time

# EVOLVE-BLOCK-START

def _compute_cost_and_sums(seq):
    """
    Compute the number of duplicate block sums and the set of sums.
    Returns (num_duplicates, sums_set).
    """
    n = len(seq)
    if n == 0:
        return 0, set()
    
    sums = []
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.append(s)
    
    unique_sums = set(sums)
    num_duplicates = len(sums) - len(unique_sums)
    return num_duplicates, unique_sums

def _is_valid(seq):
    """Check if sequence is strictly increasing in [1,256] and has distinct block sums."""
    if not seq:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    
    cost, _ = _compute_cost_and_sums(seq)
    return cost == 0

def _greedy_construct():
    """Greedy construction to get a good initial sequence."""
    values = []
    sums = set()
    for x in range(1, 257):
        tails = []
        total = x
        tails.append(total)
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _sa_extend(current_seq, target_len, max_iters=20000, start_temp=10.0, end_temp=0.1):
    """
    Attempt to extend current_seq to target_len using Simulated Annealing.
    If target_len > len(current_seq), we initialize with random extra elements.
    Returns the best valid sequence of length target_len found, or None.
    """
    n_current = len(current_seq)
    if target_len < n_current:
        return None
    
    # Initialize state
    if target_len == n_current:
        base = current_seq[0]
        diffs = [current_seq[i+1] - current_seq[i] for i in range(n_current - 1)]
    else:
        # Extend with random values to reach target_len
        # Keep the prefix, add random increasing suffix
        seq = list(current_seq)
        last = seq[-1]
        for _ in range(target_len - n_current):
            # Pick a random next value > last, <= 256
            if last >= 256:
                # Can't extend further
                return None
            # Bias towards larger jumps to avoid collisions? Or random?
            # Let's pick a random value in [last+1, 256]
            # To make it strictly increasing, we just pick a value.
            # But we represent as diffs.
            # Let's just build the seq first, then convert to base/diffs
            next_val = random.randint(last + 1, 256)
            seq.append(next_val)
            last = next_val
        
        base = seq[0]
        diffs = [seq[i+1] - seq[i] for i in range(len(seq) - 1)]

    def get_seq(base, diffs):
        s = [base]
        for d in diffs:
            s.append(s[-1] + d)
        return s

    current_seq = get_seq(base, diffs)
    current_cost, _ = _compute_cost_and_sums(current_seq)
    
    best_seq = None
    best_cost = current_cost
    
    temp = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / max_iters)
    
    # We want to find a state with cost 0.
    # If we find one, we can return immediately (or keep searching for robustness? No, cost 0 is optimal).
    
    for it in range(max_iters):
        temp *= cooling
        
        new_base = base
        new_diffs = list(diffs)
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Change one difference
            idx = random.randrange(len(new_diffs))
            delta = random.randint(-5, 5)
            if delta != 0:
                new_diffs[idx] = max(1, new_diffs[idx] + delta)
        elif move_type < 0.6:
            # Change base value
            delta = random.randint(-3, 3)
            if delta != 0:
                new_base = max(1, base + delta)
        elif move_type < 0.8:
            # Swap two differences
            if len(new_diffs) >= 2:
                i, j = random.sample(range(len(new_diffs)), 2)
                new_diffs[i], new_diffs[j] = new_diffs[j], new_diffs[i]
        else:
            # Randomize one difference significantly
            idx = random.randrange(len(new_diffs))
            new_diffs[idx] = random.randint(1, 20)

        new_seq = get_seq(new_base, new_diffs)
        
        # Validate bounds and strictly increasing
        if not new_seq or new_seq[0] < 1 or new_seq[-1] > 256:
            continue
        if any(new_seq[i] >= new_seq[i+1] for i in range(len(new_seq)-1)):
            continue
            
        new_cost, _ = _compute_cost_and_sums(new_seq)
        
        # Acceptance
        if new_cost < current_cost:
            # Always accept improvement
            base = new_base
            diffs = new_diffs
            current_seq = new_seq
            current_cost = new_cost
            
            if current_cost == 0:
                best_seq = list(current_seq)
                # We found a valid sequence. We can stop or continue to find a "better" one?
                # The task is just to maximize length. Any valid sequence of this length is good.
                # However, a "tighter" sequence might be easier to extend further?
                # For now, return immediately to save time.
                return best_seq
        else:
            # Probabilistic acceptance
            if temp > 0:
                prob = math.exp(-new_cost / temp)
                if random.random() < prob:
                    base = new_base
                    diffs = new_diffs
                    current_seq = new_seq
                    current_cost = new_cost
                    
                    if current_cost == 0:
                        best_seq = list(current_seq)
                        return best_seq
                        
    return best_seq

def construct():
    """Construct a strictly increasing sequence in [1,256] with distinct consecutive block sums."""
    random.seed(42)
    
    # 1. Get a good initial sequence via greedy
    seq = _greedy_construct()
    
    # 2. Try to extend it using SA
    # We try to extend one by one.
    # If greedy got stuck at L, we try to force L+1 via SA.
    
    current_len = len(seq)
    
    # Try to extend up to a reasonable limit (e.g., 15, though 256 bound limits it)
    # The theoretical max is small.
    for target_len in range(current_len + 1, 16):
        # Try multiple restarts for each target length
        found = None
        for attempt in range(10):
            # If attempt 0, use current seq.
            # If attempt > 0, perturb current seq slightly to escape local minima?
            # Or just run SA again. SA is stochastic.
            
            # To improve chances, we can also try to "shuffle" the current valid seq
            # before extending, but that risks losing validity.
            # Better: just run SA on the extension.
            
            # Sometimes, the greedy prefix is too "rigid".
            # We can allow SA to modify the prefix too.
            # The _sa_extend function handles the whole sequence.
            
            result = _sa_extend(seq, target_len, max_iters=5000, start_temp=15.0, end_temp=0.1)
            if result is not None:
                found = result
                break
        
        if found:
            seq = found
        else:
            # If we couldn't extend to target_len, we might still be able to extend to target_len-1?
            # We already have seq of length target_len-1.
            # So we stop trying to extend further.
            break
            
    return seq

# EVOLVE-BLOCK-END
