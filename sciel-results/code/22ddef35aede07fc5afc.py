import time
import random

def _check_valid(seq):
    """Check that seq is strictly increasing, all elements in [1,256], and all consecutive block sums distinct."""
    if not seq:
        return True
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    if any(x < 1 or x > 256 for x in seq):
        return False
    n = len(seq)
    # Compute all block sums
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    expected = n * (n + 1) // 2
    if len(sums) != expected:
        return False
    return True

def _greedy_build(start_value, max_time=2.0, strategy="smallest"):
    """
    Build a sequence starting with start_value using a greedy strategy.
    strategy: "smallest", "largest"
    """
    start_time = time.time()
    seq = [start_value]
    sums_set = {start_value}
    last = start_value
    
    while True:
        if time.time() - start_time > max_time:
            break
        
        n = len(seq)
        # Compute suffix sums for conflict check
        suffix_sums = [0] * (n + 1)
        s = 0
        for k in range(n - 1, -1, -1):
            s += seq[k]
            suffix_sums[k] = s
        suffix_sums[n] = 0
        
        valid_next = []
        for x in range(last + 1, 257):
            conflict = False
            for k in range(n + 1):
                if suffix_sums[k] + x in sums_set:
                    conflict = True
                    break
            if not conflict:
                valid_next.append(x)
        
        if not valid_next:
            break
        
        if strategy == "smallest":
            best_x = valid_next[0]
        elif strategy == "largest":
            best_x = valid_next[-1]
        else:
            best_x = random.choice(valid_next)
        
        seq.append(best_x)
        # Update sums
        for k in range(n + 1):
            sums_set.add(suffix_sums[k] + best_x)
        last = best_x
        
    return seq

def _random_build(max_window=None, max_time=2.0):
    """
    Build a sequence using randomized selection from a window of candidates.
    """
    start_time = time.time()
    seq = []
    sums_set = set()
    last = 0
    
    while True:
        if time.time() - start_time > max_time:
            break
        
        n = len(seq)
        # Compute suffix sums
        suffix_sums = [0] * (n + 1)
        s = 0
        for k in range(n - 1, -1, -1):
            s += seq[k]
            suffix_sums[k] = s
        suffix_sums[n] = 0
        
        # Determine candidate range
        if max_window is None:
            upper = 256
        else:
            upper = min(256, last + max_window)
        
        if upper <= last:
            break
        
        candidates = list(range(last + 1, upper + 1))
        random.shuffle(candidates)
        
        found = False
        for x in candidates:
            conflict = False
            for k in range(n + 1):
                if suffix_sums[k] + x in sums_set:
                    conflict = True
                    break
            if not conflict:
                seq.append(x)
                for k in range(n + 1):
                    sums_set.add(suffix_sums[k] + x)
                last = x
                found = True
                break
        
        if not found:
            break
    
    return seq

def construct():
    """
    Main entry point. Returns a strictly increasing sequence in [1,256] with distinct consecutive-block sums.
    Combines greedy multi-start with randomized restarts to explore the solution space.
    """
    total_start = time.time()
    deadline = total_start + 55.0  # Safety margin for 60s limit
    
    best_seq = []
    best_len = 0
    
    # 1. Deterministic Greedy Passes
    # Try starting from 1, 2, 3... with different strategies
    for start_val in range(1, 15):
        if time.time() > deadline - 10:
            break
        for strategy in ["smallest", "largest"]:
            seq = _greedy_build(start_val, max_time=1.0, strategy=strategy)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
    
    # 2. Randomized Restarts
    # Use random window sizes to escape local optima
    windows = [10, 20, 30, 50, 75, 100, 150, 200, 256]
    while time.time() < deadline:
        # Randomly pick a window size
        w = random.choice(windows)
        # Sometimes start from a random value to diversify
        if random.random() < 0.3 and best_seq:
            # Try to extend a random prefix of the best sequence found so far
            # This is a heuristic to explore neighborhoods of good solutions
            prefix_len = random.randint(1, max(1, len(best_seq) - 2))
            prefix = best_seq[:prefix_len]
            start_val = prefix[-1]
            # Rebuild from this prefix using random strategy
            # Note: _random_build starts from empty, so we need a variant or just use it as is
            # For simplicity, we just run random builds from scratch or with a fixed start
            # Let's just run random builds from scratch for now, as implementing prefix extension
            # in _random_build requires more code.
            pass
        
        seq = _random_build(max_window=w, max_time=1.0)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
        
        # Occasionally try a greedy build from a random start
        if random.random() < 0.2:
            start_val = random.randint(1, 20)
            strategy = random.choice(["smallest", "largest"])
            seq = _greedy_build(start_val, max_time=0.5, strategy=strategy)
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]

    # Verify result
    if not _check_valid(best_seq):
        # Fallback
        best_seq = [1, 2, 5]
        
    return best_seq
