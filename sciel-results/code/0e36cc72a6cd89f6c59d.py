import time
import sys
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0
    MAX_VAL = 256
    random.seed(42)

    def get_consecutive_sums(seq):
        """Get all consecutive block sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def is_valid(seq):
        """Check if sequence is valid (strictly increasing, in [1,256], all consecutive sums distinct)."""
        if not seq:
            return True
        if seq[0] < 1 or seq[-1] > MAX_VAL:
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        sums = get_consecutive_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    def greedy_randomized(seed_val=0):
        """Greedy with randomization to create diverse initial sequences."""
        rng = random.Random(seed_val)
        best = []
        sums = set()
        # Shuffle the range
        candidates = list(range(1, MAX_VAL + 1))
        rng.shuffle(candidates)
        for x in candidates:
            tails = []
            total = x
            for a in reversed(best):
                total += a
                tails.append(total)
            tails.append(x)
            if not sums.intersection(tails):
                best.append(x)
                sums.update(tails)
        best.sort()
        return best

    def local_improve(seq, time_limit):
        """Try to improve sequence by inserting elements or swapping."""
        seq = list(seq)
        current_len = len(seq)
        sums = get_consecutive_sums(seq)
        n_sums = len(sums)
        expected_sums = current_len * (current_len + 1) // 2
        assert n_sums == expected_sums, f"Invalid initial seq: {n_sums} vs {expected_sums}"
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try inserting each possible value at each position
            for val in range(1, MAX_VAL + 1):
                if val in seq:
                    continue
                for pos in range(len(seq) + 1):
                    # Check strictly increasing
                    if pos > 0 and seq[pos - 1] >= val:
                        continue
                    if pos < len(seq) and val >= seq[pos]:
                        continue
                    
                    new_seq = seq[:pos] + [val] + seq[pos:]
                    new_sums = get_consecutive_sums(new_seq)
                    if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                        seq = new_seq
                        sums = new_sums
                        improved = True
                        break
                if improved:
                    break
        return seq

    # Generate multiple diverse starting sequences
    best_seq = None
    best_len = 0
    
    # First, deterministic greedy as baseline
    det_seq = greedy_randomized(0)
    # The deterministic greedy uses natural order, let's do that separately
    det_best = []
    det_sums = set()
    for x in range(1, MAX_VAL + 1):
        tails = []
        total = x
        for a in reversed(det_best):
            total += a
            tails.append(total)
        tails.append(x)
        if not det_sums.intersection(tails):
            det_best.append(x)
            det_sums.update(tails)
    
    if len(det_best) > best_len:
        best_seq = det_best
        best_len = len(det_best)
    
    # Try multiple randomized greedy starts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > TIME_LIMIT - 5.0:
            break
        seq = greedy_randomized(i + 1)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    # Local improvement on the best sequences
    if best_seq and time.time() < TIME_LIMIT - 5.0:
        improved = local_improve(best_seq, TIME_LIMIT - 5.0)
        if len(improved) > best_len:
            best_seq = improved
            best_len = len(improved)
    
    # Try local improvement on a few more diverse sequences
    if time.time() < TIME_LIMIT - 5.0:
        for i in range(min(10, num_restarts)):
            if time.time() > TIME_LIMIT - 5.0:
                break
            seq = greedy_randomized(i + 1)
            if len(seq) >= best_len - 2:  # Only try if close to best
                improved = local_improve(seq, TIME_LIMIT - 5.0)
                if len(improved) > best_len:
                    best_seq = improved
                    best_len = len(improved)
    
    if best_seq is None:
        best_seq = det_best
    
    return best_seq
