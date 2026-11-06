import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0
    MAX_VAL = 256

    def get_block_sums(seq):
        """Get all consecutive block sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid(seq):
        """Check if all consecutive block sums are distinct."""
        n = len(seq)
        if n == 0:
            return True
        sums = []
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.append(s)
        return len(sums) == len(set(sums))

    def can_extend(prefix, new_val):
        """Check if adding new_val to prefix keeps all block sums distinct."""
        # New block sums are: new_val alone, and prefix suffixes + new_val
        # We need to check against all existing block sums
        n = len(prefix)
        # Existing sums
        existing_sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += prefix[j]
                existing_sums.add(s)
        
        # New sums involving new_val
        new_sums = [new_val]
        s = new_val
        for i in range(n-1, -1, -1):
            s += prefix[i]
            new_sums.append(s)
        
        # Check no overlap
        for ns in new_sums:
            if ns in existing_sums:
                return False
        # Check new_sums are distinct among themselves
        if len(new_sums) != len(set(new_sums)):
            return False
        return True

    # Start with a good initial sequence using exponential growth
    # Then try to extend it greedily and with local search
    
    # Strategy 1: Greedy with exponential base
    best = []
    sums_set = set()
    
    # Build initial sequence greedily
    for x in range(1, MAX_VAL + 1):
        if can_extend(best, x):
            best.append(x)
            # Update sums_set
            n = len(best) - 1
            s = x
            sums_set.add(s)
            for i in range(n - 1, -1, -1):
                s += best[i]
                sums_set.add(s)
    
    initial_len = len(best)
    
    # Strategy 2: Try different starting points and growth patterns
    # Use a parameterized approach: start with a, grow by factor
    
    def try_sequence(start, ratio_num, ratio_den, max_len):
        """Try a geometric-like sequence."""
        seq = []
        current = start
        for _ in range(max_len):
            if current > MAX_VAL:
                break
            if not seq or current > seq[-1]:
                if can_extend(seq, current):
                    seq.append(current)
            # Next value
            current = (current * ratio_num + ratio_den - 1) // ratio_den
            if current <= seq[-1] if seq else False:
                current = (seq[-1] if seq else 0) + 1
        return seq
    
    # Try various geometric sequences
    for start in range(1, 20):
        for num in range(2, 10):
            for den in range(1, num):
                seq = try_sequence(start, num, den, 30)
                if len(seq) > len(best):
                    best = seq
    
    # Strategy 3: Local search - try to replace elements to allow more additions
    def local_search(seq, time_budget):
        """Try to improve the sequence by local modifications."""
        seq = list(seq)
        best_seq = list(seq)
        best_len = len(seq)
        
        while time.time() - start_time < time_budget:
            improved = False
            
            # Try adding elements at the end
            for x in range(seq[-1] + 1, MAX_VAL + 1) if seq else range(1, MAX_VAL + 1):
                if can_extend(seq, x):
                    seq.append(x)
                    improved = True
                    break
            
            if not improved:
                # Try removing last element and trying alternatives
                if len(seq) > 1:
                    removed = seq.pop()
                    for x in range(removed + 1, MAX_VAL + 1):
                        if can_extend(seq, x):
                            seq.append(x)
                            improved = True
                            break
                    if not improved:
                        seq.append(removed)
            
            if len(seq) > best_len:
                best_seq = list(seq)
                best_len = len(seq)
            
            if not improved:
                break
        
        return best_seq
    
    best = local_search(best, TIME_LIMIT - 5)
    
    # Strategy 4: Multi-start with different initial sequences
    for _ in range(20):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Create a random-ish starting sequence
        import random
        random.seed(int(time.time() * 1000) % 10000)
        seq = []
        current = random.randint(1, 10)
        for _ in range(15):
            if current > MAX_VAL:
                break
            if not seq or current > seq[-1]:
                if can_extend(seq, current):
                    seq.append(current)
            current += random.randint(1, 5)
        
        if len(seq) > len(best):
            best = local_search(seq, TIME_LIMIT - 5)
    
    # Final greedy extension
    extended = list(best)
    for x in range(extended[-1] + 1, MAX_VAL + 1) if extended else range(1, MAX_VAL + 1):
        if can_extend(extended, x):
            extended.append(x)
    
    if len(extended) > len(best):
        best = extended
    
    return best
