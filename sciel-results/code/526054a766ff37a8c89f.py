import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    def is_valid_sequence(seq):
        """Check if a sequence is strictly increasing and has distinct consecutive block sums."""
        if not seq:
            return True
        # Check strictly increasing
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        # Check distinct consecutive block sums
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_consecutive_sums(seq):
        """Return set of all nonempty consecutive block sums."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def greedy_randomized():
        """Generate a candidate sequence using randomized greedy approach."""
        values = []
        used_sums = set()
        remaining = list(range(1, 257))
        random.shuffle(remaining)
        
        for x in remaining:
            if x <= values[-1] if values else False:
                continue
            # Calculate new tails if we add x
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            
            if not used_sums.intersection(tails):
                values.append(x)
                used_sums.update(tails)
        
        # Ensure strictly increasing (should be by construction if we check)
        # But the greedy above doesn't enforce increasing order properly in all cases
        # Let's re-implement more carefully
        
        return values

    def greedy_strictly_increasing():
        """Proper greedy that maintains strictly increasing order."""
        values = []
        used_sums = set()
        
        # We need to choose values in increasing order
        # So we iterate from 1 to 256, but with randomization in which ones we "try" first?
        # No, the sequence must be strictly increasing, so the order is fixed by value.
        # Randomization must come from skipping or different starting points?
        # Actually, the current seed iterates 1..256 and adds if valid.
        # To randomize, we can:
        # 1. Start with a random subset
        # 2. Use a different heuristic for which numbers to consider (but they must be increasing)
        
        # Let's try: Randomly decide whether to skip a number, or use a different "pressure"
        # Better: Generate a random permutation of 1..256, then filter to increasing? No.
        
        # Alternative: The sequence is a subset of 1..256. 
        # We can use a randomized greedy where we pick the next element from a random valid candidate?
        # But "next element" must be larger than the previous one.
        
        # Let's try a different randomization: 
        # Start with an empty set. 
        # At each step, pick the next element from the set of valid candidates (larger than last, no sum conflict).
        # Randomly choose among valid candidates?
        
        last_val = 0
        while True:
            candidates = []
            for x in range(last_val + 1, 257):
                tails = [x]
                total = x
                for a in reversed(values):
                    total += a
                    tails.append(total)
                if not used_sums.intersection(tails):
                    candidates.append(x)
            
            if not candidates:
                break
            
            # Randomly pick a candidate
            x = random.choice(candidates)
            values.append(x)
            
            # Update used_sums
            tails = [x]
            total = x
            for a in reversed(values[:-1]):  # values[:-1] because x is last
                total += a
                tails.append(total)
            used_sums.update(tails)
            
            last_val = x
            
            if time.time() > deadline - 5:
                break
        
        return values

    def local_search(seq):
        """Try to improve the sequence by swapping or replacing elements."""
        if not seq:
            return seq
        
        best_seq = list(seq)
        best_len = len(best_seq)
        
        for _ in range(100):  # Limited iterations
            if time.time() > deadline - 5:
                break
            
            improved = False
            n = len(best_seq)
            
            # Try to replace an element with a smaller valid one? No, must be increasing.
            # Try to swap adjacent elements? No, must be increasing.
            # Try to remove an element and see if we can add two?
            # Or try to replace an element with another value that maintains validity.
            
            # Let's try: For each position i, try to replace best_seq[i] with a different value v
            # such that the sequence remains strictly increasing and valid.
            # This is expensive. Let's try a simpler move:
            # Try to insert a new element between two existing ones?
            
            # For simplicity, let's just try to extend the sequence further if possible
            last_val = best_seq[-1]
            used_sums = get_consecutive_sums(best_seq)
            
            for x in range(last_val + 1, 257):
                tails = [x]
                total = x
                for a in reversed(best_seq):
                    total += a
                    tails.append(total)
                
                if not used_sums.intersection(tails):
                    new_seq = best_seq + [x]
                    if len(new_seq) > best_len:
                        best_seq = new_seq
                        best_len = len(new_seq)
                        improved = True
                        break
            
            if not improved:
                # Try to remove an element and see if we can add more
                for i in range(len(best_seq)):
                    trial = best_seq[:i] + best_seq[i+1:]
                    if not trial:
                        continue
                    # Check if trial is valid (it should be if original was)
                    # Try to extend trial
                    last_val = trial[-1]
                    used_sums = get_consecutive_sums(trial)
                    
                    for x in range(last_val + 1, 257):
                        tails = [x]
                        total = x
                        for a in reversed(trial):
                            total += a
                            tails.append(total)
                        
                        if not used_sums.intersection(tails):
                            new_seq = trial + [x]
                            if len(new_seq) > best_len:
                                best_seq = new_seq
                                best_len = len(new_seq)
                                improved = True
                                break
                    if improved:
                        break
        
        return best_seq

    best_sequence = []
    best_len = 0
    
    # First, run the original greedy as a baseline
    values = []
    used_sums = set()
    for x in range(1, 257):
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not used_sums.intersection(tails):
            values.append(x)
            used_sums.update(tails)
    
    if len(values) > best_len:
        best_sequence = values
        best_len = len(values)
    
    # Now, run randomized greedy multiple times
    while time.time() < deadline - 10:
        seq = greedy_strictly_increasing()
        if len(seq) > best_len:
            best_len = len(seq)
            best_sequence = seq
        
        # Try local search on the best sequence
        if time.time() < deadline - 5:
            improved = local_search(best_sequence)
            if len(improved) > best_len:
                best_len = len(improved)
                best_sequence = improved
    
    return best_sequence
