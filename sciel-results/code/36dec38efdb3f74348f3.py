import time
import random
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _compute_sums(values):
    """Compute all consecutive block sums for a sequence."""
    n = len(values)
    sums = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            sums.add(total)
    return sums

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    if not current:
        return current
    cur_sums = set()
    for i in range(len(current)):
        total = 0
        for j in range(i, len(current)):
            total += current[j]
            cur_sums.add(total)
    
    for x in range((values[-1] if values else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _try_insert(seq, index, val, cur_sums):
    """
    Try to insert val at index in seq.
    Returns (new_seq, new_sums) if valid, else (None, None).
    """
    if val in seq:
        return None, None
    if index > 0 and seq[index-1] >= val:
        return None, None
    if index < len(seq) and seq[index] <= val:
        return None, None
    
    # Calculate new sums introduced by insertion
    # New blocks are those that include the new element and end before or at the new element's position?
    # Actually, inserting at index 'index' means the new element is at 'index'.
    # The sequence becomes: seq[0:index] + [val] + seq[index:]
    
    # We need to check if any new consecutive block sum collides with existing ones or other new ones.
    # New blocks:
    # 1. Blocks ending at val (start from 0 to index)
    # 2. Blocks starting at val (end from index to n)
    # 3. Blocks passing through val (start < index, end > index)
    
    # Efficient check:
    # Let's just construct the new sequence and check the sums involving val.
    # Sums NOT involving val remain unchanged and were already distinct.
    # So we only need to check sums involving val against each other and against the old sums.
    
    new_seq = seq[:index] + [val] + seq[index:]
    
    # Generate sums involving val
    # val is at index 'index' in new_seq
    new_sums = set()
    
    # Sums ending at val
    s = 0
    for k in range(index, -1, -1):
        s += new_seq[k]
        new_sums.add(s)
        
    # Sums starting at val
    s = 0
    for k in range(index, len(new_seq)):
        s += new_seq[k]
        new_sums.add(s)
        
    # Sums passing through val
    # start < index, end > index
    # prefix_sum_end = sum(new_seq[index:end+1])
    # suffix_sum_start = sum(new_seq[start:index])
    # total = prefix + suffix + val? No, prefix includes val if we define carefully.
    
    # Let's be precise.
    # Block [start, end] with start < index < end.
    # Sum = sum(new_seq[start:index]) + val + sum(new_seq[index+1:end+1])
    
    # Precompute suffix sums of new_seq starting from index+1?
    # Or just iterate. Since we are in a search loop, efficiency matters.
    # Max length ~15-20. O(n^2) per insertion check is fine if n is small.
    
    for start in range(0, index):
        left_sum = 0
        for k in range(start, index):
            left_sum += new_seq[k]
        for end in range(index + 1, len(new_seq)):
            right_sum = 0
            for k in range(index + 1, end + 1):
                right_sum += new_seq[k]
            new_sums.add(left_sum + val + right_sum)
            
    # Check for collisions
    if new_sums & cur_sums:
        return None, None
    if len(new_sums) != len(new_sums): # Internal collision
        return None, None
        
    # Also need to check if new_sums has internal duplicates
    # The set construction above handles uniqueness of the new sums themselves.
    # But we must ensure they don't collide with each other.
    # Since we added to a set, duplicates are merged. 
    # Wait, if two different blocks produce the same sum, the set size will be smaller than the number of blocks generated.
    # How many blocks involve val?
    # Ending at val: index+1 blocks
    # Starting at val: len(new_seq)-index blocks
    # Passing through val: index * (len(new_seq)-index-1) blocks
    # Total blocks involving val = (index+1) + (len(new_seq)-index) + index*(len(new_seq)-index-1)
    # = index + 1 + N - index + index*N - index - index*index - index + 1 ... this is getting messy.
    
    # Simpler: Count the number of blocks generated and compare to len(new_sums).
    # Blocks ending at val: index+1
    # Blocks starting at val: N-index
    # Blocks passing through val: index * (N-index-1)
    expected_count = (index + 1) + (N - index) + index * (N - index - 1)
    # Note: The block [index, index] (just val) is counted in both ending and starting.
    # So expected_count = (index+1) + (N-index) + index*(N-index-1) - 1
    if len(new_sums) < expected_count:
        return None, None
        
    return new_seq, new_sums | cur_sums

def _simulated_annealing(start_seq, max_val=256, deadline=None):
    """
    Simulated Annealing to maximize sequence length.
    Moves:
    1. Insert a random valid number at a random position.
    2. Remove a random number.
    3. Swap two numbers (if it maintains order, effectively a move).
    """
    if not start_seq:
        return start_seq
    
    current_seq = list(start_seq)
    current_sums = _compute_sums(current_seq)
    best_seq = list(current_seq)
    best_sums = set(current_sums)
    
    # Temperature parameters
    T = 1.0
    cooling = 0.995
    min_T = 0.01
    
    rng = random.Random(42)
    
    while time.time() < deadline and T > min_T:
        move_type = rng.random()
        new_seq = None
        new_sums = None
        
        if move_type < 0.7:
            # Insert
            # Pick a random position and a random value
            pos = rng.randint(0, len(current_seq))
            # Determine valid range for value based on neighbors
            low = current_seq[pos-1] + 1 if pos > 0 else 1
            high = current_seq[pos] - 1 if pos < len(current_seq) else max_val
            
            if low <= high:
                val = rng.randint(low, high)
                new_seq, new_sums = _try_insert(current_seq, pos, val, current_sums)
                
        elif move_type < 0.9:
            # Remove
            if len(current_seq) > 1:
                pos = rng.randint(0, len(current_seq) - 1)
                new_seq = current_seq[:pos] + current_seq[pos+1:]
                new_sums = _compute_sums(new_seq)
                
        else:
            # Swap (only if it preserves increasing order, which is rare for non-adjacent,
            # so we just do adjacent swaps or skip if not valid)
            # Actually, swapping breaks increasing order unless we re-sort, which changes the sequence identity.
            # Let's stick to Insert/Remove as they are the primary drivers of length change.
            # Or "Shift": Move an element to another position.
            if len(current_seq) > 1:
                pos1 = rng.randint(0, len(current_seq) - 1)
                pos2 = rng.randint(0, len(current_seq) - 1)
                if pos1 != pos2:
                    val = current_seq[pos1]
                    # Remove and insert
                    temp_seq = current_seq[:pos1] + current_seq[pos1+1:]
                    # Determine valid insertion pos for val in temp_seq
                    # We want to keep it increasing.
                    # Find where val fits in temp_seq
                    insert_pos = -1
                    for i in range(len(temp_seq)):
                        if i > 0 and temp_seq[i-1] < val < temp_seq[i]:
                            insert_pos = i
                            break
                        if i == 0 and val < temp_seq[0]:
                            insert_pos = 0
                            break
                        if i == len(temp_seq)-1 and val > temp_seq[i]:
                            insert_pos = i + 1
                            break
                    if insert_pos != -1:
                        new_seq = temp_seq[:insert_pos] + [val] + temp_seq[insert_pos:]
                        new_sums = _compute_sums(new_seq)
        
        if new_seq is None or new_sums is None:
            continue
            
        # Evaluate move
        # Objective: Maximize length.
        # If lengths are equal, we might prefer "more spread out" or just accept to explore.
        # Let's use length as primary.
        delta_len = len(new_seq) - len(current_seq)
        
        # Acceptance probability
        if delta_len > 0:
            # Always accept improvement
            accept = True
        elif delta_len < 0:
            # Probability of accepting worsening move
            # Since we care about length, a drop of 1 is significant.
            # exp(delta_len / T) = exp(-1/T)
            accept = rng.random() < (2.718 ** (delta_len / T))
        else:
            # Same length, accept with some probability to explore
            accept = rng.random() < 0.5
            
        if accept:
            current_seq = new_seq
            current_sums = new_sums
            
            if len(current_seq) > len(best_seq):
                best_seq = list(current_seq)
                best_sums = set(current_sums)
                
        T *= cooling
        
    return best_seq

def _random_valid_sequence(max_val=256, min_len=5, rng=None):
    """Generate a random valid sequence of a given minimum length."""
    if rng is None:
        rng = random.Random()
    
    for _ in range(100):
        seq = []
        cur_sums = set()
        # Start with a small number
        x = rng.randint(1, 10)
        seq.append(x)
        cur_sums.add(x)
        
        for _ in range(20): # Try to add up to 20 more
            if seq[-1] >= max_val:
                break
            # Try random candidates
            found = False
            for _ in range(10):
                cand = rng.randint(seq[-1] + 1, max_val)
                # Check validity quickly
                tails = [cand]
                total = cand
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                if not cur_sums.intersection(tails):
                    seq.append(cand)
                    cur_sums.update(tails)
                    found = True
                    break
            if not found:
                break
                
        if len(seq) >= min_len:
            return seq
    return [1, 2, 4] # Fallback

def construct():
    deadline = time.time() + 80  # 80 second safety margin (90s kill, 10s buffer)
    
    best = []
    
    # 1. Get a good starting point via Greedy
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > len(best):
        best = list(greedy_seq)
        
    # 2. Try Simulated Annealing from the greedy start
    if time.time() < deadline - 5:
        sa_seq = _simulated_annealing(greedy_seq, deadline=deadline)
        if len(sa_seq) > len(best):
            best = list(sa_seq)
            
    # 3. Try Simulated Annealing from random starts
    rng = random.Random()
    for _ in range(10): # Limit number of restarts to save time
        if time.time() > deadline - 10:
            break
        rand_start = _random_valid_sequence(min_len=5, rng=rng)
        if len(rand_start) > 3:
            sa_seq = _simulated_annealing(rand_start, deadline=deadline)
            if len(sa_seq) > len(best):
                best = list(sa_seq)
                
    # 4. Final Greedy polish on the best found
    extended = _greedy_forward(best)
    if len(extended) > len(best):
        best = extended
        
    # 5. Ensure validity
    if not _is_valid_full(best):
        # Fallback to greedy
        best = _greedy_forward([])
        
    return best

# EVOLVE-BLOCK-END
