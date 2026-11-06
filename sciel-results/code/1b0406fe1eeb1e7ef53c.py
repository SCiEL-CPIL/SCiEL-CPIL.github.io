"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Use simulated annealing on the set of selected indices to find a strictly
    increasing sequence of integers in [1,256] such that all nonempty consecutive
    block sums are distinct. Maximize sequence length.
    
    Key improvements:
    - Efficient prefix-sum based validity check
    - Smart insertion-focused moves with incremental conflict detection
    - Multiple restarts with adaptive cooling
    """
    
    def get_block_sums(seq):
        """Compute all nonempty consecutive block sums using prefix sums."""
        n = len(seq)
        if n == 0:
            return set()
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
        sums = set()
        for i in range(n):
            for j in range(i+1, n+1):
                sums.add(prefix[j] - prefix[i])
        return sums
    
    def is_valid(seq):
        """Check if sequence is strictly increasing and all consecutive block sums are distinct."""
        n = len(seq)
        if n == 0:
            return True
        for i in range(1, n):
            if seq[i] <= seq[i-1]:
                return False
        for v in seq:
            if v < 1 or v > 256:
                return False
        sums = get_block_sums(seq)
        expected = n * (n + 1) // 2
        return len(sums) == expected
    
    def check_insertion_conflict(seq, pos, val):
        """
        Check if inserting val at position pos in seq creates any duplicate block sum.
        Returns True if the insertion would be valid (no conflicts).
        Uses prefix sums for efficiency.
        """
        n = len(seq)
        # Build prefix sums of original seq
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
        
        # Existing block sums
        existing_sums = set()
        for i in range(n):
            for j in range(i+1, n+1):
                existing_sums.add(prefix[j] - prefix[i])
        
        # New block sums created by inserting val at pos
        # These are blocks that include the new element
        new_sums = set()
        # Blocks starting at pos (the new element) and extending right
        for end in range(pos, n):
            s = val
            for k in range(pos, end+1):
                s += seq[k]
            new_sums.add(s)
        # Blocks ending at pos (extending left to the new element)
        for start in range(0, pos+1):
            s = val
            for k in range(start, pos):
                s += seq[k]
            new_sums.add(s)
        
        # Check for conflicts: any new sum that's already in existing_sums
        # or any duplicate among new_sums themselves
        for s in new_sums:
            if s in existing_sums:
                return False
        
        # Check if new_sums have internal duplicates
        if len(new_sums) != len(list(new_sums)):
            # This can't happen with a set, but let's count properly
            pass
        
        # Count: total blocks after insertion = (n+1)*(n+2)/2
        # We have |existing_sums| + |new_sums| blocks, but some existing blocks
        # that end before pos or start after pos remain unchanged.
        # Actually, all existing blocks are still present. The new blocks are those
        # that include position pos (the inserted element).
        # Number of existing blocks: n*(n+1)/2
        # Number of new blocks (including val): (pos+1) * (n-pos)
        # But we need to check that all (n+1)*(n+2)/2 block sums are distinct.
        
        # Simpler: just check that new_sums don't overlap with existing_sums
        # and that new_sums are all distinct from each other.
        # Since new_sums is a set, internal distinctness is guaranteed.
        # So we just need: new_sums ∩ existing_sums = ∅
        
        return True
    
    def greedy_solution():
        """Greedy: pick smallest available value that doesn't create conflicts."""
        values = []
        sums = set()
        for x in range(1, 257):
            # Compute all new block sums if we append x
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(values):
                total += a
                new_sums.append(total)
            if not any(s in sums for s in new_sums):
                values.append(x)
                sums.update(new_sums)
        return values
    
    def simulated_annealing(initial_seq, max_iters=20000, start_temp=30.0, end_temp=0.01, cooling=0.99995):
        """Simulated annealing to maximize sequence length while maintaining validity."""
        current = list(initial_seq)
        current_len = len(current)
        best = list(current)
        best_len = current_len
        temp = start_temp
        
        # Precompute existing sums for fast checking
        def compute_sums(seq):
            n = len(seq)
            if n == 0:
                return set()
            prefix = [0] * (n + 1)
            for i in range(n):
                prefix[i+1] = prefix[i] + seq[i]
            sums = set()
            for i in range(n):
                for j in range(i+1, n+1):
                    sums.add(prefix[j] - prefix[i])
            return sums
        
        current_sums = compute_sums(current)
        
        for iteration in range(max_iters):
            temp = max(end_temp, temp * cooling)
            
            # Choose a move type: heavily favor insertion
            r = random.random()
            
            if r < 0.55:
                # Try to insert a new element
                if len(current) == 0:
                    new_val = random.randint(1, 256)
                    new_seq = [new_val]
                else:
                    pos = random.randint(0, len(current))
                    low = current[pos-1] + 1 if pos > 0 else 1
                    high = current[pos] - 1 if pos < len(current) else 256
                    if low > high:
                        continue
                    # Try several candidate values at this position
                    candidates = []
                    if high - low <= 10:
                        candidates = list(range(low, high+1))
                        random.shuffle(candidates)
                    else:
                        # Sample a few candidates
                        for _ in range(3):
                            candidates.append(random.randint(low, high))
                    
                    inserted = False
                    for new_val in candidates:
                        # Quick check: does inserting create a conflict?
                        # Compute new block sums involving the inserted element
                        # and check against current_sums
                        n = len(current)
                        # New blocks: those that include position pos
                        # Left part sum (elements from pos-1 down to some start)
                        # Right part sum (elements from pos up to some end)
                        
                        # Build prefix sums
                        prefix = [0] * (n + 1)
                        for i in range(n):
                            prefix[i+1] = prefix[i] + current[i]
                        
                        conflict = False
                        # For each possible start <= pos and end >= pos (in new indexing),
                        # compute the sum and check if it's in current_sums
                        # In new seq, the inserted element is at index pos.
                        # A block [i, j] in new seq includes the inserted element iff i <= pos <= j.
                        # The sum = (sum of current[0..pos-1] from i to pos-1) + val + (sum of current[pos..n-1] from pos to j-1)
                        
                        # Let left_sum(a, b) = sum of current[a:b] (a inclusive, b exclusive)
                        # For block starting at i (0 <= i <= pos) and ending at j (pos <= j <= n):
                        # sum = (prefix[pos] - prefix[i]) + val + (prefix[j] - prefix[pos])
                        #     = prefix[j] - prefix[i] + val
                        
                        # We need to check all (i, j) with 0 <= i <= pos and pos <= j <= n
                        for i in range(0, pos+1):
                            for j in range(pos, n+1):
                                s = prefix[j] - prefix[i] + new_val
                                if s in current_sums:
                                    conflict = True
                                    break
                            if conflict:
                                break
                        
                        if not conflict:
                            new_seq = current[:pos] + [new_val] + current[pos:]
                            # Also need to check that the new block sums among themselves are distinct
                            # But since we're adding to a set, if any two new blocks have same sum,
                            # that's a conflict too. Let's verify.
                            # Actually, two different blocks (i1,j1) and (i2,j2) both containing pos
                            # have sums prefix[j1]-prefix[i1]+val and prefix[j2]-prefix[i2]+val.
                            # They're equal iff prefix[j1]-prefix[i1] = prefix[j2]-prefix[i2].
                            # This means the left part + right part without val are equal.
                            # Since current is strictly increasing and all its block sums are distinct,
                            # this shouldn't happen unless the blocks are the same.
                            # But to be safe, let's just do a full check for small sequences.
                            if len(new_seq) <= 20:
                                if not is_valid(new_seq):
                                    continue
                            inserted = True
                            break
                    
                    if not inserted:
                        continue
            
            elif r < 0.75:
                # Try to remove an element
                if len(current) <= 1:
                    continue
                pos = random.randint(0, len(current) - 1)
                new_seq = current[:pos] + current[pos+1:]
            
            elif r < 0.9:
                # Try to change a value
                if len(current) == 0:
                    continue
                pos = random.randint(0, len(current) - 1)
                low = current[pos-1] + 1 if pos > 0 else 1
                high = current[pos+1] - 1 if pos < len(current) - 1 else 256
                if low >= high:
                    continue
                new_val = random.randint(low, high)
                if new_val == current[pos]:
                    continue
                new_seq = current[:pos] + [new_val] + current[pos+1:]
            
            else:
                # Try to shift an element
                if len(current) <= 2:
                    continue
                pos = random.randint(0, len(current) - 1)
                # Shift left or right by a small amount
                direction = random.choice([-1, 1])
                new_pos = pos + direction
                if new_pos < 0 or new_pos >= len(current):
                    continue
                val = current[pos]
                # Check bounds
                if new_pos > 0 and val <= current[new_pos-1]:
                    continue
                if new_pos < len(current)-1 and val >= current[new_pos+1]:
                    continue
                new_seq = current[:]
                new_seq.pop(pos)
                new_seq.insert(new_pos, val)
            
            # Validate new sequence
            if not is_valid(new_seq):
                continue
            
            new_len = len(new_seq)
            delta = new_len - current_len
            
            if delta >= 0:
                current = new_seq
                current_len = new_len
                current_sums = compute_sums(current)
                if new_len > best_len:
                    best = new_seq
                    best_len = new_len
            else:
                prob = np.exp(delta / temp)
                if random.random() < prob:
                    current = new_seq
                    current_len = new_len
                    current_sums = compute_sums(current)
        
        return best
    
    # Get greedy solution as a strong baseline
    greedy = greedy_solution()
    best_overall = list(greedy)
    best_overall_len = len(greedy)
    
    # Run simulated annealing with multiple restarts
    random.seed(42)
    
    # Time budget: leave some margin
    start_time = time.time()
    time_budget = 50.0  # seconds
    
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Create initial sequence: subset of greedy or random
        if restart % 2 == 0:
            # Subset of greedy
            if len(greedy) > 4:
                k = random.randint(max(3, len(greedy)//2), min(len(greedy)-1, len(greedy)))
                idx = random.sample(range(len(greedy)), k)
                idx.sort()
                initial = [greedy[i] for i in idx]
            else:
                initial = list(greedy)
        else:
            # Random valid subset
            attempts = 0
            initial = []
            while attempts < 200:
                k = random.randint(4, min(12, len(greedy)))
                candidates = random.sample(range(1, 257), k)
                candidates.sort()
                if is_valid(candidates):
                    initial = candidates
                    break
                attempts += 1
            if not initial:
                if len(greedy) > 3:
                    k = random.randint(3, len(greedy)-1)
                    idx = random.sample(range(len(greedy)), k)
                    idx.sort()
                    initial = [greedy[i] for i in idx]
                else:
                    initial = list(greedy)
        
        if len(initial) == 0:
            continue
        
        # Adaptive parameters based on restart
        max_iters = 15000 if restart < 4 else 10000
        start_temp = 25.0 + restart * 5
        cooling = 0.99997
        
        result = simulated_annealing(initial, max_iters=max_iters, start_temp=start_temp, end_temp=0.01, cooling=cooling)
        
        if len(result) > best_overall_len:
            best_overall = result
            best_overall_len = len(result)
    
    # Final verification
    if not is_valid(best_overall):
        # Fallback to greedy
        best_overall = greedy
        best_overall_len = len(greedy)
    
    return best_overall
# EVOLVE-BLOCK-END
