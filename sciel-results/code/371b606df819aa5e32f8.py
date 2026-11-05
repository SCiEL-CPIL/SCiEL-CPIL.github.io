import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit to be safe
    
    # Fast validity check using prefix sums
    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        P = [0] * (n + 1)
        for i in range(n):
            P[i + 1] = P[i] + seq[i]
        
        sums = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                s = P[j] - P[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Compute the set of all consecutive block sums for a sequence
    def get_sums(seq):
        n = len(seq)
        P = [0] * (n + 1)
        for i in range(n):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                sums.add(P[j] - P[i])
        return sums

    # Greedy build: try to append x to the end of current sequence
    # Returns new sequence if valid, else None
    def try_append(seq, x, sums_set):
        n = len(seq)
        # New sums are: x, x + seq[n-1], x + seq[n-1] + seq[n-2], ...
        # i.e., x + suffix_sum_k for k=0..n
        # suffix_sum_0 = 0
        # suffix_sum_k = sum(seq[n-k:n])
        
        # Calculate suffix sums
        # Actually, the new sums when appending x are:
        # Block ending at new last index (n):
        # sum(seq[n-1:n]) + x ? No.
        # The sequence becomes seq[0..n-1] + [x].
        # New blocks are those that include x.
        # These are sum(seq[i:n]) + x for i in 0..n.
        # Let S_i = sum(seq[i:n]).
        # New sums = { S_i + x for i in 0..n }
        
        # We can compute these efficiently.
        # S_n = 0
        # S_{n-1} = seq[n-1]
        # ...
        
        total = 0
        new_sums = []
        # S_n = 0 -> sum = x
        new_sums.append(x)
        
        # Iterate backwards
        for i in range(n - 1, -1, -1):
            total += seq[i]
            new_sums.append(total + x)
            
        # Check for collisions with existing sums
        # And internal collisions (shouldn't happen if seq is strictly increasing and x > seq[-1])
        # Since seq is strictly increasing, S_i are strictly decreasing as i decreases?
        # S_i = sum(seq[i:n]). Since seq[k] > 0, S_i > S_{i+1}.
        # So S_i + x are all distinct.
        
        for ns in new_sums:
            if ns in sums_set:
                return None, None
                
        # Also check if x itself is valid (x is in new_sums, covered)
        
        # If we get here, it's valid
        new_seq = seq + [x]
        new_sums_set = sums_set.union(new_sums)
        return new_seq, new_sums_set

    # 1. Baseline Greedy (Ascending)
    # This is the most reliable way to get a good starting point
    best_seq = []
    best_sums = set()
    for x in range(1, 257):
        new_seq, new_sums = try_append(best_seq, x, best_sums)
        if new_seq is not None:
            best_seq = new_seq
            best_sums = new_sums
            
    best_len = len(best_seq)
    current_seq = best_seq[:]
    current_sums = best_sums.copy()
    current_set = set(current_seq)
    
    # 2. Local Search with efficient sum updates
    # We will use a mix of add/remove/swap.
    # For remove, we need to recalculate sums or do complex delta updates.
    # Given the small N (~15-20), full recalculation of sums on change is O(N^2) which is fast enough.
    # N=20 -> 210 sums. 55s / 210 ops... we can do millions of ops.
    
    def rebuild_sums(seq):
        return get_sums(seq)

    rng = random.Random(42)
    stall_counter = 0
    last_improvement_time = start_time
    
    # Pre-compute all candidates
    all_candidates = set(range(1, 257))
    
    while time.time() < deadline:
        r = rng.random()
        
        if r < 0.6:
            # ADD
            available = list(all_candidates - current_set)
            if not available:
                break
            x = rng.choice(available)
            
            # Try to insert x into the sorted sequence at the correct position
            # Since we must maintain strictly increasing order, we can just binary search insert
            # But we need to check validity of the *new* sequence.
            # Efficient way:
            # 1. Find insertion index
            # 2. Construct candidate sequence
            # 3. Check validity
            
            # Optimization: Instead of full check, we can check collisions.
            # But inserting in the middle changes suffix sums for all elements to the right.
            # This makes incremental update hard.
            # Given N is small, full check is fine.
            
            # Insert x
            pos = 0
            while pos < len(current_seq) and current_seq[pos] < x:
                pos += 1
            candidate_seq = current_seq[:pos] + [x] + current_seq[pos:]
            
            if is_valid(candidate_seq):
                current_seq = candidate_seq
                current_set.add(x)
                current_sums = rebuild_sums(current_seq)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    last_improvement_time = time.time()
                    stall_counter = 0
                else:
                    stall_counter += 1
            else:
                stall_counter += 1
                
        elif r < 0.8:
            # REMOVE
            if current_seq:
                x = rng.choice(current_seq)
                current_seq.remove(x)
                current_set.remove(x)
                current_sums = rebuild_sums(current_seq)
                stall_counter += 1
        else:
            # SWAP
            if current_seq:
                x_to_remove = rng.choice(current_seq)
                available = list(all_candidates - current_set)
                if available:
                    x_to_add = rng.choice(available)
                    if x_to_add != x_to_remove:
                        new_seq = [v for v in current_seq if v != x_to_remove]
                        # Insert x_to_add
                        pos = 0
                        while pos < len(new_seq) and new_seq[pos] < x_to_add:
                            pos += 1
                        new_seq.insert(pos, x_to_add)
                        
                        if is_valid(new_seq):
                            current_seq = new_seq
                            current_set.remove(x_to_remove)
                            current_set.add(x_to_add)
                            current_sums = rebuild_sums(current_seq)
                            if len(current_seq) > best_len:
                                best_len = len(current_seq)
                                best_seq = current_seq[:]
                                last_improvement_time = time.time()
                                stall_counter = 0
                            else:
                                stall_counter += 1
                        else:
                            stall_counter += 1
        
        # Periodic Greedy Polish
        # Try to fill gaps greedily
        if stall_counter > 20 or (time.time() - last_improvement_time > 1.0 and stall_counter > 5):
            # Try to add any available number in sorted order
            improved = False
            test_seq = current_seq[:]
            test_set = set(test_seq)
            
            # We'll try to add numbers in ascending order to the current sequence
            # This mimics the initial greedy but on the current state
            for x in sorted(all_candidates - test_set):
                # Try to insert x
                pos = 0
                while pos < len(test_seq) and test_seq[pos] < x:
                    pos += 1
                candidate_seq = test_seq[:pos] + [x] + test_seq[pos:]
                
                if is_valid(candidate_seq):
                    test_seq = candidate_seq
                    test_set.add(x)
                    improved = True
            
            if len(test_seq) > len(current_seq):
                current_seq = test_seq
                current_set = test_set
                current_sums = rebuild_sums(current_seq)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    last_improvement_time = time.time()
                stall_counter = 0
            else:
                if stall_counter > 100:
                    # Random Restart
                    # Try a different greedy order?
                    # Or just random shuffle greedy
                    order = list(range(1, 257))
                    rng.shuffle(order)
                    restart_seq = []
                    for x in order:
                        # Insert x
                        pos = 0
                        while pos < len(restart_seq) and restart_seq[pos] < x:
                            pos += 1
                        candidate_seq = restart_seq[:pos] + [x] + restart_seq[pos:]
                        if is_valid(candidate_seq):
                            restart_seq = candidate_seq
                            
                    if len(restart_seq) > len(current_seq):
                        current_seq = restart_seq
                        current_set = set(current_seq)
                        current_sums = rebuild_sums(current_seq)
                    else:
                        # Perturb current
                        if len(current_seq) > 2:
                            to_remove = rng.sample(current_seq, min(3, len(current_seq)))
                            for x in to_remove:
                                current_seq.remove(x)
                                current_set.remove(x)
                            current_sums = rebuild_sums(current_seq)
                    stall_counter = 0

    return best_seq

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)
