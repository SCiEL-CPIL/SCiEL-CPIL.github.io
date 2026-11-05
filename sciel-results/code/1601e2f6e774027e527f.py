import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def compute_block_sums(seq):
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def can_add_incremental(seq, x, existing_sums):
        """Check if adding x to sorted seq keeps all block sums distinct.
        Only checks new blocks that include x."""
        n = len(seq)
        pos = 0
        while pos < n and seq[pos] < x:
            pos += 1
        
        # Build new sequence
        new_seq = seq[:pos] + [x] + seq[pos:]
        
        # Check all blocks that include the new element at position pos
        # Block [i, j] includes x iff i <= pos <= j
        for i in range(pos + 1):
            s = 0
            for j in range(i, pos + 1):
                s += new_seq[j]
                if s in existing_sums:
                    return False, None
        return True, new_seq

    def greedy_build(order):
        selected = []
        existing_sums = set()
        for x in order:
            if x in selected:
                continue
            can, new_seq = can_add_incremental(selected, x, existing_sums)
            if can:
                selected = new_seq
                existing_sums = compute_block_sums(selected)
        return selected

    best_seq = []
    seq_asc = greedy_build(list(range(1, 257)))
    if len(seq_asc) > len(best_seq):
        best_seq = seq_asc
    seq_desc = greedy_build(list(range(256, 0, -1)))
    if len(seq_desc) > len(best_seq):
        best_seq = seq_desc
        
    best_len = len(best_seq)
    all_candidates = set(range(1, 257))
    rng = random.Random(42)
    
    current_seq = best_seq[:]
    current_set = set(current_seq)
    current_sums = compute_block_sums(current_seq)
    
    stall_counter = 0
    last_improvement_time = start_time
    
    while time.time() < deadline:
        r = rng.random()
        
        if r < 0.7:
            available = list(all_candidates - current_set)
            if not available:
                break
            x = rng.choice(available)
            can, new_seq = can_add_incremental(current_seq, x, current_sums)
            if can:
                current_seq = new_seq
                current_set.add(x)
                current_sums = compute_block_sums(current_seq)
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    last_improvement_time = time.time()
                    stall_counter = 0
                else:
                    stall_counter += 1
            else:
                stall_counter += 1
        elif r < 0.85:
            if current_seq:
                x = rng.choice(current_seq)
                pos = current_seq.index(x)
                new_seq = current_seq[:pos] + current_seq[pos+1:]
                current_seq = new_seq
                current_set.remove(x)
                current_sums = compute_block_sums(current_seq)
                stall_counter += 1
        else:
            if current_seq:
                x_to_remove = rng.choice(current_seq)
                available = list(all_candidates - current_set)
                if available:
                    x_to_add = rng.choice(available)
                    if x_to_add != x_to_remove:
                        pos = current_seq.index(x_to_remove)
                        new_seq = current_seq[:pos] + current_seq[pos+1:]
                        new_sums = compute_block_sums(new_seq)
                        can, final_seq = can_add_incremental(new_seq, x_to_add, new_sums)
                        if can:
                            current_seq = final_seq
                            current_set.remove(x_to_remove)
                            current_set.add(x_to_add)
                            current_sums = compute_block_sums(current_seq)
                            if len(current_seq) > best_len:
                                best_len = len(current_seq)
                                best_seq = current_seq[:]
                                last_improvement_time = time.time()
                                stall_counter = 0
                            else:
                                stall_counter += 1
                        else:
                            stall_counter += 1
        
        if stall_counter > 50 or (time.time() - last_improvement_time > 2.0 and stall_counter > 10):
            test_seq = current_seq[:]
            test_set = set(test_seq)
            test_sums = compute_block_sums(test_seq)
            
            for x in sorted(all_candidates - test_set):
                can, new_seq = can_add_incremental(test_seq, x, test_sums)
                if can:
                    test_seq = new_seq
                    test_set.add(x)
                    test_sums = compute_block_sums(test_seq)
            
            if len(test_seq) > len(current_seq):
                current_seq = test_seq
                current_set = test_set
                current_sums = test_sums
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq = current_seq[:]
                    last_improvement_time = time.time()
                stall_counter = 0
            else:
                if stall_counter > 200:
                    order = list(range(1, 257))
                    rng.shuffle(order)
                    restart_seq = greedy_build(order)
                    if len(restart_seq) > len(current_seq):
                        current_seq = restart_seq
                        current_set = set(current_seq)
                        current_sums = compute_block_sums(current_seq)
                    else:
                        if len(current_seq) > 3:
                            to_remove = rng.sample(current_seq, 3)
                            for x in to_remove:
                                pos = current_seq.index(x)
                                current_seq = current_seq[:pos] + current_seq[pos+1:]
                                current_set.remove(x)
                            current_sums = compute_block_sums(current_seq)
                    stall_counter = 0
                    
    return best_seq

if __name__ == "__main__":
    res = construct()
    pass
