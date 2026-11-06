import time
import sys
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    TIMEOUT = 55.0  # Safety margin for 60s limit

    def get_all_sums(seq):
        """Get all non-empty consecutive block sums of seq."""
        sums = set()
        for i in range(len(seq)):
            s = 0
            for j in range(i, len(seq)):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid_sequence(seq):
        """Check if a sequence has all distinct consecutive block sums."""
        if len(seq) == 0:
            return True
        sums = get_all_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    def greedy_build(direction='ascending'):
        """Greedy construction of a valid sequence."""
        vals = []
        sums_set = set()
        
        if direction == 'ascending':
            candidates = range(1, 257)
        else:
            candidates = range(256, 0, -1)
        
        for x in candidates:
            # Calculate new suffix sums if we append x
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(vals):
                total += a
                new_sums.append(total)
            
            if not sums_set.intersection(new_sums):
                vals.append(x)
                sums_set.update(new_sums)
        
        # Return in increasing order
        return sorted(vals)

    # Multiple greedy starts with different directions
    best_seq = []
    best_len = 0
    
    for direction in ['ascending', 'descending']:
        seq = greedy_build(direction)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
    
    # Local search refinement: try to improve by removing and re-inserting elements
    def local_search(seq, max_iterations=1000):
        """Improve sequence via local search."""
        seq = list(seq)
        current_len = len(seq)
        improved = True
        
        while improved and time.time() - start_time < TIMEOUT:
            improved = False
            
            # Try removing each element and re-inserting at a better position
            for i in range(len(seq)):
                old_elem = seq[i]
                seq_without = seq[:i] + seq[i+1:]
                
                # Try inserting old_elem at various positions
                for j in range(len(seq_without) + 1):
                    candidate = seq_without[:j] + [old_elem] + seq_without[j:]
                    if not is_valid_sequence(candidate):
                        continue
                    
                    # Try extending from this candidate
                    extended = try_extend(candidate)
                    if len(extended) > current_len:
                        seq = extended
                        current_len = len(seq)
                        improved = True
                        break
                
                if improved:
                    break
            
            # Also try direct extension
            extended = try_extend(seq)
            if len(extended) > current_len:
                seq = extended
                current_len = len(seq)
                improved = True
        
        return seq

    def try_extend(seq):
        """Try to extend a sequence greedily."""
        seq = list(seq)
        sums_set = get_all_sums(seq)
        
        last_val = seq[-1] if seq else 0
        for x in range(last_val + 1, 257):
            new_sums = []
            total = x
            new_sums.append(total)
            for a in reversed(seq):
                total += a
                new_sums.append(total)
            
            if not sums_set.intersection(new_sums):
                seq.append(x)
                sums_set.update(new_sums)
        
        return seq

    # Apply local search to best sequence
    if best_seq:
        refined = local_search(best_seq)
        if len(refined) > best_len:
            best_len = len(refined)
            best_seq = refined

    # Maximum clique approach: build compatibility graph
    # Two values a < b can coexist if appending b to [a] doesn't create sum conflicts
    # But we need to consider the full sequence context, so use a different approach
    
    # Alternative: iterative deepening with pruning
    def dfs_improve(seq, sums_set, suffix_sums, depth, max_depth):
        """DFS to try to extend sequence to max_depth."""
        nonlocal best_seq, best_len, start_time, TIMEOUT
        
        if time.time() - start_time > TIMEOUT:
            return False
        
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)
        
        if len(seq) >= max_depth:
            return True
        
        # Pruning: check if we can possibly reach max_depth
        if seq:
            max_possible = len(seq) + (256 - seq[-1])
        else:
            max_possible = 256
        
        if max_possible < max_depth:
            return False
        
        last_val = seq[-1] if seq else 0
        
        # Try each candidate
        for x in range(last_val + 1, 257):
            new_sums_list = [x + s for s in [0] + suffix_sums]
            
            if any(s in sums_set for s in new_sums_list):
                continue
            
            new_sums_set = sums_set.union(new_sums_list)
            new_suffix_sums = new_sums_list
            
            seq.append(x)
            found = dfs_improve(seq, new_sums_set, new_suffix_sums, depth + 1, max_depth)
            seq.pop()
            
            if found:
                return True
            
            if time.time() - start_time > TIMEOUT:
                return False
        
        return False

    # Try to improve using DFS with increasing depth targets
    current_target = best_len + 1
    while current_target <= best_len + 5 and time.time() - start_time < TIMEOUT:
        # Start from best_seq and try to reach current_target
        initial_sums = get_all_sums(best_seq)
        initial_suffixes = []
        total = 0
        for a in reversed(best_seq):
            total += a
            initial_suffixes.append(total)
        
        found = dfs_improve(list(best_seq), initial_sums, initial_suffixes, 0, current_target)
        if found:
            current_target += 1
        else:
            break

    return best_seq
# EVOLVE-BLOCK-END
