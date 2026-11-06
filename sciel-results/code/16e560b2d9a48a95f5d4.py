import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit with safety margin
    
    def get_all_block_sums(seq):
        """Get all non-empty consecutive block sums of a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def is_valid(seq):
        if not seq:
            return True
        if not all(seq[i] < seq[i+1] for i in range(len(seq)-1)):
            return False
        sums = set()
        for i in range(len(seq)):
            total = 0
            for j in range(i, len(seq)):
                total += seq[j]
                if total in sums:
                    return False
                sums.add(total)
        return True

    def greedy_build(start_val=1):
        values = []
        sums = set()
        for x in range(start_val, 257):
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

    def local_search_insert(seq):
        if not seq:
            return seq
        seq_set = set(seq)
        improved = True
        while improved:
            improved = False
            for x in range(1, 257):
                if x in seq_set:
                    continue
                for i in range(len(seq) + 1):
                    if i > 0 and seq[i-1] >= x:
                        continue
                    if i < len(seq) and seq[i] <= x:
                        continue
                    
                    new_seq = seq[:i] + [x] + seq[i:]
                    if is_valid(new_seq):
                        seq = new_seq
                        seq_set.add(x)
                        improved = True
                        break
                if improved:
                    break
                if time.time() > deadline:
                    return seq
        return seq

    def cp_search_with_gaps(max_len_target, initial_seq=None, max_backtracks=50000):
        """
        Constraint programming approach using gaps between consecutive elements.
        Variables are gaps g_i = a_{i+1} - a_i >= 1.
        Forward checking prunes based on sum constraints.
        """
        if initial_seq is None:
            initial_seq = [1]
        
        best_found = list(initial_seq)
        n0 = len(initial_seq)
        used_sums = get_all_block_sums(initial_seq)
        last_val = initial_seq[-1]
        
        # We'll build up to max_len_target
        # Use iterative deepening: try to extend by 1, then 2, etc.
        
        def backtrack(seq, used_sums, depth, max_depth, max_backtracks_remaining):
            nonlocal best_found
            if time.time() > deadline:
                return None, max_backtracks_remaining
            if len(seq) > len(best_found):
                best_found = list(seq)
            
            if len(seq) >= max_depth:
                return seq, max_backtracks_remaining
            
            if max_backtracks_remaining <= 0:
                return None, max_backtracks_remaining
            
            last = seq[-1]
            # Try adding next element: last + gap where gap >= 1
            # Forward check: the new element and all new block sums must not conflict
            
            candidates = []
            for gap in range(1, 257 - last):
                new_val = last + gap
                # Check: new_val itself as a block sum
                if new_val in used_sums:
                    continue
                # Check: new_val + suffix sums
                new_sums = {new_val}
                s = new_val
                for a in reversed(seq):
                    s += a
                    if s in used_sums or s in new_sums:
                        break
                    new_sums.add(s)
                else:
                    candidates.append((new_val, new_sums))
            
            if not candidates:
                return None, max_backtracks_remaining
            
            # Order by "most constrained" - prefer larger gaps (more room)
            # Actually, prefer candidates that leave more room
            candidates.sort(key=lambda x: -x[0])
            
            for new_val, new_sums in candidates:
                max_backtracks_remaining -= 1
                new_seq = seq + [new_val]
                new_used = used_sums | new_sums
                result, remaining = backtrack(new_seq, new_used, depth + 1, max_depth, max_backtracks_remaining)
                if result is not None:
                    return result, remaining
                if time.time() > deadline:
                    return None, 0
                if max_backtracks_remaining <= 0:
                    return None, 0
            
            return None, max_backtracks_remaining
        
        # Iterative deepening
        current_len = len(initial_seq)
        for target_len in range(current_len + 1, max_len_target + 1):
            if time.time() > deadline:
                break
            result, _ = backtrack(initial_seq, used_sums, 0, target_len, max_backtracks)
            if result is not None and len(result) >= target_len:
                initial_seq = list(result)
                used_sums = get_all_block_sums(initial_seq)
            else:
                # Couldn't extend further, stop
                break
        
        return best_found

    def randomized_cp_restart(initial_seq, num_restarts=5):
        """Try multiple randomized CP searches from different starting points."""
        best = list(initial_seq)
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            # Create a perturbed initial sequence
            perturbed = list(initial_seq)
            # Remove a random middle element
            if len(perturbed) > 2:
                idx = random.randint(1, len(perturbed) - 2)
                perturbed = perturbed[:idx] + perturbed[idx+1:]
            
            result = cp_search_with_gaps(len(initial_seq) + 3, perturbed, max_backtracks=30000)
            if len(result) > len(best):
                best = result
        
        return best

    # Phase 1: Greedy to get a good initial solution
    best_seq = []
    max_len = 0
    
    for start_val in [1, 2, 3, 4, 5]:
        if time.time() > deadline:
            break
        seq = greedy_build(start_val)
        seq = local_search_insert(seq)
        if len(seq) > max_len:
            max_len = len(seq)
            best_seq = seq
    
    # Phase 2: CP search with gaps to extend
    if best_seq and time.time() < deadline:
        extended = cp_search_with_gaps(max_len + 5, best_seq, max_backtracks=100000)
        if len(extended) > max_len:
            best_seq = extended
            max_len = len(best_seq)
    
    # Phase 3: Randomized restarts
    if best_seq and time.time() < deadline:
        improved = randomized_cp_restart(best_seq, num_restarts=8)
        if len(improved) > max_len:
            best_seq = improved
            max_len = len(best_seq)
    
    return best_seq
