import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_seq = []
    best_len = 0
    
    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq.copy()
    
    def get_block_sums(seq):
        """Get all consecutive block sums as a set."""
        n = len(seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums
    
    def is_valid_insertion(seq, pos, x, existing_sums):
        """Check if inserting x at position pos keeps all block sums distinct."""
        n = len(seq)
        new_sums = set(existing_sums)
        
        # Check new block sums involving x
        # Suffixes ending at x (from left)
        s = x
        for i in range(pos - 1, -1, -1):
            if s in new_sums:
                return False
            new_sums.add(s)
            s += seq[i]
        
        # Prefixes starting at x (to right) - x itself already added
        s = x
        for j in range(pos + 1, n):
            if s in new_sums:
                return False
            new_sums.add(s)
            s += seq[j]
        
        # Blocks spanning x: a[i..pos-1] + x + a[pos+1..j]
        # Left partial sums
        left_sums = [0]
        for i in range(pos - 1, -1, -1):
            left_sums.append(left_sums[-1] + seq[i])
        
        # Right partial sums  
        right_sums = [0]
        for j in range(pos, n):
            right_sums.append(right_sums[-1] + seq[j])
        
        # For each left extension and right extension
        for ls in left_sums:
            for rs in right_sums:
                total = ls + x + rs
                if ls == 0 and rs == 0:
                    continue  # Already checked (just x)
                if total in new_sums:
                    return False
                new_sums.add(total)
        
        return True
    
    def greedy_construct(seed=None):
        """Greedy construction with randomization."""
        if seed is not None:
            random.seed(seed)
        
        seq = []
        used = set()
        existing_sums = set()
        
        for _ in range(256):
            if time.time() > deadline:
                break
            
            candidates = []
            for pos in range(len(seq) + 1):
                # Determine valid range for x at this position
                if pos > 0:
                    min_x = seq[pos - 1] + 1
                else:
                    min_x = 1
                if pos < len(seq):
                    max_x = seq[pos] - 1
                else:
                    max_x = 256
                
                if min_x > max_x:
                    continue
                
                for x in range(min_x, max_x + 1):
                    if x in used:
                        continue
                    if is_valid_insertion(seq, pos, x, existing_sums):
                        candidates.append((pos, x))
            
            if not candidates:
                break
            
            # Pick randomly from candidates (or first for deterministic)
            if seed is not None:
                pos, x = random.choice(candidates)
            else:
                pos, x = candidates[0]
            
            seq.insert(pos, x)
            used.add(x)
            existing_sums = get_block_sums(seq)
        
        return seq
    
    def local_search(seq):
        """Iterative local search: try inserting elements at various positions."""
        if time.time() > deadline:
            return seq
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(seq)
            existing_sums = get_block_sums(seq)
            
            # Try all possible insertions
            for pos in range(n + 1):
                if time.time() > deadline:
                    return seq
                
                if pos > 0:
                    min_x = seq[pos - 1] + 1
                else:
                    min_x = 1
                if pos < n:
                    max_x = seq[pos] - 1
                else:
                    max_x = 256
                
                if min_x > max_x:
                    continue
                
                # Try candidates in random order for diversity
                candidates = list(range(min_x, max_x + 1))
                random.shuffle(candidates)
                
                for x in candidates:
                    if x in seq:
                        continue
                    if is_valid_insertion(seq, pos, x, existing_sums):
                        new_seq = seq[:pos] + [x] + seq[pos:]
                        seq = new_seq
                        improved = True
                        update_best(seq)
                        existing_sums = get_block_sums(seq)
                        break  # Move to next position
        
        return seq
    
    # Strategy 1: Deterministic greedy
    if time.time() < deadline:
        g_seq = greedy_construct(seed=None)
        update_best(g_seq)
    
    # Strategy 2: Multiple randomized greedy + local search
    for seed in range(20):
        if time.time() > deadline:
            break
        g_seq = greedy_construct(seed=seed)
        if len(g_seq) >= best_len:
            ls_seq = local_search(g_seq)
            update_best(ls_seq)
    
    # Strategy 3: Local search on best found
    if time.time() < deadline and best_seq:
        ls_seq = local_search(best_seq.copy())
        update_best(ls_seq)
    
    # Strategy 4: More aggressive local search with multiple passes
    if time.time() < deadline and best_seq:
        for _ in range(5):
            if time.time() > deadline:
                break
            ls_seq = local_search(best_seq.copy())
            update_best(ls_seq)
    
    return best_seq
