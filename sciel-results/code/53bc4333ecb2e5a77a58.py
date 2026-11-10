import time
import random

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before 60s
    
    # 1. Greedy Initialization
    def greedy_construct():
        values = []
        products = set()
        for x in range(2, 129):
            valid = True
            p = x
            tails = [p]
            for a in reversed(values):
                p *= a
                tails.append(p)
                if p in products:
                    valid = False
                    break
            
            if valid:
                values.append(x)
                products.update(tails)
        return values

    seq = greedy_construct()
    
    # Helper to check validity of a sequence
    def is_valid(sequence):
        products = set()
        n = len(sequence)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= sequence[j]
                if p in products:
                    return False
                products.add(p)
        return True

    # Try 1-out/2-in: remove element at position i, insert two new elements x, y
    def try_1out_2in(seq):
        n = len(seq)
        used = set(seq)
        unused = [x for x in range(2, 129) if x not in used]
        
        for i in range(n):
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+1] if i < n-1 else 129
            
            candidates = [x for x in unused if left_bound < x < right_bound]
            
            if len(candidates) < 2:
                continue
            
            for idx_x in range(len(candidates)):
                x = candidates[idx_x]
                for idx_y in range(idx_x + 1, len(candidates)):
                    y = candidates[idx_y]
                    new_seq = seq[:i] + [x, y] + seq[i+1:]
                    if is_valid(new_seq):
                        return new_seq
        return None

    # Try 2-out/2-in: remove two consecutive elements at positions i, i+1, insert two new elements x, y
    def try_2out_2in(seq):
        n = len(seq)
        if n < 2:
            return None
        used = set(seq)
        unused = [x for x in range(2, 129) if x not in used]
        
        for i in range(n - 1):
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+2] if i+2 < n else 129
            
            candidates = [x for x in unused if left_bound < x < right_bound]
            
            if len(candidates) < 2:
                continue
            
            for idx_x in range(len(candidates)):
                x = candidates[idx_x]
                for idx_y in range(idx_x + 1, len(candidates)):
                    y = candidates[idx_y]
                    new_seq = seq[:i] + [x, y] + seq[i+2:]
                    if is_valid(new_seq):
                        return new_seq
        return None

    # Try 1-out/1-in: replace element at position i with a new element x
    def try_1out_1in(seq):
        n = len(seq)
        used = set(seq)
        unused = [x for x in range(2, 129) if x not in used]
        
        for i in range(n):
            left_bound = seq[i-1] if i > 0 else 1
            right_bound = seq[i+1] if i < n-1 else 129
            
            candidates = [x for x in unused if left_bound < x < right_bound]
            
            for x in candidates:
                new_seq = seq[:i] + [x] + seq[i+1:]
                if is_valid(new_seq):
                    return new_seq
        return None

    # Main loop
    best_seq = seq
    while time.time() - start_time < timeout:
        improved = False
        
        # Try 1-out/2-in first (increases length)
        new_seq = try_1out_2in(seq)
        if new_seq is not None:
            seq = new_seq
            if len(seq) > len(best_seq):
                best_seq = seq
            improved = True
        
        # Try 2-out/2-in (same length, may enable future improvements)
        if not improved:
            new_seq = try_2out_2in(seq)
            if new_seq is not None:
                seq = new_seq
                improved = True
        
        # Try 1-out/1-in (same length, may enable future improvements)
        if not improved:
            new_seq = try_1out_1in(seq)
            if new_seq is not None:
                seq = new_seq
                improved = True
        
        if not improved:
            break
    
    return best_seq
