import time
import random

class SubsetSumTracker:
    __slots__ = ['offset', 'max_sum', 'mask', 'max_k', 'bits', 'current_k']

    def __init__(self, max_val=512, max_items=50):
        # Max possible sum difference: 512 * 50 = 25600
        # Add margin
        self.offset = 26000
        self.max_k = max_items
        self.mask = (1 << (2 * self.offset + 1)) - 1
        self.bits = [0] * (self.max_k + 1)
        self.bits[0] = 1 << self.offset
        self.current_k = 0

    def add(self, val):
        if self.current_k >= self.max_k:
            return
        
        old_bits = self.bits[:]
        
        # Update from high k to low k? 
        # No, we use old_bits, so order doesn't matter for correctness, 
        # but we must ensure we don't overwrite bits[k] before using it for bits[k+1] if we were in-place.
        # Since we use old_bits, we can iterate 1..current_k+1.
        
        for k in range(1, self.current_k + 2):
            if k > self.max_k:
                break
            
            if k == 1:
                shifted = old_bits[0]
            else:
                shifted = old_bits[k-1]
            
            if val >= 0:
                new_part = (shifted << val) & self.mask
            else:
                new_part = (shifted >> (-val)) & self.mask
            
            self.bits[k] = (old_bits[k] | new_part) & self.mask
        
        self.current_k += 1

    def can_make(self, target, min_count):
        target_idx = target + self.offset
        if target_idx < 0 or target_idx > 2 * self.offset:
            return False
        bit_mask = 1 << target_idx
        for k in range(min_count, self.current_k + 1):
            if self.bits[k] & bit_mask:
                return True
        return False

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    best_set = []
    best_len = 0
    
    def try_ordering(order):
        nonlocal best_set, best_len
        if time.time() > deadline:
            return
        
        A = []
        trackers = {}
        
        for x in order:
            if time.time() > deadline:
                break
            
            valid = True
            
            # Check 1: x is mean of subset of A
            if len(A) >= 2:
                t_temp = SubsetSumTracker()
                for a in A:
                    t_temp.add(a - x)
                if t_temp.can_make(0, 2):
                    valid = False
            
            # Check 2: x breaks existing a
            if valid:
                for a in A:
                    target = a - x
                    if trackers[a].can_make(target, 1):
                        valid = False
                        break
            
            if valid:
                A.append(x)
                
                # Update trackers
                for a in A[:-1]:
                    trackers[a].add(x - a)
                
                t_x = SubsetSumTracker()
                for a in A[:-1]:
                    t_x.add(a - x)
                trackers[x] = t_x
                
                if len(A) > best_len:
                    best_len = len(A)
                    best_set = A[:]
                
                if best_len == 48:
                    return

    # Strategy 1: Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509]
    try_ordering(primes)
    
    if best_len == 48: return best_set

    # Strategy 2: Powers of 2 (scaled to fit)
    # 1, 2, 4, ..., 256
    pows = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # Add some variations
    try_ordering(pows)
    
    if best_len == 48: return best_set

    # Strategy 3: Random restarts
    for i in range(1000):
        if time.time() > deadline:
            break
        order = list(range(1, 513))
        random.shuffle(order)
        try_ordering(order)
        if best_len == 48:
            return best_set
            
    return best_set
