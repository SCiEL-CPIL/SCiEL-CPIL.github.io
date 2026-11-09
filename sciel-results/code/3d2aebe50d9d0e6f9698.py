"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # 55s safety margin
    
    def get_block_sums(seq):
        """Generate all nonempty consecutive block sums for seq."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums
    
    def try_extend(seq, max_val=256):
        """Try to extend seq by one more element. Return new seq or None."""
        existing_sums = get_block_sums(seq)
        last = seq[-1] if seq else 0
        for x in range(last + 1, max_val + 1):
            # Check new sums: x, and x + suffix sums of seq
            new_sums = {x}
            suffix = 0
            for a in reversed(seq):
                suffix += a
                new_sums.add(x + suffix)
            if not existing_sums.intersection(new_sums):
                return seq + [x]
        return None
    
    def greedy_construct():
        """Greedy: always pick smallest valid next value."""
        seq = []
        while True:
            new_seq = try_extend(seq)
            if new_seq is None:
                break
            seq = new_seq
        return seq
    
    # Phase 1: Greedy construction
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # Phase 2: Local search - try to extend by modifying the tail
    # Strategy: remove last k elements, try different combinations
    while time.time() < deadline:
        improved = False
        for k in range(1, min(4, len(best_seq))):
            if time.time() >= deadline:
                break
            # Remove last k elements
            prefix = best_seq[:-k] if k < len(best_seq) else []
            existing_sums = get_block_sums(prefix)
            last = prefix[-1] if prefix else 0
            
            # Try to find k new values to append
            def backtrack_extend(prefix, existing_sums, k_remaining, max_val=256):
                if time.time() > deadline:
                    return None
                if k_remaining == 0:
                    return prefix
                last = prefix[-1] if prefix else 0
                for x in range(last + 1, max_val + 1):
                    new_sums = {x}
                    suffix = 0
                    for a in reversed(prefix):
                        suffix += a
                        new_sums.add(x + suffix)
                    if existing_sums.intersection(new_sums):
                        continue
                    new_prefix = prefix + [x]
                    new_existing = existing_sums | new_sums
                    result = backtrack_extend(new_prefix, new_existing, k_remaining - 1, max_val)
                    if result is not None:
                        return result
                return None
            
            result = backtrack_extend(prefix, existing_sums, k)
            if result is not None and len(result) > best_len:
                best_seq = result
                best_len = len(result)
                improved = True
                break
        if not improved and time.time() > deadline - 5:
            break
    
    return best_seq
# EVOLVE-BLOCK-END
