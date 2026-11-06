"""Direct construction of a strictly increasing sequence in [1,256]
with all nonempty consecutive-block sums distinct.
Uses bitset-based DFS with aggressive pruning to maximize sequence length.
Deadline guard ensures we return best result before 90s wall-clock limit."""
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    
    # Precompute: for a sequence of length n, there are n*(n+1)//2 consecutive block sums.
    # We need all these sums to be distinct and within [1, 256*n].
    # Max possible n: need n*(n+1)//2 distinct sums, each in [1, 256*n].
    # So n*(n+1)//2 <= 256*n => (n+1)//2 <= 256 => n <= 511. But values are in [1,256].
    # Actually the constraint is tighter because values must be in [1,256] and strictly increasing.
    # Max length is bounded by 256 (all values distinct in [1,256]).
    # But with the sum-distinctness constraint, it's much smaller.
    
    # Key insight: use DFS with bitset representation of used sums.
    # For a sequence a[0..k-1], the set of all consecutive block sums can be computed.
    # When trying to append value v, the new sums are: v, v+a[k-1], v+a[k-1]+a[k-2], etc.
    
    best_sequence = []
    best_len = 0
    
    def elapsed():
        return time.time() - start_time
    
    def check_time():
        return elapsed() > deadline
    
    # DFS approach: build sequence left to right, trying each candidate value.
    # Use a set of used sums for quick conflict checking.
    
    def dfs(current_seq, used_sums, max_val):
        nonlocal best_sequence, best_len
        
        if check_time():
            return
        
        n = len(current_seq)
        if n > best_len:
            best_len = n
            best_sequence = list(current_seq)
        
        # Try to extend: next value must be > current_seq[-1] and <= 256
        if n == 0:
            start_candidate = 1
        else:
            start_candidate = current_seq[-1] + 1
        
        if start_candidate > 256:
            return
        
        for v in range(start_candidate, 257):
            if check_time():
                return
            
            # Compute all new consecutive block sums ending at v
            new_sums = []
            total = v
            new_sums.append(total)
            for a in reversed(current_seq):
                total += a
                new_sums.append(total)
            
            # Check for conflicts with existing sums
            conflict = False
            for s in new_sums:
                if s in used_sums:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Also check new_sums don't have internal duplicates (they can't since sequence is increasing)
            # Actually they could: e.g., if current_seq = [1,3], v=2: new sums = [2, 5, 6] - all distinct
            # But wait, v must be > current_seq[-1], so v > all previous elements.
            # new_sums[0] = v
            # new_sums[1] = v + current_seq[-1]
            # new_sums[2] = v + current_seq[-1] + current_seq[-2]
            # Since all elements are positive and increasing, new_sums is strictly increasing.
            # So no internal conflicts.
            
            # Add to sequence and recurse
            current_seq.append(v)
            used_sums.update(new_sums)
            
            dfs(current_seq, used_sums, 256)
            
            current_seq.pop()
            used_sums.difference_update(new_sums)
        
        return
    
    # Start DFS
    dfs([], set(), 256)
    
    # If DFS didn't find anything (shouldn't happen), fall back to greedy
    if not best_sequence:
        values = []
        sums = set()
        for x in range(1, 257):
            if check_time():
                break
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        best_sequence = values
    
    return best_sequence
