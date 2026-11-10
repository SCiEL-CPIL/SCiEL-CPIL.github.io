import time
import sys

# Increase recursion limit just in case, though n is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    best_seq = []
    
    # used_sums will be a set of integers
    # seq will be a list of integers
    
    def dfs(seq, used_sums, last_val):
        nonlocal best_seq, deadline
        
        # Check time
        if time.time() > deadline:
            return
        
        # Update best if current is better
        if len(seq) > len(best_seq):
            best_seq = seq.copy()
            
        # Pruning: Can we possibly beat best_seq?
        # Max possible length is bounded by 256 - last_val + 1 (if we could pick consecutive ints)
        # But we need distinct sums, which is harder.
        # A simple bound: if len(seq) + (256 - last_val) <= len(best_seq), we can't beat it.
        # This bound is very loose but helps in deep, narrow branches.
        if len(seq) + (256 - last_val) <= len(best_seq):
            return

        # Try to extend
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
                
            # Calculate new sums
            # New sums are x, last_val + x, prev_last + last_val + x, ...
            # We can maintain a list of "suffix sums" of the current seq to make this faster?
            # Or just calculate on the fly.
            # Let's calculate on the fly for simplicity first, it's O(n) per check.
            
            new_sums = []
            s = 0
            # We need sums of:
            # 1. x
            # 2. a[-1] + x
            # 3. a[-2] + a[-1] + x
            # ...
            # k. a[0]...a[-1] + x
            
            # It is more efficient to precompute suffix sums of the current sequence?
            # Let's pass a list of suffix sums to avoid recalculating.
            # suffix_sums[i] = sum(seq[-i-1:]) for i=0..len(seq)-1
            # Actually, let's just recalculate. n is small.
            
            # Optimization: Check if x itself is in used_sums
            if x in used_sums:
                continue
                
            # Check other sums
            valid = True
            s = x
            new_sums.append(s)
            
            # We need to check sums ending at x.
            # Let's iterate backwards through seq
            # s = x
            # s += seq[-1]
            # s += seq[-2]
            # ...
            
            # To avoid creating a list of new_sums and then checking, we can check one by one.
            # But we need to add them all to used_sums if valid.
            
            # Let's compute the sums and check intersection.
            # We can create a temporary set of new sums?
            # Or just check each one.
            
            # Let's use a helper to get the sums efficiently.
            # Actually, maintaining a `suffix_sums` list in the state is better.
            # Let's refactor to pass `suffix_sums`.
            # suffix_sums[0] = 0 (empty)
            # suffix_sums[1] = seq[-1]
            # suffix_sums[2] = seq[-1] + seq[-2]
            # ...
            # New sums will be x + suffix_sums[i] for i=0..len(seq)
            
            # Wait, I didn't pass suffix_sums in the initial draft.
            # Let's modify dfs signature or compute inside.
            # Computing inside is O(n^2) per node if we do it naively.
            # With suffix sums passed, it's O(n) per node to generate candidates, O(n) to check.
            
            # Let's stick to the simpler version first, but optimize the sum check.
            
            # Check x
            # Check x + seq[-1]
            # Check x + seq[-1] + seq[-2]
            # ...
            
            s = x
            if s in used_sums:
                continue
            new_sums_to_add = [s]
            
            for i in range(len(seq) - 1, -1, -1):
                s += seq[i]
                if s in used_sums:
                    valid = False
                    break
                new_sums_to_add.append(s)
            
            if not valid:
                continue
                
            # All new sums are distinct from existing ones.
            # Are they distinct from each other?
            # Since seq is strictly increasing, suffix sums are strictly increasing.
            # x + suffix_sums[i] are strictly increasing as i increases (going backwards in seq).
            # Wait, suffix_sums[i] is sum of last i elements.
            # suffix_sums[0] = 0
            # suffix_sums[1] = a_n
            # suffix_sums[2] = a_{n-1} + a_n
            # Since a_i > 0, suffix_sums are strictly increasing.
            # So x + suffix_sums[i] are strictly increasing.
            # Thus, new sums are distinct from each other.
            
            # Recurse
            new_seq = seq + [x]
            new_used = used_sums.union(new_sums_to_add)
            
            dfs(new_seq, new_used, x)
            
            # Backtracking is implicit as we use new objects (immutable style)
            # But creating new sets and lists is expensive.
            # For n ~ 10-15, this should be fine in Python within 50s?
            # Number of nodes could be large.
            # Let's try to optimize by using mutable state and backtracking properly.

    # Let's rewrite with mutable state for performance.
    
    def dfs_mutable(seq, used_sums, last_val, suffix_sums):
        nonlocal best_seq, deadline
        
        if time.time() > deadline:
            return
            
        if len(seq) > len(best_seq):
            best_seq = seq.copy()
            
        # Pruning
        if len(seq) + (256 - last_val) <= len(best_seq):
            return

        # Try candidates
        # Heuristic: Try smaller values first?
        # If we try smaller values, we might fill the "low" sum space quickly.
        # If we try larger values, we might jump over collisions?
        # Usually, for "distinct sums", smaller values are safer for length?
        # Let's try ascending order.
        
        for x in range(last_val + 1, 257):
            if time.time() > deadline:
                return
                
            # Check validity
            # New sums: x + s for s in suffix_sums (which includes 0)
            # suffix_sums is a list of sums of the last k elements, for k=0..len(seq)
            # suffix_sums[0] = 0
            # suffix_sums[1] = seq[-1]
            # ...
            
            valid = True
            new_sums = []
            for s in suffix_sums:
                ns = x + s
                if ns in used_sums:
                    valid = False
                    break
                new_sums.append(ns)
                
            if not valid:
                continue
                
            # Add x to seq
            seq.append(x)
            
            # Update used_sums
            for ns in new_sums:
                used_sums.add(ns)
                
            # Update suffix_sums
            # New suffix sums:
            # 0 (for empty, but we don't store 0 in the list passed to next? 
            # Let's define suffix_sums passed to next call:
            # It should contain sums of last k elements of the NEW sequence.
            # New sequence ends with x.
            # Last 0 elements: 0
            # Last 1 element: x
            # Last 2 elements: x + old_last_1
            # Last k elements: x + old_last_{k-1}
            
            # So new_suffix_sums = [0] + [x + s for s in old_suffix_sums]
            # Note: old_suffix_sums already contains 0?
            # If old_suffix_sums = [0, a_n, a_{n-1}+a_n, ...],
            # Then new_suffix_sums = [0, x, x+a_n, x+a_{n-1}+a_n, ...]
            # This matches the required structure.
            
            new_suffix_sums = [0]
            for s in suffix_sums:
                new_suffix_sums.append(x + s)
                
            dfs_mutable(seq, used_sums, x, new_suffix_sums)
            
            # Backtrack
            seq.pop()
            for ns in new_sums:
                used_sums.remove(ns)
                
    # Initial call
    # seq = []
    # used_sums = set()
    # last_val = 0
    # suffix_sums = [0]
    
    dfs_mutable([], set(), 0, [0])
    
    return best_seq

# Test locally?
# The greedy baseline is length 9.
# Backtracking should find 10, 11, etc.

# One potential issue: The search space might be too large if we don't prune well.
# The bound `len(seq) + (256 - last_val) <= len(best_seq)` is weak.
# A better bound:
# The number of distinct sums required for a sequence of length L is L(L+1)/2.
# The maximum possible sum is roughly L * 256.
# This doesn't help much with pruning the search tree directly.

# Let's add a simple heuristic:
# If we are deep and can't extend, we backtrack.
# To find the MAX length, we should explore all branches that *could* beat the current best.
# The current implementation does this via the pruning check.

# Is there a risk of timeout?
# If the optimal length is, say, 12, and the tree is huge, we might time out.
# However, the constraints are tight.
# Let's run it.

# To be safe, I'll add a node counter and stop if it gets too high?
# No, time check is better.

# Let's refine the candidate order.
# If we want to maximize length, maybe we should try to keep values small?
# Yes, ascending order is a good bet.

# Final check on the code structure.
# The function `construct` must return the sequence.

# One optimization: 
# Instead of `new_suffix_sums = [0] + [x + s for s in suffix_sums]`, 
# we can just pass the list and modify it? No, we need a new list for the recursive call 
# because we backtrack. Creating a new list of size O(n) is fine.

# Let's ensure the initial `suffix_sums` is correct.
# For empty seq, suffix sums of last 0 elements is 0.
# So `[0]` is correct.

# The code looks solid.
