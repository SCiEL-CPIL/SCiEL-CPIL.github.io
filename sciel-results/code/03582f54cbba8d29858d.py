import time
import sys

def construct():
    start_time = time.time()
    N = 256
    deadline = start_time + 55  # 55s safety margin before 60s kill
    
    # We want to maximize length. 
    # A strictly increasing sequence a_1 < a_2 < ... < a_k.
    # Consecutive block sums must be distinct.
    
    # Strategy: Backtracking with aggressive pruning.
    # State: current sequence `seq`, set of used sums `used_sums`.
    # When adding a new element `x` to the end of `seq`:
    # The new sums generated are:
    #   x
    #   x + seq[-1]
    #   x + seq[-1] + seq[-2]
    #   ...
    #   x + sum(seq)
    # These are x + suffix_sums(seq), where suffix_sums includes the full sum.
    # Let S = sum(seq). Let suffixes = [seq[-1], seq[-1]+seq[-2], ..., S].
    # New sums = {x + s for s in suffixes}.
    # Condition: (x + suffixes) must be disjoint from `used_sums`.
    # Also x > seq[-1] (if seq not empty) and x <= N.
    
    # Pruning Insight:
    # If we are at depth k, we have used k*(k+1)/2 sums.
    # The maximum possible number of sums for length L is L*(L+1)/2.
    # We can estimate an upper bound on the final length based on the remaining "sum space"?
    # Not directly, because sums can be up to 256*L.
    
    # However, we can use a different pruning:
    # If we pick x, it blocks x, x+s1, x+s2, etc.
    # To maximize length, we generally want to keep x as small as possible relative to the gaps?
    # Actually, for increasing sequences, picking smaller x allows more room for future elements?
    # Or does picking larger x push the sums out of the "dense" region?
    
    # Let's try a Depth-First Search with:
    # 1. Ordering of candidates: Try smallest valid x first (to keep sequence tight/increasing slowly?)
    #    Or try to balance?
    # 2. Pruning: If the current length + max_possible_additional_length <= best_found, prune.
    #    How to estimate max_possible_additional_length?
    #    The sums must be distinct integers.
    #    The smallest possible sum for the next element x is x_min + 1 (if we pick x_min).
    #    This is hard to bound tightly.
    
    # Alternative: 
    # The problem is equivalent to finding a long path in a graph where nodes are states (seq, used_sums).
    # But state space is huge.
    
    # Let's look at the structure of sums.
    # Sum of block [i..j] = P[j] - P[i-1] where P is prefix sum.
    # Distinct consecutive block sums means all differences P[j] - P[i-1] for 0 <= i-1 < j <= k are distinct.
    # This is exactly the definition of a "Sidon sequence" variant? No, Sidon is sums of two elements.
    # This is a "Consecutive Block Sidon Sequence" or similar.
    
    # Let's stick to the constructive backtracking but optimize the candidate selection.
    # Key optimization: Pre-compute suffix sums incrementally.
    
    best_seq = []
    
    def dfs(seq, used_sums, suffix_sums, current_depth):
        nonlocal best_seq
        # Pruning: If current_depth + optimistic_bound <= len(best_seq), return
        # Optimistic bound: We can always add at least 1 more if there's space.
        # A crude bound: The maximum number of elements is limited by N.
        # But we want to prune deep branches that can't beat the best.
        # If current_depth + (N - seq[-1]) <= len(best_seq): return (very loose)
        
        if len(seq) > len(best_seq):
            best_seq = seq[:]
            
        if not seq:
            # Start: can pick any x in 1..N
            # To maximize length, should we start small or large?
            # Starting small allows more elements.
            # Let's try x=1, then x=2, etc.
            # But we need to ensure we don't get stuck.
            # Actually, if we always pick the smallest valid x, we might get a very long sequence.
            # Let's try "Smallest Valid X" heuristic first.
            
            # We will iterate through possible next values.
            # To avoid deep recursion overhead, we can use a stack or just recursive with sys.setrecursionlimit.
            
            # Let's refine the candidate generation.
            # If seq is empty, candidates are 1..N.
            # If seq is not empty, x > seq[-1].
            
            # We want to explore candidates that allow the most future growth.
            # Heuristic: Smaller x leaves more room for larger x later? 
            # But smaller x creates smaller sums, which might collide with other small sums.
            
            # Let's try a bidirectional approach or just a strong greedy with backtracking.
            
            # For the first element, try 1.
            # If that fails to reach a good length, try 2, etc.
            
            # Actually, let's just do a DFS that tries candidates in increasing order.
            # This tends to produce long sequences because it packs the values tightly.
            
            # Optimization: 
            # If we pick x, the new sums are x, x+suffix1, x+suffix2...
            # We need to check if any of these are in used_sums.
            
            # Let's implement the DFS loop.
            pass # Handled by the main loop below
            
    # Let's implement a iterative deepening or just a single deep DFS with "Smallest Valid" priority.
    # If "Smallest Valid" gets stuck, we backtrack and try the next valid value.
    
    def solve():
        nonlocal best_seq
        seq = []
        used_sums = set()
        suffix_sums = [] # [seq[-1], seq[-1]+seq[-2], ...]
        
        # We will use a stack to simulate recursion to avoid overhead and allow deep search.
        # But Python recursion is fine if we set limit.
        sys.setrecursionlimit(10000)
        
        def backtrack(seq, used_sums, suffix_sums):
            nonlocal best_seq
            if len(seq) > len(best_seq):
                best_seq = seq[:]
                
            # If we've used all numbers 1..N, stop.
            if seq and seq[-1] == N:
                return
                
            # Determine range for next x
            if not seq:
                start_x = 1
            else:
                start_x = seq[-1] + 1
                
            # Heuristic: Try smaller x first? 
            # Let's try to find a valid x quickly.
            
            # Pre-calculate potential conflicts?
            # For a given x, new_sums = {x + s for s in [0] + suffix_sums} 
            # Wait, if seq is [a, b], suffix_sums = [b, a+b].
            # New sums when appending x:
            # x
            # x+b
            # x+b+a
            # So yes, x + suffix_sums (where suffix_sums includes the full sum).
            # Note: suffix_sums is maintained as [last, last+second_last, ...]
            
            # To speed up, we can check if x + s is in used_sums.
            
            # Let's iterate x from start_x to N.
            # We can break early if we find a valid x and it leads to a good result?
            # No, we must explore all to ensure optimality? 
            # The task is to MAXIMIZE length. We don't need to find ALL, just the longest.
            # So we can use a greedy-like DFS:
            # 1. Find the smallest valid x.
            # 2. Go deep.
            # 3. If we hit a dead end, backtrack and try the next valid x.
            
            # Optimization: 
            # If we are already at a length L, and we know that even with the best possible packing 
            # (differences of 1) we can't beat best_len, we can prune.
            # Max possible length from current state:
            # Current last element is `last`.
            # We can add at most N - last elements.
            # So max_len = len(seq) + (N - last).
            # If max_len <= len(best_seq): return
            
            if seq:
                max_possible = len(seq) + (N - seq[-1])
                if max_possible <= len(best_seq):
                    return
            
            # Find valid x's
            # We try x in increasing order.
            # Why increasing? Because smaller x allows more elements to fit in [1, N].
            
            for x in range(start_x, N + 1):
                # Check validity
                # New sums: x, x+suffix_sums[0], x+suffix_sums[1], ...
                # If seq is empty, suffix_sums is empty, so just check x.
                
                valid = True
                # Check x itself
                if x in used_sums:
                    valid = False
                    # If x is invalid, can a larger x be valid?
                    # Yes.
                    continue
                
                # Check x + s for s in suffix_sums
                if valid:
                    for s in suffix_sums:
                        if x + s in used_sums:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Valid! Recurse
                new_suffix = []
                s_sum = x
                new_suffix.append(s_sum)
                for s in suffix_sums:
                    s_sum += s
                    new_suffix.append(s_sum)
                    
                # Update used_sums
                new_used = used_sums | {x + s for s in [0] + suffix_sums}
                
                backtrack(
                    seq + [x],
                    new_used,
                    new_suffix
                )
                
                # Time check
                if time.time() > deadline:
                    return
                
                # After backtracking from x, we continue to the next x.
                # This ensures we explore alternative branches if the first one doesn't yield the maximum.
            
        backtrack([], set(), [])
        return best_seq

    # Run the solver
    result = solve()
    
    # Fallback: If the DFS didn't find anything (shouldn't happen), return a trivial valid seq
    if not result:
        # Simple greedy fallback
        seq = []
        used = set()
        for i in range(1, N+1):
            # Check if i can be added
            # This is slow, so only as fallback
            pass
        # Just return [1]
        return [1]

    return result

# Quick test to ensure it doesn't crash immediately
if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(f"Seq: {res[:10]}...")
