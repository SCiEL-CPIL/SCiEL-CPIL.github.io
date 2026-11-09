"""Rewritten program: iterative deepening with pruning for max-length distinct-consecutive-sum sequence."""
import time
import sys

sys.setrecursionlimit(100000)


def construct():
    deadline = time.time() + 80.0  # 80s safety margin within 90s limit

    # We search for the maximum length L such that there exists a strictly
    # increasing sequence of integers in [1,256] of length L where all
    # nonempty consecutive-block sums are distinct.

    # The seed program (greedy) produces a certain length. Let's first run
    # the greedy to get a baseline, then try to improve via search.

    def greedy_sequence():
        values, sums = [], set()
        for x in range(1, 257):
            tails, total = [x], x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    best_seq = greedy_sequence()
    best_len = len(best_seq)

    # Now try iterative deepening DFS to find longer sequences.
    # State: current sequence (list), set of all consecutive-block sums.
    # We want to extend to length > best_len.

    # Key insight: for a sequence of length n, there are n*(n+1)/2 consecutive blocks.
    # Each block sum must be distinct. The minimum possible sum is 1 (single element min=1),
    # and maximum is sum of all elements.
    # For n elements in [1,256], max sum is 256+255+...+(257-n).
    # We need n*(n+1)/2 distinct sums, all within [1, max_sum].

    # Let's determine an upper bound on feasible length.
    # For n elements from [1,256], the max possible sum of any block is at most
    # 256+255+...+(257-n) = n*(513-n)/2.
    # The min sum is at least 1.
    # So we need n*(n+1)/2 <= n*(513-n)/2, i.e., n+1 <= 513-n, i.e., 2n <= 512, n <= 256.
    # That's trivial. But tighter: the sums must all be distinct positive integers
    # bounded by the total sum S = sum of all elements.
    # Actually each block sum is at most S (the total). And there are n(n+1)/2 blocks.
    # So we need n(n+1)/2 <= S. For n elements in [1,256] strictly increasing,
    # min S = 1+2+...+n = n(n+1)/2, which gives equality only for the sequence 1,2,...,n.
    # But 1,2,...,n doesn't work because e.g., block [1,2] sum=3 and block [3] sum=3.
    # So we need S > n(n+1)/2, meaning we need to use larger values.

    # Upper bound: with values in [1,256], max S for n elements is 256+255+...+(257-n).
    # Need n(n+1)/2 <= S_max = n(513-n)/2 => n+1 <= 513-n => n <= 256. Trivial.
    # But also each individual block sum must be >= 1 and <= S, and all distinct.
    # The number of distinct sums needed is n(n+1)/2, and they must all be in [1, S].
    # So n(n+1)/2 <= S. This is necessary.

    # Let's just try to find sequences of length best_len+1, best_len+2, etc.

    def dfs_extend(seq, sums, target_len):
        """Try to extend seq to target_len. Returns True if found, modifies seq/sums."""
        if len(seq) == target_len:
            return True
        if time.time() > deadline:
            return False

        last = seq[-1]
        # Try next values from last+1 to 256
        # Pruning: we need (target_len - len(seq)) more elements, all > last
        # The minimum sum we'd add is (last+1) + (last+2) + ... 
        # Let's just iterate candidates

        n_remaining = target_len - len(seq)
        # Minimum possible last element if we pick the smallest n_remaining values > last
        # last + n_remaining must be <= 256
        if last + n_remaining > 256:
            return False

        for x in range(last + 1, 257 - n_remaining + 2):
            # Check if x can be appended
            # Compute new block sums: x, x+seq[-1], x+seq[-1]+seq[-2], ...
            new_sums = [x]
            total = x
            for a in reversed(seq):
                total += a
                new_sums.append(total)
            
            # Check for conflicts
            conflict = False
            for s in new_sums:
                if s in sums:
                    conflict = True
                    break
            
            if conflict:
                continue

            # Also check: remaining elements must be > x, and we need n_remaining-1 more
            # after this one. So x + (n_remaining - 1) <= 256
            if x + (n_remaining - 1) > 256:
                continue

            # Additional pruning: the total sum of the final sequence must be at least
            # target_len*(target_len+1)/2 (since all block sums must be distinct positive ints)
            # Current sum + x + minimum remaining...
            # This is complex; let's keep it simple

            seq.append(x)
            sums.update(new_sums)
            
            if dfs_extend(seq, sums, target_len):
                return True
            
            seq.pop()
            sums.difference_update(new_sums)

        return False

    # Try to find longer sequences
    # Start from best_len+1 and go up
    # But DFS from scratch for each target length is expensive.
    # Instead, let's do a more targeted search.

    # Alternative approach: use the greedy sequence as a starting point and try
    # local search / branch and bound.

    # Let's try a different strategy: build the sequence element by element,
    # but use backtracking with better pruning.

    # First, let's understand what length the greedy achieves.
    # For now, let's try to extend by 1 using a focused search.

    # Actually, let me try a completely different approach:
    # Use a randomized/local search to try to find longer sequences.

    def try_improve(seq, max_iter=50000):
        """Try to improve the sequence length via local search."""
        best = list(seq)
        best_len = len(best)
        
        sums_set = set()
        for i in range(len(best)):
            for j in range(i+1, len(best)+1):
                sums_set.add(sum(best[i:j]))
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to insert a new element at a random position
            # or swap/modify elements to free up space for a new one
            
            # Simpler: try adding a random element at the end
            if len(best) < 256:
                last = best[-1]
                # Try a random value > last
                candidates = list(range(last+1, 257))
                if candidates:
                    x = candidates[0]  # Try smallest first for density
                    new_sums = [x]
                    total = x
                    for a in reversed(best):
                        total += a
                        new_sums.append(total)
                    
                    if not any(s in sums_set for s in new_sums):
                        best.append(x)
                        sums_set.update(new_sums)
                        best_len = len(best)
            
            # Try removing an element and replacing it with two
            # This is complex; skip for now
            
        return best

    # Let's try a more systematic approach:
    # Build sequence with backtracking, but use memoization of "forbidden sums"
    # and aggressive pruning.

    def build_with_backtracking(max_len, start_val=1):
        """Build a sequence of up to max_len using backtracking."""
        seq = []
        sums = set()
        
        def bt(pos, last_val):
            if pos == max_len:
                return True
            if time.time() > deadline:
                return False
            
            remaining = max_len - pos
            # Need remaining elements, all > last_val, max is 256
            if last_val + remaining > 256:
                return False
            
            # Pruning: total sum so far + minimum possible additional sum
            # must allow all block sums to be distinct
            # Current total sum:
            cur_sum = sum(seq)
            # Min additional sum: (last_val+1) + (last_val+2) + ... + (last_val+remaining)
            min_add = remaining * last_val + remaining*(remaining+1)//2
            total_min = cur_sum + min_add
            
            # Number of block sums in final sequence: max_len*(max_len+1)//2
            # All must be in [1, total_sum_final], so need:
            # max_len*(max_len+1)//2 <= total_sum_final
            # This is always true for reasonable sequences, so not a strong prune.
            
            # Better prune: the new block sums introduced by adding elements
            # must not conflict. Hard to check in advance.
            
            # Try candidates
            # Use a heuristic: try values that create "sparse" sums
            for x in range(last_val + 1, 257 - remaining + 2):
                new_sums = [x]
                total = x
                for a in reversed(seq):
                    total += a
                    new_sums.append(total)
                
                conflict = False
                for s in new_sums:
                    if s in sums:
                        conflict = True
                        break
                if conflict:
                    continue
                
                seq.append(x)
                sums.update(new_sums)
                
                if bt(pos + 1, x):
                    return True
                
                seq.pop()
                sums.difference_update(new_sums)
            
            return False
        
        if bt(0, start_val - 1):
            return seq
        return []

    # Try to find a sequence longer than the greedy one
    # Start with best_len + 1
    improved = False
    for target_len in range(best_len + 1, min(best_len + 5, 30)):
        if time.time() > deadline - 10:
            break
        
        # Try to build a sequence of exactly target_len
        # Use backtracking with the greedy sequence as a hint for ordering
        # Start from small values
        
        seq_found = build_with_backtracking(target_len)
        if seq_found and len(seq_found) == target_len:
            # Verify
            sset = set()
            valid = True
            for i in range(len(seq_found)):
                for j in range(i+1, len(seq_found)+1):
                    s = sum(seq_found[i:j])
                    if s in sset:
                        valid = False
                        break
                    sset.add(s)
                if not valid:
                    break
            if valid and seq_found == sorted(seq_found) and all(1 <= v <= 256 for v in seq_found):
                best_seq = seq_found
                best_len = target_len
                improved = True
                continue  # Try for even longer
        else:
            # If we couldn't find one, try a few more times with different strategies
            # or break
            break

    return best_seq


if __name__ == "__main__":
    result = construct()
    print(f"Length: {len(result)}")
    print(f"Sequence: {result}")
    # Verify
    sset = set()
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)+1):
            s = sum(result[i:j])
            if s in sset:
                valid = False
                break
            sset.add(s)
        if not valid:
            break
    print(f"Valid: {valid}, Strictly increasing: {result == sorted(result) and len(set(result))==len(result)}")
    print(f"All in [1,256]: {all(1<=v<=256 for v in result)}")
