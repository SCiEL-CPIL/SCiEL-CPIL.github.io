"""Deterministic greedy with lookahead for increasing sequences with unique consecutive-block sums."""
import time
import sys

_DEADLINE = 85.0  # safety margin before 90s kill


def _greedy_build():
    """
    Greedy construction: try each value from 1 to 256 in order.
    A value x can be appended if none of the new consecutive-block sums
    that end at x collide with existing block sums.
    
    Key insight: when appending x to sequence [a1, a2, ..., ak],
    the new block sums are:
      x, a_{k}+x, a_{k-1}+a_k+x, ..., a1+a2+...+ak+x
    i.e., x + suffix_sum_j for each suffix of the current sequence.
    """
    seq = []
    sums = set()  # all existing consecutive block sums
    
    for x in range(1, 257):
        # Compute new sums if we append x
        # New sums: x + (suffix sum of seq ending at position i) for i = 0..len(seq)-1
        # Suffix sums: sum(seq[i:]) for i in 0..len(seq)-1, plus just x (empty suffix = 0)
        new_sums = []
        total = 0
        # Build suffix sums from the end
        # suffix_j = sum(seq[j:])
        # New block sums when appending x: for each j from 0 to len(seq)-1,
        #   sum(seq[j:] + [x]) = sum(seq[j:]) + x
        # Also the block [x] itself = x
        # So new sums = {x} union {sum(seq[j:]) + x : j in 0..len(seq)-1}
        
        # Efficiently: walk backwards
        total = 0
        candidate_sums = [x]  # just x
        for j in range(len(seq) - 1, -1, -1):
            total += seq[j]
            candidate_sums.append(total + x)
        
        # Check for collision with existing sums
        if not any(s in sums for s in candidate_sums):
            seq.append(x)
            sums.update(candidate_sums)
    
    return seq


def _greedy_with_lookahead(max_lookahead=3):
    """
    Greedy with small lookahead: at each step, try the smallest available value,
    but if it leads to a dead-end within lookahead steps, try the next value.
    """
    seq = []
    sums = set()
    
    def try_extend(seq, sums, start_val, remaining_steps, depth=0):
        """Try to extend seq starting from start_val for up to remaining_steps values."""
        if remaining_steps <= 0 or depth >= 50:
            return seq, sums, True
        
        for x in range(start_val, 257):
            # Compute new sums
            total = 0
            candidate_sums = [x]
            for j in range(len(seq) - 1, -1, -1):
                total += seq[j]
                candidate_sums.append(total + x)
            
            if not any(s in sums for s in candidate_sums):
                new_seq = seq + [x]
                new_sums = sums | set(candidate_sums)
                success = True
                # Don't recurse deep; just commit
                return new_seq, new_sums, success
        
        return seq, sums, False
    
    # Simple greedy: just append the smallest valid value each time
    for x in range(1, 257):
        total = 0
        candidate_sums = [x]
        for j in range(len(seq) - 1, -1, -1):
            total += seq[j]
            candidate_sums.append(total + x)
        
        if not any(s in sums for s in candidate_sums):
            seq.append(x)
            sums.update(candidate_sums)
    
    return seq


def _improve_by_insertion(seq, time_limit=5.0):
    """
    Try to insert additional elements into a valid sequence by checking
    all possible insertion positions and values.
    """
    start_time = time.time()
    seq = list(seq)
    sums = set()
    
    # Recompute all block sums
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        n = len(seq)
        
        # Try inserting a new element at each position
        for pos in range(n + 1):
            if time.time() - start_time > time_limit:
                break
            
            # Find valid values to insert at position pos
            low = 1 if pos == 0 else seq[pos - 1] + 1
            high = 256 if pos == n else seq[pos] - 1
            
            if low > high:
                continue
            
            for x in range(low, high + 1):
                # Check if inserting x at pos creates a valid sequence
                # We need to check all block sums that include the new element
                # and verify no collisions
                
                # New sequence would be: seq[:pos] + [x] + seq[pos:]
                # Block sums that include x:
                # For each i in [0..pos-1] (or -1 for just x) and j in [pos..n-1] (or pos for just x):
                #   sum(seq[i:pos] + [x] + seq[pos:j+1])
                
                # More carefully: in the new sequence of length n+1, x is at index pos.
                # Blocks containing x: start index s in [max(0, 0)..pos], end index e in [pos..n]
                # Wait, let me re-index. New sequence has indices 0..n.
                # x is at index pos.
                # A block [s..e] contains x iff s <= pos <= e.
                # For s from max(0, pos - pos) = 0 to pos, and e from pos to n:
                #   The sum = (sum of seq[s-1:pos-1] if s>0 else 0) + x + (sum of seq[pos:e] if e>pos else 0)
                #   Wait, this is getting complicated. Let me just check all new block sums.
                
                # Actually, let's be more efficient. The block sums that change are those
                # that include the new element x at position pos.
                # In the original sequence, blocks are [i..j] for 0<=i<=j<n.
                # In the new sequence (length n+1), x is at position pos.
                # New blocks: all original blocks not containing position pos, plus new blocks containing pos.
                # Original blocks that become shifted: blocks starting at or after pos in original
                #   become blocks starting at pos+1 or later in new sequence, with same sum.
                # Original blocks entirely before pos stay the same.
                # New blocks containing x at position pos:
                #   For start s in [0..pos] and end e in [pos..n-1] in new indexing:
                #     If s < pos: sum = sum(new_seq[s:pos]) + x + sum(new_seq[pos+1:e+1])
                #     If s == pos: sum = x + sum(new_seq[pos+1:e+1]) if e > pos else x
                #     If e == pos: sum = sum(new_seq[s:pos]) + x if s < pos else x
                #     If s == pos and e == pos: sum = x
                
                # Let me define:
                # left_sum(s) = sum(seq[s:pos]) for s in 0..pos (sum of elements before x, from s to pos-1)
                # right_sum(e) = sum(seq[pos:e+1]) for e in pos..n-1 (sum of elements after x, from pos to e)
                # Wait, in the original sequence, positions pos..n-1 are the elements after where we insert.
                
                # Let me re-derive. Original seq has n elements: seq[0], ..., seq[n-1].
                # Insert x at position pos (0 <= pos <= n).
                # New sequence: seq[0], ..., seq[pos-1], x, seq[pos], ..., seq[n-1]
                # New sequence has n+1 elements.
                # x is at index pos in the new sequence.
                
                # Block sums in new sequence that include x (at index pos):
                # For start index s (0 <= s <= pos) and end index e (pos <= e <= n-1):
                #   sum = (sum of new_seq[s:pos]) + x + (sum of new_seq[pos+1:e+1])
                #   where new_seq[s:pos] = seq[s:pos] if s < pos, else []
                #   and new_seq[pos+1:e+1] = seq[pos:e] if e >= pos, else []
                #   
                #   More precisely:
                #   left_part = sum(seq[s:pos]) if s < pos else 0  (elements from s to pos-1 in original)
                #   right_part = sum(seq[pos:e]) if e >= pos else 0  (elements from pos to e-1 in original)
                #   block_sum = left_part + x + right_part
                
                # We need to check that none of these new block sums collide with existing sums
                # (excluding the new sums themselves, since multiple new sums can be the same... 
                # wait, no, we need ALL block sums to be distinct, including among the new ones).
                
                # So we need:
                # 1. No new block sum equals any existing block sum (from original sequence)
                # 2. All new block sums are distinct from each other
                
                # For efficiency, let's compute all new block sums and check.
                
                left_sums = [0] * (pos + 1)  # left_sums[s] = sum(seq[s:pos]) for s in 0..pos
                # left_sums[pos] = 0 (empty)
                # left_sums[pos-1] = seq[pos-1]
                # etc.
                for s in range(pos - 1, -1, -1):
                    left_sums[s] = left_sums[s + 1] + seq[s]
                
                right_sums = [0] * (n - pos + 1)  # right_sums[e] = sum(seq[pos:e]) for e in pos..n
                # right_sums[n] = 0 (empty, e=n means sum from pos to n-1... wait)
                # Let me redefine: right_part for end index e (in new sequence) corresponds to
                # elements seq[pos] through seq[e-1] in original, which is sum(seq[pos:e])
                # So for e in pos..n-1: right_sum = sum(seq[pos:e+1-1]) = sum(seq[pos:e])
                # Wait, I'm overcomplicating.
                
                # In the new sequence, x is at index pos.
                # For a block starting at s and ending at e (s <= pos <= e):
                #   Elements are: new_seq[s], ..., new_seq[pos-1], new_seq[pos]=x, new_seq[pos+1], ..., new_seq[e]
                #   = seq[s], ..., seq[pos-1], x, seq[pos], ..., seq[e-1]
                #   (when s < pos and e > pos, using original indices)
                #   Sum = sum(seq[s:pos]) + x + sum(seq[pos:e])
                
                # So left_sum(s) = sum(seq[s:pos]) for s in 0..pos
                # right_sum(e) = sum(seq[pos:e]) for e in pos..n  (where e is the end index in new seq)
                #   Note: e ranges from pos to n-1 in new seq, but sum(seq[pos:e]) where e is new seq end index
                #   means we sum original seq[pos] through seq[e-1], which is sum(seq[pos:e])
                
                # Let me just compute:
                # For s in 0..pos: L[s] = sum(seq[s:pos])
                # For e in pos..n: R[e] = sum(seq[pos:e])  (R[pos] = 0, R[n] = sum(seq[pos:n]))
                
                L = [0] * (pos + 1)
                for s in range(pos - 1, -1, -1):
                    L[s] = L[s + 1] + seq[s]
                
                R = [0] * (n - pos + 1)
                for e in range(pos + 1, n + 1):
                    R[e] = R[e - 1] + seq[e - 1]
                # R[pos] = 0 already
                
                # New block sums: L[s] + x + R[e] for s in 0..pos, e in pos..n
                new_sums_set = set()
                valid = True
                for s in range(pos + 1):
                    for e in range(pos, n + 1):
                        bs = L[s] + x + R[e]
                        if bs in sums:
                            valid = False
                            break
                        if bs in new_sums_set:
                            valid = False
                            break
                        new_sums_set.add(bs)
                    if not valid:
                        break
                
                if valid:
                    # Insert x at position pos
                    seq.insert(pos, x)
                    sums.update(new_sums_set)
                    improved = True
                    break  # Restart the position loop with updated seq and sums
        
        if time.time() - start_time > time_limit:
            break
    
    return seq


def construct():
    start_time = time.time()
    
    # Start with greedy baseline
    best_seq = _greedy_build()
    
    # Try to improve by insertion
    time_remaining = _DEADLINE - (time.time() - start_time)
    if time_remaining > 2:
        improved = _improve_by_insertion(best_seq, time_limit=min(time_remaining * 0.5, 20.0))
        if len(improved) > len(best_seq):
            best_seq = improved
    
    # Final validation
    n = len(best_seq)
    if n > 0:
        # Check strictly increasing
        for i in range(1, n):
            if best_seq[i] <= best_seq[i - 1]:
                best_seq = _greedy_build()
                break
        
        # Check range
        for v in best_seq:
            if v < 1 or v > 256:
                best_seq = _greedy_build()
                break
        
        # Check distinct block sums
        sums = set()
        valid = True
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += best_seq[j]
                if s in sums:
                    valid = False
                    break
                sums.add(s)
            if not valid:
                break
        
        if not valid:
            best_seq = _greedy_build()
    
    return best_seq
