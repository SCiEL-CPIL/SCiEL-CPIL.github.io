import sys
import time
import bisect

def construct():
    start_time = time.time()
    LIMIT = 10000
    
    # 1. Generate primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    n = len(primes)
    
    # states[i] will store a list of (gap, length, prev_idx)
    # The list is sorted by gap, and length is strictly increasing.
    # This represents the Pareto frontier of sequences ending at primes[i].
    states = [[] for _ in range(n)]
    # gap_lists[i] will store just the gaps from states[i] for bisect
    gap_lists = [[] for _ in range(n)]
    
    best_len = 0
    best_end_idx = -1
    best_end_gap = 0
    
    for j in range(n):
        p_j = primes[j]
        cands = []
        
        # Base case: sequence of length 1 starting at p_j
        # We use gap 0 as a placeholder, but it's only valid for length 1.
        # Actually, for extension, we need the last gap.
        # If we start a new sequence, the "last gap" is effectively 0 (or -infinity) 
        # so that any positive gap G >= 0 can extend it.
        # However, the first gap in the sequence is p_2 - p_1.
        # If we define the state as (last_gap, length, prev_idx), 
        # for a sequence of length 1, last_gap is 0.
        cands.append((0, 1, -1))
        
        # Extend from previous primes
        for i in range(j):
            p_i = primes[i]
            G = p_j - p_i
            
            # Query states[i] for max length with gap <= G
            if not gap_lists[i]:
                continue
                
            # Binary search for the largest gap <= G
            pos = bisect.bisect_right(gap_lists[i], G)
            if pos > 0:
                # The best state is at index pos - 1
                state = states[i][pos - 1]
                g_old, L_old, prev_idx = state
                new_len = L_old + 1
                cands.append((G, new_len, i))
        
        # Consolidate cands into states[j]
        # Sort by gap
        cands.sort(key=lambda x: x[0])
        
        filtered = []
        max_len_so_far = 0
        
        for c in cands:
            gap, length, prev_idx = c
            if length > max_len_so_far:
                filtered.append((gap, length, prev_idx))
                max_len_so_far = length
        
        states[j] = filtered
        gap_lists[j] = [s[0] for s in filtered]
        
        if max_len_so_far > best_len:
            best_len = max_len_so_far
            best_end_idx = j
            # The last element in filtered has the max length
            best_end_gap = filtered[-1][0]
            
        # Time check
        if time.time() - start_time > 50:
            break
            
    if best_end_idx == -1:
        return [primes[0]] if primes else []
        
    # Reconstruct sequence
    seq = []
    curr_idx = best_end_idx
    curr_gap = best_end_gap
    
    while curr_idx != -1:
        seq.append(primes[curr_idx])
        
        # Find the state in states[curr_idx] with gap curr_gap
        # Since states[curr_idx] is sorted by gap, we can bisect
        # But curr_gap is the gap *used to enter* this state?
        # No, `curr_gap` is the last gap of the sequence ending at `curr_idx`.
        # In `states[curr_idx]`, we store (last_gap, length, prev_idx).
        # So we need to find the state where `last_gap == curr_gap`.
        
        # Since gaps in states are unique (due to filtering), we can bisect
        pos = bisect.bisect_left(gap_lists[curr_idx], curr_gap)
        
        # Verify
        if pos < len(gap_lists[curr_idx]) and gap_lists[curr_idx][pos] == curr_gap:
            state = states[curr_idx][pos]
        else:
            # Fallback to linear search if bisect fails (should not happen)
            state = None
            for s in states[curr_idx]:
                if s[0] == curr_gap:
                    state = s
                    break
            if state is None:
                break

        prev_idx = state[2]
        prev_gap = state[0] # This is the gap of the *previous* step? 
        # Wait.
        # State at j is (G_j, L_j, i).
        # G_j = p_j - p_i.
        # This G_j is the gap between p_i and p_j.
        # So the "last gap" of the sequence ending at j is G_j.
        # The previous prime is p_i.
        # The "last gap" of the sequence ending at i was G_i (which is stored in states[i]).
        # So when we move back to i, the curr_gap for i should be the gap that ended the sequence at i.
        # Which is state[0] of the state in states[i] that we extended.
        
        # In the reconstruction loop:
        # We are at `curr_idx` (which is j).
        # We found the state `(G_j, L_j, i)` in `states[j]`.
        # `G_j` is the gap $p_j - p_i$.
        # The sequence ending at $i$ had some last gap $G_i$.
        # We need to update `curr_gap` to $G_i$ and `curr_idx` to $i$.
        
        # How do we get $G_i$?
        # We don't store it directly in the state at j.
        # But we know the state at i that was used.
        # Wait, the state at j only stores `prev_idx` (which is i).
        # It does NOT store the specific state at i that was used.
        # However, the state at i that was used is the one with max length and gap <= G_j.
        # Let's call it `state_i_used`.
        # `state_i_used` is at index `pos_i - 1` in `states[i]` where `pos_i = bisect_right(gap_lists[i], G_j)`.
        
        # So we can re-compute which state at i was used.
        
        curr_idx = prev_idx
        if curr_idx == -1:
            break
            
        # Find the state in states[curr_idx] that was extended
        # The gap used to extend was G_j (which is `curr_gap` before update? No, `curr_gap` is G_j).
        # So we look in states[curr_idx] for the best state with gap <= G_j.
        
        G_ext = curr_gap # This is the gap p_j - p_i
        pos_i = bisect.bisect_right(gap_lists[curr_idx], G_ext)
        
        if pos_i > 0:
            state_i = states[curr_idx][pos_i - 1]
            curr_gap = state_i[0] # Update curr_gap to the last gap of the sequence ending at i
        else:
            # Should not happen if logic is correct
            break

    seq.reverse()
    
    return seq
