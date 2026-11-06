"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # 1. Generate a strong baseline using forward greedy
    seq = _forward_greedy(1, 256)
    
    # 2. Try to extend from the back (high numbers) to see if we can get a longer sequence
    #    Sometimes starting high allows for different sum patterns?
    #    Actually, the constraint is on sums of consecutive blocks.
    #    Let's try a bidirectional approach.
    
    # Baseline length
    best_len = len(seq)
    
    # 3. Attempt to improve by backtracking from the end of the greedy sequence
    #    and trying alternative choices.
    #    We will use a randomized restart strategy with a focus on "deep" backtracking.
    
    # To make this materially different and potentially stronger:
    # We will use a "meet-in-the-middle" style construction.
    # Construct a prefix greedily.
    # Construct a suffix greedily (from 256 downwards).
    # Then try to bridge the gap with a search.
    
    # However, simple bidirectional might not be enough if the greedy is already optimal or near-optimal.
    # Let's try a more aggressive local search on the greedy result.
    # We will try to "swap" elements or "insert" elements to see if we can fit more.
    
    # Actually, let's try a different constructive heuristic:
    # Instead of always picking the smallest available number, we can pick numbers
    # that are "sparse" in terms of their contribution to block sums.
    # But the greedy already picks the smallest valid one.
    
    # Let's try a randomized restart with a different heuristic:
    # Heuristic: Pick the smallest valid number, but with a small probability,
    # pick a slightly larger valid number to "jump" over a conflict.
    
    best_seq = seq
    
    # Time budget for search
    while time.time() < deadline:
        # Randomized restart with "lookahead" or "perturbation"
        # We will try to build a sequence, but at each step, if there are multiple choices,
        # we usually take the smallest, but sometimes we take a random valid candidate.
        
        # To be efficient, we only consider candidates within a range of the greedy choice.
        # Let's try a "beam search" approach.
        
        beam = [seq[:]]  # Start with the greedy sequence as one beam? No, start from scratch.
        # Actually, restarting from scratch is expensive.
        # Let's try to modify the existing best_seq.
        
        # Local Search: Try to insert a new number into the sequence.
        # If we have a sequence of length L, can we insert a number to make it L+1?
        # We try all possible positions and values.
        
        # This is O(N^2 * 256) per iteration, which is manageable.
        
        # Let's try to find a valid insertion.
        improved = False
        current_len = len(best_seq)
        
        # Try inserting a new element x at position i
        # We iterate x from 1 to 256
        # We iterate i from 0 to current_len
        # Check if best_seq with x inserted at i is valid.
        
        # Optimization: We can precompute block sums of best_seq.
        # But insertion changes many block sums.
        
        # Let's just try a few random insertions to see if we can beat the greedy.
        # Greedy is very strong. It might be optimal.
        # If greedy is optimal, we can't beat it.
        # But maybe greedy gets stuck in a local optimum?
        
        # Let's try a "simulated annealing" style swap.
        # Pick two random indices i, j in best_seq.
        # Swap them? No, sequence must be strictly increasing.
        # So we can't just swap.
        
        # We can replace an element with a different one.
        # Pick index i.
        # Pick a new value x such that best_seq[i-1] < x < best_seq[i+1].
        # Check if the new sequence is valid.
        
        # Let's try this for a while.
        
        for _ in range(100): # 100 swaps per second roughly?
            if time.time() > deadline:
                break
            
            if len(best_seq) < 2:
                break
            
            i = random.randint(1, len(best_seq) - 2) # Don't change ends easily?
            # Actually, changing ends is fine too.
            i = random.randint(0, len(best_seq) - 1)
            
            low = best_seq[i-1] + 1 if i > 0 else 1
            high = best_seq[i+1] - 1 if i < len(best_seq) - 1 else 256
            
            if low > high:
                continue
            
            # Try a random value in range
            x = random.randint(low, high)
            if x == best_seq[i]:
                continue
            
            new_seq = best_seq[:]
            new_seq[i] = x
            
            if _is_valid(new_seq):
                # If we swapped, length is same.
                # We want to increase length.
                # So this swap is only useful if it allows a future insertion?
                # Or if we are stuck, maybe a different arrangement allows a longer sequence?
                # For now, let's just accept it if it's valid, hoping to find a better local optimum.
                best_seq = new_seq
                improved = True
                break # Restart the loop to try more swaps from this new state
        
        # Try to insert a new element
        if not improved:
            # Try to insert at random positions
            for _ in range(10):
                if time.time() > deadline:
                    break
                pos = random.randint(0, len(best_seq))
                # Find a value x that fits
                # This is harder. Let's just check all x in [1, 256] for this pos?
                # Too slow.
                pass
    
    return best_seq

def _is_valid(seq):
    """Check if sequence is strictly increasing and all consecutive block sums are unique."""
    if not seq:
        return True
    
    # Check strictly increasing
    for i in range(len(seq) - 1):
        if seq[i] >= seq[i+1]:
            return False
    
    # Check unique block sums
    sums = set()
    n = len(seq)
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _forward_greedy(start, end):
    """Standard greedy: pick smallest valid number."""
    values = []
    sums = set()
    
    for x in range(start, end + 1):
        # Calculate new block sums ending at x
        # tails[i] is sum of values[i:] + x ? No.
        # Let's reuse the logic from the seed program which was efficient.
        
        # tails[k] is the sum of the last k elements of values, plus x?
        # No, the seed program did:
        # tails, total = [x], x
        # for a in reversed(values):
        #     total += a
        #     tails.append(total)
        # This calculates sums of x, x+last, x+last+second_last, etc.
        # These are exactly the new block sums that include x.
        # Any block sum not including x is already in 'sums'.
        
        tails = [x]
        total = x
        for a in reversed(values):
            total += a
            tails.append(total)
        
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
            
    return values
# EVOLVE-BLOCK-END
