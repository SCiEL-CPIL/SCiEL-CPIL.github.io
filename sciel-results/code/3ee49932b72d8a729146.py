"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct the sequence of step heights.
    
    Returns: sequence of N step heights in [0, 1] whose sum is exactly N/2.
    """
    # We represent the sequence by intervals of 1s.
    # h is 1 on these intervals, 0 elsewhere.
    # Sum of lengths must be N/2.
    
    half = N // 2
    
    def get_intervals(h):
        """Extract intervals of 1s from binary array h."""
        intervals = []
        in_block = False
        start = 0
        for i in range(N):
            if h[i] == 1 and not in_block:
                in_block = True
                start = i
            elif h[i] == 0 and in_block:
                in_block = False
                intervals.append((start, i))
        if in_block:
            # Wrap around case
            intervals.append((start, N))
            # Also need to check if it started at 0
            # Actually, if it wraps, the first part is at the end.
            # Let's handle wrap-around separately or just linearize.
            # For circular autocorrelation, we just need the array.
            pass
        return intervals

    def intervals_to_array(intervals):
        """Convert list of (start, end) intervals to binary array."""
        h = np.zeros(N, dtype=np.int8)
        for s, e in intervals:
            if s < e:
                h[s:e] = 1
            else:
                # Wrap around
                h[s:] = 1
                h[:e] = 1
        return h

    def compute_cost(h):
        """Compute the cost: max overlap of h with shifted 1-h."""
        # Overlap_k = sum_i h_i (1 - h_{i+k}) = sum_i h_i - sum_i h_i h_{i+k}
        # sum_i h_i = N/2
        # C_k = sum_i h_i h_{i+k}
        # Cost = max_k (N/2 - C_k) = N/2 - min_k C_k
        min_corr = N # Initialize with high value
        for k in range(1, N):
            # Circular correlation
            # h shifted by -k
            shifted = np.roll(h, -k)
            corr = np.dot(h, shifted)
            if corr < min_corr:
                min_corr = corr
        return N/2.0 - min_corr

    # Initialize with random intervals
    # We need sum of lengths = N/2.
    # Let's start with K random intervals.
    # Try different K? Let's fix K for now, maybe 10-20.
    # Or start with a random binary sequence and extract intervals.
    
    # Strategy: Start with a random binary sequence with sum N/2,
    # extract intervals, then optimize interval positions.
    
    # Random init
    h = np.zeros(N, dtype=np.int8)
    indices = np.random.choice(N, half, replace=False)
    h[indices] = 1
    
    # Extract intervals
    # Simple extraction for linear, but we have circular.
    # Let's just use the array directly for moves?
    # The directive says "optimize positions of boundaries".
    # I will implement moves that shift blocks of 1s.
    
    current_cost = compute_cost(h)
    best_h = h.copy()
    best_cost = current_cost
    
    T = 1.0
    T_min = 0.001
    steps = 20000
    cooling_rate = (T_min / T) ** (1.0 / steps)
    
    for s in range(steps):
        # Pick a random 1-block to move
        # Find all 1-blocks
        # To do this efficiently, we can just pick a random index that is 1,
        # and find the block it belongs to.
        
        idx = np.random.randint(N)
        if h[idx] == 0:
            continue
            
        # Find start of this block
        start = idx
        while start > 0 and h[start-1] == 1:
            start -= 1
        if start == 0 and h[N-1] == 1:
             # Wrapping block?
             # This is tricky. Let's assume non-wrapping for simplicity in moves,
             # or handle it.
             # If it wraps, start is 0 and end is N? No.
             # Let's find end.
             end = idx
             while end < N-1 and h[end+1] == 1:
                 end += 1
             if end == N-1 and h[0] == 1:
                 # It wraps.
                 # This block covers [0, end] and [start, N-1]?
                 # Actually, if h[0]==1 and h[N-1]==1, they are the same block.
                 # Let's treat the array as circular.
                 # For simplicity, if we detect a wrap, we might skip or handle carefully.
                 # Let's just find the length of the block.
                 # Count 1s in the connected component.
                 length = 0
                 pos = idx
                 while True:
                     length += 1
                     pos = (pos + 1) % N
                     if h[pos] == 0:
                         break
                     if pos == idx:
                         break # All 1s? Unlikely.
                 
                 # Move this block by delta
                 delta = np.random.choice([-2, -1, 1, 2])
                 
                 # Create new h
                 new_h = h.copy()
                 # Zero out the block
                 # We need to identify the indices.
                 # This is getting complicated.
                 continue
             
        # Non-wrapping block case
        end = idx
        while end < N-1 and h[end+1] == 1:
            end += 1
        
        length = end - start + 1
        
        # Try to shift by delta
        delta = np.random.choice([-3, -2, -1, 1, 2, 3])
        new_start = start + delta
        new_end = end + delta
        
        # Check bounds and overlap
        valid = True
        if new_start < 0 or new_end >= N:
            valid = False
        else:
            # Check if new position overlaps with other 1s
            # The region [new_start, new_end] must be all 0s in the original h,
            # EXCEPT for the part that is the original block?
            # Actually, we are moving the block.
            # So [new_start, new_end] must be 0 in h, unless it overlaps with [start, end].
            # If it overlaps, we just change the shape? No, we shift the whole block.
            # So the destination must be empty.
            # But the source becomes empty.
            # So we need to check if [new_start, new_end] contains any 1s that are NOT in [start, end].
            
            # Simple check:
            # Temporarily set the block to 0, then check if destination is 0.
            # Or just check intersection.
            
            # Let's just create the new array and check sum and cost.
            # It's safer.
            new_h = h.copy()
            new_h[start:end+1] = 0
            new_h[new_start:new_end+1] = 1
            
            # Check if sum is preserved (it is, we moved a block)
            # Check for overlap conflicts:
            # If new_h has more 1s than N/2, it means we overlapped with another 1.
            if np.sum(new_h) != half:
                valid = False
            else:
                # Check if the move is valid (no overlap with other 1s)
                # The sum check above ensures no *extra* 1s were created.
                # But did we lose any?
                # We set source to 0 and dest to 1.
                # If dest had 1s (from other blocks), they are still 1.
                # So sum would be half + (overlap_count).
                # If sum == half, then overlap_count = 0.
                # So it's valid.
                
                new_cost = compute_cost(new_h)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    h = new_h
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_h = h.copy()
                else:
                    if np.random.rand() < np.exp((current_cost - new_cost) / T):
                        current_cost = new_cost
                        h = new_h
                    else:
                        # Reject
                        pass
        T *= cooling_rate
        
    return best_h.astype(np.float64)
# EVOLVE-BLOCK-END
