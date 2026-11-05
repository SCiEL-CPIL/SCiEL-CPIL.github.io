"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict, deque
import heapq
from itertools import combinations

def synthesize(matrix, depth_limit):
    """Synthesize an XOR circuit using Global Common Subexpression Elimination (CSE).
    
    Iteratively identifies the most beneficial common sub-expression (mask) across
    the current set of row masks, creates a wire for it, and XORs it out of all
    rows that contain it. This minimizes total gate count by maximizing sharing.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    
    mat = np.array(matrix, dtype=np.int8)
    n_outputs, n_inputs = mat.shape
    
    def row_to_mask(row):
        m = 0
        for j, v in enumerate(row):
            if v:
                m |= (1 << j)
        return m
        
    output_masks = [row_to_mask(mat[i]) for i in range(n_outputs)]
    
    def popcount(m):
        return bin(m).count('1')
        
    def bits_from_mask(m):
        bits = []
        j = 0
        while m:
            if m & 1:
                bits.append(j)
            m >>= 1
            j += 1
        return bits

    # State
    current_masks = list(output_masks)
    wire_map = {0: n_inputs}
    for i in range(n_inputs):
        wire_map[1 << i] = i
        
    next_wire = n_inputs + 1
    gates = []
    
    def create_wire_for_mask(mask):
        """Recursively create a wire for a mask, reusing existing wires if possible."""
        nonlocal next_wire
        if mask in wire_map:
            return wire_map[mask]
            
        # Find a sub-mask that is already available to reduce depth/complexity
        # Simple strategy: pick the lowest set bit, XOR it with the rest
        # Better: pick a sub-mask that is in wire_map to maximize sharing
        best_sub = None
        # Check single bits first
        for i in range(n_inputs):
            if mask & (1 << i):
                best_sub = 1 << i
                break
        
        # If we found a single bit, the rest is mask ^ best_sub
        if best_sub is not None:
            rest_mask = mask ^ best_sub
            w1 = wire_map[best_sub]
            w2 = create_wire_for_mask(rest_mask)
            gates.append([w1, w2])
            w = next_wire
            next_wire += 1
            wire_map[mask] = w
            return w
            
        # Fallback: shouldn't happen if mask is non-zero
        return n_inputs

    # Main CSE Loop
    max_iterations = 100 # Safety cap, usually converges much faster
    
    for _ in range(max_iterations):
        if time.time() > DEADLINE - 2.0:
            break
            
        # 1. Identify candidate sub-masks
        # We look for sub-masks of size 2 to min(6, max_row_popcount)
        # Count their frequency in current_masks
        freq_count = defaultdict(int)
        
        # Optimization: Only consider sub-masks from rows that are "complex" enough
        # or just iterate all
        for mask in current_masks:
            if mask == 0:
                continue
            bits = bits_from_mask(mask)
            max_size = min(6, len(bits))
            if max_size < 2:
                continue
            for size in range(2, max_size + 1):
                for combo in combinations(bits, size):
                    sub_mask = 0
                    for b in combo:
                        sub_mask |= (1 << b)
                    freq_count[sub_mask] += 1
                    
        # 2. Find the best candidate
        best_mask = None
        best_benefit = 0
        
        for sub_mask, freq in freq_count.items():
            if freq < 2:
                continue
            k = popcount(sub_mask)
            # Benefit: saving (k-1) gates for each additional row it appears in
            # Cost: (k-1) gates to create it once
            # Net Benefit: (k-1) * (freq - 1)
            benefit = (k - 1) * (freq - 1)
            if benefit > best_benefit:
                best_benefit = benefit
                best_mask = sub_mask
                
        if best_mask is None:
            break
            
        # 3. Create wire for best_mask
        w = create_wire_for_mask(best_mask)
        
        # 4. XOR it out of all rows that contain it
        changed = False
        for i in range(n_outputs):
            if (current_masks[i] & best_mask) == best_mask:
                current_masks[i] ^= best_mask
                changed = True
                
        if not changed:
            break

    # Assemble final outputs
    output_wires = []
    for i in range(n_outputs):
        mask = current_masks[i]
        if mask == 0:
            output_wires.append(n_inputs)
        else:
            # Create wire for the remaining mask
            # This will reuse any previously created wires
            w = create_wire_for_mask(mask)
            output_wires.append(w)
            
    return {'gates': gates, 'outputs': output_wires}

# EVOLVE-BLOCK-END
