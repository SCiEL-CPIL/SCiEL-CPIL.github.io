"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq

def synthesize(matrix, depth_limit):
    """
    Synthesize XOR circuit using a greedy approach with optimized split search.
    
    Structural approach:
    1. Convert matrix rows to bitmasks.
    2. Sort outputs by popcount (complexity) to build simple sub-expressions first.
    3. For each output, if the wire doesn't exist, find the optimal split (A, B) 
       such that A ^ B = Mask. The optimal split minimizes the total new gates 
       required to create A and B.
    4. Use a restricted submask search for balanced splits to find cheap decompositions.
    """
    deadline = time.time() + 50  # Safety margin for 60s limit
    
    n = len(matrix)
    m = len(matrix[0]) if n > 0 else 0
    
    if n == 0:
        return {'gates': [], 'outputs': []}
    
    # Convert matrix rows to bitmasks
    rows = []
    for i in range(n):
        mask = 0
        for j in range(m):
            if matrix[i][j]:
                mask |= (1 << j)
        rows.append(mask)
    
    # Zero wire is m, inputs are 0..m-1
    zero_wire = m
    next_wire = m + 1
    gates = []
    
    # Wire management
    wire_to_mask = {}  # wire_id -> bitmask
    mask_to_wire = {}  # bitmask -> wire_id
    
    # Initialize inputs and zero
    for j in range(m):
        wire_to_mask[j] = 1 << j
        mask_to_wire[1 << j] = j
    wire_to_mask[zero_wire] = 0
    mask_to_wire[0] = zero_wire
    
    # Precompute popcount for all possible masks (m <= 16)
    max_mask = 1 << m
    popcount = [0] * max_mask
    for i in range(1, max_mask):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Sort outputs by complexity (popcount)
    output_indices = list(range(n))
    output_indices.sort(key=lambda i: popcount[rows[i]])
    
    outputs = [0] * n
    
    def get_or_create_wire(mask):
        """
        Get wire for mask, or create one using minimum new gates.
        Uses a split search to find the most efficient decomposition.
        """
        nonlocal next_wire
        
        # Check if wire already exists
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        
        if mask == 0:
            return zero_wire
            
        k = popcount[mask]
        
        # Base case: single bit
        if k == 1:
            # This should be an input, but if it's not in mask_to_wire (shouldn't happen), create it
            # Actually, inputs are pre-registered. If we get here, something is wrong or it's a new bit?
            # In this problem, inputs are fixed 0..m-1. So k=1 masks are always inputs.
            # But just in case:
            # Find the bit index
            bit_idx = mask.bit_length() - 1
            if mask_to_wire.get(mask) is not None:
                return mask_to_wire[mask]
            # If not registered (shouldn't happen), register it as input? 
            # No, inputs are fixed. If it's not there, it's not a valid single input?
            # But mask is a submask of a row, so it's composed of valid inputs.
            # So k=1 implies it IS an input.
            return mask_to_wire[mask]

        # Search for best split A ^ B = mask
        # We want to minimize cost(A) + cost(B)
        # Cost is 0 if wire exists, else estimated by recursion.
        
        best_cost = float('inf')
        best_a = None
        best_b = None
        
        # Strategy: Iterate through submasks A of mask.
        # To keep it fast, we focus on balanced splits if k is large.
        # If k is small (<= 10), we can check all submasks.
        # If k is large, we check submasks with popcount close to k/2.
        
        # Generate submasks
        # Optimization: Only consider A where A is a proper submask (0 < A < mask)
        # and to avoid duplicates (A, B) and (B, A), we can enforce A < B or similar, 
        # but since we just need the pair, checking all A is fine, we just pick the min cost.
        
        # Limit search space for large k to avoid timeout
        # For k <= 12, 2^k is <= 4096, which is fast.
        # For k > 12, we sample balanced submasks.
        
        if k <= 12:
            # Iterate all submasks
            sub = mask
            while sub:
                # Skip sub=mask (B=0) and sub=0 (A=0) as they don't help reduce gate count below k-1
                # Actually, if B=0, cost is cost(A) + 0 + 1? No, if B=0, we just use A.
                # But A=mask, so cost is cost(mask) which is what we are calculating. Infinite recursion.
                # So we must have A != mask and B != mask (i.e., A != 0 and B != 0).
                if sub == mask:
                    sub = (sub - 1) & mask
                    continue
                    
                b = mask ^ sub
                if b == 0:
                    sub = (sub - 1) & mask
                    continue
                
                # Evaluate cost of sub and b
                # Cost is 0 if in mask_to_wire
                # Otherwise, we estimate. 
                # A safe estimate is popcount(x) - 1.
                # But we can do better: if x can be formed by 2 existing wires, cost is 1.
                
                cost_a = 0 if sub in mask_to_wire else popcount[sub] - 1
                cost_b = 0 if b in mask_to_wire else popcount[b] - 1
                
                # Heuristic improvement:
                # If one part is not in mask_to_wire, check if it can be made from two existing wires.
                # This is O(N_wires). We do this only for the best few candidates or if k is small.
                # For now, use the popcount estimate, which is a valid lower bound.
                # We will refine the cost if we pick this split.
                
                total_cost = cost_a + cost_b + 1 # +1 for the XOR gate combining them
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_a = sub
                    best_b = b
                    
                sub = (sub - 1) & mask
                
        else:
            # Large k: Sample balanced splits
            # We want popcount(A) around k/2
            target_pc = k // 2
            # Generate some random or systematic submasks with popcount near target
            # Systematic: Iterate submasks, but break early if we find a good one?
            # Or just iterate all and filter? 2^16 is 65536. That's actually fast in Python if simple.
            # Let's just iterate all for k <= 16, it's 65k iterations max.
            sub = mask
            while sub:
                if sub == mask:
                    sub = (sub - 1) & mask
                    continue
                b = mask ^ sub
                if b == 0:
                    sub = (sub - 1) & mask
                    continue
                
                cost_a = 0 if sub in mask_to_wire else popcount[sub] - 1
                cost_b = 0 if b in mask_to_wire else popcount[b] - 1
                
                total_cost = cost_a + cost_b + 1
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_a = sub
                    best_b = b
                    
                sub = (sub - 1) & mask

        # If we found a split
        if best_a is not None:
            # Recursively build A and B
            # We need to be careful about recursion depth, but k is small.
            w_a = get_or_create_wire(best_a)
            w_b = get_or_create_wire(best_b)
            
            # Create the gate
            gates.append((w_a, w_b))
            w = next_wire
            next_wire += 1
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            return w
            
        # Fallback: Chain (should not happen if best_a is found for k>1)
        # Build from scratch
        bits = [j for j in range(m) if mask & (1 << j)]
        if not bits:
            return zero_wire
        w = bits[0]
        for j in bits[1:]:
            gates.append((w, j))
            w = next_wire
            next_wire += 1
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
        return w

    # Process outputs
    for i in output_indices:
        if time.time() > deadline:
            # Emergency fallback for remaining
            for remaining in output_indices[output_indices.index(i):]:
                if outputs[remaining] != 0:
                    continue
                bits = [j for j in range(m) if rows[remaining] & (1 << j)]
                if not bits:
                    outputs[remaining] = zero_wire
                elif len(bits) == 1:
                    outputs[remaining] = bits[0]
                else:
                    w = bits[0]
                    for j in bits[1:]:
                        gates.append((w, j))
                        w = next_wire
                        next_wire += 1
                    outputs[remaining] = w
            break
        
        w = get_or_create_wire(rows[i])
        outputs[i] = w

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
