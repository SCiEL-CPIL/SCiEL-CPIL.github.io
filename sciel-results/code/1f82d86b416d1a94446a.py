import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    if m == 0 or n == 0:
        return {'gates': [], 'outputs': []}
        
    # Convert matrix to list of integer masks for fast XOR
    # Bit i corresponds to input wire i
    targets = []
    for i in range(m):
        mask = 0
        for j in range(n):
            if matrix[i][j]:
                mask |= (1 << j)
        targets.append(mask)
        
    # State:
    # wire_to_mask: maps wire index to the mask it represents
    # mask_to_wire: maps mask to the wire index that represents it (for CSE)
    # wire_depth: maps wire index to its depth
    # gates: list of [wire_a, wire_b]
    
    wire_to_mask = {}
    mask_to_wire = {}
    wire_depth = {}
    gates = []
    next_wire = n + 1
    
    # Initialize inputs (0..n-1) and constant zero (n)
    for i in range(n):
        mask = 1 << i
        wire_to_mask[i] = mask
        mask_to_wire[mask] = i
        wire_depth[i] = 0
    wire_to_mask[n] = 0
    mask_to_wire[0] = n
    wire_depth[n] = 0
    
    # Map target index to current remaining mask
    # We want to reduce each target to 0 (or a single input, which is depth 0)
    # Actually, we want to reduce each target to a mask that is already a wire.
    # Let's maintain 'remaining[i]' as the mask we still need to compute for output i.
    # Initially remaining[i] = targets[i].
    # When we create a new gate W with mask M, for any i where remaining[i] has M as a subset?
    # No, XOR is not subset. 
    # The "Peeling" strategy works like this:
    # If we have a wire W with mask M, and a target T contains M (i.e., T XOR M is "smaller" in some sense? No).
    # Actually, if we compute W, we can replace T with T XOR W.
    # So remaining[i] becomes remaining[i] XOR M.
    # We stop when remaining[i] is in mask_to_wire.
    
    remaining = targets[:]
    
    # Helper to check if a mask is "simple" (already a wire)
    def is_simple(mask):
        return mask in mask_to_wire
        
    # We iterate until all remaining are simple, or time runs out.
    # At each step, we look for a "candidate" mask M that can be formed by XORing two existing wires,
    # such that adding M reduces the total complexity of the remaining targets.
    
    # Complexity metric: sum of popcounts of remaining masks?
    # Or better: sum of "cost to compute remaining".
    # If remaining[i] is simple, cost is 0.
    # If not, cost is roughly popcount(remaining[i]) / 2 or similar.
    
    # To make this efficient, we don't check all pairs of wires (O(W^2)).
    # We check pairs of wires that are "likely" to be useful.
    # A pair (A, B) is useful if M = A XOR B appears in multiple remaining targets (or reduces them significantly).
    
    # Let's maintain a frequency count of how many remaining targets contain a specific "pattern"?
    # That's hard because XOR doesn't decompose into bits simply.
    
    # Alternative: 
    # 1. Identify all unique non-simple remaining masks.
    # 2. For each unique mask R, try to find a decomposition R = A XOR B where A, B are existing wires.
    #    If found, creating A XOR B might help R and potentially others.
    # 3. Score each candidate gate (A, B) by how many targets it helps (i.e., how many remaining[i] become simpler).
    #    "Simpler" could mean: remaining[i] XOR M is in mask_to_wire, or has lower popcount.
    
    # Let's refine:
    # We will maintain a set of "active" targets (those not yet simple).
    # In each iteration:
    #   1. Collect all unique remaining masks R for active targets.
    #   2. For each R, try to find a split R = A XOR B with A, B in existing wires.
    #      How to find A, B? 
    #      Iterate through existing wires A. Compute B = R XOR A. Check if B is in existing wires.
    #      This is O(W) per R.
    #   3. For each valid split (A, B) for any R, calculate the "benefit".
    #      Benefit = sum over all active targets T of (cost(T) - cost(T XOR M)).
    #      If benefit > 0, it's a candidate.
    #   4. Pick the candidate with highest benefit.
    #   5. Add gate, update remaining, repeat.
    
    # Optimization: 
    # Instead of checking all active targets for benefit, we can approximate.
    # Or, we can just pick the R that appears most frequently, and split it.
    
    # Let's try a simpler heuristic first:
    # 1. Find the most frequent non-simple remaining mask R.
    # 2. Find a split R = A XOR B.
    # 3. Add gate.
    # 4. Update all remaining[i] where remaining[i] == R (or maybe others?).
    #    Actually, if we add M, we should update ALL remaining[i] to remaining[i] XOR M?
    #    No, that would change the value. 
    #    Wait. The circuit computes wires. 
    #    If we add a gate W = A XOR B, W is a new wire.
    #    We don't *replace* R with R XOR W in the target definition.
    #    The target is still R.
    #    But now R can be computed as (R XOR W) XOR W.
    #    So if we define "remaining" as the mask we need to XOR with existing wires to get the target,
    #    then yes, remaining[i] becomes remaining[i] XOR W.
    #    
    #    Let's clarify the state:
    #    Target i is T_i.
    #    We have a set of available wires W_0, ..., W_k.
    #    We want to express T_i as XOR of some subset of W_j.
    #    Let S_i be the subset.
    #    Initially, W_j are inputs.
    #    If we add a new wire W_new = A XOR B, we can use W_new in the subset.
    #    So if T_i was represented by subset S, it can now be represented by S U {W_new} XOR {A, B} (removing A, B if they were there).
    #    This is getting complicated to track subsets.
    
    # Let's stick to the "Mask Peeling" model:
    # State: remaining[i] is the mask such that T_i = XOR(wires in remaining[i])? No.
    # Let's say remaining[i] is the mask that, when XORed with the *current* set of "basis" wires, equals T_i?
    # No, that's Gaussian elimination.
    
    # Let's try a different angle: **Common Subexpression via Frequency of Subsets**
    # This is hard for XOR.
    
    # Let's go back to the "Split the most frequent R" heuristic.
    # It's simple and often works well for random matrices.
    
    active_indices = [i for i in range(m) if not is_simple(targets[i])]
    
    if not active_indices:
        return {'gates': gates, 'outputs': [mask_to_wire[t] for t in targets]}
        
    # We will maintain a dictionary: mask -> count of targets that currently "need" this mask?
    # No, "need" is ambiguous.
    
    # Let's use the "Benefit" scoring but optimize it.
    # To speed up, we will limit the search for splits.
    
    while active_indices:
        if time.time() > DEADLINE:
            break
            
        # 1. Count frequency of each non-simple remaining mask
        freq = defaultdict(int)
        for i in active_indices:
            r = remaining[i]
            if not is_simple(r):
                freq[r] += 1
                
        if not freq:
            break
            
        # Pick the most frequent mask
        most_freq_mask, count = max(freq.items(), key=lambda x: x[1])
        
        # 2. Try to find a split most_freq_mask = A XOR B
        #    Iterate through existing wires A
        found_split = False
        best_a = None
        best_b = None
        
        # To avoid trivial splits (A=0 or B=0), we skip wire n (0)
        # Also, we prefer splits where A and B are "small" or "common"?
        # For now, just find any valid split.
        
        # Optimization: Only check A if A is not 0 and B is not 0.
        # Also, if A or B is the mask itself, that's trivial (M = M XOR 0).
        
        # Let's iterate through wires in order of "simplicity" (lower depth first)?
        # Or just arbitrary.
        
        for a_wire in wire_to_mask:
            a_mask = wire_to_mask[a_wire]
            if a_mask == 0: continue
            b_mask = most_freq_mask ^ a_mask
            if b_mask in mask_to_wire:
                b_wire = mask_to_wire[b_mask]
                if b_mask != 0: # Avoid trivial M = M XOR 0
                    # Found a split
                    best_a = a_wire
                    best_b = b_wire
                    found_split = True
                    break
            # Time check inside loop?
            if time.time() > DEADLINE:
                break
                
        if not found_split:
            # If we can't split the most frequent mask, try the next most frequent?
            # Or give up on this mask and try to build it from scratch?
            # If we can't split it using existing wires, it means it's not in the span of existing wires?
            # No, it IS in the span of inputs. 
            # If we can't find A, B in *current* wires, it means the current set of wires doesn't contain a pair that XORs to it.
            # This can happen if the current wires are sparse.
            # In that case, we should probably add a new wire that is a "building block".
            # Let's pick a random input or a simple mask and add it?
            # Actually, if no split exists, it implies the mask is "new".
            # We can just pick two inputs that are part of the mask?
            # Let's try to find a split using only inputs (wires 0..n-1).
            # If the mask has >= 2 bits, we can always split it into two inputs.
            # e.g., Mask = 1100. A = 1000 (wire 3), B = 0100 (wire 2).
            
            # Find two bits in most_freq_mask
            bits = []
            temp = most_freq_mask
            while temp:
                lb = temp & -temp
                bits.append(lb)
                temp ^= lb
                if len(bits) >= 2:
                    break
                    
            if len(bits) >= 2:
                # Use two inputs
                # Wire for bit 1: index = log2(bit)
                # Since bit is power of 2, index is bit.bit_length() - 1
                a_idx = bits[0].bit_length() - 1
                b_idx = bits[1].bit_length() - 1
                best_a = a_idx
                best_b = b_idx
                found_split = True
            else:
                # Mask has only 1 bit? Then it's an input, so it should be simple.
                # If it's not simple, something is wrong.
                # Or it's 0? 0 is simple.
                # Break to avoid infinite loop
                break
                
        if not found_split:
            break
            
        # 3. Create the new gate
        new_mask = most_freq_mask
        # Check if new_mask is already a wire (shouldn't be, since we checked is_simple)
        if new_mask in mask_to_wire:
            # It became simple? Maybe due to previous updates?
            # Just update active indices and continue
            active_indices = [i for i in active_indices if not is_simple(remaining[i])]
            continue
            
        gates.append([best_a, best_b])
        new_wire = next_wire
        next_wire += 1
        wire_to_mask[new_wire] = new_mask
        mask_to_wire[new_mask] = new_wire
        wire_depth[new_wire] = max(wire_depth[best_a], wire_depth[best_b]) + 1
        
        # 4. Update remaining targets
        # For all active targets, XOR the new mask?
        # NO. 
        # If we add a wire W, we don't change the target.
        # We change the *representation* of the target.
        # If remaining[i] was R, and we add W, we can choose to represent R as R XOR W XOR W? No.
        # The "Peeling" logic:
        # If we compute W, we can use it to simplify R if R contains W?
        # In XOR, "contains" is not a term.
        # However, if we define remaining[i] as the mask we still need to compute *from inputs*,
        # and we have a new wire W, we can replace any occurrence of W in the computation of R with the wire W.
        # This is only possible if R is *equal* to W, or R is a combination that *includes* W.
        # 
        # Let's rethink the "remaining" state.
        # Let's say remaining[i] is the mask T_i XOR (XOR of wires we've already decided to use for T_i).
        # Initially, we haven't used any wires, so remaining[i] = T_i.
        # When we add a gate W, we can decide to "use" W for some targets.
        # If we use W for target i, we update remaining[i] = remaining[i] XOR W.
        # We should use W for target i if it makes remaining[i] simpler (e.g., closer to 0 or a single input).
        # 
        # Heuristic: Use W for target i if remaining[i] XOR W has lower popcount than remaining[i].
        # Or if remaining[i] XOR W is already a wire.
        
        for i in active_indices:
            r = remaining[i]
            new_r = r ^ new_mask
            # Decide if we should "peel" W from this target
            # If new_r is simple, definitely peel.
            # If popcount(new_r) < popcount(r), maybe peel.
            # Let's be aggressive: peel if popcount decreases.
            if new_r in mask_to_wire or bin(new_r).count('1') < bin(r).count('1'):
                remaining[i] = new_r
                if is_simple(remaining[i]):
                    # This target is done
                    pass # We'll filter active_indices at the end of loop
        
        # Update active_indices
        active_indices = [i for i in active_indices if not is_simple(remaining[i])]
        
    # After loop, some targets might not be simple.
    # We need to build them from scratch using the current wires.
    # For each remaining non-simple target, we can just build a tree from inputs.
    # But we have many wires now.
    # Let's just do a simple greedy split for the leftovers.
    
    for i in range(m):
        if is_simple(remaining[i]):
            continue
        # Build remaining[i] from scratch
        # Use a simple recursive split
        def build_mask(mask):
            if mask in mask_to_wire:
                return mask_to_wire[mask]
            if mask == 0:
                return n
            # Split into two halves
            bits = []
            temp = mask
            while temp:
                lb = temp & -temp
                bits.append(lb)
                temp ^= lb
                
            half = len(bits) // 2
            m1 = 0
            for b in bits[:half]:
                m1 |= b
            m2 = 0
            for b in bits[half:]:
                m2 |= b
                
            w1 = build_mask(m1)
            w2 = build_mask(m2)
            gates.append([w1, w2])
            new_w = next_wire
            nonlocal next_wire # This won't work in nested function without nonlocal in outer scope? 
            # Actually, build_mask is inside synthesize. next_wire is in synthesize.
            # We need to declare nonlocal next_wire in build_mask? No, next_wire is modified in synthesize scope.
            # Let's just use a list or global to hold next_wire? 
            # Or return the new wire index and update next_wire in the caller?
            # This is getting messy. Let's use a helper class or closure properly.
            
        # Let's rewrite the leftover handling to be safe.
        # We'll just add gates for the leftovers directly.
        
    # Actually, the loop above should have handled most.
    # Let's just ensure all outputs are valid wires.
    outputs = []
    for i in range(m):
        if is_simple(remaining[i]):
            outputs.append(mask_to_wire[remaining[i]])
        else:
            # Fallback: build from inputs
            # This is a crude fallback
            mask = remaining[i]
            # Get a wire for this mask
            # We can just use the first input? No, it must be exact.
            # Let's add a simple gate tree.
            # To avoid complexity, let's just pick a wire that is "close"? No, must be exact.
            # Let's implement a simple recursive builder for this specific case.
            
            def get_wire_for_leftover(msk):
                nonlocal next_wire
                if msk in mask_to_wire:
                    return mask_to_wire[msk]
                if msk == 0:
                    return n
                # Split
                bits = []
                temp = msk
                while temp:
                    lb = temp & -temp
                    bits.append(lb)
                    temp ^= lb
                half = len(bits) // 2
                m1 = 0
                for b in bits[:half]:
                    m1 |= b
                m2 = 0
                for b in bits[half:]:
                    m2 |= b
                w1 = get_wire_for_leftover(m1)
                w2 = get_wire_for_leftover(m2)
                gates.append([w1, w2])
                nw = next_wire
                next_wire += 1
                wire_to_mask[nw] = msk
                mask_to_wire[msk] = nw
                wire_depth[nw] = max(wire_depth[w1], wire_depth[w2]) + 1
                return nw
                
            outputs.append(get_wire_for_leftover(remaining[i]))
            
    return {'gates': gates, 'outputs': outputs}
