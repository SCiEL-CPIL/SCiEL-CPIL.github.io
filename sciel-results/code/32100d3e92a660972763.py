"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    matrix = np.array(matrix, dtype=np.int64)
    m, n = matrix.shape
    
    # Wire representation: 0..n-1 are inputs, n is zero
    # Gate k produces wire n+1+k
    # We'll track wires as bitmasks (n bits)
    
    # Available wires: dict from bitmask to wire_id
    # wire_id 0..n-1: inputs, wire_id n: zero
    wire_to_mask = {}
    mask_to_wires = {}
    
    for i in range(n):
        mask = 1 << i
        wire_to_mask[i] = mask
        mask_to_wires[mask] = i
    
    zero_mask = 0
    wire_to_mask[n] = zero_mask
    mask_to_wires[zero_mask] = n
    
    # Parse target rows as masks
    target_masks = []
    for row in matrix:
        mask = 0
        for j, val in enumerate(row):
            if val:
                mask |= (1 << j)
        target_masks.append(mask)
    
    # Count how many times each target appears (for prioritization)
    target_counts = defaultdict(int)
    for tm in target_masks:
        target_counts[tm] += 1
    
    # Sort targets: prioritize by (count descending, popcount ascending)
    # This helps with CSE - more common targets get synthesized first
    target_indices = sorted(range(m), key=lambda i: (-target_counts[target_masks[i]], bin(target_masks[i]).count('1')))
    
    gates = []  # list of (wire_a, wire_b)
    wire_id_counter = n + 1  # next wire id
    wire_depth = {}
    for i in range(n + 1):
        wire_depth[i] = 0
    
    # For each target, find best way to compute it
    output_wire_for_target = [None] * m
    
    # Precompute popcount for efficiency
    def popcount(x):
        return bin(x).count('1')
    
    for idx in target_indices:
        if time.time() > deadline:
            # Time out - return best effort
            break
            
        target = target_masks[idx]
        if target == 0:
            output_wire_for_target[idx] = n
            continue
        
        # Find best decomposition: try all pairs of available wires
        # Cost = number of additional gates needed
        # Prefer reusing existing wires (cost 0 for the wire itself)
        best_cost = float('inf')
        best_pair = None
        best_remaining = None
        
        avail_masks = list(mask_to_wires.keys())
        num_avail = len(avail_masks)
        
        # Direct cost baseline: compute from inputs
        direct_cost = popcount(target) - 1 if popcount(target) > 0 else 0
        
        for i in range(num_avail):
            mask_a = avail_masks[i]
            wa = mask_to_wires[mask_a]
            for j in range(i + 1, num_avail):  # Only i < j to avoid duplicates
                mask_b = avail_masks[j]
                wb = mask_to_wires[mask_b]
                xor_mask = mask_a ^ mask_b
                remaining = target ^ xor_mask
                popcount_rem = popcount(remaining)
                
                # If remaining is already available, cost is just 1 gate
                if remaining in mask_to_wires:
                    cost = 1
                elif popcount_rem == 0:
                    cost = 1
                else:
                    cost = 1 + (popcount_rem - 1)
                
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (wa, wb)
                    best_remaining = remaining
        
        # If no good pair found, just compute from inputs directly
        if best_pair is None or best_cost > direct_cost:
            # Direct computation from inputs
            bits = []
            for j in range(n):
                if target & (1 << j):
                    bits.append(j)
            if not bits:
                output_wire_for_target[idx] = n
                continue
            if len(bits) == 1:
                output_wire_for_target[idx] = bits[0]
                continue
            # Chain XOR
            current_wire = bits[0]
            for b in bits[1:]:
                new_wire = wire_id_counter
                wire_id_counter += 1
                gates.append((current_wire, b))
                wire_to_mask[new_wire] = wire_to_mask[current_wire] ^ (1 << b)
                mask_to_wires[wire_to_mask[new_wire]] = new_wire
                wire_depth[new_wire] = max(wire_depth[current_wire], wire_depth[b]) + 1
                current_wire = new_wire
            output_wire_for_target[idx] = current_wire
        else:
            wa, wb = best_pair
            new_wire = wire_id_counter
            wire_id_counter += 1
            gates.append((wa, wb))
            xor_mask = wire_to_mask[wa] ^ wire_to_mask[wb]
            wire_to_mask[new_wire] = xor_mask
            mask_to_wires[xor_mask] = new_wire
            wire_depth[new_wire] = max(wire_depth[wa], wire_depth[wb]) + 1
            
            remaining = best_remaining
            if remaining == 0:
                output_wire_for_target[idx] = new_wire
            else:
                # Check if remaining is already available
                if remaining in mask_to_wires:
                    # Need one more XOR
                    existing_wire = mask_to_wires[remaining]
                    final_wire = wire_id_counter
                    wire_id_counter += 1
                    gates.append((new_wire, existing_wire))
                    final_mask = wire_to_mask[new_wire] ^ wire_to_mask[existing_wire]
                    wire_to_mask[final_wire] = final_mask
                    mask_to_wires[final_mask] = final_wire
                    wire_depth[final_wire] = max(wire_depth[new_wire], wire_depth[existing_wire]) + 1
                    output_wire_for_target[idx] = final_wire
                else:
                    # Compute remaining from inputs
                    bits = []
                    for j in range(n):
                        if remaining & (1 << j):
                            bits.append(j)
                    if not bits:
                        output_wire_for_target[idx] = new_wire
                    else:
                        current_wire = new_wire
                        for b in bits:
                            nw = wire_id_counter
                            wire_id_counter += 1
                            gates.append((current_wire, b))
                            wire_to_mask[nw] = wire_to_mask[current_wire] ^ (1 << b)
                            mask_to_wires[wire_to_mask[nw]] = nw
                            wire_depth[nw] = max(wire_depth[current_wire], wire_depth[b]) + 1
                            current_wire = nw
                        output_wire_for_target[idx] = current_wire
    
    # Handle any targets not yet assigned (if we timed out)
    for idx in range(m):
        if output_wire_for_target[idx] is None:
            target = target_masks[idx]
            if target == 0:
                output_wire_for_target[idx] = n
            else:
                # Direct computation
                bits = []
                for j in range(n):
                    if target & (1 << j):
                        bits.append(j)
                if not bits:
                    output_wire_for_target[idx] = n
                elif len(bits) == 1:
                    output_wire_for_target[idx] = bits[0]
                else:
                    current_wire = bits[0]
                    for b in bits[1:]:
                        new_wire = wire_id_counter
                        wire_id_counter += 1
                        gates.append((current_wire, b))
                        wire_to_mask[new_wire] = wire_to_mask[current_wire] ^ (1 << b)
                        mask_to_wires[wire_to_mask[new_wire]] = new_wire
                        wire_depth[new_wire] = max(wire_depth[current_wire], wire_depth[b]) + 1
                        current_wire = new_wire
                    output_wire_for_target[idx] = current_wire
    
    outputs = output_wire_for_target
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
