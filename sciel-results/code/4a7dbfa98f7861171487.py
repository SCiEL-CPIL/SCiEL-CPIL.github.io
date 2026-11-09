"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict
import numpy as np

def synthesize(matrix, depth_limit):
    """
    XOR-circuit synthesis via iterative maximum-benefit pairwise matching.
    
    Instead of building each output independently (even with caching), we
    repeatedly find the pair of existing wires whose XOR creates the most
    useful new intermediate (one that helps complete the most pending outputs
    or enables cheaper future constructions). This is a global greedy approach
    that maximizes sub-expression sharing across all outputs simultaneously.
    """
    start_time = time.time()
    deadline = start_time + 75  # 15s safety margin before 90s kill
    
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
    
    zero_wire = m
    next_wire = m + 1
    gates = []
    
    # Wire management
    wire_to_mask = {}
    mask_to_wire = {}
    for j in range(m):
        wire_to_mask[j] = 1 << j
        mask_to_wire[1 << j] = j
    wire_to_mask[zero_wire] = 0
    mask_to_wire[0] = zero_wire
    
    # Precompute popcount
    max_mask = 1 << m
    popcount = np.zeros(max_mask, dtype=np.int32)
    for i in range(1, max_mask):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Track which outputs are done
    outputs = [None] * n
    outputs_done = [False] * n
    
    # For each pending output, track its mask
    pending_masks = {}  # mask -> list of output indices
    for i in range(n):
        if rows[i] == 0:
            outputs[i] = zero_wire
            outputs_done[i] = True
        elif rows[i] in mask_to_wire:
            outputs[i] = mask_to_wire[rows[i]]
            outputs_done[i] = True
        else:
            if rows[i] not in pending_masks:
                pending_masks[rows[i]] = []
            pending_masks[rows[i]].append(i)
    
    # Available wires (initially just inputs + zero)
    available = list(wire_to_mask.keys())
    
    # Helper: try to build a mask using two existing wires
    def try_build_from_existing(mask):
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        if mask == 0:
            return zero_wire
        # Check if mask can be formed by XOR of two existing wires
        for w1 in available:
            m1 = wire_to_mask[w1]
            m2 = mask ^ m1
            if m2 in mask_to_wire:
                w2 = mask_to_wire[m2]
                if w1 != w2:
                    gates.append((w1, w2))
                    w_new = next_wire
                    next_wire += 1
                    wire_to_mask[w_new] = mask
                    mask_to_wire[mask] = w_new
                    available.append(w_new)
                    return w_new
        return None
    
    # Helper: build a mask directly using a chain
    def build_direct(mask):
        if mask in mask_to_wire:
            return mask_to_wire[mask]
        if mask == 0:
            return zero_wire
        bits = [j for j in range(m) if mask & (1 << j)]
        if not bits:
            return zero_wire
        if len(bits) == 1:
            w = bits[0]
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            return w
        w = bits[0]
        for j in bits[1:]:
            gates.append((w, j))
            w = next_wire
            next_wire += 1
            wire_to_mask[w] = mask
            mask_to_wire[mask] = w
            available.append(w)
        return w
    
    # Iterative maximum-benefit matching
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        if not pending_masks:
            break
        
        # Score all possible pairs of available wires
        # Benefit = number of pending outputs that would be completed or helped
        benefits = []
        avail_list = list(available)
        num_avail = len(avail_list)
        
        # For efficiency, limit pair evaluation
        max_pairs = 500
        if num_avail > 40:
            # Sample pairs
            indices = list(range(num_avail))
            np.random.shuffle(indices)
            indices = indices[:40]
            avail_list = [avail_list[i] for i in indices]
            num_avail = len(avail_list)
        
        for i in range(num_avail):
            if time.time() > deadline:
                break
            w1 = avail_list[i]
            m1 = wire_to_mask[w1]
            for j in range(i + 1, num_avail):
                w2 = avail_list[j]
                m2 = wire_to_mask[w2]
                new_mask = m1 ^ m2
                if new_mask == 0:
                    continue
                if new_mask in mask_to_wire:
                    continue
                
                # Compute benefit: how many pending outputs does this help?
                benefit = 0
                # Direct completion
                if new_mask in pending_masks:
                    benefit += len(pending_masks[new_mask]) * 10  # High weight for direct completion
                
                # Indirect benefit: new_mask is a submask of some pending output
                # or a superset that could be XORed with something useful
                for pmask in pending_masks:
                    if benefit >= 20:  # Cap benefit computation
                        break
                    # If new_mask XOR pmask has low popcount, it's a useful intermediate
                    xor_result = new_mask ^ pmask
                    if popcount[xor_result] <= 2:
                        benefit += 3
                    elif popcount[xor_result] <= 4:
                        benefit += 1
                
                if benefit > 0:
                    benefits.append((benefit, w1, w2, new_mask))
        
        if not benefits:
            # No beneficial pairs found; build remaining outputs directly
            break
        
        # Sort by benefit descending
        benefits.sort(key=lambda x: -x[0])
        
        # Greedy matching: select non-conflicting pairs
        used_wires = set()
        selected = []
        
        for benefit, w1, w2, new_mask in benefits:
            if w1 in used_wires or w2 in used_wires:
                continue
            used_wires.add(w1)
            used_wires.add(w2)
            selected.append((w1, w2, new_mask, benefit))
            if len(selected) >= 20:
                break
        
        if not selected:
            break
        
        # Execute selected pairs
        for w1, w2, new_mask, benefit in selected:
            if time.time() > deadline:
                break
            gates.append((w1, w2))
            w_new = next_wire
            next_wire += 1
            wire_to_mask[w_new] = new_mask
            mask_to_wire[new_mask] = w_new
            available.append(w_new)
            
            # Check if this completes any pending outputs
            if new_mask in pending_masks:
                for idx in pending_masks[new_mask]:
                    outputs[idx] = w_new
                    outputs_done[idx] = True
                del pending_masks[new_mask]
    
    # Final cleanup: build any remaining outputs directly
    for i in range(n):
        if not outputs_done[i]:
            if time.time() > deadline:
                # Emergency: minimal construction
                mask = rows[i]
                if mask == 0:
                    outputs[i] = zero_wire
                else:
                    bits = [j for j in range(m) if mask & (1 << j)]
                    if not bits:
                        outputs[i] = zero_wire
                    elif len(bits) == 1:
                        outputs[i] = bits[0]
                    else:
                        w = bits[0]
                        for j in bits[1:]:
                            gates.append((w, j))
                            w = next_wire
                            next_wire += 1
                        outputs[i] = w
            else:
                w = try_build_from_existing(rows[i])
                if w is None:
                    w = build_direct(rows[i])
                outputs[i] = w
            outputs_done[i] = True
    
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
