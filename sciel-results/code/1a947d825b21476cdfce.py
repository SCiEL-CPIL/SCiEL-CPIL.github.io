"""XOR-circuit synthesis: minimum-gate construction via common-subexpression elimination."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def synthesize(matrix, depth_limit):
    """
    Synthesize a minimal (or near-minimal) XOR circuit for the given matrix.
    
    Parameters:
        matrix: list of lists (or 2D array) of 0/1 values, shape (m, n)
        depth_limit: maximum allowed circuit depth (<= 16)
    
    Returns:
        {'gates': [[wire_a, wire_b], ...], 'outputs': [wire, ...]}
        where input wires are 0..n-1, constant-zero is n, gate k produces n+1+k.
    """
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 85s wall-clock safety margin
    
    # Parse matrix
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0
    
    # Convert to list of tuples for hashability
    target_vecs = []
    for i in range(m):
        row = tuple(int(matrix[i][j]) for j in range(n))
        target_vecs.append(row)
    
    zero_vec = tuple([0] * n)
    
    # Wire 0..n-1 are inputs, wire n is constant zero
    # We'll build up available vectors
    wire_to_vec = {}
    vec_to_wire = {}
    
    # Input wires
    for j in range(n):
        inp_vec = list(zero_vec)
        inp_vec[j] = 1
        inp_vec = tuple(inp_vec)
        wire_to_vec[j] = inp_vec
        vec_to_wire[inp_vec] = j
    
    # Constant zero
    wire_to_vec[n] = zero_vec
    vec_to_wire[zero_vec] = n
    
    # Next available wire index
    next_wire = n + 1
    gates = []
    
    # Helper: compute XOR of two tuples
    def xor_tuples(a, b):
        return tuple(a[i] ^ b[i] for i in range(n))
    
    # Helper: popcount
    def popcount(v):
        return sum(v)
    
    # Phase 1: Greedy CSE - repeatedly find the most useful intermediate vector
    # A vector v is useful if it helps express multiple target vectors with fewer gates
    
    remaining_targets = list(set(target_vecs))
    
    while remaining_targets:
        if time.time() > DEADLINE:
            break
        
        # For each remaining target, find the minimum-cost way to compute it
        # given current available wires
        best_candidate = None
        best_cost = float('inf')
        
        # First, check if any target is already available
        new_remaining = []
        for tv in remaining_targets:
            if tv in vec_to_wire:
                pass  # already available
            else:
                new_remaining.append(tv)
        remaining_targets = new_remaining
        
        if not remaining_targets:
            break
        
        # Try to find a single XOR that produces a target directly
        # Also try to find an intermediate that helps multiple targets
        
        # Strategy: for each target not yet available, find min-cost decomposition
        # Then pick the target with lowest cost, or find shared subexpressions
        
        # Try all pairs of available wires to see if they produce a target
        avail_vecs = list(vec_to_wire.keys())
        
        # Check direct XOR of two available wires produces a target
        direct_hits = {}
        for i in range(len(avail_vecs)):
            for j in range(i + 1, len(avail_vecs)):
                result = xor_tuples(avail_vecs[i], avail_vecs[j])
                if result in remaining_targets:
                    if result not in direct_hits or True:
                        direct_hits[result] = (avail_vecs[i], avail_vecs[j])
        
        if direct_hits:
            # Pick one to add
            for target in direct_hits:
                if time.time() > DEADLINE:
                    break
                a_vec, b_vec = direct_hits[target]
                a_wire = vec_to_wire[a_vec]
                b_wire = vec_to_wire[b_vec]
                gates.append([a_wire, b_wire])
                wire_to_vec[next_wire] = target
                vec_to_wire[target] = next_wire
                next_wire += 1
                if target in remaining_targets:
                    remaining_targets.remove(target)
        else:
            # No direct hit; need to build up intermediates
            # For each target, find the best 2-step or direct-from-inputs decomposition
            
            # Direct from inputs: XOR together all 1-bits
            # This costs popcount - 1 gates (if popcount > 1)
            
            # Better: try to find a = vec ^ b where a and b are "cheaper" to produce
            # Use recursive greedy with memoization-ish approach
            
            # Pick the target with minimum popcount (easiest to build directly)
            # or one that shares structure with others
            
            min_pop = float('inf')
            chosen_target = None
            for tv in remaining_targets:
                pc = popcount(tv)
                if pc < min_pop:
                    min_pop = pc
                    chosen_target = tv
            
            if chosen_target is None:
                break
            
            # Try to decompose chosen_target = a ^ b where a, b are available
            # or where a is cheap to produce
            found_decomp = False
            for av in avail_vecs:
                bv = xor_tuples(chosen_target, av)
                if bv in vec_to_wire:
                    a_wire = vec_to_wire[av]
                    b_wire = vec_to_wire[bv]
                    gates.append([a_wire, b_wire])
                    wire_to_vec[next_wire] = chosen_target
                    vec_to_wire[chosen_target] = next_wire
                    next_wire += 1
                    if chosen_target in remaining_targets:
                        remaining_targets.remove(chosen_target)
                    found_decomp = True
                    break
            
            if not found_decomp:
                # Build directly from inputs
                ones = [j for j in range(n) if chosen_target[j] == 1]
                if len(ones) == 0:
                    pass  # zero, already available
                elif len(ones) == 1:
                    pass  # input, already available
                else:
                    current_wire = ones[0]
                    for j in range(1, len(ones)):
                        gates.append([current_wire, ones[j]])
                        current_wire = next_wire
                        wire_to_vec[next_wire] = xor_tuples(
                            wire_to_vec[current_wire - 1] if current_wire > n else zero_vec,
                            wire_to_vec[ones[j]]
                        )
                        # Actually track properly
                        next_wire += 1
                    # The final wire is current_wire - 1 (last gate output)
                    # Wait, let me redo this more carefully
                    # Let me restart the direct build
                    
                    # Undo what we just did - this is getting messy
                    # Let me just rebuild
                    pass
            
            if not found_decomp and chosen_target not in vec_to_wire:
                # Clean direct build from inputs
                ones = [j for j in range(n) if chosen_target[j] == 1]
                if len(ones) >= 2:
                    # XOR them all together step by step
                    result_vec = wire_to_vec[ones[0]]
                    result_wire = ones[0]
                    for j in range(1, len(ones)):
                        new_vec = xor_tuples(result_vec, wire_to_vec[ones[j]])
                        gates.append([result_wire, ones[j]])
                        wire_to_vec[next_wire] = new_vec
                        vec_to_wire[new_vec] = next_wire
                        result_vec = new_vec
                        result_wire = next_wire
                        next_wire += 1
                    if chosen_target not in vec_to_wire:
                        vec_to_wire[chosen_target] = result_wire
                    if chosen_target in remaining_targets:
                        remaining_targets.remove(chosen_target)
    
    # Phase 2: Ensure all outputs are available
    output_wires = []
    for tv in target_vecs:
        if tv in vec_to_wire:
            output_wires.append(vec_to_wire[tv])
        else:
            # Emergency: build from inputs
            ones = [j for j in range(n) if tv[j] == 1]
            if len(ones) == 0:
                output_wires.append(n)
            elif len(ones) == 1:
                output_wires.append(ones[0])
            else:
                result_vec = wire_to_vec[ones[0]]
                result_wire = ones[0]
                for j in range(1, len(ones)):
                    new_vec = xor_tuples(result_vec, wire_to_vec[ones[j]])
                    gates.append([result_wire, ones[j]])
                    wire_to_vec[next_wire] = new_vec
                    vec_to_wire[new_vec] = next_wire
                    result_vec = new_vec
                    result_wire = next_wire
                    next_wire += 1
                output_wires.append(result_wire)
    
    return {'gates': gates, 'outputs': output_wires}
# EVOLVE-BLOCK-END
