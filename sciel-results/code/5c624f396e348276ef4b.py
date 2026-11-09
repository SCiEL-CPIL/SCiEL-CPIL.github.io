"""
Size-efficient sorting network construction for N=16.
Strategy: 
1. Generate a valid sorting network using Batcher's Bitonic Sort.
2. Apply greedy deletion of redundant comparators, verifying correctness 
   via the zero-one principle after each removal.
3. Return the minimized network.
"""
import time
import random

N = 16
MAX_TIME = 60.0 # Safety margin for evaluator limit of 90s

# EVOLVE-BLOCK-START

def _verify_network(network, n):
    """
    Verify if the network sorts all 2^n boolean inputs using the zero-one principle.
    Returns True if the network is a valid sorting network.
    """
    if not network:
        return n <= 1
    
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        for i, j in network:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check if sorted
        for i in range(n - 1):
            if wires[i] > wires[i + 1]:
                return False
    return True

def _bitonic_sort_network(n):
    """
    Generate a bitonic sorting network for n wires (n must be a power of 2).
    Returns a list of (i, j) comparators.
    """
    if n <= 1:
        return []
    
    network = []
    
    # Recursive bitonic sort
    def rec_bitonic(start, length, ascending):
        if length <= 1:
            return
        half = length // 2
        
        # Split into two halves
        rec_bitonic(start, half, True)
        rec_bitonic(start + half, half, False)
        
        # Merge the two halves
        for i in range(length):
            network.append((start + i, start + i + half))
        
    # Initial generation: Create bitonic sequence
    # Standard bitonic sort generation
    def rec_merge(start, length, ascending):
        if length <= 1:
            return
        half = length // 2
        
        # Recursive merge
        rec_merge(start, half, ascending)
        rec_merge(start + half, half, ascending)
        
        # Compare and swap
        for i in range(half):
            if ascending:
                network.append((start + i, start + i + half))
            else:
                # Note: standard compare-exchange is min to lower index.
                # If we want descending order, we need to be careful with logic or just rely on the final state.
                # Actually, for bitonic sort, we generate a network that sorts in ascending order.
                # The recursive step usually builds a bitonic sequence and then merges.
                # Let's use a standard recursive generator for the *comparators* needed for bitonic sort.
                pass 

    # Simpler approach: Use the standard recursive definition for Bitonic Sort Network generation
    def gen_bitonic(start, length, direction):
        """
        Generates comparators for a bitonic sort network.
        direction: True for ascending, False for descending.
        """
        if length <= 1:
            return []
        
        comparators = []
        half = length // 2
        
        # Recursively generate bitonic networks for halves
        # One half ascending, one descending to create a bitonic sequence
        comparators.extend(gen_bitonic(start, half, True))
        comparators.extend(gen_bitonic(start + half, half, False))
        
        # Merge the bitonic sequence
        for i in range(half):
            comparators.append((start + i, start + i + half))
            
        # Note: The standard bitonic sort network sorts in ascending order.
        # The recursive step above creates a bitonic sequence (up then down) and merges it.
        # However, the "direction" parameter in standard bitonic sort networks usually refers to the final output order.
        # If direction is False, we might need to swap inputs/outputs or use reverse logic.
        # For simplicity and correctness verification, we will generate the standard ascending bitonic sort network.
        # If we need descending, we can just reverse the comparison logic in the final pass or swap i,j in the comparators if the logic allowed, 
        # but compare-exchange is fixed as min->i, max->j.
        # Standard bitonic sort produces ascending order.
        
        return comparators

    # The function gen_bitonic above is slightly flawed for the recursive structure of standard bitonic sort.
    # Let's use a clearer iterative or correct recursive approach.
    
    # Correct Recursive Bitonic Sort Network Generator
    def gen_bsort(start, length):
        if length <= 1:
            return []
        
        comps = []
        half = length // 2
        
        # Generate bitonic sequences
        # Left half: ascending
        comps.extend(gen_bitonic_seq(start, half, True))
        # Right half: descending
        comps.extend(gen_bitonic_seq(start + half, half, False))
        
        # Merge
        for i in range(half):
            comps.append((start + i, start + i + half))
            
        return comps
        
    def gen_bitonic_seq(start, length, ascending):
        """
        Generate a network that produces a bitonic sequence (ascending if ascending=True, else descending)
        from arbitrary input? No, bitonic sort assumes the input is split into two sorted sequences.
        Actually, the standard construction is:
        1. Split into two halves.
        2. Recursively sort first half ascending, second half descending.
        3. Merge the two sorted halves into a single sorted sequence.
        """
        if length <= 1:
            return []
        
        comps = []
        half = length // 2
        
        # Recursive step: Sort halves in opposite directions
        comps.extend(gen_bsort(start, half))
        comps.extend(gen_bsort(start + half, half))
        
        # Merge step: Compare elements i distance apart
        for i in range(half):
            comps.append((start + i, start + i + half))
            
        return comps
        
    # The function gen_bsort above recursively calls itself, which is the standard definition.
    # It generates a network that sorts n elements.
    
    network = gen_bsort(0, n)
    
    # Verify just in case, though it should be correct
    if not _verify_network(network, n):
        # Fallback to odd-even transposition if bitonic generation is buggy
        return _odd_even_transposition(n)
        
    return network

def _odd_even_transposition(n):
    """Fallback sorting network."""
    network = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, n - 1, 2):
                network.append((i, i + 1))
    return network

def construct():
    start_time = time.time()
    
    # 1. Generate initial network using Bitonic Sort
    network = _bitonic_sort_network(N)
    
    # 2. Greedy Deletion of Redundant Comparators
    # We try to remove each comparator if the remaining network is still valid.
    # We iterate multiple times until no more comparators can be removed.
    
    changed = True
    while changed:
        changed = False
        current_time = time.time()
        
        # Iterate backwards to avoid index issues when removing
        for i in range(len(network) - 1, -1, -1):
            # Deadline check
            if time.time() - start_time > MAX_TIME - 5:
                break
            
            # Try removing comparator i
            candidate = network[:i] + network[i+1:]
            
            # Verify candidate
            if _verify_network(candidate, N):
                network = candidate
                changed = True
                # We can break early or continue checking others in the new list.
                # Continuing allows us to remove multiple redundants in one pass if we were careful,
                # but since we modified 'network', the indices for the next iteration in the outer loop
                # would be stale if we continued the inner loop with old 'i'.
                # So we break the inner loop and restart the outer loop.
                break
                
        # Check deadline again
        if time.time() - start_time > MAX_TIME - 5:
            break

    # Final verification
    if not _verify_network(network, N):
        # Should not happen, but return a known good one if it does
        return _odd_even_transposition(N)

    return network

# EVOLVE-BLOCK-END
