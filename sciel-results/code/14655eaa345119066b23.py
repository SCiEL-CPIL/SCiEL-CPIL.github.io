"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import random
import time

def _count_comparators(net):
    return len(net)

def _depth(net):
    """Compute parallel depth of a sorting network."""
    if not net:
        return 0
    wire_last_layer = [-1] * N
    max_layer = 0
    for i, j in net:
        L = max(wire_last_layer[i], wire_last_layer[j]) + 1
        wire_last_layer[i] = L
        wire_last_layer[j] = L
        max_layer = max(max_layer, L)
    return max_layer

def _is_valid_sorter_fast(net, N=18, timeout=20):
    """Fast validity check using random sampling first, then exact check if needed."""
    start_time = time.time()
    
    # 1. Random Sampling (Fast)
    # If it fails random, it's invalid.
    # If it passes random, it might still be invalid, so we do exact check.
    # But for local search, we can rely on random for speed and do exact at the end.
    
    # For the final result, we MUST do exact check.
    # For intermediate steps in local search, we can use random.
    
    # This function is used for FINAL validation.
    # So we must be exact.
    # To speed up exact check, we can use bitmask simulation.
    
    # Optimization: Pre-compute the effect of each comparator? No.
    # Just use a tight loop.
    
    # We can skip masks that are already sorted? No, they are rare.
    
    # We can use the fact that if the network is valid, it sorts all permutations.
    # But Boolean is sufficient.
    
    # Let's try to optimize the inner loop.
    # wires = [0]*N
    # for mask in range(1<<N):
    #    ...
    
    # A faster way:
    # Iterate through all masks.
    # For each mask, simulate.
    
    # We can break early if we find a failure.
    
    for mask in range(1 << N):
        if time.time() - start_time > timeout:
            return False
            
        # Fast simulation
        # Use a list of integers
        wires = [0] * N
        # Set bits
        for i in range(N):
            if (mask >> i) & 1:
                wires[i] = 1
        
        for i, j in net:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        
        # Check sorted
        # A sorted Boolean array has all 0s then all 1s.
        # So it must be non-decreasing.
        for k in range(N - 1):
            if wires[k] > wires[k + 1]:
                return False
                
    return True

def construct():
    """Construct a sorting network on 18 wires using Local Search from OET seed."""
    
    start_time = time.time()
    timeout = 55  # Leave 5s for safety
    
    # Start with odd-even transposition
    current_net = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Verify it's valid (should be, but fast check)
    # We trust OET is valid, so we skip full check initially to save time.
    # But we should be safe.
    
    # Local Search: Try to remove comparators.
    # Order of removal: Random or from the end?
    # Removing from the end is often safer as they are "redundant" in OET?
    # OET is a sequence of phases.
    
    # We will iterate through the network and try to remove each comparator.
    # If removal keeps the network valid (checked via random sampling for speed), we remove it.
    # At the end, we do a full exact check.
    
    # To speed up random check, we use a small number of samples.
    
    def _random_check(net, samples=100):
        for _ in range(samples):
            mask = random.getrandbits(N)
            wires = [(mask >> i) & 1 for i in range(N)]
            for i, j in net:
                if wires[i] > wires[j]:
                    wires[i], wires[j] = wires[j], wires[i]
            for k in range(N - 1):
                if wires[k] > wires[k + 1]:
                    return False
        return True

    # Shuffle the indices to try removing in random order?
    # Or just iterate backwards.
    
    # We will try to remove each comparator once.
    # If it passes random check, we remove it.
    # We do this in a single pass.
    
    new_net = []
    for idx, comp in enumerate(current_net):
        # Try removing comp
        # Create a temporary net without comp
        # This is O(N) to copy.
        # Better: Just check if current_net[:idx] + current_net[idx+1:] is valid.
        
        # To be faster, we can just keep a list and try to append.
        # But we need to check the *whole* network without this comp.
        
        # Strategy:
        # Keep a list of accepted comparators.
        # For each new comparator, try to add it.
        # If the network with it is valid (random), keep it.
        # If not, skip it.
        
        # This is "Greedy Construction" from the seed list.
        
        candidate_net = new_net + [comp]
        
        if time.time() - start_time > timeout - 5:
            # Time running out, just append rest
            new_net.extend(current_net[idx:])
            break
        
        # Random check
        if _random_check(candidate_net, samples=50):
            new_net.append(comp)
        else:
            # Skip comp
            pass
            
    # Final exact check
    if time.time() - start_time < timeout - 2:
        if not _is_valid_sorter_fast(new_net, N, timeout - 2):
            # If invalid, return the original OET (which is valid)
            return current_net
    else:
        # Timeout, return new_net (risky) or original?
        # Better to return original if we can't verify.
        # But original is 153.
        # If we can't verify new_net, we should return original to ensure score > 0.
        return current_net
        
    return new_net
# EVOLVE-BLOCK-END
