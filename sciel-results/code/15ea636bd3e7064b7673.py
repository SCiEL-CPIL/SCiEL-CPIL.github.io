import time
import random

N = 16
DEADLINE = 85.0  # Leave 5s margin for the evaluator's 90s limit

def _init_masks():
    """
    Initialize the 16 wire masks.
    Mask w is a 65536-bit integer where bit k is 1 if wire w has value 1 in input k.
    Input k is a 16-bit integer. Wire w corresponds to bit w of k.
    """
    masks = []
    for w in range(N):
        m = 0
        # Period is 2^(w+1). In the first half (0 to 2^w - 1), bit is 0.
        # In the second half (2^w to 2^(w+1) - 1), bit is 1.
        # This pattern repeats every 2^(w+1).
        period = 1 << (w + 1)
        half = 1 << w
        
        # Construct the mask for one period
        # Bits 0..half-1 are 0, bits half..period-1 are 1.
        # This is (2^half - 1) << half
        period_mask = ((1 << half) - 1) << half
        
        # Repeat this mask for the full 2^N range
        # Number of repetitions is 2^N / period = 2^(N - w - 1)
        reps = 1 << (N - w - 1)
        
        for r in range(reps):
            m |= period_mask << (r * period)
        masks.append(m)
    return masks

def _verify_network(network, init_masks=None):
    """
    Verify the network using bitmask parallelism.
    Returns True if the network sorts all inputs.
    """
    if init_masks is None:
        masks = _init_masks()
    else:
        # Copy the masks to avoid modifying the original
        masks = init_masks[:]
    
    for i, j in network:
        a = masks[i]
        b = masks[j]
        masks[i] = a & b
        masks[j] = a | b
    
    # Check if sorted: masks[k] must be a subset of masks[k-1]
    # i.e., masks[k-1] & ~masks[k] should be 0
    for k in range(1, N):
        if (masks[k-1] & ~masks[k]) != 0:
            return False
    return True

def construct():
    start_time = time.time()
    
    # 1. Generate initial network: Odd-Even Transposition (120 comparators)
    # This is a valid sorting network.
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify initial network (should be True)
    if not _verify_network(network):
        # Fallback if something went wrong (shouldn't happen)
        # Just return the network as is, or a simple bubble sort
        return [(i, i+1) for i in range(N-1)]

    # 2. Greedy Pruning
    # We iterate through the network and try to remove each comparator.
    # To speed up, we can maintain the state of the network?
    # No, because removing a comparator changes the state for all subsequent comparators.
    # So we must re-verify from scratch for each candidate removal.
    
    # Optimization:
    # We can pre-compute the "prefix" states?
    # State after step k depends on steps 0..k.
    # If we remove step k, the state for step k+1 is different.
    # So we can't easily reuse prefix states for removal.
    
    # However, we can try removing from the end to the beginning.
    # If we remove a comparator at the end, the prefix states are unchanged.
    # So we can maintain the state up to the end and just check the final condition?
    # No, because the final condition depends on the last few steps.
    # But if we remove step k, steps k+1..end are applied to a different state.
    
    # Let's stick to full re-verification for correctness and simplicity.
    # With bitmask verification, it's very fast.
    
    # We will do multiple passes.
    # In each pass, we try to remove every comparator.
    # If we remove one, we restart the pass (or continue, but the list changes).
    # Restarting is safer and often converges faster.
    
    # To improve efficiency, we can try to identify "independent" comparators?
    # Or just rely on the speed of the bitmask check.
    
    # Let's estimate the cost.
    # 120 comparators.
    # Each check: 120 AND/OR ops on 65536-bit ints.
    # Python big ints are efficient.
    # 120 * 2 * 1000 bytes (approx) = 240KB of ops.
    # Should be < 1ms per check.
    # 120 checks per pass. < 100ms per pass.
    # We can do many passes.
    
    while time.time() - start_time < DEADLINE:
        removed = False
        # Iterate in reverse to allow safe removal
        for idx in range(len(network) - 1, -1, -1):
            if time.time() - start_time > DEADLINE:
                return network
            
            # Try removing network[idx]
            # Create candidate
            # To avoid list slicing overhead, we can just check
            # but slicing is O(N). N is small (<=120).
            
            candidate = network[:idx] + network[idx+1:]
            
            if _verify_network(candidate):
                network = candidate
                removed = True
                # Break and restart pass? Or continue?
                # If we continue, we are checking a modified network.
                # Since we are going backwards, the indices before idx are unchanged.
                # But the network length changed.
                # It's safer to restart the pass to ensure we don't miss interactions.
                # However, restarting might be slower if we remove many.
                # Let's try continuing first. If it fails to remove more, we'll restart.
                # Actually, if we remove idx, the next index to check is idx-1.
                # The network at idx-1 is the same.
                # So we can just continue.
                
        if not removed:
            # No more single removals possible.
            # Try 2-opt or swaps?
            # Or just stop, as single removal is the main gain.
            # The prompt says "as few as possible".
            # Greedy single removal is a good baseline.
            
            # Let's try a simple local search:
            # Try swapping adjacent comparators?
            # Or try removing a comparator and adding a different one?
            # This is more complex.
            
            # For now, let's just do a few more passes to be sure.
            # Sometimes removing one allows another to be removed that wasn't removable before?
            # Yes, that's why we loop.
            # If `removed` is False, it means NO comparator could be removed.
            # So we are at a local minimum for single removals.
            break
            
    return network

# Test the construction
if __name__ == "__main__":
    net = construct()
    print(f"Comparators: {len(net)}")
    print("Valid:", _verify_network(net))
    # Print depth for info
    depth = 0
    last_time = 0
    # This is a rough estimate, not exact parallel depth
    # Exact depth requires topological sort
    pass
