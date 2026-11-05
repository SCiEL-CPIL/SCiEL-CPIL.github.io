"""Size-efficient sorting network search on 16 wires using bit-parallel verification."""
import time
import random
import numpy as np

N = 16
NUM_BOOLS = 1 << N
DEADLINE = time.time() + 85.0  # 85s safety margin (90s limit)

def verify_network_bitparallel(network):
    """
    Check all 65536 Boolean inputs using zero-one principle with bit-parallelism.
    Returns True if the network sorts all inputs.
    """
    if not network:
        return False
        
    # State: array of shape (N, NUM_BOOLS)
    # state[i, j] is 1 if wire i has value 1 for the j-th input
    # We represent the j-th input as bits in the integer index j.
    # Actually, it's easier to think of each column as one input.
    # state[i] is an array of NUM_BOOLS bits, where bit k is the value of wire i for input k.
    
    # Initialize state
    # For input k (0 to 65535), wire i is (k >> i) & 1.
    # We can generate this efficiently.
    # state[i] should be a uint8 or bool array of length NUM_BOOLS.
    
    # Let's use uint8 for speed.
    # state[i, k] = 1 if wire i is 1 for input k.
    # Input k is an integer. Bit i of k is the value of wire i.
    # So state[i, k] = (k >> i) & 1.
    
    # We can generate k = arange(NUM_BOOLS)
    # Then state[i] = (k >> i) & 1
    
    # But doing this for every verification is expensive.
    # However, we only need to initialize once per verification call.
    # The verification itself involves applying comparators.
    
    # Let's precompute the base bits for all wires? No, the state changes.
    # But the *initial* state is fixed for all verifications.
    
    # Initial state:
    # k = np.arange(NUM_BOOLS, dtype=np.uint32)
    # state = np.zeros((N, NUM_BOOLS), dtype=np.uint8)
    # for i in range(N):
    #     state[i] = (k >> i) & 1
    
    # This initialization takes ~16 * 65536 ops, which is fast.
    
    # Optimization: Precompute the initial states outside the function?
    # Yes, let's do that.
    
    global _INITIAL_STATE
    if not hasattr(verify_network_bitparallel, '_INITIAL_STATE'):
        k = np.arange(NUM_BOOLS, dtype=np.uint32)
        _init = np.zeros((N, NUM_BOOLS), dtype=np.uint8)
        for i in range(N):
            _init[i] = (k >> i) & 1
        verify_network_bitparallel._INITIAL_STATE = _init
    
    # Copy initial state
    state = verify_network_bitparallel._INITIAL_STATE.copy()
    
    # Apply comparators
    for i, j in network:
        # Comparator (i, j): min to i, max to j.
        # If state[i, k] > state[j, k], swap.
        # Since values are 0 or 1:
        # min = state[i] & state[j]
        # max = state[i] | state[j]
        # So new state[i] = state[i] & state[j]
        # new state[j] = state[i] | state[j]
        
        # Wait, this is only correct if we use the *old* values for both.
        # min(a,b) = a & b for bits.
        # max(a,b) = a | b for bits.
        # Yes, this is correct for 0/1 values.
        
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
        
    # Check if sorted:
    # For all k, state[0, k] <= state[1, k] <= ... <= state[N-1, k]
    # Since values are 0/1, sorted means once a 1 appears, all subsequent must be 1.
    # Equivalently, state[i] <= state[i+1] element-wise.
    # state[i] <= state[i+1] is equivalent to (state[i] & ~state[i+1]) == 0.
    # Or simply: np.all(state[:-1] <= state[1:])
    
    # Let's check:
    # if np.any(state[:-1] > state[1:]): return False
    
    # Even faster:
    # A sorted sequence of 0s and 1s has the form 0...01...1.
    # This is equivalent to: for all i, if state[i, k] == 0 then state[i-1, k] == 0? No.
    # If state[i, k] == 1 then state[i+1, k] == 1.
    # So state[i] & ~state[i+1] must be 0 for all i.
    
    # Let's just use the comparison.
    if np.any(state[:-1] > state[1:]):
        return False
        
    return True

def odd_even_network():
    """Odd-even transposition sorting network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def try_remove(network, idx):
    """Try removing comparator at idx, return new network if still valid."""
    if len(network) <= 1:
        return None
    new_net = network[:idx] + network[idx+1:]
    if verify_network_bitparallel(new_net):
        return new_net
    return None

def try_insert(network, i, j, pos):
    """Try inserting comparator (i,j) at position pos, return new network if valid."""
    if not (0 <= i < j < N):
        return None
    new_net = network[:pos] + [(i, j)] + network[pos:]
    if verify_network_bitparallel(new_net):
        return new_net
    return None

def construct():
    start_time = time.time()
    
    # Start with known-good baseline
    best = odd_even_network()
    best_count = len(best)
    
    # Phase 1: Local search to remove redundant comparators from baseline
    # We will loop until deadline.
    # Each iteration:
    # 1. Try to remove a random comparator.
    # 2. If successful, continue.
    # 3. If not, try to insert a random comparator, then try to remove a different one (2-opt style).
    
    while time.time() < DEADLINE - 5:  # Leave 5s for final verification and safety
        improved = False
        
        # Attempt removal
        # Pick a random index
        idx = random.randint(0, len(best) - 1)
        new_net = try_remove(best, idx)
        if new_net is not None:
            best = new_net
            best_count = len(best)
            improved = True
            continue  # Restart with new length
            
        # If removal failed, try a "swap" or "insert-then-remove"
        # Insert a random comparator
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(best))
        
        # Check if this exact comparator already exists at this position?
        # Not strictly necessary, but might waste time.
        
        temp_net = best[:pos] + [(i, j)] + best[pos:]
        
        # Now try to remove a random comparator from temp_net (could be the one just added, or another)
        # If we remove the one just added, we are back to best (which we know is valid but didn't improve size).
        # We want to remove a *different* one to keep the size same but potentially change structure,
        # or remove the inserted one if it allows removing another? No, we want to reduce size.
        # So we insert 1, then remove 1. If we remove the inserted one, size is same.
        # If we remove another one, size is same, but structure changed.
        # This is a "move".
        
        # To reduce size, we need to insert 0 and remove 1 (which is just removal, already tried)
        # OR insert 1 and remove 2? No, that's complex.
        
        # Let's stick to:
        # 1. Remove 1 (if valid, size decreases).
        # 2. If not, try "Move": Insert 1 at pos1, Remove 1 at pos2 (pos2 != pos1).
        #    This keeps size constant but might escape local minima for future removals.
        
        # Let's try a few random moves.
        for _ in range(10):
            if time.time() > DEADLINE:
                break
            # Pick a random index to remove from temp_net
            # Avoid removing the one we just added if we want to change structure?
            # Actually, if we remove the one we just added, we are back to `best`.
            # So we should pick an index != pos.
            remove_idx = random.randint(0, len(temp_net) - 1)
            if remove_idx == pos:
                remove_idx = (remove_idx + 1) % len(temp_net)
                
            # Construct net with insertion and removal
            # temp_net has len(best) + 1
            # We remove one, so len becomes len(best)
            
            # Efficiently construct:
            # temp_net = best[:pos] + [(i,j)] + best[pos:]
            # We want to remove element at remove_idx from temp_net.
            
            # This is getting complicated to build the list.
            # Let's just build it.
            new_net = temp_net[:remove_idx] + temp_net[remove_idx+1:]
            
            # Verify
            if verify_network_bitparallel(new_net):
                # If valid, accept. Size is same as best.
                # This allows us to explore the landscape of same-size networks.
                best = new_net
                improved = True
                break
                
        if not improved:
            # If we are stuck, maybe try a more aggressive restart?
            # Or just keep trying removals.
            pass

    # Final verification
    if not verify_network_bitparallel(best):
        # Fallback to baseline
        best = odd_even_network()
    
    return best
