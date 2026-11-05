import numpy as np
import time
import random

N = 16
MAX_INPUTS = 2 ** N
DEADLINE = 85.0  # Safety margin for 90s limit

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    # 1. Generate the baseline Odd-Even Transposition Network (120 comparators)
    # This is a valid sorting network. We will attempt to reduce it.
    # Pattern: phases 0..N-1. Phase p uses comparators (i, i+1) where i % 2 == p % 2.
    # Actually, standard odd-even transposition for N wires:
    # Phase 0: (0,1), (2,3), ..., (N-2, N-1)
    # Phase 1: (1,2), (3,4), ..., (N-3, N-2)
    # ...
    # Phase N-1: depends on parity.
    
    # Let's generate the 120 comparator network
    network = []
    for phase in range(N):
        if phase % 2 == 0:
            # Even phase: compare (0,1), (2,3), ...
            for i in range(0, N - 1, 2):
                network.append((i, i + 1))
        else:
            # Odd phase: compare (1,2), (3,4), ...
            for i in range(1, N - 1, 2):
                network.append((i, i + 1))
                
    # Verify baseline just in case (should be 120)
    if len(network) != 120:
        # Fallback to a known valid construction if generation logic was off
        # But for N=16, the above generates exactly 16 phases * 8 comparators = 120.
        pass

    # 2. Prepare Bit-Parallel Verification Data
    # We represent the state of all 65536 inputs for all 16 wires simultaneously.
    # `state` is a numpy array of shape (MAX_INPUTS, N) with values 0 or 1.
    # Or, more efficiently for bit-parallel ops, we can use integer bitmasks.
    # Let's use an array of uint64 integers where each integer represents the 16-bit input.
    # Actually, to apply comparators in parallel, it's easier to keep the (MAX_INPUTS, N) matrix 
    # but use vectorized numpy operations.
    # 
    # Better approach for speed: 
    # Represent the current value of each wire for all inputs as a column of bits.
    # Let W be an array of shape (N, MAX_INPUTS) of bools or ints.
    # W[w, k] is the value of wire w for input k.
    
    # Initialize W for all 2^16 inputs
    # Input k is the binary representation of k.
    # Wire w corresponds to bit w.
    # We can generate this quickly.
    
    # Create a range of all inputs
    inputs = np.arange(MAX_INPUTS, dtype=np.uint64)
    
    # Extract bits for each wire
    # W[w] = (inputs >> w) & 1
    # We want shape (N, MAX_INPUTS)
    # W = np.array([(inputs >> w) & 1 for w in range(N)], dtype=np.uint8)
    
    # This creates a large array: 16 * 65536 = 1,048,576 bytes. Very fast to create.
    W = np.empty((N, MAX_INPUTS), dtype=np.uint8)
    for w in range(N):
        W[w] = (inputs >> w) & 1

    # Helper to apply a comparator (i, j) to the state W
    # min goes to i, max goes to j
    # For each input k:
    # a = W[i, k], b = W[j, k]
    # W[i, k] = min(a, b)
    # W[j, k] = max(a, b)
    # 
    # Vectorized:
    # a = W[i], b = W[j]
    # W[i] = np.minimum(a, b)
    # W[j] = np.maximum(a, b)
    
    def apply_comparator(W_state, i, j):
        a = W_state[i].copy() # Need to be careful with in-place if i/j overlap? No, i != j.
        b = W_state[j].copy()
        W_state[i] = np.minimum(a, b)
        W_state[j] = np.maximum(a, b)

    # Helper to check if the network is sorted
    # Sorted means for all k, W[0, k] <= W[1, k] <= ... <= W[15, k]
    # i.e., W[w, k] >= W[w+1, k] is FALSE for all w? 
    # No, sorted ascending: W[0] <= W[1] <= ...
    # So we need W[w, k] <= W[w+1, k] for all w, k.
    # Equivalently: sum(W) is constant (it is, 8 ones per input? No, sum varies by input).
    # Just check the inequalities.
    
    def is_sorted(W_state):
        # Check if W_state[w] <= W_state[w+1] for all w
        # diff = W_state[1:] - W_state[:-1]
        # If any diff < 0, it's not sorted.
        # Since values are 0 or 1, diff can be -1, 0, 1.
        # We just need to ensure no -1s.
        # Or simply: np.all(W_state[:-1] <= W_state[1:])
        return np.all(W_state[:-1] <= W_state[1:])

    # 3. Local Search: Try to remove comparators
    # We iterate through the network from the end to the beginning.
    # Removing a comparator is only safe if the network remains sorted.
    # Since comparators earlier in the network affect later ones, 
    # removing a comparator at index idx might break sorting if later comparators
    # relied on the ordering established by idx.
    # However, if we remove from the END to the BEGINNING, we can check validity
    # immediately. If removing comparator k breaks sorting, we can't remove it.
    # But wait, if we remove comparator k, the state changes for all subsequent comparators.
    # So we must re-simulate the rest of the network to check if it's still sorted.
    
    # Strategy:
    # 1. Start with the full network.
    # 2. Re-simulate to get the final sorted state (which we know is sorted).
    # 3. Iterate backwards through the list of comparators.
    # 4. For each comparator at index i, temporarily remove it.
    # 5. Re-simulate the network from scratch? No, that's O(N^2).
    #    Better: Re-simulate from comparator i to the end?
    #    Actually, the state before comparator i is fixed by the prefix (0..i-1).
    #    So, we can pre-compute the state at every step? 
    #    Memory: 120 steps * 16 * 65536 bytes = ~125 MB. Acceptable.
    #    
    # Let's use a "Checkpoint" approach.
    # We store the state W at the beginning of each comparator step.
    # state_before[i] is the state before applying network[i].
    
    states = []
    current_W = W.copy()
    for comp in network:
        states.append(current_W.copy())
        apply_comparator(current_W, comp[0], comp[1])
    
    # Now we have states[0..119].
    # states[i] is the state before network[i] is applied.
    # The final state is current_W (after last comparator).
    # We know current_W is sorted.
    
    # We want to remove comparators.
    # Let's try removing them one by one from the end to the start.
    # If we remove comparator at index i, the new network is network[:i] + network[i+1:].
    # The state before comparator i+1 (in the new network) will be the state AFTER comparator i (in old network)?
    # No.
    # If we remove comparator i, the input to comparator i+1 becomes the state BEFORE comparator i.
    # So we can simulate from state i forward through the remaining comparators (i+1 .. end)
    # and check if the result is sorted.
    
    # To make this fast, we can just re-simulate the suffix.
    # Suffix length is up to 120. Simulation of 120 comparators is fast (vectorized).
    # 120 * (few numpy ops on 65k array) is very fast (< 1ms).
    # So we can afford to try removing every comparator multiple times.
    
    # Optimization:
    # We'll iterate backwards. 
    # If we remove a comparator, we update the list.
    # We don't need to re-simulate the whole thing if we are smart, 
    # but given the small size, re-simulating the suffix is simple and robust.
    
    current_network = network[:]
    
    # We'll use a while loop to try removing comparators until no more can be removed.
    # To avoid getting stuck in local minima, we can randomize the order or use multiple passes.
    # But first, a simple backward pass.
    
    changed = True
    while changed:
        changed = False
        # Iterate backwards
        for i in range(len(current_network) - 1, -1, -1):
            if time.time() > DEADLINE:
                break
            
            # Try removing current_network[i]
            # New network: current_network[:i] + current_network[i+1:]
            # We need to check if this new network sorts all inputs.
            
            # Simulation:
            # Start with initial W.
            # Apply all comparators in the new network.
            # Check if sorted.
            
            # To speed this up, we can start the simulation from the state 
            # BEFORE comparator i in the ORIGINAL network?
            # No, because if we remove comparator i, the state entering i+1 is different.
            # The state entering i+1 in the new network is the state BEFORE i in the old network.
            # So we can take states[i] (state before i in old network) 
            # and apply comparators i+1..end from the OLD network?
            # Wait, if we removed i, the subsequent comparators are the same objects, 
            # but their inputs are different.
            # So yes, we can take states[i] and apply current_network[i+1:] to it.
            
            # Let's verify this logic.
            # Old: S_i -> C_i -> S_{i+1} -> C_{i+1} -> ...
            # New: S_i -> C_{i+1} -> ...
            # So we start with S_i and apply C_{i+1}, C_{i+2}...
            
            test_W = states[i].copy()
            for comp in current_network[i+1:]:
                apply_comparator(test_W, comp[0], comp[1])
                
            if is_sorted(test_W):
                # Safe to remove
                del current_network[i]
                changed = True
                # Since we removed an item, the indices shift. 
                # We are iterating backwards, so i-1 is the next one to check.
                # The states array is now "stale" for indices < i? 
                # states[i] was state before i. 
                # If we remove i, the state before i+1 (now at index i in new list) is states[i].
                # The state before i (now at index i-1 in new list) is states[i-1].
                # So states[k] for k <= i is still valid as "state before comparator k in the NEW network"?
                # Let's check.
                # New network: C_0 ... C_{i-1}, C_{i+1} ...
                # State before C_{i-1} is same as old.
                # State before C_{i+1} (now at pos i) is the output of C_{i-1}, which is states[i].
                # So yes, states[k] for k < i is valid. states[i] is valid for the new index i.
                # But states[i+1] is NOT valid for the new index i+1 (which is old i+2).
                # So we must be careful. 
                # Actually, since we are iterating backwards, and we just removed i,
                # the next index to check is i-1.
                # The state before i-1 is states[i-1], which is unchanged.
                # So we can continue the loop with i-1.
                # However, if we remove multiple comparators, the `states` array becomes increasingly invalid 
                # for higher indices. But since we only look at `states[i]` where i is the current index,
                # and we are moving downwards, we only need states[i] to be the state before the comparator 
                # currently at index i.
                # 
                # Wait, if I removed index 10, the comparator now at index 10 is old 11.
                # The state before old 11 was states[11].
                # But in the new network, the state before old 11 is the output of old 10? No, old 10 is removed.
                # So the state before old 11 is the state before old 10, which is states[10].
                # So `states[10]` is the correct starting state for the suffix starting at new index 10.
                # 
                # So the logic holds: `states[i]` is always the state before the comparator currently at index i.
                # Why?
                # Initially true.
                # If we remove comparator at i, the comparator at i becomes old i+1.
                # The state before old i+1 in the new network is the output of old i-1 (since old i is gone).
                # The output of old i-1 is states[i].
                # So yes, states[i] is correct.
                
                # We continue the loop, i will decrement to i-1 in the for loop.
                pass
            else:
                # Cannot remove
                pass
        
        if time.time() > DEADLINE:
            break

    # If we have time, we can try to shuffle and re-optimize to escape local minima.
    # But for now, the backward removal from a structured network is a strong baseline.
    # Let's add a simple random restart/shuffle if time permits.
    
    # Check time
    if time.time() < DEADLINE - 5:
        # Try a few random shuffles and re-optimizations
        # This is a heuristic to find better local minima.
        for _ in range(5): # Max 5 restarts
            if time.time() > DEADLINE - 2:
                break
            
            # Shuffle the current network
            shuffled_net = current_network[:]
            random.shuffle(shuffled_net)
            
            # Re-run the backward removal on the shuffled network
            # We need to recompute states for the shuffled network
            states_shuf = []
            current_W_shuf = W.copy()
            for comp in shuffled_net:
                states_shuf.append(current_W_shuf.copy())
                apply_comparator(current_W_shuf, comp[0], comp[1])
            
            changed_shuf = True
            while changed_shuf:
                changed_shuf = False
                for i in range(len(shuffled_net) - 1, -1, -1):
                    if time.time() > DEADLINE - 2:
                        break
                    
                    test_W = states_shuf[i].copy()
                    for comp in shuffled_net[i+1:]:
                        apply_comparator(test_W, comp[0], comp[1])
                    
                    if is_sorted(test_W):
                        del shuffled_net[i]
                        changed_shuf = True
                        # states_shuf[i] is still valid for the new index i
                        pass
                
                if time.time() > DEADLINE - 2:
                    break
            
            # If the shuffled network is smaller, adopt it
            if len(shuffled_net) < len(current_network):
                current_network = shuffled_net

    return current_network

# EVOLVE-BLOCK-END
