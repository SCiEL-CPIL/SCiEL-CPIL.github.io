"""
Strategy: Greedy Construction + Local Search
1. Start with an empty network.
2. Use the Zero-One Principle: A network sorts all inputs iff it sorts all 2^16 binary inputs.
3. We track the set of "bad" binary inputs (represented as 16-bit integers).
4. Greedy Phase: Repeatedly find the comparator (i, j) that, when added to the current network, 
   fixes the largest number of currently "bad" inputs. Add it.
5. Repeat until no bad inputs remain (network is valid).
6. Local Search Phase: Try to remove comparators one by one if the network remains valid.
7. Return the final list of comparators.
"""

import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096

# Precompute all possible binary inputs (0 to 2^16 - 1)
# Represent each input as a 16-bit integer.
# Bit k is 1 if wire k has value 1.

def apply_comparator(state, i, j):
    """
    Apply compare-exchange to a single 16-bit state.
    min to i, max to j.
    """
    bit_i = (state >> i) & 1
    bit_j = (state >> j) & 1
    
    if bit_i > bit_j:
        # Swap: i gets 0, j gets 1 (since 1 > 0)
        # Actually: min(0,1)=0 goes to i, max(0,1)=1 goes to j.
        # If i=1, j=0: new i=0, new j=1.
        # state ^= (1 << i) | (1 << j)
        state ^= (1 << i) | (1 << j)
    return state

def is_sorted(state):
    """
    Check if the 16-bit state is sorted (non-decreasing).
    Sorted means all 0s come before all 1s.
    So it looks like 00...011...1.
    """
    # If state is 0 or (2^16 - 1), it is sorted.
    # More generally, state must be of the form 2^k - 1 for some k? No.
    # 000111 is sorted. 010011 is not.
    # A binary string is sorted if it has at most one transition from 0 to 1.
    # Or simply: it is 0...01...1.
    # This is equivalent to: state == 0 or state == (1 << N) - 1 or 
    # (state & (state + 1)) == 0 ? No, that's for power of 2.
    
    # Let's check: 
    # 000111 (7) -> sorted
    # 001011 (11) -> not sorted
    # 011100 (28) -> not sorted
    
    # A number is "sorted" in this context if all 1 bits are contiguous and at the LSB end?
    # No, wire 0 is min. So 1s should be at the HIGH end?
    # Wait, "min to wire i". 
    # If input is [1, 0], i=0, j=1. min=0 to 0, max=1 to 1. Result [0, 1].
    # So wire 0 is smallest, wire 15 is largest.
    # Sorted state: 0, 0, ..., 0, 1, 1, ..., 1.
    # In binary representation where bit 0 is wire 0:
    # [0, 1] is 10 (binary) = 2.
    # [0, 0, 1] is 100 (binary) = 4.
    # [0, 1, 1] is 110 (binary) = 6.
    # [1, 1, 1] is 111 (binary) = 7.
    # [0, 0, 0] is 0.
    
    # So a state is sorted if it is of the form: 
    # bits 0..k-1 are 0, bits k..15 are 1.
    # This means state = 2^16 - 2^(16-k) = 2^16 - 2^m for some m.
    # Or state = 0.
    
    # Let's verify:
    # k=0 (all 1s): 2^16 - 1 = 65535.
    # k=1 (one 0, rest 1s): 2^16 - 2^15 = 32768. (1000...0) -> Wire 0 is 0, others 1. Correct.
    # k=16 (all 0s): 2^16 - 2^16 = 0.
    
    # So state is sorted iff state == 0 OR (state > 0 AND (state & (state - 1)) == 0 ? No.
    # The numbers are: 0, 32768, 49152, 57344, ...
    # These are numbers where the set bits form a contiguous block ending at the MSB.
    # Equivalently, the complement (~state) has a contiguous block of 1s ending at the LSB.
    # Let x = ~state & 0xFFFF.
    # x must be of the form 2^m - 1.
    # i.e., x & (x + 1) == 0.
    
    x = (~state) & 0xFFFF
    return (x & (x + 1)) == 0

def verify_network(network):
    """
    Verify if the network sorts all 2^16 inputs.
    Returns True if valid, False otherwise.
    """
    # To be fast, we can just check a subset first, but for correctness in the final step,
    # we must check all. However, 65536 * 120 ops is ~8M ops, which is fast in Python if optimized?
    # No, pure Python loop over 65536 states * 120 comparators is ~7.8M iterations.
    # Each iteration has bit shifts and logic. This might take ~0.5-1s.
    # We have 90s budget.
    
    # Optimization: We can use numpy to vectorize the verification.
    # But applying comparators sequentially is hard to vectorize easily without storing all intermediate states.
    # Storing 65536 states of 16 bits is small.
    
    # Let's try a vectorized approach for verification.
    # states = np.arange(2**16, dtype=np.uint32)
    # For each comparator (i, j):
    #    bit_i = (states >> i) & 1
    #    bit_j = (states >> j) & 1
    #    # If bit_i > bit_j, swap
    #    # swap_mask = bit_i & (~bit_j)
    #    # states = np.where(swap_mask, states ^ (1<<i) | (1<<j), states)
    
    # This is O(Comparators * 2^16). 
    # 120 * 65536 = 7.8M vector operations. NumPy handles this in milliseconds.
    
    states = np.arange(2**16, dtype=np.uint32)
    for i, j in network:
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        # If bit_i is 1 and bit_j is 0, we need to swap.
        # swap_mask is 1 where bit_i=1 and bit_j=0.
        swap_mask = bit_i & (~bit_j) & 1
        # Apply swap: XOR with (1<<i) | (1<<j)
        # But only where swap_mask is 1.
        # states = states ^ (swap_mask * ((1<<i) | (1<<j)))
        # Note: swap_mask is 0 or 1.
        # We can just do:
        # states[swap_mask == 1] ^= (1<<i) | (1<<j)
        # This is fast.
        mask = (swap_mask == 1)
        if np.any(mask):
            states[mask] ^= ((1 << i) | (1 << j))
            
    # Check if all states are sorted.
    # Sorted if x = (~state) & 0xFFFF is of form 2^m - 1.
    # i.e., x & (x + 1) == 0.
    x = (~states) & 0xFFFF
    sorted_mask = (x & (x + 1)) == 0
    return np.all(sorted_mask)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # 1. Greedy Construction
    # Start with empty network
    network = []
    
    # We need to know which inputs are currently "bad".
    # Instead of re-verifying the whole network every time (O(C * 2^16)),
    # we can track the set of bad inputs incrementally?
    # No, adding a comparator changes the state of ALL inputs.
    # So we must re-evaluate or track states.
    
    # Strategy: Keep the current states for all 65536 inputs.
    # Initially, states = identity (0..65535).
    # Bad inputs are those not sorted.
    
    states = np.arange(2**16, dtype=np.uint32)
    
    # Function to get mask of unsorted states
    def get_unsorted_mask():
        x = (~states) & 0xFFFF
        return (x & (x + 1)) != 0
        
    unsorted_mask = get_unsorted_mask()
    
    # If already sorted (impossible for identity unless N=1), return empty.
    if not np.any(unsorted_mask):
        return []
        
    # Precompute all possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Greedy Loop
    while np.any(unsorted_mask):
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_comp = None
        best_new_states = None
        best_new_unsorted = None
        
        # For each candidate comparator, estimate how many unsorted inputs it fixes.
        # To be exact, we should apply it and count.
        # 120 comparators * 65536 states is too slow to do for ALL ~120 candidates in Python loops.
        # But we can vectorize the evaluation of candidates?
        # No, candidates are applied sequentially.
        
        # Optimization: Only consider comparators that involve wires with "potential" to fix.
        # Or, just iterate through a random subset of comparators?
        # Better: Iterate through all, but use NumPy to calculate the effect.
        
        # Let's try to speed up the candidate evaluation.
        # We can't easily parallelize the "apply comparator" step for 120 different comparators 
        # on the same `states` array in a single vectorized op without broadcasting to 120x65536.
        # 120 * 65536 * 4 bytes = ~30MB. This is feasible.
        
        # Let's create a batch of candidate comparators.
        # Actually, let's just pick the "most promising" comparators first.
        # Heuristic: Comparators between adjacent wires are usually more useful?
        # Or random sampling.
        
        # Given the time constraint, let's try a simpler heuristic:
        # Try all adjacent comparators first, then random others.
        
        candidates = []
        # Adjacent
        for i in range(N-1):
            candidates.append((i, i+1))
        # Random others
        random.shuffle(all_comparators)
        for c in all_comparators:
            if c not in candidates:
                candidates.append(c)
            if len(candidates) > 50: # Limit candidates per step to save time
                break
                
        for i, j in candidates:
            # Simulate adding (i, j)
            # We need to copy states? Or calculate delta?
            # Calculating delta:
            # new_states = states.copy()
            # This is 65536 * 4 bytes = 256KB copy. 
            # Doing this 50 times = 12MB memory bandwidth. Fast.
            
            new_states = states.copy()
            bit_i = (new_states >> i) & 1
            bit_j = (new_states >> j) & 1
            swap_mask = bit_i & (~bit_j) & 1
            mask = (swap_mask == 1)
            if np.any(mask):
                new_states[mask] ^= ((1 << i) | (1 << j))
                
            # Count newly sorted
            x = (~new_states) & 0xFFFF
            new_sorted_mask = (x & (x + 1)) == 0
            # We want to maximize (new_sorted_mask & unsorted_mask).sum()
            # i.e., how many previously unsorted are now sorted.
            fixed_count = np.sum(new_sorted_mask & unsorted_mask)
            
            if fixed_count > best_gain:
                best_gain = fixed_count
                best_comp = (i, j)
                best_new_states = new_states
                best_new_unsorted = ~new_sorted_mask
                
        if best_comp is None or best_gain == 0:
            # No progress? This shouldn't happen if we search all comparators.
            # If we limited candidates, we might get stuck.
            # If stuck, break and hope the rest is sorted? No, must be valid.
            # Force a random comparator if stuck?
            # Or expand candidate list.
            break
            
        # Commit the best comparator
        network.append(best_comp)
        states = best_new_states
        unsorted_mask = best_new_unsorted
        
        # Check if done
        if not np.any(unsorted_mask):
            break
            
    # 2. Local Search (Removal)
    # Try to remove comparators from the end (or random order) if valid.
    # Verification is expensive (O(C * 2^16)). 
    # We can do a few iterations.
    
    # Optimization for removal: 
    # If we remove a comparator, we can check validity.
    # To speed up, we can check a random subset of inputs first.
    
    def is_valid_subset(network, num_samples=1000):
        # Sample random inputs
        indices = np.random.choice(2**16, num_samples, replace=False)
        states_sample = indices.astype(np.uint32)
        
        for i, j in network:
            bit_i = (states_sample >> i) & 1
            bit_j = (states_sample >> j) & 1
            swap_mask = bit_i & (~bit_j) & 1
            mask = (swap_mask == 1)
            if np.any(mask):
                states_sample[mask] ^= ((1 << i) | (1 << j))
                
        x = (~states_sample) & 0xFFFF
        sorted_mask = (x & (x + 1)) == 0
        return np.all(sorted_mask)
        
    # Try removing from the end
    # Note: Order matters. Removing from end is safer?
    # Actually, any order.
    
    # We will try to remove each comparator in reverse order.
    # If removal keeps it valid (on full set), keep it removed.
    # Full verification is ~0.1s? 
    # If we have 100 comparators, 100 * 0.1s = 10s. Feasible.
    
    # Let's do a limited number of removals to stay within time.
    max_removal_attempts = 20
    attempts = 0
    i = len(network) - 1
    while i >= 0 and attempts < max_removal_attempts and time.time() < deadline:
        attempts += 1
        # Try removing network[i]
        candidate_network = network[:i] + network[i+1:]
        
        # Quick check on subset
        if is_valid_subset(candidate_network, 500):
            # Full check
            if verify_network(candidate_network):
                network = candidate_network
                # Do not decrement i, because the next item is now at index i
                # Wait, if we remove i, the list shrinks. 
                # The item that was i+1 is now i.
                # So we check the new i again.
            else:
                i -= 1
        else:
            i -= 1
            
    return network

# To ensure the function is robust, we wrap it in a try-except if needed, 
# but the logic above seems sound.
# One risk: `states` array modification in greedy phase.
# `states` is a numpy array. `new_states = states.copy()` is safe.

# Let's add a fallback to the standard odd-even network if the greedy search fails or times out.
def _fallback_construct():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# We will use the greedy construct, but if it returns an invalid network, 
# we should probably return the fallback or the best found.
# The task says "return the best result found".
# If greedy fails to find a valid network, it returns a partial one which is invalid -> Score 0.
# So we MUST ensure validity.
# If we break out of the greedy loop with unsorted inputs remaining, the network is invalid.
# We should check validity at the end. If invalid, return fallback?
# Or continue search?
# Given the 90s limit, if greedy doesn't finish, it's likely stuck or slow.
# Let's modify `construct` to check validity and fall back if invalid.

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # 10s margin for overhead
    
    try:
        net = _greedy_construct(deadline)
        # Verify the result
        if verify_network(net):
            return net
        else:
            # If greedy failed to produce a valid network, return fallback
            # Fallback is 120 comparators.
            # Greedy might find < 120.
            # If greedy fails, it's better to return the valid 120-comparator network 
            # than an invalid one (score 0).
            return _fallback_construct()
    except Exception as e:
        # In case of any error, return fallback
        return _fallback_construct()

def _greedy_construct(deadline):
    N = 16
    network = []
    states = np.arange(2**16, dtype=np.uint32)
    
    def get_unsorted_mask():
        x = (~states) & 0xFFFF
        return (x & (x + 1)) != 0
        
    unsorted_mask = get_unsorted_mask()
    
    if not np.any(unsorted_mask):
        return []
        
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Greedy Loop
    while np.any(unsorted_mask):
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_comp = None
        best_new_states = None
        best_new_unsorted = None
        
        # To save time, we don't check ALL 120 comparators every step if we can avoid it.
        # But 120 * 65536 is ~8M ops. 
        # If we do this 100 times, that's 800M ops. 
        # NumPy vectorized: 800M / (100M ops/s) = 8s? 
        # It depends on the overhead of Python loop for the 120 candidates.
        # 120 iterations of Python loop, each doing ~10 numpy ops on 65k array.
        # 120 * 10 * (65k * 4 bytes) memory bandwidth ~ 300MB. 
        # This should be fast (<1s per step).
        # So 100 steps = 100s? Too slow.
        
        # Optimization: Reduce candidate set.
        # Only consider comparators that touch wires involved in unsorted patterns?
        # Or just random sample of 20 comparators per step.
        
        candidates = random.sample(all_comparators, 20)
        
        for i, j in candidates:
            new_states = states.copy()
            bit_i = (new_states >> i) & 1
            bit_j = (new_states >> j) & 1
            swap_mask = bit_i & (~bit_j) & 1
            mask = (swap_mask == 1)
            if np.any(mask):
                new_states[mask] ^= ((1 << i) | (1 << j))
                
            x = (~new_states) & 0xFFFF
            new_sorted_mask = (x & (x + 1)) == 0
            fixed_count = np.sum(new_sorted_mask & unsorted_mask)
            
            if fixed_count > best_gain:
                best_gain = fixed_count
                best_comp = (i, j)
                best_new_states = new_states
                best_new_unsorted = ~new_sorted_mask
                
        if best_comp is None or best_gain == 0:
            # Stuck with random candidates. Try all?
            # If we are stuck, we might need to try more.
            # But if we run out of time, we break.
            # Let's try a few more random samples.
            stuck_count = 0
            while stuck_count < 5 and time.time() < deadline:
                candidates = random.sample(all_comparators, 20)
                for i, j in candidates:
                    new_states = states.copy()
                    bit_i = (new_states >> i) & 1
                    bit_j = (new_states >> j) & 1
                    swap_mask = bit_i & (~bit_j) & 1
                    mask = (swap_mask == 1)
                    if np.any(mask):
                        new_states[mask] ^= ((1 << i) | (1 << j))
                        
                    x = (~new_states) & 0xFFFF
                    new_sorted_mask = (x & (x + 1)) == 0
                    fixed_count = np.sum(new_sorted_mask & unsorted_mask)
                    
                    if fixed_count > best_gain:
                        best_gain = fixed_count
                        best_comp = (i, j)
                        best_new_states = new_states
                        best_new_unsorted = ~new_sorted_mask
                stuck_count += 1
                if best_gain > 0:
                    break
                    
            if best_gain == 0:
                break # Truly stuck or time out
                
        network.append(best_comp)
        states = best_new_states
        unsorted_mask = best_new_unsorted
        
    # Local Search
    # Try to remove
    # We'll do a simple pass.
    # To save time, we limit removal attempts.
    
    def is_valid_subset(network, num_samples=200):
        indices = np.random.choice(2**16, num_samples, replace=False)
        states_sample = indices.astype(np.uint32)
        
        for i, j in network:
            bit_i = (states_sample >> i) & 1
            bit_j = (states_sample >> j) & 1
            swap_mask = bit_i & (~bit_j) & 1
            mask = (swap_mask == 1)
            if np.any(mask):
                states_sample[mask] ^= ((1 << i) | (1 << j))
                
        x = (~states_sample) & 0xFFFF
        sorted_mask = (x & (x + 1)) == 0
        return np.all(sorted_mask)
        
    # Attempt removals
    # Start from end
    i = len(network) - 1
    attempts = 0
    max_attempts = 50
    while i >= 0 and attempts < max_attempts and time.time() < deadline:
        attempts += 1
        candidate_network = network[:i] + network[i+1:]
        
        if is_valid_subset(candidate_network, 100):
            if verify_network(candidate_network):
                network = candidate_network
                # i stays same, check new element at i
            else:
                i -= 1
        else:
            i -= 1
            
    return network

# Final wrapper to ensure signature and safety
def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    try:
        net = _greedy_construct(deadline)
        # Final verification
        if verify_network(net):
            return net
        else:
            # If greedy produced invalid, fallback
            return _fallback_construct()
    except Exception:
        return _fallback_construct()

# Ensure N is defined globally for the fallback
N = 16

def _fallback_construct():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# Re-define construct to use the local _greedy_construct and _fallback_construct
# The previous definition of construct was overwritten? No, I defined it twice.
# Let's clean up the code structure.

import time
import random
import numpy as np

N = 16

def _verify_network_full(network):
    """
    Verify if the network sorts all 2^16 inputs.
    Returns True if valid, False otherwise.
    """
    states = np.arange(2**16, dtype=np.uint32)
    for i, j in network:
        bit_i = (states >> i) & 1
        bit_j = (states >> j) & 1
        swap_mask = bit_i & (~bit_j) & 1
        mask = (swap_mask == 1)
        if np.any(mask):
            states[mask] ^= ((1 << i) | (1 << j))
            
    x = (~states) & 0xFFFF
    sorted_mask = (x & (x + 1)) == 0
    return np.all(sorted_mask)

def _verify_network_subset(network, num_samples=500):
    """
    Fast approximate verification.
    """
    indices = np.random.choice(2**16, num_samples, replace=False)
    states_sample = indices.astype(np.uint32)
    
    for i, j in network:
        bit_i = (states_sample >> i) & 1
        bit_j = (states_sample >> j) & 1
        swap_mask = bit_i & (~bit_j) & 1
        mask = (swap_mask == 1)
        if np.any(mask):
            states_sample[mask] ^= ((1 << i) | (1 << j))
            
    x = (~states_sample) & 0xFFFF
    sorted_mask = (x & (x + 1)) == 0
    return np.all(sorted_mask)

def _fallback_construct():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    try:
        # 1. Greedy Construction
        network = []
        states = np.arange(2**16, dtype=np.uint32)
        
        def get_unsorted_mask():
            x = (~states) & 0xFFFF
            return (x & (x + 1)) != 0
            
        unsorted_mask = get_unsorted_mask()
        
        if not np.any(unsorted_mask):
            return []
            
        all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
        
        # Greedy Loop
        while np.any(unsorted_mask):
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_comp = None
            best_new_states = None
            best_new_unsorted = None
            
            # Sample candidates to save time
            # If we sample too few, we might get stuck. 
            # Let's try 30 candidates.
            candidates = random.sample(all_comparators, 30)
            
            for i, j in candidates:
                new_states = states.copy()
                bit_i = (new_states >> i) & 1
                bit_j = (new_states >> j) & 1
                swap_mask = bit_i & (~bit_j) & 1
                mask = (swap_mask == 1)
                if np.any(mask):
                    new_states[mask] ^= ((1 << i) | (1 << j))
                    
                x = (~new_states) & 0xFFFF
                new_sorted_mask = (x & (x + 1)) == 0
                fixed_count = np.sum(new_sorted_mask & unsorted_mask)
                
                if fixed_count > best_gain:
                    best_gain = fixed_count
                    best_comp = (i, j)
                    best_new_states = new_states
                    best_new_unsorted = ~new_sorted_mask
                    
            if best_comp is None or best_gain == 0:
                # Stuck with current sample. Try a few more samples.
                stuck = False
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    candidates = random.sample(all_comparators, 30)
                    for i, j in candidates:
                        new_states = states.copy()
                        bit_i = (new_states >> i) & 1
                        bit_j = (new_states >> j) & 1
                        swap_mask = bit_i & (~bit_j) & 1
                        mask = (swap_mask == 1)
                        if np.any(mask):
                            new_states[mask] ^= ((1 << i) | (1 << j))
                            
                        x = (~new_states) & 0xFFFF
                        new_sorted_mask = (x & (x + 1)) == 0
                        fixed_count = np.sum(new_sorted_mask & unsorted_mask)
                        
                        if fixed_count > best_gain:
                            best_gain = fixed_count
                            best_comp = (i, j)
                            best_new_states = new_states
                            best_new_unsorted = ~new_sorted_mask
                            
                if best_gain == 0:
                    break # Truly stuck
                    
            network.append(best_comp)
            states = best_new_states
            unsorted_mask = best_new_unsorted
            
        # 2. Local Search (Removal)
        # Try to remove comparators
        # Start from end
        i = len(network) - 1
        attempts = 0
        max_attempts = 100
        while i >= 0 and attempts < max_attempts and time.time() < deadline:
            attempts += 1
            candidate_network = network[:i] + network[i+1:]
            
            # Quick check
            if _verify_network_subset(candidate_network, 200):
                # Full check
                if _verify_network_full(candidate_network):
                    network = candidate_network
                    # i stays same
                else:
                    i -= 1
            else:
                i -= 1
                
        # 3. Final Verification
        if _verify_network_full(network):
            return network
        else:
            # If greedy failed to produce a valid network, return fallback
            return _fallback_construct()
            
    except Exception:
        return _fallback_construct()

# The code above defines construct() which calls _verify_network_full and _fallback_construct.
# It uses N=16.
# It returns a list of tuples.

# One detail: The task says "Keep the function names and signatures of the seed program."
# Seed program has `def construct():`
# My program has `def construct():`
# It also has `N = 16` at the top level.

# I should ensure that `random` and `numpy` are imported.
# I used `import numpy as np` and `import random` and `import time`.

# Let's double check the logic for "sorted".
# Wire 0 is min. Wire 15 is max.
# Sorted: 0, 0, ..., 0, 1, 1, ..., 1.
# Bit 0 is LSB.
# State 0b1100 (12) -> Wire 0=0, 1=0, 2=1, 3=1.
# Is 0,0,1,1 sorted? Yes.
# My check: x = ~12 & 15 = 0011 (3).
# x & (x+1) = 3 & 4 = 0. Correct.
# State 0b1010 (10) -> Wire 0=0, 1=1, 2=0, 3=1.
# Is 0,1,0,1 sorted? No.
# My check: x = ~10 & 15 = 0101 (5).
# x & (x+1) = 5 & 6 = 4 != 0. Correct.

# The logic holds.

# The greedy approach might not find the optimal 60-comparator network, 
# but it should find a valid one with fewer than 120 comparators.
# Odd-even transposition is 120.
# A random greedy build might find ~100-110.
# Local search might reduce it further.
# This is a different strategy from "static template".

# One potential issue: `states` is large (65536 * 4 bytes = 256KB).
# Copying it 30 times per step is 7.5MB.
# 100 steps = 750MB memory traffic.
# This is very fast on modern hardware (< 1s).
# The bottleneck is the Python loop overhead for the 30 candidates.
# 30 * 100 = 3000 iterations.
# Each iteration does ~5 numpy ops on 65k array.
# This should be well within 85s.

# I will output the code now.
