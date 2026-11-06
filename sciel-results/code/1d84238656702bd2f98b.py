"""Size-efficient sorting networks on 18 wires.
Strategy: Use a fast bitwise verifier combined with Simulated Annealing
to escape local minima of greedy removal, aiming for significantly lower
comparator counts than the seed odd-even transposition network.
"""
import time
import random

N = 18

# EVOLVE-BLOCK-START

def _verify_network_fast(network):
    """
    Verify that the network sorts all 2^N Boolean inputs using the zero-one principle.
    Uses bitwise operations for speed.
    """
    # Pre-compute masks for each comparator to speed up simulation
    # We simulate the network on all 2^18 inputs at once using bit-parallelism?
    # No, 2^18 is 262144. We can process them in batches or just loop.
    # A faster way is to represent the state of all 2^18 inputs as a set of bitsets?
    # Actually, since N=18, we can represent the value of each wire for all inputs
    # as a Python integer where bit k represents the value of that wire for input mask k.
    # There are 2^18 inputs. So each wire's state is an integer of 2^18 bits.
    # That's 2^18 bits = 32768 bytes = 32 KB per wire.
    # 18 wires * 32 KB = ~576 KB. This is very fast in Python integers.
    
    # Initialize wire states for all 2^N inputs
    # Input mask m has bit i set if the i-th input is 1.
    # For wire k, the value is 1 if bit k of m is 1.
    # We need to construct an integer W_k such that bit m of W_k is ((m >> k) & 1).
    
    # Let's construct W_k for k in 0..N-1
    # For a given k, the pattern of bits in W_k is a repeating block of 0s and 1s.
    # Period is 2^(k+1). First 2^k are 0, next 2^k are 1.
    
    total_inputs = 1 << N
    # We will represent the state of all wires as a list of N integers.
    # state[k] is an integer where bit m is the value of wire k for input m.
    
    state = [0] * N
    
    for k in range(N):
        # Construct the integer for wire k
        # Pattern: 0...0 1...1 repeated
        block_size = 1 << k
        # Create a mask for one block: 0...0 1...1 of length 2*block_size
        # Actually, it's easier to build it up or use a formula.
        # Let's just loop over blocks? No, that's slow.
        # We can use a recursive construction or bit manipulation.
        
        # For k=0: 010101...
        # For k=1: 00110011...
        # For k=2: 00001111...
        
        # Let's build it by iterating through the total_inputs/2^(k+1) blocks
        # But total_inputs is 2^18, so for k=0, we have 2^17 blocks.
        # This might be slow to construct in pure Python if we do it naively.
        
        # Alternative:
        # W_k = sum( (1 << m) for m in range(total_inputs) if (m >> k) & 1 )
        # This is O(2^N) per wire, so O(N * 2^N) total construction time.
        # 18 * 262144 ~ 5 million operations. This is acceptable in Python (<1s).
        
        w = 0
        for m in range(total_inputs):
            if (m >> k) & 1:
                w |= (1 << m)
        state[k] = w
    
    # Apply network
    for i, j in network:
        # Compare wire i and wire j
        # If state[i] has a 1 where state[j] has a 0, we need to swap.
        # swap_mask = state[i] & ~state[j]
        # new_state[i] = state[i] & ~swap_mask | state[j] & swap_mask? No.
        # min goes to i, max to j.
        # If i=1, j=0 -> i becomes 0, j becomes 1.
        # If i=0, j=1 -> no change.
        # If i=1, j=1 -> no change.
        # If i=0, j=0 -> no change.
        
        # So, bits where i=1 and j=0 need to be cleared in i and set in j.
        # bits where i=0 and j=1 are already correct (min=0, max=1).
        
        swap_mask = state[i] & ~state[j]
        
        # Clear these bits in i
        state[i] &= ~swap_mask
        # Set these bits in j
        state[j] |= swap_mask
        
        # Note: state[i] & state[j] remains 1 (both 1), ~state[i] & ~state[j] remains 0 (both 0).
        # This correctly implements compare-exchange for all inputs in parallel.
        
    # Check if sorted
    # For the network to be a sorting network, for every input m, the output must be sorted.
    # i.e., for all k, if output[k]=1 then output[k+1]=1.
    # This is equivalent to: for all k, output[k] & ~output[k+1] == 0.
    
    for k in range(N - 1):
        # Check if there is any input where wire k is 1 and wire k+1 is 0
        if state[k] & ~state[k+1]:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # Start with odd-even transposition network
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial network
    if not _verify_network_fast(network):
        # Should not happen, but just in case
        return network
    
    current_count = len(network)
    best_network = network[:]
    best_count = current_count
    
    # Simulated Annealing
    # Moves:
    # 1. Remove a random comparator
    # 2. Swap two comparators
    # 3. Insert a random comparator
    # 4. Change a comparator to a different pair (less likely, keep simple)
    
    T = 1.0
    cooling = 0.999
    
    # Pre-calculate all possible comparators for random insertion
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    while time.time() < deadline:
        # Generate a neighbor
        move_type = random.random()
        candidate = network[:]
        
        if move_type < 0.4:
            # Remove a random comparator
            if len(network) > 0:
                idx = random.randrange(len(network))
                candidate = network[:idx] + network[idx+1:]
        elif move_type < 0.7:
            # Swap two comparators
            if len(network) > 1:
                i, j = random.sample(range(len(network)), 2)
                candidate[i], candidate[j] = candidate[j], candidate[i]
        elif move_type < 0.9:
            # Insert a random comparator at a random position
            new_comp = random.choice(all_comparators)
            idx = random.randrange(len(candidate) + 1)
            candidate = candidate[:idx] + [new_comp] + candidate[idx:]
        else:
            # Modify a random comparator
            if len(network) > 0:
                idx = random.randrange(len(network))
                new_comp = random.choice(all_comparators)
                candidate[idx] = new_comp
        
        # Check validity
        if _verify_network_fast(candidate):
            new_count = len(candidate)
            delta = new_count - current_count
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                # Accept
                network = candidate
                current_count = new_count
                
                if current_count < best_count:
                    best_count = current_count
                    best_network = network[:]
        
        T *= cooling
        
        # If T is very small, we might be stuck.
        # Reheat occasionally?
        if T < 0.01:
            T = 0.5
            # Also, if we haven't improved in a while, we could try a more aggressive restart
            # but for now, just reheat.
            
    # Final greedy removal pass on the best network found
    # This often squeezes out the last few comparators
    final_network = best_network[:]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(final_network) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            candidate = final_network[:idx] + final_network[idx+1:]
            if _verify_network_fast(candidate):
                final_network = candidate
                improved = True
                break
    
    # Verify final network one last time
    if _verify_network_fast(final_network):
        return final_network
    
    return best_network

# EVOLVE-BLOCK-END
