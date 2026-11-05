import numpy as np
import time
import random

N = 16
MAX_INPUTS = 2 ** N
DEADLINE = 85.0  # Safety margin for 90s limit

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    
    # 1. Setup Vectorized Verification Infrastructure
    # We represent the network state by a permutation of the 2^N bitmasks.
    # `state[k]` is the current bitmask of the k-th input after applying the network.
    # Initially, state[k] = k.
    # Goal: state[k] must equal target[k] for all k.
    # target[k] is the sorted version of k (all 1s in lowest positions).
    
    inputs = np.arange(MAX_INPUTS, dtype=np.uint32)
    
    # Precompute popcount for 0..65535
    # Using a lookup table approach for speed, or vectorized bit ops.
    # NumPy doesn't have popcount for uint32 in all versions, but we can use
    # a clever trick or just precompute once.
    # Since N=16, we can use a precomputed array of size 65536.
    popcounts = np.zeros(MAX_INPUTS, dtype=np.uint8)
    for i in range(1, MAX_INPUTS):
        popcounts[i] = popcounts[i >> 1] + (i & 1)
        
    # Target sorted mask for each input k
    # If k has p ones, the sorted mask is (1 << p) - 1
    targets = np.zeros(MAX_INPUTS, dtype=np.uint32)
    for i in range(MAX_INPUTS):
        targets[i] = (1 << popcounts[i]) - 1
        
    # Helper to apply a comparator (i, j) to a state array
    # Returns the new state array.
    # Comparator: min to i, max to j.
    # Bit i becomes min(bit_i, bit_j), Bit j becomes max(bit_i, bit_j).
    def apply_comp_vectorized(state, i, j):
        # Extract bits
        bit_i = (state >> i) & 1
        bit_j = (state >> j) & 1
        
        # Calculate new bits
        new_i = bit_i & bit_j  # min
        new_j = bit_i | bit_j  # max
        
        # Clear old bits and set new ones
        # Mask to clear bits i and j
        # Note: ~ on numpy uint32 works as expected for 32-bit width
        clear_mask = np.uint32(~((1 << i) | (1 << j)))
        
        new_state = (state & clear_mask) | ((new_i << i) | (new_j << j))
        return new_state.astype(np.uint32)
    
    # Helper to calculate "energy" (number of mismatches)
    def get_energy(state):
        # Count how many state[k] != targets[k]
        return np.sum(state != targets)
    
    # Helper to verify if a network is valid (energy == 0)
    def is_valid(state):
        return np.all(state == targets)
    
    # 2. Generate Initial Valid Network: Bitonic Sort
    # Bitonic sort on N=16 uses N/2 * (N/2 + 1) / 2 * 2 ? 
    # Standard bitonic sort depth is log^2(N). 
    # Number of comparators for Bitonic Sort on N=16:
    # Phases: 2, 4, 8, 16
    # Phase size 2: 8 comparators (1 layer)
    # Phase size 4: 8 comparators (2 layers)
    # Phase size 8: 8 comparators (3 layers)
    # Phase size 16: 8 comparators (4 layers)
    # Total layers: 1+2+3+4 = 10 layers.
    # Comparators per layer: 8.
    # Total comparators: 80.
    
    def generate_bitonic_network():
        network = []
        def bitonic_sort(k, d, dir):
            # k: size of the sequence being sorted (must be power of 2)
            # d: step size (initially k/2)
            # dir: 1 for ascending, 0 for descending
            if k <= 1:
                return
            bitonic_sort(k // 2, d, 1) # Sort first half ascending
            bitonic_sort(k // 2, d, 0) # Sort second half descending
            # Merge
            for i in range(0, k, 2):
                # Compare i and i+d
                # If dir is 1 (asc), min goes to i, max to i+d
                # If dir is 0 (desc), max goes to i, min to i+d
                # Wait, standard bitonic merge:
                # We compare elements at distance d apart.
                # If dir=1 (ascending merge):
                #   if a[i] > a[i+d], swap
                #   So min goes to i, max to i+d.
                # If dir=0 (descending merge):
                #   if a[i] < a[i+d], swap
                #   So max goes to i, min to i+d.
                
                # In our comparator model:
                # compare(i, j) always puts min in i, max in j.
                # So for ascending (dir=1): compare(i, i+d) is correct.
                # For descending (dir=0): we want max in i, min in i+d.
                # This is equivalent to compare(i+d, i) but our comparator requires i < j.
                # So for descending, we compare(i, i+d) but we interpret the result differently?
                # No, the hardware comparator is fixed: min to lower index, max to higher index.
                # To achieve descending order in the merge step of bitonic sort,
                # we need to reverse the comparison direction?
                # Actually, standard bitonic sort implementations handle this by
                # comparing pairs and swapping if they are in the wrong order for the desired direction.
                # Since our comparator is "min to low, max to high", it inherently sorts ascending.
                # To sort descending, we would need "max to low, min to high".
                # We can simulate "max to low, min to high" by swapping the indices in the pair?
                # If we call compare(j, i) with j > i, it's invalid.
                # We must call compare(i, j) with i < j.
                # Result: wire i gets min, wire j gets max.
                # If we want wire i to get max and wire j to get min, we can't do it directly with a single comparator.
                # HOWEVER, bitonic sort can be implemented using only "ascending" comparators if we structure the network correctly?
                # No, bitonic sort requires both ascending and descending comparators.
                # Let's check the definition of the task comparator.
                # "Each compare-exchange writes min to wire i and max to j."
                # This is a fixed direction.
                # Does this restrict us to only ascending sorts? No, because we can permute wires? No, wires are fixed.
                # Wait, if I have values [2, 1] on wires [0, 1].
                # compare(0, 1) -> min(2,1)=1 to wire 0, max(2,1)=2 to wire 1. Result [1, 2]. Ascending.
                # If I want [2, 1] (Descending), I cannot achieve it with compare(0, 1) alone.
                # I would need compare(1, 0) which is not allowed (i<j).
                # So, can we sort descending with only min-to-low comparators?
                # Yes, if we reverse the input? No.
                # Actually, the Zero-One principle allows us to verify.
                # If a network sorts all inputs ascending, it's valid.
                # Bitonic sort *as typically described* uses both.
                # But there is a version of Bitonic Sort that uses only "min" comparators?
                # No, the merge step fundamentally requires direction.
                
                # Let's re-read the comparator definition carefully.
                # "min to wire i and max to j".
                # This is an "ascending" comparator.
                # Can we build a sorting network with only ascending comparators?
                # Yes, ANY sorting network can be built with only ascending comparators?
                # No. If you only have ascending comparators, you can only move 1s to the right (higher indices) and 0s to the left?
                # Wait.
                # Input: 1 0 (Wire 0=1, Wire 1=0).
                # Compare(0,1): min(1,0)=0 to Wire 0, max(1,0)=1 to Wire 1.
                # Result: 0 1.
                # So 1s move to higher indices, 0s to lower indices.
                # This effectively sorts in ASCENDING order (0s first, then 1s).
                # So, a network of ONLY ascending comparators can ONLY produce an output where 0s are on the left and 1s are on the right?
                # No, it can produce any permutation?
                # Let's trace:
                # Start: 1 0 1 0
                # Comp(0,1) -> 0 1 1 0
                # Comp(2,3) -> 0 1 0 1
                # Comp(1,2) -> 0 0 1 1
                # It seems they always tend to push 1s to the right.
                # Is it possible to get 1 0 0 1?
                # Start 0 0 1 1.
                # Comp(1,2) -> 0 0 1 1 (no change)
                # Comp(0,1) -> 0 0 1 1
                # It seems if 1s are on the right, they stay there?
                # No, if I have 1 0 0 0.
                # Comp(0,1) -> 0 1 0 0.
                # Comp(1,2) -> 0 0 1 0.
                # Comp(2,3) -> 0 0 0 1.
                # So 1s move right.
                # Can a 1 move left?
                # Only if it meets a 0 on its left?
                # Compare(i, j) with i<j.
                # If Wire i=0, Wire j=1.
                # min(0,1)=0 to i, max(0,1)=1 to j.
                # No change.
                # If Wire i=1, Wire j=0.
                # min(1,0)=0 to i, max(1,0)=1 to j.
                # 1 moves from i to j (Right).
                # So 1s ALWAYS move Right or stay. 0s ALWAYS move Left or stay.
                # This implies that with ONLY ascending comparators, you can ONLY sort in Ascending order (0s then 1s).
                # You cannot create a pattern like 1 0 1 0 from 0 1 0 1?
                # Start 0 1 0 1.
                # Comp(0,1) -> 0 1 0 1.
                # Comp(1,2) -> 0 0 1 1.
                # Once sorted, it stays sorted?
                # If input is 1 1 0 0.
                # Comp(1,2) -> 1 0 1 0.
                # Comp(0,1) -> 0 1 1 0.
                # Comp(2,3) -> 0 1 0 1.
                # Comp(1,2) -> 0 0 1 1.
                # It seems we can always reach the sorted state (0...01...1).
                # So, a network of ONLY ascending comparators is sufficient to sort any input into Ascending order.
                # Therefore, we can use Bitonic Sort logic, but we must ensure that every "descending" requirement is handled by the fact that we are sorting ascending?
                # Wait, Bitonic Sort's power comes from sorting sub-arrays in opposite directions to facilitate merging.
                # If we can ONLY sort ascending, can we still use Bitonic Sort?
                # Standard Bitonic Sort:
                # 1. Split.
                # 2. Sort first half Ascending, second half Descending.
                # 3. Merge.
                # If we can't sort Descending, we can't do Step 2.
                # However, there is a "Bitonic" variant that works with only one type of comparator?
                # Actually, the "Odd-Even Transposition" network uses only adjacent comparators.
                # Adjacent comparators (i, i+1) are ascending.
                # Odd-Even Transposition sorts in Ascending order.
                # It works.
                # So, we don't need "descending" comparators. We just need a network that sorts ascending.
                # Bitonic Sort *as a sequence of comparators* usually includes "reverse" comparators.
                # But we can implement a "Bitonic-like" structure using only ascending comparators?
                # Or just use the known valid Odd-Even Transposition (120 comps) or a pruned version.
                
                # Let's stick to generating a valid network first.
                # Odd-Even Transposition is easy to generate and valid.
                # It has 120 comparators.
                # Bitonic Sort (if implemented correctly with only ascending comps? No, standard bitonic requires both).
                # Wait, if I use a "Bitonic" generator that outputs comparators (i, j) where i<j, and I just apply them,
                # does it sort?
                # If the generator produces a sequence that sorts for the "ascending" comparator, yes.
                # But standard bitonic sort generators produce a mix.
                # Let's just use Odd-Even Transposition as the seed. It's robust.
                # 120 comparators.
                
                # We can also try to generate a "Shellsort" or "Batcher's" network?
                # Batcher's Bitonic Sort on 16 inputs:
                # It is known that Batcher's network uses 80 comparators?
                # Let's check: N=16.
                # Phase 1 (size 2): 8 comps.
                # Phase 2 (size 4): 8 comps (2 layers).
                # Phase 3 (size 8): 8 comps (3 layers).
                # Phase 4 (size 16): 8 comps (4 layers).
                # Total 80.
                # Does Batcher's network use only ascending comparators?
                # No, it uses both.
                # BUT, if I just list the pairs (i, j) with i < j, and the logic is "min to i, max to j",
                # then a "descending" step in bitonic sort would be implemented by comparing (i, j) but expecting the result to be descending?
                # No, the hardware is fixed.
                # So I CANNOT use a standard Bitonic Sort network if it requires descending comparators.
                # I MUST use a network that sorts ascending using only ascending comparators.
                # Odd-Even Transposition does this.
                # Are there smaller ones?
                # The "AKS" network? Too complex.
                # Let's start with Odd-Even Transposition (120) and prune.
                
                pass # Placeholder

    # Generate Odd-Even Transposition Network
    def generate_oe_network():
        net = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    net.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    net.append((i, i + 1))
        return net

    network = generate_oe_network()
    
    # Apply network to get initial state
    current_state = inputs.copy()
    for i, j in network:
        current_state = apply_comp_vectorized(current_state, i, j)
        
    # Verify
    if not is_valid(current_state):
        # Fallback: return OE network (should be valid)
        return network

    current_energy = get_energy(current_state)
    best_network = network.copy()
    best_energy = current_energy
    
    # 3. Local Search with Simulated Annealing
    # Moves:
    # 1. Remove a comparator.
    # 2. Insert a comparator at a random position.
    # 3. Swap two comparators.
    # 4. Change a comparator to a different pair.
    
    T = 50.0 # Initial temperature
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute all possible comparator pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < DEADLINE:
        # Generate a move
        move_type = random.choice(['remove', 'insert', 'swap', 'change'])
        
        if len(network) == 0:
            move_type = 'insert'
            
        if move_type == 'remove':
            idx = random.randint(0, len(network) - 1)
            removed = network.pop(idx)
            # Calculate new energy
            # We need to re-simulate? 
            # Optimization: We can't easily update state incrementally for removal 
            # unless we store intermediate states or re-simulate.
            # Re-simulation of 120 comparators on 65536 items is fast (~1ms).
            # 1000 moves/sec is feasible.
            new_state = inputs.copy()
            for i, j in network:
                new_state = apply_comp_vectorized(new_state, i, j)
            new_energy = get_energy(new_state)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_network = network.copy()
                current_state = new_state
            else:
                # Accept or reject based on SA
                delta = new_energy - current_energy
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        current_state = new_state
                        current_energy = new_energy
                    else:
                        network.insert(idx, removed)
                else:
                    current_state = new_state
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_network = network.copy()
                        
        elif move_type == 'insert':
            idx = random.randint(0, len(network))
            new_comp = random.choice(all_pairs)
            network.insert(idx, new_comp)
            
            new_state = inputs.copy()
            for i, j in network:
                new_state = apply_comp_vectorized(new_state, i, j)
            new_energy = get_energy(new_state)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_network = network.copy()
                current_state = new_state
            else:
                delta = new_energy - current_energy
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        current_state = new_state
                        current_energy = new_energy
                    else:
                        network.pop(idx)
                else:
                    current_state = new_state
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_network = network.copy()

        elif move_type == 'swap':
            if len(network) < 2:
                continue
            idx1, idx2 = random.sample(range(len(network)), 2)
            network[idx1], network[idx2] = network[idx2], network[idx1]
            
            new_state = inputs.copy()
            for i, j in network:
                new_state = apply_comp_vectorized(new_state, i, j)
            new_energy = get_energy(new_state)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_network = network.copy()
                current_state = new_state
            else:
                delta = new_energy - current_energy
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        current_state = new_state
                        current_energy = new_energy
                    else:
                        network[idx1], network[idx2] = network[idx2], network[idx1]
                else:
                    current_state = new_state
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_network = network.copy()

        elif move_type == 'change':
            if len(network) == 0:
                continue
            idx = random.randint(0, len(network) - 1)
            old_comp = network[idx]
            new_comp = random.choice(all_pairs)
            if new_comp == old_comp:
                continue
            network[idx] = new_comp
            
            new_state = inputs.copy()
            for i, j in network:
                new_state = apply_comp_vectorized(new_state, i, j)
            new_energy = get_energy(new_state)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_network = network.copy()
                current_state = new_state
            else:
                delta = new_energy - current_energy
                if delta > 0:
                    if random.random() < np.exp(-delta / T):
                        current_state = new_state
                        current_energy = new_energy
                    else:
                        network[idx] = old_comp
                else:
                    current_state = new_state
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_network = network.copy()

        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final Verification
    final_state = inputs.copy()
    for i, j in best_network:
        final_state = apply_comp_vectorized(final_state, i, j)
        
    if not is_valid(final_state):
        # If the best found is invalid (shouldn't happen if we track valid states, 
        # but SA might have accepted a bad state if we messed up the logic), 
        # fallback to the initial valid one.
        return network # The initial OE network

    return best_network

# EVOLVE-BLOCK-END
