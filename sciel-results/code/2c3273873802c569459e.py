import time
import random
import sys

N = 16
NUM_INPUTS = 1 << N  # 65536

# Global cache for precomputed masks
_INITIAL_MASKS = None

def _get_masks():
    """
    Precompute bit-parallel masks for the 16 wires.
    mask[i] is a 65536-bit integer where bit k is 1 if wire i is 1 in input k.
    Input k is a 16-bit integer, so bit i of k determines wire i's value.
    """
    global _INITIAL_MASKS
    if _INITIAL_MASKS is None:
        _INITIAL_MASKS = [0] * N
        for i in range(N):
            m = 0
            # Optimization: Construct mask using bit patterns
            # For wire i, the bit is 1 for k where (k >> i) & 1 == 1
            # This is a repeating pattern of 0s and 1s.
            # Period is 2^(i+1), half period is 2^i.
            # Pattern: 2^i zeros, then 2^i ones.
            # We can build this efficiently.
            
            # Alternative simple loop (16 * 65536 = 1M ops, ~0.1s in Python, acceptable)
            # But let's try to be faster if possible.
            
            # Let's stick to the simple loop for reliability first, it's fast enough.
            for k in range(NUM_INPUTS):
                if (k >> i) & 1:
                    m |= 1 << k
            _INITIAL_MASKS[i] = m
    return _INITIAL_MASKS

def fast_verify(network):
    """
    Verify if the network sorts all inputs using the zero-one principle.
    Returns True if valid, False otherwise.
    """
    masks = _get_masks()
    state = masks[:]
    
    for i, j in network:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    # Check if sorted: state[i] <= state[i+1] for all i
    # In bit-parallel representation, this means state[i] & ~state[i+1] == 0
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
    """
    Generate Batcher's bitonic sorting network for n wires.
    NOTE: Standard bitonic sort requires comparators that can sort in either direction.
    Our comparators are fixed: min to lower index, max to higher index.
    
    If we only have "increasing" comparators (min to i, max to j where i<j),
    we cannot directly implement the "decreasing" sort steps of standard bitonic sort 
    on the same wire indices without permutation.
    
    However, the Odd-Even Transposition network (120 comparators) is a valid baseline 
    that uses only increasing comparators.
    
    Let's implement a correct baseline that works with fixed-direction comparators.
    Actually, we can implement Bitonic Sort by permuting the wires?
    No, the network is a fixed list of pairs.
    
    Let's look at the inspiration: "Start from the known 64-comparator Batcher bitonic sorter".
    Does Batcher's bitonic sorter for N=16 use 64 comparators?
    Formula: (N/2) * log2(N) * (log2(N)+1) / 2 ?
    For N=16: 8 * 4 * 5 / 2 = 80?
    Or is it (N/2) * (log2(N) * (log2(N)+1)/2) ?
    8 * (4*5/2) = 80.
    
    Wait, the inspiration says 64.
    Let's check the number of comparators in Batcher's bitonic sort.
    Depth is (log N)^2 / 2.
    Number of comparators is (N/2) * (log N * (log N + 1) / 2).
    For N=16, log N = 4.
    Comparators = 8 * (4 * 5 / 2) = 8 * 10 = 80.
    
    Why does the inspiration say 64?
    Maybe it's a specific optimized version or I'm misremembering the formula.
    Or maybe the inspiration is slightly inaccurate about the count but correct about the method.
    
    Regardless, if I can't easily construct a valid bitonic sorter with fixed-direction comparators 
    (because it needs decreasing sorts), I should fall back to a known valid network 
    that uses only increasing comparators, like Odd-Even Transposition, 
    OR I can try to construct a "0-1" sorting network.
    
    Actually, there is a way to do bitonic sort with fixed comparators if we permute the wires 
    such that the "decreasing" part is mapped to "increasing" on different wires?
    No, that's complex.
    
    Let's use the Odd-Even Transposition network as the guaranteed valid starting point (120 comparators).
    It's known to be correct for fixed-direction comparators.
    
    def odd_even_network():
        net = []
        for phase in range(N):
            if phase % 2 == 0:
                for i in range(0, N - 1, 2):
                    net.append((i, i + 1))
            else:
                for i in range(1, N - 1, 2):
                    net.append((i, i + 1))
        return net
    
    Let's use this.
    """
    # Since constructing a valid bitonic sorter with fixed-direction comparators is non-trivial 
    # and the inspiration's 64-comparator claim might rely on bidirectional comparators 
    # or a specific permutation trick not detailed, we will use the robust Odd-Even Transposition 
    # network (120 comparators) as our baseline. It is guaranteed to work with (i, j) where i<j.
    
    net = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                net.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                net.append((i, i + 1))
    return net

def greedy_reduce(network, deadline):
    """
    Repeatedly try to remove each comparator. If the network remains valid, keep it removed.
    This is a local search step.
    """
    changed = True
    # Iterate until no changes or deadline
    while changed and time.time() < deadline:
        changed = False
        # Iterate backwards to handle index shifts easily, or just restart
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            candidate = network[:idx] + network[idx+1:]
            if fast_verify(candidate):
                network = candidate
                changed = True
                break # Restart the pass since the network changed
    return network

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s. We should aim to finish by 85s.
    deadline = start_time + 85.0
    
    # 1. Generate initial network
    # Using Odd-Even Transposition (120 comparators)
    net = bitonic_sorter(N) # This function name is kept, but content is Odd-Even for safety
    
    # Verify initial network just in case
    if not fast_verify(net):
        # This shouldn't happen, but if it does, we return empty or a known good one?
        # Odd-Even is always good.
        pass
        
    best_net = net[:]
    best_count = len(net)
    
    # 2. Greedy Reduction
    # This is the main driver for improvement.
    net = greedy_reduce(net, deadline)
    if len(net) < best_count:
        best_net = net[:]
        best_count = len(net)
    
    # 3. Local Search with Insertions (Escape from local minima)
    # If greedy reduction stalls, try adding a random comparator, then reducing again.
    # This explores the neighborhood of the current local minimum.
    
    random.seed(42)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # We can try multiple iterations of:
    # a) Add random comparator
    # b) Greedy reduce
    # c) If better, keep
    
    current_net = best_net[:]
    
    # Time check
    if time.time() > deadline - 10:
        return best_net
        
    # Number of attempts limited by time
    attempts = 0
    max_attempts = 5000 # Cap to prevent infinite loop if verify is fast
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        # Pick a random comparator to add
        # To avoid duplicates, we could check, but for simplicity, just add.
        # Greedy reduce will remove it if it's not useful.
        rand_pair = random.choice(all_pairs)
        
        # Insert at a random position? Or just append?
        # Appending is simpler. The position might matter for efficiency of reduction?
        # Greedy reduction checks all positions, so position doesn't matter for correctness,
        # only for how fast it finds the removal.
        candidate_net = current_net[:] + [rand_pair]
        
        # Reduce the candidate
        reduced_net = greedy_reduce(candidate_net, deadline)
        
        if len(reduced_net) < len(current_net):
            current_net = reduced_net
            if len(current_net) < best_count:
                best_net = current_net[:]
                best_count = len(best_net)
                
        # If we didn't improve, we might be stuck.
        # We can also try to remove a random comparator from current_net if valid?
        # Greedy reduce already does that.
        
        # Optional: Try to swap two comparators? Probably not worth the complexity.
        
    # Final check: one last greedy pass
    final_net = greedy_reduce(best_net, deadline)
    if len(final_net) < best_count:
        best_net = final_net
        best_count = len(best_net)
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(len(net))
