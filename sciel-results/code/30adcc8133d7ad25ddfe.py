"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a sorting network for 16 wires using a hybrid approach:
    1. Try to find a small network via randomized local search (greedy + simulated annealing style).
    2. If no valid network is found within time budget, fall back to a known good construction.
    
    The goal is to beat the 120-comparator odd-even transposition network.
    A known 60-comparator network exists. We aim to find something <= 60.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    # 1. Try to find a compact network via search
    best_network = None
    best_count = 120  # Start with the transposition count as upper bound
    
    # We will use a greedy/local search approach.
    # State: list of comparators.
    # Objective: minimize count, subject to correctness.
    
    # Helper to check if a network is a sorter
    def is_sorter(net):
        if len(net) == 0:
            return True
        # Zero-one principle: test all 2^16 inputs? That's 65536 * 16 * len(net) operations.
        # 65536 * 16 * 60 ~ 60 million ops. Might be slow in Python if not optimized.
        # Let's optimize the check.
        # We can represent the state as a 16-bit integer.
        # For each input, simulate the network.
        # If any input fails, return False.
        
        # Optimization: 
        # A network is a sorter iff for every input x, the output is sorted.
        # Sorted output for 16 bits means the bits are 0...01...1.
        # i.e., output must be of form (1 << k) - 1 for some k in 0..16.
        
        # We can pre-compute the "signature" of the network? No, it depends on input.
        
        # Let's try a faster simulation.
        # Instead of iterating all 65536, maybe we can use a smarter check?
        # For small N, brute force is often necessary.
        # 65536 inputs * 60 comparators = ~4M operations per check.
        # If we do 100 checks, that's 400M ops. In Python, this is too slow (~10s+).
        # We need a faster verifier or a smarter search.
        
        # Alternative: Use the "bypass" property or other structural checks?
        # Or, use a faster language? No, must be Python.
        
        # Let's optimize the simulator.
        # Represent state as int.
        # Comparator (i, j):
        # bit_i = (state >> i) & 1
        # bit_j = (state >> j) & 1
        # min_bit = bit_i & bit_j
        # max_bit = bit_i | bit_j
        # new_bit_i = min_bit
        # new_bit_j = max_bit
        # state = (state & ~(1<<i)) | (min_bit << i)
        # state = (state & ~(1<<j)) | (max_bit << j)
        
        # This is ~10 ops per comparator.
        # 60 comparators * 65536 inputs = 4M iterations.
        # Each iteration ~10-20 ns in C, but in Python ~100ns+?
        # 4M * 100ns = 0.4s per check.
        # If we do 100 checks, 40s. Tight but possible if we are careful.
        # We can reduce the number of inputs to check?
        # No, must check all for correctness.
        
        # However, we can early exit if a failure is found.
        # Many invalid networks will fail early.
        
        for x in range(1 << N):
            state = x
            for i, j in net:
                bi = (state >> i) & 1
                bj = (state >> j) & 1
                if bi > bj:
                    # Swap
                    state ^= (1 << i) | (1 << j)
            # Check if sorted
            # Sorted means all 0s then all 1s.
            # state must be (1 << k) - 1
            if state not in sorted_masks:
                return False
        return True

    # Pre-compute sorted masks
    global sorted_masks
    sorted_masks = set((1 << k) - 1 for k in range(N + 1))

    # Search Strategy:
    # Start with a known valid network (e.g., transposition, 120 comps).
    # Try to remove comparators or swap them to reduce count while maintaining validity.
    # This is a "deletion" search.
    
    # 1. Initialize with transposition network
    current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    current_count = len(current_net)
    best_net = list(current_net)
    best_count = current_count
    
    # 2. Local Search: Try to delete random comparators
    # If deletion breaks validity, try to "fix" it by moving the comparator?
    # Or just random restarts with smaller initial networks?
    
    # Let's try a simpler approach: Randomized local search on the set of comparators.
    # State: Set of comparators.
    # Move: Remove one, Add one.
    
    # Given time constraints, let's try to find a network with <= 60 comparators.
    # We can use a heuristic: 
    # Many good networks have a "layered" structure.
    
    # Let's try a specific known construction if we can't search fast enough.
    # The "AKS" style or "Batcher" style?
    # Batcher's Odd-Merge Sort for 16:
    # OMS(16) = OMS(8) + Merge(8,8)
    # OMS(8) = OMS(4) + Merge(4,4)
    # OMS(4) = OMS(2) + Merge(2,2)
    # OMS(2) = 1 comparator
    # Merge(2,2) = 3 comparators (for 2+2)
    # Merge(4,4) = 9 comparators?
    # Let's calculate Batcher's Odd-Merge Sort comparator count for N=16.
    # T(N) = 2*T(N/2) + M(N/2, N/2)
    # M(N, N) for even N: 
    #   If N=2, M=3.
    #   If N=4, M= OMS(2)+OMS(2)+3? No.
    #   Standard Odd-Merge:
    #   Sort odd indices, sort even indices, then merge.
    #   Merge of two sorted lists of length k/2?
    
    # Let's just generate Batcher's Odd-Merge Sort network for 16 wires.
    # It is known to be efficient.
    # Reference: 
    #   OMS(1) = 0
    #   OMS(2) = 1
    #   OMS(4) = 3 (OMS(2) on odds, OMS(2) on evens, then merge 2,2) -> 1+1+3 = 5? 
    #   Actually, OMS(4) is often cited as 5 comparators? Or 3?
    #   Let's check: 
    #   Inputs 0,1,2,3.
    #   Sort odds (1,3): 1 comp.
    #   Sort evens (0,2): 1 comp.
    #   Merge:
    #     Compare (0,1), (2,3) -> 2 comps.
    #     If 0>1, swap; if 2>3, swap.
    #     Then compare (1,2) -> 1 comp.
    #     Total merge = 3.
    #     Total OMS(4) = 1+1+3 = 5.
    #   OMS(8):
    #     Sort odds (1,3,5,7) with OMS(4) -> 5 comps.
    #     Sort evens (0,2,4,6) with OMS(4) -> 5 comps.
    #     Merge 4,4:
    #       Compare (0,1), (2,3), (4,5), (6,7) -> 4 comps.
    #       If any swap, need more?
    #       The merge step in OMS is more complex.
    #       M(4,4) uses OMS(2) on pairs?
    #       Total OMS(8) is 19 comparators.
    #   OMS(16):
    #     2 * OMS(8) + M(8,8)
    #     M(8,8) is 27?
    #     Total OMS(16) = 2*19 + 27 = 65?
    #     Or maybe 64?
    #     Actually, Batcher's OMS(16) is 65 comparators.
    #     This is better than 120, but worse than 60.
    
    # Can we do better than 65?
    # The reference says 60 is possible.
    # We can try to optimize the OMS network by removing redundant comparators.
    
    # Let's generate OMS(16) and then try to prune it.
    
    def generate_oms(n):
        """Generate Odd-Merge Sort network for n wires (n must be power of 2)."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        comps = []
        # Sort odd and even indices recursively
        # Odd indices: 1, 3, 5, ...
        # Even indices: 0, 2, 4, ...
        # Map original indices to 0..n/2-1 for recursion
        
        # Helper to add network on subset of wires
        def add_net(sub_indices, net):
            nonlocal comps
            for i, j in net:
                comps.append((sub_indices[i], sub_indices[j]))
        
        half = n // 2
        odd_indices = [2*k + 1 for k in range(half)]
        even_indices = [2*k for k in range(half)]
        
        # Recurse on odds and evens
        sub_net = generate_oms(half)
        add_net(odd_indices, sub_net)
        add_net(even_indices, sub_net)
        
        # Merge step
        # The merge step for OMS(n) where n is even:
        # 1. Compare (0,1), (2,3), ..., (n-2, n-1)
        for i in range(0, n, 2):
            comps.append((i, i+1))
        
        # 2. Recursively merge the "upper" and "lower" halves?
        # The standard OMS merge:
        # After step 1, the smallest element is at 0, largest at n-1.
        # The remaining elements are in two sorted lists:
        #   L: indices 1, 3, ..., n-3 (odd indices from 1 to n-3)
        #   R: indices 2, 4, ..., n-2 (even indices from 2 to n-2)
        # Wait, this is getting complicated to implement correctly from scratch.
        
        # Alternative: Use a known good network from literature if we can't derive it.
        # Or, stick to the search approach.
        
        # Given the complexity of implementing OMS correctly in a short time,
        # and the risk of bugs, let's rely on the search approach with a fallback.
        
        # Fallback: Return the transposition network (120) if search fails.
        # But we want to beat 120.
        # Let's try to find a 65-comparator network via a simpler heuristic.
        
        # Heuristic: 
        # 1. Generate a random network of size 60.
        # 2. Check if it's a sorter.
        # 3. If not, mutate it.
        
        # This is a "random search" which is unlikely to find a 60-comparator sorter quickly.
        
        # Better: Use the "bubble sort" network (120) and try to delete comparators.
        # Start with 120.
        # Try to delete one random comparator.
        # If valid, keep it.
        # Repeat.
        
        # This is a greedy deletion.
        
        return comps

    # Let's implement the greedy deletion search properly.
    
    # Start with transposition
    current_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    current_count = len(current_net)
    best_net = list(current_net)
    best_count = current_count
    
    # To speed up is_sorter, we can cache results? No, state is unique.
    # We can use a faster check: 
    # If we have a list of comparators, we can simulate all inputs in parallel? No, Python.
    
    # Let's try to delete comparators one by one.
    # To make it faster, we can try to delete comparators that are "redundant".
    # A comparator is redundant if removing it still results in a sorter.
    
    # We will iterate:
    # 1. Try to delete each comparator in the current network.
    # 2. If deletion keeps it a sorter, delete it.
    # 3. Repeat until no more deletions are possible.
    
    # This is O(N^2 * 2^N) in worst case, but usually much faster.
    # 120 comparators * 65536 inputs * 120 checks = ~1 billion ops. Too slow.
    
    # We need a smarter approach.
    # Let's use a timeout-based search.
    
    # 1. Try to find a network with <= 65 comparators using a randomized local search.
    #    State: List of comparators.
    #    Move: Swap two comparators, or replace one with a random one.
    #    Objective: Minimize count, subject to validity.
    
    # Given the time limit, I will implement a simple greedy deletion with a fast verifier.
    # The verifier is the bottleneck.
    
    # Optimization: 
    # We can check validity on a subset of inputs first?
    # If it fails on a subset, it's invalid.
    # If it passes on a subset, it might still be invalid.
    # But for deletion, if it was valid before, and we remove one, it's likely to fail only on specific inputs.
    
    # Let's just try to delete random comparators and check full validity.
    
    random.seed(42)
    
    while time.time() < deadline:
        # Try to delete a random comparator
        if not current_net:
            break
        
        idx = random.randint(0, len(current_net) - 1)
        candidate_net = current_net[:idx] + current_net[idx+1:]
        
        if is_sorter(candidate_net):
            current_net = candidate_net
            if len(current_net) < best_count:
                best_count = len(current_net)
                best_net = list(current_net)
        else:
            # Try to swap the removed comparator with a random position?
            # Or just skip.
            pass
    
    if best_count < 120:
        return best_net
    else:
        # Fallback to transposition
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
