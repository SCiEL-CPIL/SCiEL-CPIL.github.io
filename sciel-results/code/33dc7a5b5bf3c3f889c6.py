import sys
import random

N = 18

def construct():
    """
    Constructs a sorting network for N=18 wires using a local search 
    heuristic starting from Batcher's Merge Sort.
    """
    
    def batcher_merge_sort(n):
        """Returns a list of comparators for sorting n wires using Batcher's Merge Sort."""
        if n <= 1:
            return []
        
        k = n // 2
        left = batcher_merge_sort(k)
        right = [(i + k, j + k) for (i, j) in batcher_merge_sort(n - k)]
        merge = batcher_odd_merge(k, n - k)
        
        return left + right + merge
    
    def batcher_odd_merge(n, m):
        """
        Returns comparators to merge two sorted lists of sizes n and m.
        The first list occupies indices 0..n-1, the second occupies indices n..n+m-1.
        Uses Batcher's Odd-Merge algorithm.
        """
        if n == 0 or m == 0:
            return []
        
        if n == 1 and m == 1:
            return [(0, 1)]
        
        n_odd = n // 2
        n_even = n - n_odd
        m_odd = m // 2
        m_even = m - m_odd
        
        merge_odd = batcher_odd_merge(n_odd, m_even)
        
        def map_odd(pos):
            if pos < n_odd:
                return 2 * pos + 1
            else:
                return n + 2 * (pos - n_odd)
        
        mapped_odd = [(map_odd(i), map_odd(j)) for (i, j) in merge_odd]
        
        merge_even = batcher_odd_merge(n_even, m_odd)
        
        def map_even(pos):
            if pos < n_even:
                return 2 * pos
            else:
                return n + 2 * (pos - n_even) + 1
        
        mapped_even = [(map_even(i), map_even(j)) for (i, j) in merge_even]
        
        total = n + m
        final_comps = [(2 * i, 2 * i + 1) for i in range(total // 2)]
        
        return mapped_odd + mapped_even + final_comps

    def is_sorting(network, inputs):
        """
        Checks if the network sorts all given inputs (0/1 masks).
        Returns True if all outputs are sorted.
        """
        for inp in inputs:
            state = inp
            for i, j in network:
                # Compare bits i and j
                # If bit i is 1 and bit j is 0, swap them (set i to 0, j to 1)
                # min goes to i, max goes to j.
                # So if state has 1 at i and 0 at j, we want 0 at i and 1 at j.
                bit_i = (state >> i) & 1
                bit_j = (state >> j) & 1
                if bit_i > bit_j:
                    state ^= (1 << i) | (1 << j)
            
            # Check if state is sorted: all 0s then all 1s
            # This means there is at most one transition from 0 to 1.
            # Or, state & (state - 1) == 0 is for powers of 2.
            # Sorted 0/1 sequence looks like 000...011...1.
            # This is equivalent to: state is of the form (2^k - 1) << (N-k)? No.
            # It's of the form: lower bits are 1, upper are 0? 
            # Wait, compare-exchange writes min to i, max to j.
            # So 0s go to lower indices? 
            # If i < j, and we put min at i, max at j.
            # So 0 < 1. 0 goes to lower index, 1 goes to higher index.
            # So sorted state has 0s at the bottom (low indices) and 1s at the top (high indices).
            # Example: 000111.
            # This is NOT a power of 2 mask.
            # It is a mask where bits 0..k-1 are 0 and k..N-1 are 1.
            # Or simply, once a 1 is seen, all subsequent bits must be 1.
            
            # Efficient check:
            # Find the first 1. All bits before must be 0. All bits after must be 1.
            # Or: state must be equal to ((1 << N) - 1) ^ ((1 << k) - 1) for some k?
            # If k=0, state = 2^N - 1 (all 1s).
            # If k=N, state = 0 (all 0s).
            # If k=1, state = 111...110 (bit 0 is 0, rest 1).
            
            # Let's just iterate or use a trick.
            # Trick: (state + 1) & state == 0 is for 1000...
            # For 000111, state is 111. state+1 is 1000. (111 & 1000) = 0.
            # For 001110 (not sorted), state=14 (1110). state+1=15 (1111). 1110 & 1111 = 1110 != 0.
            # For 010101 (not sorted), state=21 (10101). state+1=22 (10110). 10101 & 10110 = 10100 != 0.
            # Is it true that a number is of the form 000...111...1 if and only if (state + 1) & state == 0?
            # Let's check 0 (000). 0+1=1. 0&1=0. Yes.
            # Let's check 1 (001). 1+1=2 (010). 001&010=0. Yes.
            # Let's check 3 (011). 3+1=4 (100). 011&100=0. Yes.
            # Let's check 7 (111). 7+1=8 (1000). 111&1000=0. Yes.
            # Let's check 2 (010). 2+1=3 (011). 010&011=010 != 0. No.
            # So yes, the condition `(state + 1) & state == 0` checks if the 1s are contiguous and at the LSB?
            # Wait. 1 is 001. 1s are at LSB.
            # 3 is 011. 1s are at LSB.
            # 7 is 111. 1s are at LSB.
            # But we want 0s at LSB and 1s at MSB.
            # So we want 000111.
            # This is `~state` being of that form?
            # Or `state` being of the form `2^N - 2^k`.
            # `2^N - 2^k` has 1s from k to N-1.
            # Example N=3, k=1: 8 - 2 = 6 (110).
            # Check 6 (110). 6+1=7 (111). 110 & 111 = 110 != 0.
            # So the condition `(state + 1) & state == 0` checks for 1s at LSB.
            # We need 1s at MSB.
            # So we check if `~state` (within N bits) has 1s at LSB?
            # `~state` for 110 is 001. 001 is 1. (1+1)&1 = 2&1=0. Yes.
            # So condition: `((~state) & ((1 << N) - 1))` is a power of 2 minus 1?
            # i.e., `inv_state = (~state) & ((1 << N) - 1)`
            # `inv_state` must be of form `2^k - 1`.
            # Which means `(inv_state + 1) & inv_state == 0`.
            
            inv_state = (~state) & ((1 << N) - 1)
            if (inv_state + 1) & inv_state != 0:
                return False
        
        return True

    # Generate initial network
    network = batcher_merge_sort(N)
    
    # Local Search
    # Use a fixed seed for reproducibility in search, but random samples
    random.seed(42)
    sample_size = 2000
    samples = [random.getrandbits(N) for _ in range(sample_size)]
    
    improved = True
    while improved:
        improved = False
        for i in range(len(network)):
            # Try removing comparator i
            trial_net = network[:i] + network[i+1:]
            if is_sorting(trial_net, samples):
                network = trial_net
                improved = True
                break # Restart loop as network structure changed
                
    # Final Full Verification
    # Generate all 2^N inputs
    # This might be memory heavy if we store them, so we iterate.
    # But is_sorting takes a list. 
    # 2^18 = 262,144 integers. List of ints is fine.
    all_inputs = list(range(1 << N))
    
    if is_sorting(network, all_inputs):
        return network
    else:
        # Fallback to the initial Batcher network if the heuristic failed 
        # (which shouldn't happen if the heuristic is sound, but safety first)
        return batcher_merge_sort(N)

# EVOLVE-BLOCK-END
