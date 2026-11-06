"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 18

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a compare-exchange sorting network for 18 wires.
    
    Strategy: Use a known efficient construction based on AKS-inspired or
    carefully designed networks. Since we can't easily implement AKS in Python
    within time limits, we'll use a hybrid approach:
    
    1. Start with a bitonic sorting network structure (known to be efficient)
    2. Optimize by removing redundant comparators through verification
    
    Bitonic sort for 16 wires uses 40 comparators. For 18 wires, we can use
    a merge-based approach or extend the bitonic network.
    
    Actually, let's use a proven small network. The best known for 18 wires 
    is around 77 comparators. Let's try to construct one using a divide-and-conquer
    merge strategy with verified sub-networks.
    
    Approach: Use a "merge" of two 9-wire sorting networks, then merge the results.
    But 9-wire optimal is 30 comparators (known), so 2*30 + merge(9,9).
    Merge of two sorted 9-element sequences needs at most 9*9/2 ~ 40+ comparators.
    That's too many.
    
    Better: Use known optimal/near-optimal small networks as building blocks.
    
    Let's use the following approach:
    - For n <= 4, use known optimal networks
    - For larger n, use a recursive bitonic-like construction but prune
      redundant comparators via Boolean verification
    
    Actually, let me just implement a clean bitonic sorting network for 18 wires
    and then try to remove redundant comparators.
    
    Bitonic sort for n=18:
    - Round up to power of 2? No, bitonic works for any n but is less efficient.
    - Standard bitonic for n=16 is 40 comparators. For n=18, we can do:
      Sort first 16, then merge last 2 into the sorted 16.
    
    Alternative: Use the "AKS" construction is too complex. Let's use a 
    well-known practical network.
    
    Let me implement a simple but effective approach:
    1. Generate a bitonic sorting network for 18 wires
    2. Verify and prune redundant comparators
    
    Bitonic sort construction for arbitrary n:
    - If n == 1: return []
    - If n is even: 
      - Sort first half, sort second half
      - Merge them using bitonic merge
    - If n is odd:
      - Handle by padding or special case
    
    Let's implement bitonic sort properly for n=18.
    """
    def bitonic_sort(n):
        """Generate bitonic sorting network for n wires."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n == 3:
            return [(0, 1), (1, 2), (0, 1)]
        if n == 4:
            return [(0, 1), (2, 3), (1, 2), (0, 1), (2, 3)]
        
        # For larger n, use recursive bitonic construction
        # Bitonic sort: split into halves, sort each half (one ascending, one descending),
        # then merge
        
        def bitonic_merge(n, reverse=False):
            """Merge two sorted halves of size n/2 into a sorted sequence of size n."""
            if n <= 1:
                return []
            comparators = []
            # The bitonic merge for size n
            # Compare elements i and i + n//2 for i in range(n//2)
            half = n // 2
            # Standard bitonic merge
            for i in range(half):
                a = i
                b = i + half
                if reverse:
                    a, b = b, a
                comparators.append((a, b))
            # Recursively merge the two halves
            left = bitonic_merge(half, reverse=False)
            right = bitonic_merge(half, reverse=True)
            # Adjust indices for right half
            right = [(i + half, j + half) for i, j in right]
            comparators.extend(left)
            comparators.extend(right)
            return comparators
        
        def bitonic_sort_rec(n):
            """Recursively generate bitonic sorting network."""
            if n <= 1:
                return []
            if n == 2:
                return [(0, 1)]
            if n == 3:
                return [(0, 1), (1, 2), (0, 1)]
            if n == 4:
                return [(0, 1), (2, 3), (1, 2), (0, 1), (2, 3)]
            
            half = n // 2
            # Sort first half ascending, second half descending
            left = bitonic_sort_rec(half)
            right = bitonic_sort_rec(n - half)
            # Adjust right indices
            right = [(i + half, j + half) for i, j in right]
            # Now merge: the concatenation is bitonic
            # To merge a bitonic sequence of size n:
            merge = bitonic_merge(n, reverse=False)
            return left + right + merge
        
        return bitonic_sort_rec(n)
    
    # Generate initial bitonic network
    network = bitonic_sort(N)
    
    # Now try to prune redundant comparators
    # Verify that the network sorts all Boolean inputs
    def verify_network(net):
        """Verify that the network sorts all 2^n Boolean inputs."""
        n = N
        for mask in range(1 << n):
            wires = [1 if (mask >> i) & 1 else 0 for i in range(n)]
            for i, j in net:
                if wires[i] > wires[j]:
                    wires[i], wires[j] = wires[j], wires[i]
            # Check if sorted
            for i in range(n - 1):
                if wires[i] > wires[i + 1]:
                    return False
        return True
    
    # Try to remove comparators one by one if possible
    # This is a greedy pruning approach
    improved = list(network)
    changed = True
    while changed:
        changed = False
        for idx in range(len(improved) - 1, -1, -1):
            # Try removing comparator at idx
            candidate = improved[:idx] + improved[idx+1:]
            if verify_network(candidate):
                improved = candidate
                changed = True
    
    return improved
# EVOLVE-BLOCK-END
