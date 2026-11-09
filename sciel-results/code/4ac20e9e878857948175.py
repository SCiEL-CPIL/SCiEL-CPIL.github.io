"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import random

def _is_valid_sorter(net, N=18, timeout=5):
    """Check if the network sorts all 2^N Boolean inputs using Zero-One Principle."""
    start_time = time.time()
    
    # Optimization: Pre-compute the sequence of operations?
    # No, just simulate.
    
    # We can optimize the inner loop by using a list of wires
    # and swapping in place.
    
    for mask in range(1 << N):
        if time.time() - start_time > timeout:
            return False
            
        # Extract bits
        # Using list comprehension is fast enough
        wires = [(mask >> i) & 1 for i in range(N)]
        
        # Apply comparators
        for i, j in net:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        
        # Check sorted
        # Since it's boolean, sorted means all 0s then all 1s
        # We can check if it's non-decreasing
        for k in range(N - 1):
            if wires[k] > wires[k + 1]:
                return False
                
    return True

def _bitonic_sort(N):
    """Generate Batcher's Bitonic Sort network for N wires (N must be power of 2)."""
    if N <= 1:
        return []
    if N == 2:
        return [(0, 1)]
    
    def _bitonic_merge(net, start, length, reverse=False):
        """
        Merge two sorted sequences of length length/2 starting at 'start'.
        The first half is sorted in ascending order, the second in descending (or vice versa).
        This function adds comparators to 'net'.
        """
        if length <= 1:
            return
        
        # Compare pairs
        half = length // 2
        for i in range(half):
            # Compare (start + i) and (start + i + half)
            # If reverse is False, we want the result to be sorted ascending?
            # Bitonic merge:
            # Compare (i, i+half).
            # If reverse, swap the indices?
            # Standard: 
            # If not reverse: min goes to i, max to i+half.
            # If reverse: max goes to i, min to i+half.
            a = start + i
            b = start + i + half
            if not reverse:
                net.append((a, b))
            else:
                net.append((b, a)) # Note: comparator (i,j) puts min at i, max at j.
                # So (b,a) puts min at b, max at a.
                # If we want max at a (start), we put max at a.
                # So min at b, max at a. Correct for reverse sort.
                
        # Recursively merge
        _bitonic_merge(net, start, half, reverse)
        _bitonic_merge(net, start + half, half, not reverse)

    net = []
    # Generate the full bitonic sequence
    # 1. Sort first half ascending, second half descending
    _bitonic_merge(net, 0, N, False) 
    # Wait, the initial setup for Bitonic Sort is:
    # Sort first half ascending, second half descending.
    # Then merge the whole thing.
    
    # Correct Bitonic Sort structure:
    # def bitonic_sort(N):
    #   if N=1 return
    #   sort first half ascending
    #   sort second half descending
    #   merge whole
    
    # My _bitonic_merge assumes the input is already bitonic (first half asc, second half desc).
    # So I need to generate the bitonic pattern first.
    
    # Let's re-implement cleanly.
    
    net = []
    
    def _sort_bitonic(start, length, reverse):
        """Sort a bitonic sequence of length 'length' starting at 'start'."""
        if length <= 1:
            return
        
        half = length // 2
        # Compare (i, i+half)
        for i in range(half):
            a = start + i
            b = start + i + half
            if not reverse:
                net.append((a, b))
            else:
                net.append((b, a))
        
        # Recursively sort the two halves
        _sort_bitonic(start, half, reverse)
        _sort_bitonic(start + half, half, not reverse)
        
    # To sort the whole array:
    # 1. Make it bitonic:
    #    - Sort first half ascending
    #    - Sort second half descending
    # 2. Merge the whole array
    
    # We need a helper to sort a subarray into a specific direction, assuming it's already bitonic?
    # No, standard Bitonic Sort:
    # Sort(A, N, reverse):
    #   if N=1 return
    #   Sort(A, N/2, False)
    #   Sort(A+N/2, N/2, True)
    #   Merge(A, N, reverse)
    
    # This is getting confusing. Let's use the standard recursive definition from literature.
    
    # Reset net
    net = []
    
    def _bitonic_sort_recursive(start, length, reverse):
        if length <= 1:
            return
        
        half = length // 2
        
        # Sort the two halves in opposite directions
        _bitonic_sort_recursive(start, half, False)
        _bitonic_sort_recursive(start + half, half, True)
        
        # Merge the two halves
        # Compare (start+i, start+i+half)
        for i in range(half):
            a = start + i
            b = start + i + half
            if not reverse:
                net.append((a, b))
            else:
                net.append((b, a))
                
        # Wait, the merge step in Bitonic Sort is just the comparisons?
        # No, after comparisons, the result is not fully sorted if the input was just "bitonic".
        # The recursive calls in the merge step are needed?
        # Actually, the standard Bitonic Merge is:
        # 1. Compare (i, i+n/2)
        # 2. Recursively merge the first half and second half.
        
        # So the structure is:
        # def bitonic_merge(start, length, reverse):
        #   if length <= 1: return
        #   half = length // 2
        #   for i in range(half):
        #      compare(start+i, start+i+half)
        #   bitonic_merge(start, half, reverse)
        #   bitonic_merge(start+half, half, not reverse)
        
        # And Bitonic Sort is:
        # def bitonic_sort(start, length, reverse):
        #   if length <= 1: return
        #   half = length // 2
        #   bitonic_sort(start, half, False)
        #   bitonic_sort(start+half, half, True)
        #   bitonic_merge(start, length, reverse)
        
        # Let's implement this.
        
    # Re-define functions inside construct to avoid global state issues or just use nested functions.
    
    return net # Placeholder, will be replaced in construct

def construct():
    start_time = time.time()
    timeout = 55
    
    # 1. Generate Bitonic Sort for N=32
    N_pad = 32
    net = []
    
    def _bitonic_merge(start, length, reverse):
        if length <= 1:
            return
        half = length // 2
        for i in range(half):
            a = start + i
            b = start + i + half
            if not reverse:
                net.append((a, b))
            else:
                net.append((b, a))
        _bitonic_merge(start, half, reverse)
        _bitonic_merge(start + half, half, not reverse)

    def _bitonic_sort(start, length, reverse):
        if length <= 1:
            return
        half = length // 2
        _bitonic_sort(start, half, False)
        _bitonic_sort(start + half, half, True)
        _bitonic_merge(start, length, reverse)

    _bitonic_sort(0, N_pad, False)
    
    # 2. Prune to N=18
    net = [(i, j) for i, j in net if i < N and j < N]
    
    # 3. Validate
    if not _is_valid_sorter(net, N, timeout=5):
        # If invalid, return odd-even transposition
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    # 4. Greedy Removal
    # Calculate criticality
    def _get_swap_counts(net, N=18, num_samples=500):
        counts = {idx: 0 for idx in range(len(net))}
        for _ in range(num_samples):
            wires = [random.randint(0, 1) for _ in range(N)]
            for idx, (i, j) in enumerate(net):
                if wires[i] > wires[j]:
                    wires[i], wires[j] = wires[j], wires[i]
                    counts[idx] += 1
        return counts

    # Loop removal
    removed_any = True
    while removed_any and time.time() - start_time < timeout:
        removed_any = False
        counts = _get_swap_counts(net, N)
        # Sort indices by count
        sorted_indices = sorted(counts.keys(), key=lambda x: counts[x])
        
        for idx in sorted_indices:
            if time.time() - start_time > timeout:
                break
                
            # Try to remove
            new_net = [comp for i, comp in enumerate(net) if i != idx]
            
            # Validate
            if _is_valid_sorter(new_net, N, timeout=2):
                net = new_net
                removed_any = True
                break # Restart loop with new network
                
    return net
# EVOLVE-BLOCK-END
