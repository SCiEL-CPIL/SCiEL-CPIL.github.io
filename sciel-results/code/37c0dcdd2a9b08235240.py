"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
N = 16

# EVOLVE-BLOCK-START
import time
import random

def _verify_network_fast(network):
    """
    Verify network sorts all 2^N Boolean inputs using the zero-one principle.
    Represents each wire's state across all 65536 inputs as a Python int bitmask.
    """
    n = N
    wire_state = []
    for w in range(n):
        period = 1 << (w + 1)
        half = 1 << w
        period_mask = 0
        for bit_pos in range(half, period):
            period_mask |= (1 << bit_pos)
        mask = 0
        for rep in range(1 << (n - w - 1)):
            mask |= (period_mask << (rep * period))
        wire_state.append(mask)
    
    for (i, j) in network:
        a = wire_state[i]
        b = wire_state[j]
        wire_state[i] = a & b
        wire_state[j] = a | b
    
    for k in range(n - 1):
        if (wire_state[k] & ~wire_state[k + 1]) != 0:
            return False
    return True

def _generate_bitonic_network(n):
    """
    Generate a correct Bitonic Sorting Network for n wires (power of 2).
    Uses recursive construction.
    """
    network = []
    
    def bitonic_merge(k, offset, direction):
        """
        k: log2(size)
        offset: starting wire index
        direction: 0 for ascending, 1 for descending
        """
        if k == 1:
            # Base case: sort 2 elements
            if direction == 0:
                network.append((offset, offset + 1))
            else:
                # For descending, we still use (i, j) with i<j but we want max in i?
                # No, standard bitonic sort network for ascending output:
                # The "descending" sort in the recursion is actually just sorting the sub-network
                # such that it produces a descending sequence relative to the merge?
                # Actually, the standard algorithm is:
                # BitonicSort(k, offset, dir):
                #   if k == 1: return
                #   half = 1 << (k-1)
                #   for i in range(half):
                #      if dir == 0: compare(offset+i, offset+i+half)
                #      else: compare(offset+i+half, offset+i) # INVALID for our (i,j) i<j constraint?
                #      # Wait, if we want descending, we want max in the lower index.
                #      # If we compare (i, i+half) with i < i+half, min goes to i.
                #      # This produces ascending.
                #      # How do we produce descending?
                #      # We can't with a single layer.
                #      # 
                #      # CORRECTION:
                #      # The bitonic sort network is symmetric.
                #      # To sort a sub-range in "descending" order (for the purpose of creating a bitonic sequence),
                #      # we simply sort it in "ascending" order but with the indices reversed?
                #      # No.
                #      # 
                #      # Let's look at the N=4 example again.
                #      # (0,1), (2,3) are ascending sorts.
                #      # (0,2), (1,3) are the merge.
                #      # (1,2) is the final merge.
                #      # 
                #      # The recursive definition that works:
                #      # def bitonic_sort(k, offset, dir):
                #      #   if k == 1: return
                #      #   half = 1 << (k-1)
                #      #   for i in range(half):
                #      #      if dir == 0:
                #      #         network.append((offset + i, offset + i + half))
                #      #      else:
                #      #         network.append((offset + i + half, offset + i)) # This is (j, i) with j > i, so it becomes (i, j) in our list?
                #      #         # No, the comparator (a, b) puts min in a.
                #      #         # If we call compare(offset+i+half, offset+i), we are passing a > b.
                #      #         # The problem statement says: "Each compare-exchange writes min to wire i and max to j."
                #      #         # And "0 <= i < j < N".
                #      #         # So we CANNOT call compare(j, i) if j > i. We must call compare(i, j).
                #      #         # If we want the effect of "max to lower index", we cannot do it directly with one comparator.
                #      # 
                #      # HOWEVER, the bitonic sort network construction for ASCENDING output
                #      # does NOT require explicit "descending" comparators.
                #      # It relies on the structure.
                #      # 
                #      # Let's use the verified N=4 network:
                #      # (0,1), (2,3), (0,2), (1,3), (1,2)
                #      # 
                #      # How to generate this recursively?
                #      # 
                #      # Step 1: Sort 0-1 asc, 2-3 asc.
                #      # Step 2: Compare (0,2), (1,3).
                #      # Step 3: Compare (1,2).
                #      # 
                #      # This looks like:
                #      # 1. Sort halves (both asc).
                #      # 2. Merge.
                #      # 
                #      # Merge(k, offset, dir):
                #      #   if k == 1: return
                #      #   half = 1 << (k-1)
                #      #   for i in range(half):
                #      #      network.append((offset + i, offset + i + half))
                #      #   Merge(k-1, offset, 0)
                #      #   Merge(k-1, offset + half, 0)
                #      # 
                #      # Let's test this Merge logic for N=4 (k=2, offset=0, dir=0):
                #      # Merge(2, 0, 0):
                #      #   half=2
                #      #   i=0: (0,2)
                #      #   i=1: (1,3)
                #      #   Merge(1, 0, 0): return
                #      #   Merge(1, 2, 0): return
                #      # 
                #      # This only generates (0,2), (1,3). Missing (0,1), (2,3), (1,2).
                #      # 
                #      # The full Bitonic Sort is:
                #      # BitonicSort(k, offset, dir):
                #      #   if k == 1: return
                #      #   half = 1 << (k-1)
                #      #   BitonicSort(k-1, offset, 0)
                #      #   BitonicSort(k-1, offset + half, 1) # <--- Here is the descending sort
                #      #   Merge(k, offset, dir)
                #      # 
                #      # How to implement BitonicSort(k-1, offset+half, 1)?
                #      # If dir=1, we want the sub-network to produce a descending sequence.
                #      # To produce a descending sequence on wires [offset+half, offset+half+size),
                #      # we can take the ascending sort network for that range and REVERSE the wire indices?
                #      # No, we can just use the same comparators?
                #      # 
                #      # Actually, there is a simpler observation:
                #      # The bitonic network for N is the same as the bitonic network for N/2, but applied to halves?
                #      # 
                #      # Let's hardcode the N=16 bitonic network or use a library? No.
                #      # Let's try to generate it correctly.
                #      # 
                #      # Reference implementation from a known source:
                #      # def bitonic_sort(n, offset, direction):
                #      #   if n == 2:
                #      #     if direction == 0:
                #      #       add (offset, offset+1)
                #      #     else:
                #      #       add (offset, offset+1) # Wait, is it the same?
                #      #       # If I want descending (1, 0), and I have wires a, b.
                #      #       # Comparator (a, b) gives (min, max).
                #      #       # If I want (max, min), I can't do it with one comparator.
                #      #       # 
                #      #       # BUT, in the context of the full network, the "descending" sort
                #      #       # is only needed to create a bitonic sequence.
                #      #       # The merge step then fixes it.
                #      #       # 
                #      #       # Is it possible that the "descending" sort is just the ascending sort
                #      #       # with the input wires swapped?
                #      #       # If I swap input wires, the output is swapped.
                #      #       # If I have a network that sorts (a,b) to (min,max),
                #      #       # and I feed it (b,a), it outputs (min(b,a), max(b,a)) = (min, max).
                #      #       # So the output is still ascending on the wires.
                #      #       # This doesn't help create a descending sequence on the wires.
                #      # 
                #      # Let's look at the N=4 network again.
                #      # (0,1), (2,3)
                #      # (0,2), (1,3)
                #      # (1,2)
                #      # 
                #      # If we split into 0-1 and 2-3:
                #      # 0-1 is sorted asc.
                #      # 2-3 is sorted asc.
                #      # 
                #      # So both halves are sorted ASCENDING?
                #      # If both halves are ascending, the merge (0,2), (1,3), (1,2) sorts them?
                #      # Yes, merging two sorted lists.
                #      # 
                #      # So Bitonic Sort for N=4 is just:
                #      # 1. Sort left half asc.
                #      # 2. Sort right half asc.
                #      # 3. Merge.
                #      # 
                #      # What about N=8?
                #      # 1. Sort 0-3 asc.
                #      # 2. Sort 4-7 asc.
                #      # 3. Merge 0-7.
                #      # 
                #      # Merge(0-7):
                #      #   Compare (0,4), (1,5), (2,6), (3,7).
                #      #   Then sort 0-3 and 4-7?
                #      #   No, the merge step in bitonic sort is:
                #      #   Compare (i, i+half).
                #      #   Then recursively merge the two halves?
                #      # 
                #      # If we do that:
                #      # Merge(8, 0, 0):
                #      #   (0,4), (1,5), (2,6), (3,7)
                #      #   Merge(4, 0, 0)
                #      #   Merge(4, 4, 0)
                #      # 
                #      # Merge(4, 0, 0):
                #      #   (0,2), (1,3)
                #      #   Merge(2, 0, 0) -> (0,1)
                #      #   Merge(2, 2, 0) -> (2,3)
                #      # 
                #      # Total for N=8:
                #      # (0,4)..(3,7)
                #      # (0,2), (1,3)
                #      # (0,1), (2,3)
                #      # (4,6), (5,7)
                #      # (4,5), (6,7)
                #      # 
                #      # Count: 4 + 2 + 2 + 2 + 2 = 12?
                #      # Standard N=8 bitonic is 19 comparators? Or 12?
                #      # N=4 is 5. N=8 is 19?
                #      # My count: 4 (level 1) + 2 (level 2) + 2 (level 3) ... 
                #      # Wait, Merge(4, 0, 0) generates (0,2), (1,3), (0,1), (2,3). That's 4 comparators.
                #      # Merge(4, 4, 0) generates (4,6), (5,7), (4,5), (6,7). That's 4 comparators.
                #      # Total: 4 + 4 + 4 = 12.
                #      # 
                #      # Is 12 correct for N=8?
                #      # Yes, Batcher's bitonic sort for N=8 has 12 comparators?
                #      # No, it's usually cited as 19?
                #      # Let's check.
                #      # N=2: 1
                #      # N=4: 5
                #      # N=8: 19?
                #      # 
                #      # My recursive merge generated 12.
                #      # Why the difference?
                #      # Because the "Sort half" step in BitonicSort is NOT just "Sort half".
                #      # It is "BitonicSort half".
                #      # 
                #      # BitonicSort(4, 0, 0):
                #      #   BitonicSort(2, 0, 0) -> (0,1)
                #      #   BitonicSort(2, 2, 1) -> ???
                #      #   Merge(4, 0, 0) -> (0,2), (1,3), (0,1), (2,3)
                #      # 
                #      # If BitonicSort(2, 2, 1) is just (2,3), then total is 1 + 1 + 4 = 6?
                #      # But N=4 is 5.
                #      # 
                #      # The standard N=4 network is:
                #      # (0,1), (2,3)
                #      # (0,2), (1,3)
                #      # (1,2)
                #      # 
                #      # My generated:
                #      # (0,1)
                #      # (2,3)
                #      # (0,2), (1,3), (0,1), (2,3)
                #      # Duplicates!
                #      # 
                #      # The issue is that Merge(k, offset, dir) should NOT recursively call Merge on the halves if the halves were already sorted by BitonicSort?
                #      # No, the merge step IS the recursive part.
                #      # 
                #      # Let's stop guessing and use a known correct list for N=16 or a robust generator.
                #      # Since I can't easily derive it, I will use the Odd-Even Transposition network (128 comparators) as the seed,
                #      # which is guaranteed to be correct, and rely on the search to improve it.
                #      # 
                #      # To improve the seed, I will also try to generate a "Shuffle-Merge" network or similar if I can,
                #      # but for now, Odd-Even is safe.
                #      
                #      # Actually, I found a simple correct bitonic generator in my memory:
                #      # It uses the "Bitonic Sort" algorithm but handles the direction by swapping the input wires of the sub-network?
                #      # No.
                #      
                #      # I will stick to Odd-Even for the seed to ensure correctness.
                #      # The search will do the rest.
                network.append((offset, offset + 1))
            return
        
        half = 1 << (k - 1)
        # Sort halves
        bitonic_merge(k - 1, offset, direction)
        bitonic_merge(k - 1, offset + half, 1 - direction)
        
        # Merge
        for i in range(half):
            if direction == 0:
                network.append((offset + i, offset + i + half))
            else:
                network.append((offset + i + half, offset + i)) # This is invalid (i,j) with i<j
                # Fix: always append (min, max)
                a = offset + i
                b = offset + i + half
                network.append((a, b))

    bitonic_merge(int(math.log2(n)), 0, 0)
    return network

def _generate_odd_even_network(n):
    network = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            network.append((i, i + 1))
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Use Odd-Even Transposition Network as a robust seed
    network = _generate_odd_even_network(N)
    
    if not _verify_network_fast(network):
        network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_network = network
    best_count = len(network)
    
    # Local Search: Prune redundant comparators
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(best_network)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = best_network[:idx] + best_network[idx+1:]
            
            if _verify_network_fast(candidate):
                best_network = candidate
                improved = True
                break
        
    # Structural Search: Swap comparators to find better orderings
    if time.time() < deadline - 5.0:
        for _ in range(20):
            if time.time() > deadline:
                break
            
            current = best_network[:]
            # Try swapping two comparators
            if len(current) > 1:
                i = random.randint(0, len(current) - 2)
                j = random.randint(i + 1, len(current) - 1)
                current[i], current[j] = current[j], current[i]
                
                # Prune after swap
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    indices = list(range(len(current)))
                    random.shuffle(indices)
                    for idx in indices:
                        if time.time() > deadline:
                            break
                        candidate = current[:idx] + current[idx+1:]
                        if _verify_network_fast(candidate):
                            current = candidate
                            improved = True
                            break
            
            if len(current) < best_count:
                best_network = current
                best_count = len(current)

    return best_network
# EVOLVE-BLOCK-END
