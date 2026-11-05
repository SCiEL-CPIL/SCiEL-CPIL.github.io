"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    # Initialize wire states: wire w for input k is bit w of k
    wire_state = []
    for w in range(n):
        # Build bitmask where bit k = (k >> w) & 1 for k in 0..2^n-1
        period = 1 << (w + 1)
        half = 1 << w
        period_mask = 0
        for bit_pos in range(half, period):
            period_mask |= (1 << bit_pos)
        mask = 0
        for rep in range(1 << (n - w - 1)):
            mask |= (period_mask << (rep * period))
        wire_state.append(mask)
    
    # Apply each comparator
    for (i, j) in network:
        a = wire_state[i]
        b = wire_state[j]
        wire_state[i] = a & b  # min for 0/1
        wire_state[j] = a | b  # max for 0/1
    
    # Check sorted: wire_state[k] & ~wire_state[k+1] must be 0 for all k
    for k in range(n - 1):
        if (wire_state[k] & ~wire_state[k + 1]) != 0:
            return False
    return True

def _generate_bitonic_network(n):
    """
    Generate a Bitonic Sorting Network for n wires.
    n must be a power of 2.
    """
    network = []
    
    def bitonic_sort(k, offset, direction):
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
                network.append((offset, offset + 1)) # Note: comparator (i,j) puts min in i, max in j.
                # If we want descending, we want max in i, min in j.
                # So we compare (offset, offset+1) but swap? 
                # Standard comparator (i,j) -> min(i), max(j).
                # To get descending at offset, we want max at offset, min at offset+1.
                # So we should use comparator (offset+1, offset) if we allow i>j? 
                # The interface says 0 <= i < j < N.
                # So we must use (offset, offset+1).
                # If we use (offset, offset+1), min goes to offset. That's ascending.
                # To get descending, we need to invert the logic? 
                # No, the comparator is fixed. 
                # Bitonic sort requires comparing with alternating directions.
                # If we can only use (i, j) with i < j, we can only do "min to i".
                # This means we can only sort in ascending order locally?
                # No, the "direction" in bitonic sort refers to the order of the bitonic sequence itself.
                # A bitonic sequence is one that increases then decreases.
                # The merge step compares elements at distance 2^(k-1).
                # If the left half is ascending and right half is descending, we compare (i, i+dist).
                # If we want the result to be ascending, we compare such that min goes to the lower index.
                # If we want the result to be descending, we compare such that max goes to the lower index?
                # But our comparator (i, j) with i < j puts min at i.
                # So we can ONLY produce ascending results at the lower indices.
                # This implies we can only build ascending sort networks?
                # Wait. If I have a network that sorts ascending, it is valid.
                # Bitonic sort algorithms often describe "descending" phases by swapping the roles of wires or inverting comparisons.
                # Since we are restricted to (i, j) with i < j, we are restricted to "min to i".
                # This means every comparator enforces that wire i has a value <= wire j.
                # This is a partial order constraint.
                # The final state must be fully sorted ascending.
                # So we don't need to generate "descending" sub-sorts explicitly in a way that violates i<j.
                # We just need to ensure the network sorts ascending.
                # Standard bitonic construction:
                # 1. Recursively sort first half ascending, second half descending.
                # 2. Merge.
                # How to sort second half descending using only (i, j) with i < j?
                # We can sort it ascending, and then... we can't reverse it easily without more comparators.
                # Actually, the standard bitonic network for ascending sort uses comparators that are always (i, j) with i < j?
                # Let's check.
                # For N=4:
                # (0,1), (2,3) -> Sort pairs ascending.
                # (0,2), (1,3) -> Compare 0 with 2, 1 with 3.
                # (1,2) -> Compare 1 with 2.
                # This is 5 comparators.
                # Does this sort all inputs?
                # Yes, this is the standard 5-comparator network for 4 wires.
                # Note that all comparators are (i, j) with i < j.
                # So we don't need "descending" comparators. The "direction" is handled by the topology.
                
                return

        half = 1 << (k - 1)
        
        # Step 1: Sort first half ascending, second half descending?
        # Actually, for the final merge to produce ascending, we need:
        # Left half ascending, Right half descending.
        # But we can't sort Right half descending with only (i, j) i<j?
        # Wait, if we sort Right half ascending, and Left half ascending, 
        # the merge step (comparing i with i+half) will not necessarily sort the whole thing.
        
        # Let's look at the standard Bitonic Sort algorithm for N=4 (k=2):
        # 1. Sort (0,1) and (2,3). (Ascending)
        # 2. Compare (0,2) and (1,3).
        # 3. Compare (1,2).
        
        # For N=8 (k=3):
        # 1. Bitonic sort 0-3 (ascending) and 4-7 (descending).
        #    How to sort 4-7 descending?
        #    We can sort 4-7 ascending, and then... 
        #    Actually, the "descending" sort in bitonic sort is often implemented by swapping the wire indices in the recursive call, or by inverting the comparators.
        #    Since we can't invert comparators (min must go to lower index), we must rely on the structure.
        
        # Let's trace N=4 again.
        # If we just use the 5-comparator network, it works.
        # The 5-comparator network is:
        # (0,1), (2,3)
        # (0,2), (1,3)
        # (1,2)
        
        # For N=8:
        # We need to merge two 4-wire bitonic sequences.
        # One ascending (0-3), one descending (4-7).
        # To get a descending 4-wire sequence from 4-7 using only (i, j) i<j:
        # We can sort 4-7 ascending, but that gives ascending.
        # Is there a network that sorts 4 wires in descending order using only (i, j) i<j?
        # No, because (i, j) forces wire i <= wire j. 
        # If we want wire 4 >= wire 5, we need a comparator that puts max in 4. 
        # But (4,5) puts min in 4. 
        # So we CANNOT create a local descending order on adjacent wires 4,5 using a single comparator (4,5).
        # However, the "descending" part of bitonic sort doesn't require adjacent wires to be descending in the final state of the sub-sort. It requires the sequence to be bitonic.
        
        # Actually, the standard bitonic network for N wires uses comparators (i, j) where i < j.
        # The "direction" is handled by the fact that the second half is sorted in "reverse" order relative to the merge.
        # But how?
        
        # Let's look at a known 8-wire bitonic network.
        # It has 19 comparators.
        
        # Alternative: Use a known generator.
        # For N=16, we can use the recursive definition:
        # Bitonic(N) = Bitonic(N/2) on 0..N/2-1 (Asc) 
        #              + Bitonic(N/2) on N/2..N-1 (Desc)
        #              + Merge(N/2)
        
        # How to implement Desc?
        # If we call Bitonic(N/2) on the second half, it will sort it Ascending.
        # If we want it Descending, we can map the wires: 
        # Wire N/2 + i corresponds to logical wire N-1-i in the "descending" frame?
        # No, the wires are physical.
        
        # Let's try a different approach: 
        # We will just use the **Odd-Even Transposition** as the starting point but apply a much better search.
        # Or, we can use a **Randomized Local Search** starting from a random valid network? No, generating a valid one is hard.
        
        # Let's stick to the current seed's starting point (Odd-Even) but improve the search.
        # The seed's search is too weak.
        
        pass

    return network

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Start with odd-even transposition (120 comparators)
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    if not _verify_network_fast(network):
        return network
    
    best_network = network
    best_count = len(network)
    
    # Aggressive Local Search with Restarts
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        current = best_network[:]
        
        # Phase 1: Prune
        # Use a faster loop: try to remove, if successful, restart pass.
        # To speed up, we can check time less frequently? No, keep it safe.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(current)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                
                # Optimization: Check if comparator is unique? 
                # If duplicate, removing one might be safe if the other is present?
                # Not necessarily, order matters.
                
                candidate = current[:idx] + current[idx+1:]
                if _verify_network_fast(candidate):
                    current = candidate
                    improved = True
                    break # Restart scan after change
        
        # Update best
        if len(current) < best_count:
            best_network = current
            best_count = len(current)
            if best_count <= 60:
                break
        
        # Phase 2: Perturb (if not optimal)
        if best_count > 60:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(current))
            current.insert(pos, (i, j))
            
            # Quick prune of the perturbed network
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
                if best_count <= 60:
                    break

    return best_network
# EVOLVE-BLOCK-END
