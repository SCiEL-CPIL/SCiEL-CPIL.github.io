import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 112s out of 120s limit to allow for cleanup/return
    deadline = start_time + 112.0 

    # 1. Precomputation
    # Map every t-subset to an integer id
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute masks for fast gain calculation
    # masks[x] is a dict: key is a (t-1)-tuple of points, value is a bitmask of t-subsets 
    # that contain x and that (t-1)-tuple.
    masks = [dict() for _ in range(v)]
    
    # Iterate through all t-subsets to populate masks
    # For each t-subset S, and each point x in S, the other t-1 points form a key.
    # We OR the bit for S into masks[x][key].
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i in range(t):
            # Create the (t-1) tuple of points excluding s[i]
            others = s[:i] + s[i+1:]
            # Ensure the key is consistent (tuple of sorted ints, though s is already sorted from combinations)
            key = others 
            x = s[i]
            masks[x][key] = masks[x].get(key, 0) | bit

    covered = 0  # Bitset of covered t-subsets
    blocks = []
    
    # Helper: Count bits in an integer
    # Python 3.10+ has int.bit_count()
    def popcount(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    def gain_of_add(x, B_tuple):
        """
        Calculate the number of new t-subsets covered by adding x to the current block B_tuple.
        B_tuple is a tuple of points currently in the block.
        """
        g_mask = 0
        # Iterate over all (t-1) subsets of B_tuple
        # We need to check if adding x covers any new t-subset.
        # A t-subset is formed by {x} U S where S is a (t-1)-subset of B_tuple.
        for S in itertools.combinations(B_tuple, t - 1):
            key = S
            m = masks[x].get(key)
            if m is not None:
                g_mask |= m
        # Intersect with currently uncovered subsets
        return popcount(g_mask & ~covered)

    def build_block_from(seed_points):
        """
        Grow a block from initial set (seed_points) to size k by max-gain greedy.
        Returns the block as a list of ints.
        """
        B = list(seed_points)
        B_set = set(B)
        available = [x for x in range(v) if x not in B_set]
        
        # To speed up iteration, we can maintain a list of available points.
        # However, removing from a list is O(n). 
        # Given v is small (24), O(n) is fine.
        
        while len(B) < k:
            best_x = -1
            best_gain = -1
            # We can potentially prune available points if time is tight, 
            # but for now, evaluate all.
            for x in available:
                g = gain_of_add(x, tuple(B))
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x == -1:
                break
            
            B.append(best_x)
            B_set.add(best_x)
            available.remove(best_x)
            
        return B

    def total_new_coverage(block):
        """
        Count how many t-subsets in the block are newly covered.
        """
        total = 0
        for s in itertools.combinations(sorted(block), t):
            idx = subset_to_index[s]
            if not (covered >> idx) & 1:
                total += 1
        return total

    # Main Loop: Strong Greedy
    # We repeatedly pick the block that covers the most new t-subsets.
    # To do this, we try to build a block from several uncovered seeds and pick the best one.
    
    # Optimization: Precompute the list of all points
    all_points = list(range(v))
    
    while True:
        if time.time() > deadline:
            break

        # Check if all covered
        # If n_subsets is large, (1 << n_subsets) - 1 is huge. 
        # But we can just check if covered has all bits set.
        # Alternatively, we can maintain a count of uncovered.
        # Let's stick to bitset check for correctness, it's fast enough for 24,7,3 (2024 subsets).
        uncov_mask = ((1 << n_subsets) - 1) & ~covered
        if uncov_mask == 0:
            break

        # Collect uncovered subset indices
        # We want to try building blocks from these seeds.
        # To limit time, we cap the number of seeds we try.
        # Previous code used 300. With faster gain, we can try more, say 1000.
        uncovered_indices = []
        m = uncov_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            m ^= lsb
            if len(uncovered_indices) >= 1000:
                break
        
        if not uncovered_indices:
            break

        best_block = None
        best_gain_total = -1
        
        # Shuffle or sample? 
        # The order of uncovered_indices is deterministic (LSB first).
        # To avoid bias, we could randomize, but deterministic is often stable.
        # However, trying a diverse set of seeds is key.
        
        for idx in uncovered_indices:
            if time.time() > deadline:
                break
            
            seed = all_subsets[idx]
            # Build block from this seed
            B = build_block_from(list(seed))
            
            # Calculate total new coverage for this block
            # We can optimize this: we already know it covers the seed.
            # But total_new_coverage is O(k^t) which is small (7^3=343).
            total = total_new_coverage(B)
            
            if total > best_gain_total:
                best_gain_total = total
                best_block = B
        
        if best_block is None:
            break

        bl = sorted(best_block)
        
        # Mark covered
        # Instead of iterating all t-subsets of bl, we can just OR the masks?
        # No, we need to update the global 'covered' bitset.
        for s in itertools.combinations(bl, t):
            covered |= 1 << subset_to_index[s]
            
        blocks.append(bl)

    return blocks
