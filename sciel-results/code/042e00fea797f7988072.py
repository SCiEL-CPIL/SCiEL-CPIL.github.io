import time
import random

DEADLINE_SECONDS = 85.0

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS

    def time_left():
        return deadline - time.time()

    best_size = 0
    best_set = []

    # Bitwise DP: bitsets[k] is an integer where bit i is set iff sum i is achievable at cardinality k
    # Max possible sum: 60 * 256 = 15360, so we need bits 0..15360
    MAX_SUM = 60 * 256
    NUM_BITS = MAX_SUM + 1  # bits 0 through MAX_SUM inclusive

    def try_add(bitsets, x):
        """
        Try to add element x to current set represented by bitsets.
        bitsets[k] is an int where bit s is set iff sum s achievable at cardinality k.
        Returns (success, new_bitsets) or (False, None).
        """
        n = len(bitsets) - 1  # current max cardinality
        
        # Compute new sums: new_sums[k] = bitsets[k-1] << x
        # Check collisions: for each k, new_sums[k] & (union of bitsets[j] for j != k) must be 0
        # But we also need to account for the fact that bitsets[k] already has those sums.
        # Actually: the new sums at cardinality k are (bitsets[k-1] << x).
        # A collision occurs if any of these new sums is already in bitsets[j] for j != k.
        # Note: if a new sum is in bitsets[k] already, that's fine (same cardinality).
        
        # Build mask of all sums at cardinalities != k for each k
        # Efficient approach: compute total_mask = OR of all bitsets, then for each k,
        # collision_mask[k] = total_mask & ~bitsets[k], and check (new_sums[k] & collision_mask[k]) == 0
        
        # First, compute total_mask
        total_mask = 0
        for k in range(n + 1):
            total_mask |= bitsets[k]
        
        # Precompute collision masks for each cardinality
        # collision_mask[k] = total_mask & ~bitsets[k] (sums that appear at some cardinality but not k)
        # We need to check: for each k from 1 to n+1, (bitsets[k-1] << x) & (total_mask & ~bitsets[k]) == 0
        # But wait, bitsets[k] might not exist yet for k > n. Let's handle that.
        
        for k in range(1, n + 2):
            # New sums at cardinality k
            if k - 1 > n:
                continue
            new_sums = bitsets[k - 1] << x
            if new_sums == 0:
                continue
            
            # Sums at cardinality k (existing)
            existing_k = bitsets[k] if k <= n else 0
            
            # Collision: new_sums that are in total_mask but NOT in existing_k
            # i.e., new_sums & (total_mask & ~existing_k)
            # But we also need to be careful: ~existing_k in Python gives negative numbers for large ints.
            # Use: collision = new_sums & total_mask & ~existing_k
            # To avoid issues with ~, we can do: (new_sums & total_mask) & (total_mask ^ (total_mask & existing_k))
            # Actually simpler: check if (new_sums & total_mask) has any bits not in existing_k
            # That is: (new_sums & total_mask) & ~existing_k != 0
            # In Python, ~existing_k works correctly for the bits we care about since total_mask bounds it.
            
            # The collision is: bits in new_sums that are in total_mask but not in existing_k
            # = (new_sums & total_mask) & ~existing_k
            # But ~existing_k in Python is an infinite two's complement, so we mask with total_mask:
            # collision = (new_sums & total_mask) & (total_mask ^ (total_mask & existing_k))
            # = (new_sums & total_mask) & (total_mask & ~existing_k)
            
            # Let's just compute: forbidden = total_mask & ~existing_k
            # Then check new_sums & forbidden
            # In Python, ~existing_k gives -existing_k-1, which has all bits set except those in existing_k.
            # So total_mask & ~existing_k correctly gives bits in total_mask not in existing_k.
            
            forbidden = total_mask & ~existing_k
            if new_sums & forbidden:
                return False, None
        
        # No collision, build new bitsets
        new_bitsets = [0] * (n + 2)
        for k in range(n + 1):
            new_bitsets[k] = bitsets[k]
        for k in range(1, n + 2):
            if k - 1 <= n:
                new_bitsets[k] |= (bitsets[k - 1] << x)
        
        return True, new_bitsets

    # Strategy 1: Structured candidates (powers of 2, geometric, etc.)
    # Strategy 2: Random shuffles
    # Try multiple orderings

    for attempt in range(200):
        if time_left() < 1.0:
            break

        if attempt == 0:
            # Ascending
            candidates = list(range(1, 257))
        elif attempt == 1:
            # Descending
            candidates = list(range(256, 0, -1))
        elif attempt == 2:
            # Powers of 2 and near-powers
            candidates = []
            for p in range(8):
                candidates.append(1 << p)
            # Add some structured values
            for base in [3, 5, 7, 9, 11, 13]:
                for mult in range(1, 257 // base + 1):
                    v = base * mult
                    if 1 <= v <= 256 and v not in candidates:
                        candidates.append(v)
            candidates = list(dict.fromkeys(candidates))  # Remove duplicates, preserve order
        elif attempt == 3:
            # Geometric progression
            candidates = []
            ratio = 1.15
            val = 1.0
            while val <= 256:
                candidates.append(int(round(val)))
                val *= ratio
            # Fill in remaining
            for i in range(1, 257):
                if i not in candidates:
                    candidates.append(i)
        elif attempt % 2 == 0:
            # Even-indexed attempts: random
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        else:
            # Odd-indexed attempts: sorted by some hash for diversity
            candidates = list(range(1, 257))
            random.shuffle(candidates)

        current = []
        # bitsets[0] = 1 (bit 0 set, sum 0 at cardinality 0)
        bitsets = [0] * 62  # indices 0..61, we'll use up to 61
        bitsets[0] = 1  # sum 0 is achievable at cardinality 0

        for x in candidates:
            if time_left() < 0.3:
                break
            if len(current) >= 60:
                break

            success, new_bitsets = try_add(bitsets, x)
            if success:
                current.append(x)
                bitsets = new_bitsets

        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]

    # Local search: try to improve best_set by swapping elements
    if time_left() > 3 and best_size > 0:
        for iteration in range(100):
            if time_left() < 1.0:
                break

            # Try removing one element and trying to add a different one
            improved = False
            for i in range(len(best_set)):
                if time_left() < 0.5:
                    break
                # Remove element i
                reduced = best_set[:i] + best_set[i+1:]
                # Rebuild bitsets for reduced set
                bitsets = [0] * 62
                bitsets[0] = 1
                for x in reduced:
                    success, new_bitsets = try_add(bitsets, x)
                    if not success:
                        break
                    bitsets = new_bitsets
                else:
                    # All elements added successfully, now try to add a new element
                    for x in range(1, 257):
                        if x in reduced:
                            continue
                        success, new_bitsets = try_add(bitsets, x)
                        if success:
                            # We found a swap that keeps same size but different element
                            # Or we could try to extend
                            new_set = reduced + [x]
                            # Try to extend further
                            extended = new_set[:]
                            ext_bitsets = new_bitsets
                            for x2 in range(1, 257):
                                if time_left() < 0.3:
                                    break
                                if len(extended) >= 60:
                                    break
                                if x2 in extended:
                                    continue
                                success2, new_ext = try_add(ext_bitsets, x2)
                                if success2:
                                    extended.append(x2)
                                    ext_bitsets = new_ext
                            if len(extended) > best_size:
                                best_size = len(extended)
                                best_set = extended
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break

    # Final validation
    if best_set:
        S = best_set
        n = len(S)
        # Verify using bitwise method
        bitsets = [0] * (n + 2)
        bitsets[0] = 1
        valid = True
        for x in S:
            success, new_bitsets = try_add(bitsets, x)
            if not success:
                valid = False
                break
            bitsets = new_bitsets

        # Double-check: ensure no sum appears at two different cardinalities
        if valid:
            # Check by examining all bitsets
            all_sums = {}
            for k in range(n + 1):
                # Extract set bits from bitsets[k]
                mask = bitsets[k]
                s = 0
                while mask:
                    if mask & 1:
                        if s in all_sums:
                            if all_sums[s] != k:
                                valid = False
                                break
                        else:
                            all_sums[s] = k
                    mask >>= 1
                    s += 1
                    if not valid:
                        break
                if not valid:
                    break

        if not valid:
            # Fallback
            best_set = [1]

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
