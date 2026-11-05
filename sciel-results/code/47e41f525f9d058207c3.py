"""Sidon set extension in [1, 284] containing {1,2,5,11}."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    seed_elements = [1, 2, 5, 11]
    max_val = 284
    max_sum = 2 * max_val
    
    def compute_sums_bitset(a):
        """Compute bitset of all pairwise sums a[i]+a[j] for i<=j."""
        bits = 0
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                bits |= 1 << (a[i] + a[j])
        return bits

    def can_add(x, a, sums_bits):
        """Check if x can be added to a without creating duplicate sums."""
        # Check if 2x conflicts
        if (sums_bits >> (2 * x)) & 1:
            return False
        # Check if x+y conflicts for any y in a
        for y in a:
            s = x + y
            if (sums_bits >> s) & 1:
                return False
        return True

    def greedy_extend_bitset(a, sums_bits):
        """Greedy extension using bitsets. Returns (new_a, new_sums_bits)."""
        a = list(a)
        sums_bits = sums_bits
        # Try adding elements in increasing order
        for x in range(1, max_val + 1):
            if x in a:
                continue
            if can_add(x, a, sums_bits):
                a.append(x)
                a.sort()
                # Update sums_bits
                # New sums: 2x and x+y for y in a (excluding x)
                # Note: a already contains x, so we iterate over original a? 
                # No, we added x to a. 
                # The new sums generated are 2x and x+y for y in (a \ {x}).
                # But wait, can_add checked against the old sums_bits.
                # The new sums are not in old sums_bits.
                # We need to add them.
                sums_bits |= 1 << (2 * x)
                for y in a:
                    if y != x:
                        sums_bits |= 1 << (x + y)
        return a, sums_bits

    def local_search_bitset(a, sums_bits, max_iters=1000, deadline=None):
        """Local search: remove one, add one, then greedy extend."""
        a = list(a)
        sums_bits = sums_bits
        best_a = list(a)
        best_bits = sums_bits
        best_size = len(a)
        seed_set = set(seed_elements)
        
        candidates = [x for x in range(1, max_val + 1) if x not in a]
        
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            
            if len(a) <= len(seed_elements):
                break
            
            # Pick a random non-seed element to remove
            removable = [x for x in a if x not in seed_set]
            if not removable:
                break
                
            rem_idx = random.randint(0, len(removable) - 1)
            x_remove = removable[rem_idx]
            
            # Remove x_remove from a
            a.remove(x_remove)
            
            # Recompute sums (fast for small n)
            new_sums_bits = compute_sums_bitset(a)
            
            # Try to add a random candidate
            # We can try a few random candidates or the best greedy one
            # Let's try a random candidate from the full range
            improved = False
            # Try a few random additions
            for _ in range(10):
                if not candidates:
                    break
                x_add = random.choice(candidates)
                if x_add in a:
                    continue
                if can_add(x_add, a, new_sums_bits):
                    a.append(x_add)
                    a.sort()
                    # Update sums
                    new_sums_bits |= 1 << (2 * x_add)
                    for y in a:
                        if y != x_add:
                            new_sums_bits |= 1 << (x_add + y)
                    
                    # Greedy extend
                    a, new_sums_bits = greedy_extend_bitset(a, new_sums_bits)
                    
                    if len(a) > best_size:
                        best_a = list(a)
                        best_bits = new_sums_bits
                        best_size = len(a)
                    improved = True
                    break
            
            if not improved:
                # Restore x_remove if we didn't improve? 
                # Actually, we are exploring neighbors. 
                # If we didn't find an improvement, we can revert to previous state 
                # or just continue from the current state (which is a valid Sidon set, just smaller).
                # Reverting is safer to avoid getting stuck in small sets.
                a.append(x_remove)
                a.sort()
                new_sums_bits = compute_sums_bitset(a)
                
        return best_a, best_bits

    # Initial greedy
    best_a = list(seed_elements)
    best_bits = compute_sums_bitset(best_a)
    best_a, best_bits = greedy_extend_bitset(best_a, best_bits)
    
    random.seed(42)
    
    # Multiple restarts with local search
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Randomized greedy start
        a = list(seed_elements)
        bits = compute_sums_bitset(a)
        
        # Shuffle candidates for greedy
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in a:
                continue
            if can_add(x, a, bits):
                a.append(x)
                a.sort()
                bits |= 1 << (2 * x)
                for y in a:
                    if y != x:
                        bits |= 1 << (x + y)
                
        # Local search
        a_ls, bits_ls = local_search_bitset(a, bits, max_iters=200, deadline=deadline - 5)
        
        if len(a_ls) > len(best_a):
            best_a = a_ls
            best_bits = bits_ls
            
    # Final verification
    final_a = sorted(best_a)
    assert all(x in final_a for x in seed_elements), "Missing seed elements"
    assert all(1 <= x <= 284 for x in final_a), "Out of range"
    assert len(set(final_a)) == len(final_a), "Duplicates"
    
    # Verify Sidon property
    sums = set()
    n = len(final_a)
    valid = True
    for i in range(n):
        for j in range(i, n):
            s = final_a[i] + final_a[j]
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    assert valid, "Invalid Sidon set"
    
    return final_a
# EVOLVE-BLOCK-END
