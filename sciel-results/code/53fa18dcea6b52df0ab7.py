"""Sidon set extension in [1, 284] containing {1,2,5,11}."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    seed_elements = [1, 2, 5, 11]
    max_val = 284
    max_sum = 2 * max_val
    seed_set = frozenset(seed_elements)
    
    def compute_sums_bitset(a):
        """Compute bitset of all pairwise sums a[i]+a[j] for i<=j."""
        bits = 0
        n = len(a)
        for i in range(n):
            ai = a[i]
            for j in range(i, n):
                bits |= 1 << (ai + a[j])
        return bits

    def can_add(x, a, sums_bits):
        """Check if x can be added to a without creating duplicate sums."""
        if (sums_bits >> (2 * x)) & 1:
            return False
        for y in a:
            s = x + y
            if (sums_bits >> s) & 1:
                return False
        return True

    def greedy_extend(a, sums_bits, order=None):
        """Greedy extension. order: list of candidates to try in that order, or None for ascending."""
        a = list(a)
        sums_bits = sums_bits
        if order is None:
            order = list(range(1, max_val + 1))
        for x in order:
            if x in a:
                continue
            if can_add(x, a, sums_bits):
                a.append(x)
                a.sort()
                sums_bits |= 1 << (2 * x)
                for y in a:
                    if y != x:
                        sums_bits |= 1 << (x + y)
        return a, sums_bits

    def local_search(a, sums_bits, deadline, max_iters=5000):
        """1-swap local search with simulated annealing."""
        a = list(a)
        sums_bits = sums_bits
        best_a = list(a)
        best_bits = sums_bits
        best_size = len(a)
        
        all_candidates = [x for x in range(1, max_val + 1) if x not in a]
        
        # Temperature schedule
        temp = 5.0
        cool_rate = 0.9995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if len(a) <= len(seed_elements):
                break
            
            # Pick a random non-seed element to remove
            removable = [x for x in a if x not in seed_set]
            if not removable:
                break
                
            x_remove = random.choice(removable)
            
            # Remove x_remove
            a.remove(x_remove)
            new_sums = compute_sums_bitset(a)
            
            # Try to find a good addition
            # First try greedy: find the best candidate we can add
            # Then maybe try random ones for diversity
            found = False
            for _ in range(3):  # try up to 3 random candidates
                if not all_candidates:
                    break
                x_add = random.choice(all_candidates)
                if x_add in a:
                    continue
                if can_add(x_add, a, new_sums):
                    a.append(x_add)
                    a.sort()
                    new_sums |= 1 << (2 * x_add)
                    for y in a:
                        if y != x_add:
                            new_sums |= 1 << (x_add + y)
                    found = True
                    break
            
            if not found:
                # Revert
                a.append(x_remove)
                a.sort()
                sums_bits = compute_sums_bitset(a)
                temp *= cool_rate
                continue
            
            # Now we have a new state of size len(a)
            # Greedy extend to fill up
            a, new_sums = greedy_extend(a, new_sums)
            
            new_size = len(a)
            if new_size > best_size:
                best_a = list(a)
                best_bits = new_sums
                best_size = new_size
                temp = 5.0  # reset temperature on improvement
            else:
                # Simulated annealing: accept worse moves with probability
                delta = best_size - new_size
                if delta > 0 and random.random() < 0.5 ** delta:
                    # Accept the worse state to explore
                    pass  # already in new state
                else:
                    # Revert
                    a = list(best_a)
                    sums_bits = best_bits
            
            temp *= cool_rate
        
        return best_a, best_bits

    # Phase 1: Multiple greedy initializations with different orderings
    best_a = list(seed_elements)
    best_bits = compute_sums_bitset(best_a)
    best_size = len(best_a)
    
    random.seed(42)
    
    # Try ascending greedy
    a1, bits1 = greedy_extend(list(seed_elements), compute_sums_bitset(seed_elements), order=list(range(1, max_val + 1)))
    if len(a1) > best_size:
        best_a, best_bits, best_size = a1, bits1, len(a1)
    
    # Try descending greedy
    a2, bits2 = greedy_extend(list(seed_elements), compute_sums_bitset(seed_elements), order=list(range(max_val, 0, -1)))
    if len(a2) > best_size:
        best_a, best_bits, best_size = a2, bits2, len(a2)
    
    # Try a few random orderings
    for _ in range(5):
        order = list(range(1, max_val + 1))
        random.shuffle(order)
        a3, bits3 = greedy_extend(list(seed_elements), compute_sums_bitset(seed_elements), order=order)
        if len(a3) > best_size:
            best_a, best_bits, best_size = a3, bits3, len(a3)
    
    # Phase 2: Local search from best state
    if time.time() < deadline - 30:
        best_a, best_bits = local_search(best_a, best_bits, deadline - 5)
        best_size = len(best_a)
    
    # Phase 3: Additional restarts with local search if time permits
    remaining = deadline - time.time()
    if remaining > 20:
        num_restarts = min(50, int(remaining / 2))  # ~2s per restart
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Randomized greedy start
            a = list(seed_elements)
            bits = compute_sums_bitset(a)
            
            order = list(range(1, max_val + 1))
            random.shuffle(order)
            
            for x in order:
                if x in a:
                    continue
                if can_add(x, a, bits):
                    a.append(x)
                    a.sort()
                    bits |= 1 << (2 * x)
                    for y in a:
                        if y != x:
                            bits |= 1 << (x + y)
            
            # Local search with limited iterations
            iters = min(2000, int((deadline - 5 - time.time()) * 50))
            if iters > 100:
                a_ls, bits_ls = local_search(a, bits, deadline - 5, max_iters=iters)
                if len(a_ls) > best_size:
                    best_a = a_ls
                    best_bits = bits_ls
                    best_size = len(a_ls)
    
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
