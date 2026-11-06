import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 16
    k = 4
    all_ones = (1 << n) - 1

    # Precompute all k-subsets as bitmasks
    all_masks = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_masks.append(mask)
        
    num_masks = len(all_masks)
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Precompute submasks of size 0, 1, 2, 3 for all masks
    # submasks_by_size[mask][size] = list of submasks
    # We only need sizes 0 to k-1 = 3
    submasks_by_size = [[[] for _ in range(k)] for _ in range(1 << n)]
    
    for mask in range(1 << n):
        bits = []
        m = mask
        while m:
            lsb = m & -m
            bits.append(lsb)
            m ^= lsb
            
        for size in range(k):
            if len(bits) >= size:
                # Generate combinations of bits of given size
                # Since size <= 3 and bits <= 16, this is fast
                if size == 0:
                    submasks_by_size[mask][size] = [0]
                elif size == 1:
                    submasks_by_size[mask][size] = list(bits)
                elif size == 2:
                    res = []
                    for i in range(len(bits)):
                        for j in range(i + 1, len(bits)):
                            res.append(bits[i] | bits[j])
                    submasks_by_size[mask][size] = res
                elif size == 3:
                    res = []
                    for i in range(len(bits)):
                        for j in range(i + 1, len(bits)):
                            for l in range(j + 1, len(bits)):
                                res.append(bits[i] | bits[j] | bits[l])
                    submasks_by_size[mask][size] = res

    def greedy(order):
        counts = [0] * (1 << n)
        family = []
        
        for mask in order:
            if counts[mask] > 0:
                continue
            
            family.append(mask)
            
            for A in family[:-1]:
                J = A & mask
                U = all_ones ^ (A | mask)
                need = k - popcount[J]
                
                submasks = submasks_by_size[U][need]
                for Y in submasks:
                    M_prime = J | Y
                    counts[M_prime] += 1
                    
        return family
        
    best_family = []
    best_size = 0
    
    # First pass: natural order
    fam = greedy(all_masks)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = fam[:]
        
    # Random restarts
    # Estimate time per restart
    # Greedy is O(N * F * AvgSubmasks)
    # N=1820, F~250, AvgSubmasks~small
    # Should be fast. Let's try many.
    
    num_restarts = 5000
    for restart in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        random.seed(12345 + restart)
        shuffled = all_masks[:]
        random.shuffle(shuffled)
        
        fam = greedy(shuffled)
        
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam[:]
            
        if best_size >= 250:
            break
            
    # Local Search / Hill Climbing
    # Try to swap out a set for a better one, or just add if possible (greedy is maximal, so can't add)
    # So we must swap.
    # Or, we can try to "repair" by removing one set and adding two?
    # Simpler: Randomly pick a set in family, remove it, and try to add random sets from outside.
    
    if best_size < 250 and time.time() < deadline - 10:
        family_set = set(best_family)
        
        # Precompute conflict check for speed?
        # The greedy check is O(F^2) effectively if we do it naively.
        # But we have the counts array logic.
        
        # Let's try a simple local search:
        # 1. Pick a random member m from family
        # 2. Remove m
        # 3. Try to add random candidates from outside
        # 4. If we added >= 1, keep. Else revert.
        
        # To add a candidate, we need to check if it conflicts with the current family.
        # Conflict check:
        # For new mask S, and family F.
        # Does there exist A, B in F such that A, B, S form sunflower?
        # This is equivalent to:
        # For any pair A, B in F, let J = A & B, U = all_ones ^ (A | B).
        # If S contains J and S is disjoint from U? No.
        # Sunflower condition: A&B = A&S = B&S = J.
        # So S must contain J.
        # And S & (A \ J) = 0 and S & (B \ J) = 0.
        # i.e. S is disjoint from (A | B) \ J = A ^ B.
        # So S must be a superset of J and disjoint from A^B.
        
        # We can precompute for each pair in F the "forbidden" masks?
        # No, we just need to check if S is in the union of "bad" masks for any pair.
        
        # Let's implement a fast check using the counts logic in reverse?
        # Actually, the greedy algorithm maintains counts of "bad" masks.
        # If we remove a mask, we need to decrement counts.
        # Then we can try to add masks.
        
        # Let's maintain the counts array properly.
        
        counts = [0] * (1 << n)
        current_family = list(best_family)
        
        # Rebuild counts for current_family
        for idx, mask in enumerate(current_family):
            for A in current_family[:idx]:
                J = A & mask
                U = all_ones ^ (A | mask)
                need = k - popcount[J]
                for Y in submasks_by_size[U][need]:
                    M_prime = J | Y
                    counts[M_prime] += 1
        
        # Now counts[mask] > 0 means mask conflicts with some pair in current_family.
        # Wait, the greedy logic was:
        # When adding M, for each existing A:
        #   J = A & M
        #   U = all_ones ^ (A | M)
        #   need = k - popcount(J)
        #   For each Y in submasks(U, need):
        #       M_prime = J | Y
        #       counts[M_prime] += 1
        #       (This M_prime is a set that would form a sunflower with A and M)
        
        # So if I want to check if a new set S can be added to current_family:
        # I need to check if counts[S] == 0.
        
        # If I remove a set R from current_family:
        # I need to decrement counts for all pairs involving R.
        # And for all pairs not involving R, counts remain valid.
        
        # Let's try this local search.
        
        # We need to be careful with time.
        # Each swap attempt involves:
        # 1. Removing R: O(F * AvgSubmasks) decrements.
        # 2. Trying to add S: O(1) check (counts[S] == 0).
        # 3. If added, updating counts for S: O(F * AvgSubmasks) increments.
        
        # Let's try a few iterations.
        
        improved = True
        max_local_steps = 200
        step = 0
        
        while improved and step < max_local_steps and time.time() < deadline - 2:
            improved = False
            step += 1
            
            # Try to swap a random element
            if not current_family:
                break
                
            idx_to_remove = random.randint(0, len(current_family) - 1)
            R = current_family[idx_to_remove]
            
            # Remove R
            # Find index in list
            # Since we are modifying, let's just pop and handle counts.
            
            # Decrement counts for pairs involving R
            # We need to iterate over all other sets A in current_family
            for A in current_family:
                if A == R:
                    continue
                J = A & R
                U = all_ones ^ (A | R)
                need = k - popcount[J]
                for Y in submasks_by_size[U][need]:
                    M_prime = J | Y
                    counts[M_prime] -= 1
            
            current_family.pop(idx_to_remove)
            
            # Now try to add random candidates from outside
            # We need a list of candidates.
            # We can pick random masks from all_masks and check if they are not in current_family and counts[mask]==0.
            
            # To avoid picking from current_family, we can just check membership.
            # But picking random from 1820 and checking membership is O(1) with a set.
            
            family_set = set(current_family)
            
            added = False
            # Try up to 100 random candidates
            for _ in range(100):
                S = random.choice(all_masks)
                if S in family_set:
                    continue
                if counts[S] == 0:
                    # Can add S
                    current_family.append(S)
                    family_set.add(S)
                    
                    # Update counts for S
                    for A in current_family[:-1]:
                        J = A & S
                        U = all_ones ^ (A | S)
                        need = k - popcount[J]
                        for Y in submasks_by_size[U][need]:
                            M_prime = J | Y
                            counts[M_prime] += 1
                            
                    added = True
                    break
            
            if added:
                improved = True
                if len(current_family) > best_size:
                    best_size = len(current_family)
                    best_family = current_family[:]
                    if best_size >= 250:
                        break
            else:
                # Revert removal
                current_family.append(R)
                family_set.add(R)
                
                # Increment counts for pairs involving R
                for A in current_family:
                    if A == R:
                        continue
                    J = A & R
                    U = all_ones ^ (A | R)
                    need = k - popcount[J]
                    for Y in submasks_by_size[U][need]:
                        M_prime = J | Y
                        counts[M_prime] += 1

    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result

