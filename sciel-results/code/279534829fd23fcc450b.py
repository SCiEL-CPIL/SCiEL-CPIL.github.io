"""
Program for large integer sets avoiding equal-gcd triples in [1,128].
Strategy: Simulated Annealing with precomputed conflict structures.
Uses O(1) move evaluation for single-bit flips and O(N) for swaps.
"""

import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin from 90s limit

    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCD matrix
    gcd_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            gcd_mat[i][j] = math.gcd(elements[i], elements[j])

    # Precompute conflict counts and neighbor lists
    # conflict_count[i] = number of pairs (j, k) in [0, N-1] such that {i, j, k} is a forbidden triple
    # Note: j, k are distinct and different from i.
    # For a set S, the number of violations involving i is:
    # sum_{j in S, j != i} count_of_k_in_S_such_that_{i,j,k}_is_bad
    # This is hard to do O(1) for arbitrary S.
    # However, we can precompute for each pair (i, j) the list of k's that form a bad triple with i and j.
    # Let bad_k[i][j] be a bitmask of k's such that gcd(i,j) == gcd(i,k) == gcd(j,k).
    
    # This is N*N bitmasks, each N bits. 128*128 = 16384 integers. Very manageable.
    bad_k = [[0] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            g = gcd_mat[i][j]
            mask = 0
            for k in range(N):
                if k == i or k == j:
                    continue
                if gcd_mat[i][k] == g and gcd_mat[j][k] == g:
                    mask |= (1 << k)
            bad_k[i][j] = mask

    # Helper: Count violations in a mask
    # Violations are triples {a, b, c} in mask that are bad.
    # To count total violations efficiently for a move:
    # If we add x to S:
    #   New violations = number of pairs (a, b) in S such that {x, a, b} is bad.
    #   This is sum_{a in S} popcount(bad_k[x][a] & S)
    #   Note: bad_k[x][a] contains b's. Since b must be in S, we AND with S.
    #   We must ensure a != x (true since x not in S) and b != x (true since x not in S).
    #   Also b != a is guaranteed by bad_k definition.
    #   So new_violations = sum_{a in S} popcount(bad_k[x][a] & S)
    
    # If we remove x from S:
    #   Removed violations = number of pairs (a, b) in S\{x} such that {x, a, b} is bad.
    #   This is sum_{a in S\{x}} popcount(bad_k[x][a] & (S\{x}))
    
    # Let's implement a function to calculate the change in violations.
    # For SA, we need to evaluate the "energy" (number of violations).
    # We can maintain the current energy incrementally.
    
    # However, calculating sum_{a in S} popcount(...) is O(|S| * cost_popcount).
    # |S| is around 60-80. popcount is fast (int.bit_count()).
    # So O(80) operations per move. 100,000 moves * 80 ops = 8M ops. Should be fine in Python if optimized.
    
    # Let's precompute the list of indices for each bit in the mask to speed up iteration?
    # Or just iterate bits.
    
    def count_violations_involving_x(S_mask, x):
        """
        Count the number of bad triples {x, a, b} where a, b are in S_mask.
        Assumes x is NOT in S_mask for 'add' calculation, or x IS in S_mask for 'remove' calculation?
        Actually, let's define it as:
        Count pairs (a, b) in S_mask such that {x, a, b} is a bad triple.
        This works whether x is in S_mask or not, as long as a,b are distinct from x and each other.
        bad_k[x][a] contains b. If x is in S_mask, we must ensure a,b != x.
        bad_k[x][a] does NOT contain x (by construction, k != i, j).
        So if x is in S_mask, bad_k[x][a] & S_mask will still only contain valid b's (b != x).
        And a is from S_mask. If a == x, bad_k[x][x] is 0.
        So the formula sum_{a in S_mask} popcount(bad_k[x][a] & S_mask) counts all pairs (a,b) in S_mask 
        that form a bad triple with x.
        """
        total = 0
        temp = S_mask
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            # bad_k[x][a] contains all b such that {x,a,b} is bad.
            # We need b to be in S_mask.
            # So count bits in (bad_k[x][a] & S_mask)
            total += (bad_k[x][a] & S_mask).bit_count()
            temp ^= lsb
        # Note: This counts each pair (a,b) once for a fixed x.
        # Does it double count? 
        # For a fixed x, we sum over a. For each a, we count b's.
        # The triple is {x, a, b}. The pair (a,b) is unordered in the triple, but here we iterate a.
        # If we swap a and b, we are in a different iteration.
        # Wait. The violation is the triple {x, a, b}.
        # The count of violations *involving* x is exactly the number of such triples.
        # Each triple {x, a, b} is counted once when we pick a (and find b) OR when we pick b (and find a)?
        # No. bad_k[x][a] contains b. bad_k[x][b] contains a.
        # So if we sum over all a in S, we count the pair (a,b) when a is the index, and (b,a) when b is the index?
        # Let's check.
        # Triple {1, 2, 3} is bad.
        # x=1. a=2. bad_k[1][2] contains 3. If 3 in S, count 1.
        # x=1. a=3. bad_k[1][3] contains 2. If 2 in S, count 1.
        # Total count for x=1 is 2?
        # But there is only 1 triple {1,2,3}.
        # The "energy" is usually the number of bad triples.
        # If we define Energy = number of bad triples in S.
        # When adding x to S, the increase in Energy is the number of NEW bad triples formed.
        # New triples must include x.
        # The number of such triples is the number of pairs (a,b) in S such that {x,a,b} is bad.
        # In the sum above, for a fixed pair {a,b}, it is counted twice (once for a, once for b).
        # So we need to divide by 2?
        # Let's verify.
        # Sum_{a in S} popcount(bad_k[x][a] & S).
        # If {x,a,b} is bad, then b in bad_k[x][a] and a in bad_k[x][b].
        # So the pair {a,b} contributes 1 when a is processed (b is in mask) and 1 when b is processed (a is in mask).
        # So the sum is 2 * (number of bad triples involving x).
        # So, number of new violations = Sum // 2.
        
        return total // 2

    # Initial State
    # Start with a greedy set to have a low initial energy (ideally 0)
    # Greedy: Add elements one by one if they don't create violations.
    # To make it fast, we can just start with a random subset or empty and let SA do the work.
    # But starting with 0 violations is much better.
    
    # Let's do a quick greedy to get a valid set.
    current_mask = 0
    # Random order for diversity
    rng = random.Random(42)
    indices = list(range(N))
    rng.shuffle(indices)
    
    for i in indices:
        # Check if adding i creates violations
        # Violations = count_violations_involving_x(current_mask, i)
        # If 0, add it.
        if count_violations_involving_x(current_mask, i) == 0:
            current_mask |= (1 << i)
    
    current_energy = 0 # Since we only added if 0 violations, and previous set had 0, new set has 0.
    current_size = current_mask.bit_count()
    
    best_mask = current_mask
    best_size = current_size
    best_energy = 0
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    # We want to maximize size.
    # Objective: Maximize Size, Minimize Energy.
    # We can define a combined score or use lexicographic ordering.
    # Let's use: Accept if (Size increases) OR (Size same and Energy decreases) OR (Size decreases but Energy increases significantly? No).
    # Standard SA for Max Size with Hard Constraint (Energy=0):
    # If new Energy == 0:
    #   Accept if New Size >= Current Size (or probabilistically if smaller)
    # If new Energy > 0:
    #   Accept with probability exp(-Delta_Energy / T) if it leads to larger size?
    #   Actually, allowing temporary violations is key to escape local optima.
    
    # Let's define the acceptance criterion:
    # We prefer states with Energy=0 and larger Size.
    # We allow moves to Energy>0 if they increase Size, or if they decrease Energy, or randomly with probability based on T.
    
    # A robust metric: Score = Size * 1000 - Energy.
    # We want to maximize Score.
    # Delta_Score = New_Score - Current_Score.
    # If Delta_Score >= 0: Accept.
    # If Delta_Score < 0: Accept with prob exp(Delta_Score / T).
    
    # Since Size is integer and Energy is integer, this works well.
    # However, Energy can be large. 1000 might not be enough if Energy > 1000.
    # Max possible energy? N=128. Max triples ~ 128^3/6 ~ 350k.
    # So 1000 is too small.
    # Let's use Score = Size * 10000 - Energy.
    # Or better, just handle the logic explicitly.
    
    # Logic:
    # If New_Energy == 0 and New_Size > Current_Size: Accept
    # If New_Energy == 0 and New_Size == Current_Size: Accept (or random)
    # If New_Energy == 0 and New_Size < Current_Size: Accept with prob exp(-1/T) ? No, usually we want to keep size.
    # If New_Energy > 0:
    #   If New_Size > Current_Size: Accept with prob exp(-(New_Energy - Current_Energy)/T)
    #   If New_Size == Current_Size: Accept with prob exp(-(New_Energy - Current_Energy)/T)
    #   If New_Size < Current_Size: Unlikely to accept unless T is high.
    
    # Let's stick to the Score method with a large weight.
    # Weight = 10000.
    # Score = Size * 10000 - Energy.
    
    while time.time() < DEADLINE:
        if T < T_min:
            # Reheat or restart?
            # Let's just break and do a final local search or restart.
            # For now, break.
            break
            
        # Generate a move
        # 1. Single bit flip (Add or Remove)
        # 2. Swap (Remove one, Add one)
        
        # Randomly choose move type
        if rng.random() < 0.5:
            # Single bit flip
            # Choose a random index i
            i = rng.randint(0, N - 1)
            
            if current_mask & (1 << i):
                # Remove i
                new_mask = current_mask & ~(1 << i)
                # Energy change:
                # Current Energy includes violations involving i.
                # Violations involving i in current set: count_violations_involving_x(current_mask, i)
                # Note: current_mask contains i.
                # The function count_violations_involving_x(S, x) counts pairs (a,b) in S.
                # If x is in S, it counts pairs (a,b) in S such that {x,a,b} is bad.
                # This is exactly the number of violations that will be REMOVED.
                # So New_Energy = Current_Energy - count_violations_involving_x(current_mask, i)
                removed_violations = count_violations_involving_x(current_mask, i)
                new_energy = current_energy - removed_violations
                new_size = current_size - 1
            else:
                # Add i
                new_mask = current_mask | (1 << i)
                # Energy change:
                # New violations involving i: count_violations_involving_x(current_mask, i)
                # (Since i is not in current_mask, this counts pairs (a,b) in current_mask)
                added_violations = count_violations_involving_x(current_mask, i)
                new_energy = current_energy + added_violations
                new_size = current_size + 1
                
            # Acceptance
            delta_score = (new_size - current_size) * 10000 - (new_energy - current_energy)
            
            if delta_score >= 0:
                current_mask = new_mask
                current_energy = new_energy
                current_size = new_size
            else:
                # Probabilistic accept
                # delta_score is negative
                # prob = exp(delta_score / T)
                # delta_score can be very negative (e.g. -10000 for size drop, or -1000 for energy increase)
                # If delta_score / T < -50, exp is 0.
                if delta_score > -50 * T:
                    prob = math.exp(delta_score / T)
                    if rng.random() < prob:
                        current_mask = new_mask
                        current_energy = new_energy
                        current_size = new_size
        else:
            # Swap: Remove i, Add j (i != j)
            # Choose i from current set, j from complement
            # To be efficient, pick i randomly from set bits, j randomly from 0..N-1
            # If j is in set, retry or swap? No, must be distinct.
            
            # Get list of set bits? Expensive.
            # Just pick random i, if not in set, pick random j, if in set, swap roles?
            # Better: Pick i random. If in set, pick j random. If j in set, reject/try again.
            # If i not in set, pick j random. If j not in set, reject/try again.
            
            # This might be inefficient if set is large/small.
            # Let's just pick two distinct random indices i, j.
            # If both in set or both out, it's not a swap (it's 0 or 2 flips).
            # We want 1 remove, 1 add.
            
            # Efficient way:
            # Pick i random.
            # If i in set:
            #   Pick j random.
            #   If j in set: Continue (not a swap)
            #   If j not in set: Valid Swap (Remove i, Add j)
            # If i not in set:
            #   Pick j random.
            #   If j in set: Valid Swap (Add i, Remove j)
            #   If j not in set: Continue
            
            # Probability of valid swap is roughly 2 * |S|/N * (N-|S|)/N.
            # If |S|=64, prob = 2 * 0.5 * 0.5 = 0.5. So half the time we get a valid swap.
            # Acceptable.
            
            i = rng.randint(0, N - 1)
            j = rng.randint(0, N - 1)
            if i == j:
                continue
                
            i_in = bool(current_mask & (1 << i))
            j_in = bool(current_mask & (1 << j))
            
            if i_in == j_in:
                continue # Not a swap
            
            # Determine remove and add
            if i_in:
                remove_idx = i
                add_idx = j
            else:
                remove_idx = j
                add_idx = i
                
            # Calculate Energy Change
            # New Mask = current_mask ^ (1 << remove_idx) ^ (1 << add_idx)
            
            # Violations removed:
            # 1. Those involving remove_idx and two other elements in (current_mask \ {remove_idx})
            # 2. Those involving add_idx? No, add_idx is not in current set.
            # 3. Those involving remove_idx and add_idx? No, add_idx is not in current set.
            # Wait. The set of violations changes.
            # Old Violations: V(S)
            # New Violations: V(S \ {r} U {a})
            
            # Let's compute V(S \ {r} U {a}) from scratch? O(N) is fine.
            # Or incremental.
            
            # Violations in new set involving a:
            # Pairs (x, y) in (S \ {r}) such that {a, x, y} is bad.
            # S \ {r} is the set without r.
            # count_violations_involving_x(S \ {r}, a)
            
            # Violations in new set involving r: 0 (r is not in new set)
            
            # Violations in new set involving neither a nor r:
            # These are violations in S \ {r} that did not involve a (a wasn't there) and do not involve r.
            # Basically, V(S \ {r}) - V(S \ {r} involving r).
            # But r is not in S \ {r}. So V(S \ {r}) is just the violations of the set S without r.
            
            # So New_Energy = V(S \ {r} U {a})
            # = V(S \ {r}) + V(S \ {r} U {a} involving a) - V(S \ {r} involving a? No a not in S\r)
            # Actually, V(S \ {r} U {a}) = V(S \ {r}) + (number of new triples involving a and two elements from S \ {r})
            # Note: V(S \ {r}) includes triples that might involve a? No, a is not in S \ {r}.
            # So yes, New_Energy = V(S \ {r}) + count_violations_involving_x(S \ {r}, a)
            
            # We need V(S \ {r}).
            # V(S \ {r}) = V(S) - (number of triples in S involving r)
            # Number of triples in S involving r = count_violations_involving_x(S, r)
            
            # So:
            # V(S \ {r}) = current_energy - count_violations_involving_x(current_mask, remove_idx)
            # New_Energy = V(S \ {r}) + count_violations_involving_x(S \ {r}, add_idx)
            
            # We need count_violations_involving_x(S \ {r}, add_idx).
            # S \ {r} is current_mask with remove_idx cleared.
            mask_without_r = current_mask & ~(1 << remove_idx)
            
            v_s_minus_r = current_energy - count_violations_involving_x(current_mask, remove_idx)
            new_violations_a = count_violations_involving_x(mask_without_r, add_idx)
            
            new_energy = v_s_minus_r + new_violations_a
            new_size = current_size # Size remains same
            
            # Acceptance
            delta_score = (new_size - current_size) * 10000 - (new_energy - current_energy)
            # delta_score = - (new_energy - current_energy) = current_energy - new_energy
            
            if delta_score >= 0:
                current_mask = new_mask = (current_mask & ~(1 << remove_idx)) | (1 << add_idx)
                current_energy = new_energy
                # current_size unchanged
            else:
                if delta_score > -50 * T:
                    prob = math.exp(delta_score / T)
                    if rng.random() < prob:
                        current_mask = (current_mask & ~(1 << remove_idx)) | (1 << add_idx)
                        current_energy = new_energy

        # Update Best
        if current_energy == 0 and current_size > best_size:
            best_size = current_size
            best_mask = current_mask
        elif current_energy < best_energy and current_size == best_size:
            # Prefer lower energy if size is same?
            # But we want valid sets (Energy=0).
            # If best_energy is 0, we only care about size.
            # If best_energy > 0, we might want to keep track of best (size, -energy).
            # But the final output must be valid (Energy=0).
            # So we should only update best_mask if current_energy == 0.
            # What if we never find a valid set? We should return the one with lowest energy?
            # The task requires a valid set.
            # So best_mask should always be a valid set found so far.
            pass
            
        # Cooling
        T *= cooling_rate

    # Final Local Search / Cleanup
    # Try to remove elements to reduce energy to 0 if best_mask has energy > 0?
    # But we track best_mask only when energy == 0.
    # So best_mask is always valid.
    
    # If we haven't found any valid set (unlikely, greedy should find one), fallback.
    if best_size == 0:
        # Fallback: Powers of 2
        res = [1, 2, 4, 8, 16, 32, 64, 128]
        return res

    # Convert best_mask to list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(elements[i])
            
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Final size: {len(res)}")
    print(res)
