"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 256
    
    # Precompute for each element x:
    # - partner_mask[x]: bitmask of elements that are "partners" with x in some 3-term AP
    #   i.e., y is a partner of x if there exists z in [1,N] such that {x,y,z} forms a 3-term AP
    # - For validity: x can be added to set S iff for all y in S that are partners of x,
    #   the third element z completing the AP is NOT in S.
    # 
    # Actually, let's use a cleaner formulation:
    # For each x, precompute:
    #   conflict_pairs[x] = list of (y, z) such that x,y,z form a 3-term AP (x is one of the three)
    # But that's expensive. Instead:
    #
    # Key insight: x conflicts with set S iff:
    #   There exist a, b in S such that a, b, x form a 3-term AP.
    # This happens iff:
    #   (a) x is middle: exists a in S such that (2x-a) in S and 2x-a != a and 1 <= 2x-a <= N
    #   (b) x is first: exists b in S such that (2b-x) in S and 2b-x > b and 1 <= 2b-x <= N  
    #   (c) x is last: exists b in S such that (2b-x) in S and 2b-x < b and 1 <= 2b-x <= N
    #
    # For fast checking, precompute for each x:
    #   mid_partners[x] = bitmask where bit a is set if a < x and 2x-a in [1,N] and 2x-a > x
    #   first_partners[x] = bitmask where bit b is set if b > x and 2b-x in [1,N] and 2b-x > b
    #   last_partners[x] = bitmask where bit b is set if b < x and 2b-x in [1,N] and 2b-x < b
    #
    # Then x is valid to add to S (bitmask) iff:
    #   For all a in (S & mid_partners[x]): (2x-a) not in S
    #   For all b in (S & first_partners[x]): (2b-x) not in S
    #   For all b in (S & last_partners[x]): (2b-x) not in S
    #
    # Even faster: precompute for each x, a "conflict mask" that directly tells us
    # which elements in S would conflict. But the mapping from partner to third element
    # isn't a simple mask operation.
    #
    # Let's use a different approach: precompute for each pair (x, y) whether they
    # "conflict" in the sense that there exists z such that x,y,z form a 3-AP.
    # Then x can be added to S iff no y in S conflicts with x... wait, that's not right either.
    # Two elements x and y in S don't necessarily conflict; it's about triples.
    #
    # OK, let's just precompute the conflict pairs as before but use numpy for speed.
    import numpy as np
    
    # For each x, store:
    # mid_pairs[x]: array of (a, c) pairs where a + c = 2x, a < x < c
    # first_pairs[x]: array of (b, c) pairs where x + c = 2b, x < b < c
    # last_pairs[x]: array of (a, b) pairs where a + x = 2b, a < b < x
    
    mid_pairs = [None] * (N + 1)
    first_pairs = [None] * (N + 1)
    last_pairs = [None] * (N + 1)
    
    for x in range(1, N + 1):
        mid_list = []
        for a in range(1, x):
            c = 2 * x - a
            if c <= N and c > x:
                mid_list.append((a, c))
        mid_pairs[x] = np.array(mid_list, dtype=np.int32) if mid_list else np.empty((0, 2), dtype=np.int32)
        
        first_list = []
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c <= N and c > b:
                first_list.append((b, c))
        first_pairs[x] = np.array(first_list, dtype=np.int32) if first_list else np.empty((0, 2), dtype=np.int32)
        
        last_list = []
        for b in range(1, x):
            a = 2 * b - x
            if a >= 1 and a < b:
                last_list.append((a, b))
        last_pairs[x] = np.array(last_list, dtype=np.int32) if last_list else np.empty((0, 2), dtype=np.int32)
    
    def is_valid_addition(bitmask, x):
        """Check if x can be added to set represented by bitmask. Returns bool."""
        if (bitmask >> (x - 1)) & 1:
            return False
        
        # Check mid conflicts: for each (a, c) with a+c=2x, not both a and c in S
        mp = mid_pairs[x]
        if len(mp) > 0:
            a_idx = mp[:, 0] - 1
            c_idx = mp[:, 1] - 1
            a_bits = (bitmask >> a_idx) & 1
            c_bits = (bitmask >> c_idx) & 1
            if np.any(a_bits & c_bits):
                return False
        
        # Check first conflicts: for each (b, c) with x+c=2b, not both b and c in S
        fp = first_pairs[x]
        if len(fp) > 0:
            b_idx = fp[:, 0] - 1
            c_idx = fp[:, 1] - 1
            b_bits = (bitmask >> b_idx) & 1
            c_bits = (bitmask >> c_idx) & 1
            if np.any(b_bits & c_bits):
                return False
        
        # Check last conflicts: for each (a, b) with a+x=2b, not both a and b in S
        lp = last_pairs[x]
        if len(lp) > 0:
            a_idx = lp[:, 0] - 1
            b_idx = lp[:, 1] - 1
            a_bits = (bitmask >> a_idx) & 1
            b_bits = (bitmask >> b_idx) & 1
            if np.any(a_bits & b_bits):
                return False
        
        return True
    
    def reciprocal_sum_float(bitmask):
        """Compute float reciprocal sum from bitmask."""
        total = 0.0
        for x in range(1, N + 1):
            if (bitmask >> (x - 1)) & 1:
                total += 1.0 / x
        return total
    
    def reciprocal_sum_exact(result_list):
        """Compute exact reciprocal sum as Fraction from a list."""
        total = Fraction(0, 1)
        for x in result_list:
            total += Fraction(1, x)
        return total
    
    def greedy_fill(bitmask, order):
        """Greedy fill: add elements in given order if valid."""
        bm = bitmask
        for x in order:
            if (bm >> (x - 1)) & 1:
                continue
            if is_valid_addition(bm, x):
                bm |= (1 << (x - 1))
        return bm
    
    def simulated_annealing(bitmask, time_budget):
        """
        Simulated annealing over bitmask states.
        Moves: add one element, remove one element, swap one element.
        """
        current = bitmask
        current_val = reciprocal_sum_float(current)
        best = bitmask
        best_val = current_val
        
        # Temperature schedule
        T0 = 0.05
        Tend = 0.001
        t_start = time.time()
        
        # Precompute reciprocals
        recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
        
        while (time.time() - t_start) < time_budget:
            # Update temperature
            elapsed = time.time() - t_start
            frac = elapsed / time_budget
            T = T0 * (Tend / T0) ** frac
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random element
                candidates = [x for x in range(1, N + 1) if not ((current >> (x - 1)) & 1)]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if is_valid_addition(current, x):
                    new_bm = current | (1 << (x - 1))
                    new_val = current_val + recip[x]
                    delta = new_val - current_val
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current = new_bm
                        current_val = new_val
                        if current_val > best_val:
                            best = current
                            best_val = current_val
            elif move_type < 0.5:
                # Remove a random element
                in_set = [x for x in range(1, N + 1) if (current >> (x - 1)) & 1]
                if not in_set:
                    continue
                x = random.choice(in_set)
                new_bm = current & ~(1 << (x - 1))
                new_val = current_val - recip[x]
                delta = new_val - current_val
                # Always accept removals in SA (they enable future additions)
                if random.random() < math.exp(delta / T):
                    current = new_bm
                    current_val = new_val
                    if current_val > best_val:
                        best = current
                        best_val = current_val
            else:
                # Swap: remove one, add another
                in_set = [x for x in range(1, N + 1) if (current >> (x - 1)) & 1]
                not_in_set = [x for x in range(1, N + 1) if not ((current >> (x - 1)) & 1)]
                if not in_set or not not_in_set:
                    continue
                rem = random.choice(in_set)
                add = random.choice(not_in_set)
                # Check if add is valid in current without rem
                temp_bm = current & ~(1 << (rem - 1))
                if is_valid_addition(temp_bm, add):
                    new_bm = temp_bm | (1 << (add - 1))
                    new_val = current_val - recip[rem] + recip[add]
                    delta = new_val - current_val
                    if delta > 0 or random.random() < math.exp(delta / T):
                        current = new_bm
                        current_val = new_val
                        if current_val > best_val:
                            best = current
                            best_val = current_val
        
        return best
    
    def local_search_improved(bitmask, max_iter=200, time_limit=None):
        """
        Improved local search: try all possible single additions and 1-swaps.
        """
        if time_limit is None:
            time_limit = DEADLINE
        
        current = bitmask
        current_val = reciprocal_sum_float(current)
        best = bitmask
        best_val = current_val
        
        recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
        
        for iteration in range(max_iter):
            if (time.time() - start_time) > time_limit:
                break
            
            improved = False
            
            # Try additions: find the best valid addition (smallest x not in set)
            for x in range(1, N + 1):
                if (current >> (x - 1)) & 1:
                    continue
                if is_valid_addition(current, x):
                    new_bm = current | (1 << (x - 1))
                    new_val = current_val + recip[x]
                    if new_val > best_val + 1e-15:
                        best_val = new_val
                        best = new_bm
                        current = new_bm
                        current_val = new_val
                        improved = True
                        break
            
            # Try swaps: for each element in set, try replacing with a smaller valid element
            if not improved:
                for elem in range(1, N + 1):
                    if not ((current >> (elem - 1)) & 1):
                        continue
                    
                    best_cand = None
                    for cand in range(1, elem):
                        if (current >> (cand - 1)) & 1:
                            continue
                        temp = current & ~(1 << (elem - 1))
                        if is_valid_addition(temp, cand):
                            best_cand = cand
                            break
                    
                    if best_cand is not None:
                        new_bm = (current & ~(1 << (elem - 1))) | (1 << (best_cand - 1))
                        new_val = current_val - recip[elem] + recip[best_cand]
                        if new_val > best_val + 1e-15:
                            best_val = new_val
                            best = new_bm
                            current = new_bm
                            current_val = new_val
                            improved = True
                            break
            
            if not improved:
                break
        
        return best
    
    # Phase 1: Generate diverse initial solutions using different greedy orders
    random.seed(42)
    orders = [
        list(range(1, N + 1)),  # Ascending
        list(range(N, 0, -1)),  # Descending
    ]
    
    # Add random orders
    for _ in range(8):
        order = list(range(1, N + 1))
        random.shuffle(order)
        orders.append(order)
    
    # Also try orders weighted toward small numbers
    for _ in range(4):
        order = list(range(1, N + 1))
        # Weighted shuffle: small numbers more likely to come first
        weighted = []
        for x in range(1, N + 1):
            weighted.extend([x] * max(1, (N + 1 - x) // 16))
        random.shuffle(weighted)
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for x in weighted:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        orders.append(deduped)
    
    best_bitmask = 0
    best_float = 0.0
    
    for order in orders:
        bm = greedy_fill(0, order)
        fl = reciprocal_sum_float(bm)
        if fl > best_float:
            best_float = fl
            best_bitmask = bm
    
    # Phase 2: Local search on best initial
    best_bitmask = local_search_improved(best_bitmask, max_iter=300)
    best_float = reciprocal_sum_float(best_bitmask)
    
    # Phase 3: Simulated annealing from best
    sa_time_budget = min(15, DEADLINE - 5 - (time.time() - start_time))
    if sa_time_budget > 3:
        sa_result = simulated_annealing(best_bitmask, sa_time_budget)
        sa_float = reciprocal_sum_float(sa_result)
        if sa_float > best_float:
            best_float = sa_float
            best_bitmask = sa_result
    
    # Phase 4: More restarts with perturbation + SA
    while (time.time() - start_time) < DEADLINE - 10:
        # Perturb current best
        current = best_bitmask
        elems = [x for x in range(1, N + 1) if (current >> (x - 1)) & 1]
        
        if not elems:
            break
        
        num_remove = random.randint(3, min(8, len(elems)))
        to_remove = random.sample(elems, num_remove)
        for r in to_remove:
            current &= ~(1 << (r - 1))
        
        # Greedy fill with random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        current = greedy_fill(current, order)
        
        # Local search
        current = local_search_improved(current, max_iter=50)
        fl = reciprocal_sum_float(current)
        
        if fl > best_float + 1e-15:
            best_float = fl
            best_bitmask = current
    
    # Final polish
    remaining = DEADLINE - 2 - (time.time() - start_time)
    if remaining > 5:
        best_bitmask = local_search_improved(best_bitmask, max_iter=500, time_limit=DEADLINE - 2)
    elif remaining > 2:
        # Quick SA polish
        best_bitmask = simulated_annealing(best_bitmask, remaining)
    
    best_float = reciprocal_sum_float(best_bitmask)
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if (best_bitmask >> (x - 1)) & 1]
    
    return result
# EVOLVE-BLOCK-END
