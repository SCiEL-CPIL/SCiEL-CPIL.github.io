"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def is_valid_addition(s, candidate):
        """Check if adding candidate to s keeps the set valid.
        A triple (a,b,c) with a<b<c is forbidden if a+b, a+c, b+c are all in the set.
        Adding candidate x creates new triples where x is one of the three members.
        """
        # For each pair (a, b) in s with a < b, check if adding x would create a forbidden triple
        # Triple must have all three pair-sums in the set (including x itself potentially as a sum)
        # New triples involving x:
        # Case 1: x is the largest member: (a, b, x) where a+b, a+x, b+x all in s∪{x}
        # Case 2: x is the middle member: (a, x, b) where a<b, a+x, x+b, a+b all in s∪{x}
        # Case 3: x is the smallest member: (x, a, b) where x<a<b, x+a, x+b, a+b all in s∪{x}
        
        for a in s:
            if a == candidate:
                continue
            for b in s:
                if b <= a:
                    continue
                # Check triple (a, b, candidate)
                # The three pair sums are: a+b, a+candidate, b+candidate
                # All must be in s ∪ {candidate}
                if (a + b) in s and (a + candidate) in s and (b + candidate) in s:
                    return False
        return True

    def is_valid_addition_fast(s_set, s_list, candidate):
        """Faster check using precomputed sorted list."""
        for i in range(len(s_list)):
            a = s_list[i]
            if a == candidate:
                continue
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                if (a + b) in s_set and (a + candidate) in s_set and (b + candidate) in s_set:
                    return False
        return True

    base_A = set(range(65, 129))
    base_B = set(range(17, 33))
    all_candidates = list(range(1, 129))
    
    # Remove elements already in base
    base_set = base_A | base_B
    remaining = [x for x in all_candidates if x not in base_set]

    best_set = set(base_set)
    best_size = len(best_set)

    # Multiple greedy strategies with different orderings
    rng = random.Random(42)
    
    strategies = []
    # Ascending
    strategies.append(sorted(remaining))
    # Descending
    strategies.append(sorted(remaining, reverse=True))
    # Small first then large
    mid = len(remaining) // 2
    small = sorted(remaining[:mid])
    large = sorted(remaining[mid:], reverse=True)
    strategies.append(small + large)
    strategies.append(large + small)
    # Random orders
    for _ in range(60):
        c = remaining[:]
        rng.shuffle(c)
        strategies.append(c)

    for strat in strategies:
        if time.time() > deadline - 5:
            break
        S = set(base_set)
        S_list = sorted(S)
        for x in strat:
            if time.time() > deadline - 5:
                break
            if x in S:
                continue
            # Fast check
            valid = True
            for a in S_list:
                if a == x:
                    continue
                for b in S_list:
                    if b <= a:
                        continue
                    if (a + b) in S and (a + x) in S and (b + x) in S:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(x)
                S_list.append(x)
                S_list.sort()
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Local improvement: swap operations
    if time.time() < deadline - 3:
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            S = set(best_set)
            S_list = sorted(S)
            not_in_S = [x for x in range(1, 129) if x not in S]
            
            for x in S_list:
                if time.time() > deadline - 3:
                    break
                S_remove = S - {x}
                S_remove_list = [v for v in S_list if v != x]
                
                # Try adding one element
                for y in not_in_S:
                    if time.time() > deadline - 3:
                        break
                    # Check if y can be added to S_remove
                    valid_y = True
                    for a in S_remove_list:
                        for b in S_remove_list:
                            if b <= a:
                                continue
                            if (a + b) in S_remove and (a + y) in S_remove and (b + y) in S_remove:
                                valid_y = False
                                break
                        if not valid_y:
                            break
                    if not valid_y:
                        continue
                    
                    S_temp = S_remove | {y}
                    S_temp_list = S_remove_list + [y]
                    S_temp_list.sort()
                    
                    # Try adding a second element z
                    for z in not_in_S:
                        if z == y:
                            continue
                        if z in S_temp:
                            continue
                        # Check if z can be added to S_temp
                        valid_z = True
                        for a in S_temp_list:
                            for b in S_temp_list:
                                if b <= a:
                                    continue
                                if (a + b) in S_temp and (a + z) in S_temp and (b + z) in S_temp:
                                    valid_z = False
                                    break
                            if not valid_z:
                                break
                        if valid_z:
                            S_new = S_temp | {z}
                            if len(S_new) > best_size:
                                best_set = S_new
                                best_size = len(S_new)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # Also try 1-for-1 swaps (remove x, add y, same size but maybe enables future growth)
            if not improved and time.time() < deadline - 3:
                for x in S_list:
                    if time.time() > deadline - 3:
                        break
                    S_remove = S - {x}
                    S_remove_list = [v for v in S_list if v != x]
                    for y in not_in_S:
                        if time.time() > deadline - 3:
                            break
                        valid_y = True
                        for a in S_remove_list:
                            for b in S_remove_list:
                                if b <= a:
                                    continue
                                if (a + b) in S_remove and (a + y) in S_remove and (b + y) in S_remove:
                                    valid_y = False
                                    break
                            if not valid_y:
                                break
                        if valid_y:
                            S_new = S_remove | {y}
                            # Check if this enables adding more
                            S_new_list = sorted(S_new)
                            added = 0
                            for w in not_in_S:
                                if w == y or w in S_new:
                                    continue
                                valid_w = True
                                for a in S_new_list:
                                    for b in S_new_list:
                                        if b <= a:
                                            continue
                                        if (a + b) in S_new and (a + w) in S_new and (b + w) in S_new:
                                            valid_w = False
                                            break
                                    if not valid_w:
                                        break
                                if valid_w:
                                    S_new.add(w)
                                    S_new_list.append(w)
                                    S_new_list.sort()
                                    added += 1
                            if len(S_new) > best_size:
                                best_set = S_new
                                best_size = len(S_new)
                                improved = True
                                break
                    if improved:
                        break

    result = [int(x) for x in sorted(best_set)]
    return result
# EVOLVE-BLOCK-END
