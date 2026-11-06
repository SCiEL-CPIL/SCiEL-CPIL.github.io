import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit

    # Precompute all 4-subsets of [1,16] as bitmasks
    all_masks = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)

    N = len(all_masks)  # 1820
    mask_to_idx = {m: i for i, m in enumerate(all_masks)}

    # Precompute subsets of each mask to help with core analysis
    # For each mask C, we want to know for each K subset of C,
    # which sets A have A & C == K.
    
    # sets_with_core[C_idx][K_mask] = list of A_idx such that all_masks[A_idx] & all_masks[C_idx] == K_mask
    # Note: K_mask must be a subset of C_mask.
    
    # To speed up, we can just iterate all pairs (A, C) once?
    # N=1820, N^2 ~ 3.3M. This is fast.
    # But we need to group by K.
    
    # Let's build a structure:
    # core_groups[C_idx] = { K_mask: [A_idx, ...] }
    
    core_groups = [dict() for _ in range(N)]
    
    for c_idx in range(N):
        C = all_masks[c_idx]
        for a_idx in range(N):
            if a_idx == c_idx:
                continue
            A = all_masks[a_idx]
            K = A & C
            if K not in core_groups[c_idx]:
                core_groups[c_idx][K] = []
            core_groups[c_idx][K].append(a_idx)

    def is_sunflower(a, b, c):
        """Check if three 4-element sets (as bitmasks) form a sunflower."""
        ab = a & b
        ac = a & c
        bc = b & c
        return ab == ac == bc

    def can_add_fast(candidate_idx, family_idx_set, family_list):
        """
        Check if candidate_idx can be added to family.
        family_idx_set: set of indices in family for O(1) lookup? 
        Actually, we need to check pairs in family.
        
        Using core_groups:
        For candidate C (idx c_idx), we check if there exist A, B in family
        such that (A, B, C) is a sunflower.
        
        This happens if there is a core K subset of C such that:
        1. A, B are in core_groups[c_idx][K]
        2. A, B are in family
        3. (A \ K) & (B \ K) == 0  (which is equivalent to A & B == K)
        
        Wait, if A & C = K and B & C = K, then A & B == K is NOT automatically true.
        We need A & B == K.
        So we need to check pairs in the list L = [A in family if A in core_groups[c_idx][K]].
        For each pair A, B in L, check if A & B == K.
        
        Optimization:
        If |K| == 3: A = K U {a}, B = K U {b}. A & B == K iff a != b.
        So if there are >= 2 distinct sets in L, they form a sunflower (since A!=B implies a!=b).
        
        If |K| == 2: A = K U {a, b}, B = K U {c, d}.
        A & B == K iff {a, b} & {c, d} is empty.
        
        If |K| == 1: A = K U {a, b, c}, B = K U {d, e, f}.
        A & B == K iff tails are disjoint.
        
        If |K| == 0: A, B disjoint from C. A & B == 0.
        """
        C = all_masks[candidate_idx]
        
        # Iterate over all possible cores K
        # K must be a subset of C.
        # We can just iterate over the keys in core_groups[c_idx]
        for K, list_A in core_groups[candidate_idx].items():
            # Filter list_A to only those in family
            # Since family is a list, checking membership is O(|F|) unless we use a set.
            # We maintain family_set for O(1) membership.
            
            # How many sets in list_A are in family?
            # We can precompute the count? No, family changes.
            
            # If the number of sets in list_A that are in family is small, we can check pairs.
            # Worst case list_A size is ~ N/16?
            
            # Let's collect the family members from this group
            fam_in_group = [a for a in list_A if a in family_idx_set]
            
            if len(fam_in_group) < 2:
                continue
                
            # Check if any pair in fam_in_group has intersection K
            # This is equivalent to (A & B) == K
            # Note: A & C = K and B & C = K implies (A & B) is a superset of K?
            # No. A & B can contain elements not in C.
            # But we require A & B == K.
            # So (A & B) & ~C must be 0?
            # Actually, if A & B == K, then A & B is a subset of C (since K is subset of C).
            # So (A & B) & ~C == 0 is necessary.
            
            # Let's just check the condition directly for pairs.
            # To speed up, we can use the size of K.
            k_size = bin(K).count('1')
            
            if k_size == 3:
                # Any two distinct sets in fam_in_group work?
                # A = K U {a}, B = K U {b}.
                # A & B = K U ({a} & {b}).
                # Since a, b are in [16]\C, and A!=B, a!=b.
                # So {a} & {b} is empty.
                # So A & B = K.
                # So yes, any 2 distinct sets in fam_in_group form a sunflower.
                return False
            
            elif k_size == 2:
                # Check pairs
                # A = K U {a, b}
                # We need (A & B) == K.
                # This means (A & B) & ~K == 0.
                # Let A_tail = A & ~K. B_tail = B & ~K.
                # We need A_tail & B_tail == 0.
                # Since A_tail is 2 bits, we can precompute a bitmask of tails?
                # Or just check pairs.
                # Number of pairs is (M choose 2).
                # M is size of fam_in_group.
                # If M is large, this is slow.
                
                # Optimization: 
                # For each A in fam_in_group, compute tail_A = A & ~K.
                # We need to find if there exists B in fam_in_group such that tail_B & tail_A == 0.
                # We can store tails in a set.
                # If we see a tail T, we check if any previous tail T' has T & T' == 0.
                
                seen_tails = set()
                for a in fam_in_group:
                    A = all_masks[a]
                    tail_A = A & ~K
                    # Check if any seen tail is disjoint from tail_A
                    for t in seen_tails:
                        if tail_A & t == 0:
                            return False
                    seen_tails.add(tail_A)
                
            elif k_size == 1:
                # Tail size 3.
                seen_tails = set()
                for a in fam_in_group:
                    A = all_masks[a]
                    tail_A = A & ~K
                    for t in seen_tails:
                        if tail_A & t == 0:
                            return False
                    seen_tails.add(tail_A)
                    
            elif k_size == 0:
                # Tail size 4. A, B disjoint from C and each other.
                # A & B == 0.
                seen_tails = set()
                for a in fam_in_group:
                    A = all_masks[a]
                    tail_A = A & ~K # which is A
                    for t in seen_tails:
                        if tail_A & t == 0:
                            return False
                    seen_tails.add(tail_A)
                    
            else:
                # k_size 4 is impossible for distinct sets
                pass

        return True

    def greedy_build_deterministic(deadline):
        """Build family greedily in lexicographic order."""
        family = []
        family_set = set()
        for idx in range(N):
            if time.time() > deadline:
                break
            if can_add_fast(idx, family_set, family):
                family.append(idx)
                family_set.add(idx)
        return family, family_set

    def local_search_swap(family, family_set, deadline, max_steps=20000):
        """Local search: try to add sets by removing conflicting sets."""
        family = family[:]
        family_set = family_set.copy()
        best_size = len(family)
        steps_no_improve = 0
        random.seed(42) # For reproducibility in local search if needed, but random is fine
        
        # Precompute a list of all sets not in family to sample from?
        # Or just sample random index and check if in family.
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if steps_no_improve > 2000:
                break

            idx = random.randint(0, N - 1)

            if idx in family_set:
                steps_no_improve += 1
                continue

            # Check if idx can be added
            if can_add_fast(idx, family_set, family):
                family.append(idx)
                family_set.add(idx)
                if len(family) > best_size:
                    best_size = len(family)
                    steps_no_improve = 0
                else:
                    steps_no_improve += 1
            else:
                # Find a blocker pair to remove
                # We need to find A, B in family such that (A, B, idx) is sunflower.
                # We can reuse the logic from can_add_fast to find the first blocker.
                
                C = all_masks[idx]
                blocker_found = False
                for K, list_A in core_groups[idx].items():
                    fam_in_group = [a for a in list_A if a in family_set]
                    if len(fam_in_group) < 2:
                        continue
                    
                    k_size = bin(K).count('1')
                    
                    if k_size == 3:
                        # Any pair works
                        A_idx = fam_in_group[0]
                        B_idx = fam_in_group[1]
                        # Remove one
                        victim = A_idx
                        family.remove(victim)
                        family_set.remove(victim)
                        
                        if can_add_fast(idx, family_set, family):
                            family.append(idx)
                            family_set.add(idx)
                            # Try to add more
                            for _ in range(2):
                                if time.time() > deadline: break
                                idx2 = random.randint(0, N - 1)
                                if idx2 not in family_set and can_add_fast(idx2, family_set, family):
                                    family.append(idx2)
                                    family_set.add(idx2)
                            steps_no_improve = 0
                            blocker_found = True
                            break
                        else:
                            # Restore
                            family.append(victim)
                            family_set.add(victim)
                            # Try removing B_idx
                            victim = B_idx
                            family.remove(victim)
                            family_set.remove(victim)
                            if can_add_fast(idx, family_set, family):
                                family.append(idx)
                                family_set.add(idx)
                                for _ in range(2):
                                    if time.time() > deadline: break
                                    idx2 = random.randint(0, N - 1)
                                    if idx2 not in family_set and can_add_fast(idx2, family_set, family):
                                        family.append(idx2)
                                        family_set.add(idx2)
                                steps_no_improve = 0
                                blocker_found = True
                                break
                            else:
                                family.append(victim)
                                family_set.add(victim)
                        break # Break K loop, we tried this K
                    
                    elif k_size in [0, 1, 2]:
                        # Find a pair A, B in fam_in_group that conflict
                        # We need to find A, B such that (A&B)==K
                        # Then remove A or B.
                        found_pair = False
                        for i in range(len(fam_in_group)):
                            if found_pair: break
                            A_idx = fam_in_group[i]
                            A = all_masks[A_idx]
                            for j in range(i+1, len(fam_in_group)):
                                B_idx = fam_in_group[j]
                                B = all_masks[B_idx]
                                if (A & B) == K:
                                    # Found conflict pair
                                    # Try removing A
                                    victim = A_idx
                                    family.remove(victim)
                                    family_set.remove(victim)
                                    if can_add_fast(idx, family_set, family):
                                        family.append(idx)
                                        family_set.add(idx)
                                        for _ in range(2):
                                            if time.time() > deadline: break
                                            idx2 = random.randint(0, N - 1)
                                            if idx2 not in family_set and can_add_fast(idx2, family_set, family):
                                                family.append(idx2)
                                                family_set.add(idx2)
                                        steps_no_improve = 0
                                        found_pair = True
                                        blocker_found = True
                                        break
                                    else:
                                        family.append(victim)
                                        family_set.add(victim)
                                        # Try removing B
                                        victim = B_idx
                                        family.remove(victim)
                                        family_set.remove(victim)
                                        if can_add_fast(idx, family_set, family):
                                            family.append(idx)
                                            family_set.add(idx)
                                            for _ in range(2):
                                                if time.time() > deadline: break
                                                idx2 = random.randint(0, N - 1)
                                                if idx2 not in family_set and can_add_fast(idx2, family_set, family):
                                                    family.append(idx2)
                                                    family_set.add(idx2)
                                            steps_no_improve = 0
                                            found_pair = True
                                            blocker_found = True
                                            break
                                        else:
                                            family.append(victim)
                                            family_set.add(victim)
                                    break # Break inner j loop
                        if found_pair:
                            break # Break K loop
                        # If no pair found in this K, continue to next K?
                        # Actually if no pair found, this K doesn't block.
                        # But can_add_fast said it's blocked, so some K must block.
                        # So we should find it.
                        # If we reach here, we failed to find a blocker in this K, 
                        # but can_add_fast returned False.
                        # This implies logic error or we missed a case.
                        # For now, assume we found it or break.
                        if not found_pair:
                             # Fallback: just increment no_improve
                             pass

                if not blocker_found:
                    steps_no_improve += 1

        return family, family_set

    def try_add_all_remaining(family, family_set, deadline):
        """Try to add all remaining sets that can be added."""
        family = family[:]
        family_set = family_set.copy()
        added_any = True
        while added_any and time.time() < deadline:
            added_any = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                if idx not in family_set and can_add_fast(idx, family_set, family):
                    family.append(idx)
                    family_set.add(idx)
                    added_any = True
        return family, family_set

    best_family = []
    best_family_set = set()
    best_size = 0

    # Phase 1: Deterministic greedy
    family, family_set = greedy_build_deterministic(deadline - 15)
    if len(family) > best_size:
        best_size = len(family)
        best_family = family[:]
        best_family_set = family_set.copy()

    # Phase 2: Local search
    if time.time() < deadline - 5:
        improved_family, improved_set = local_search_swap(best_family, best_family_set, deadline - 5, max_steps=10000)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family = improved_family[:]
            best_family_set = improved_set.copy()

    # Phase 3: Try to add any remaining sets
    if time.time() < deadline - 2:
        final_family, final_set = try_add_all_remaining(best_family, best_family_set, deadline - 2)
        if len(final_family) > best_size:
            best_size = len(final_family)
            best_family = final_family
            best_family_set = final_set

    # Convert bitmasks back to sorted lists
    result = []
    for idx in best_family:
        m = all_masks[idx]
        s = [i for i in range(1, 17) if (m >> i) & 1]
        result.append(sorted(s))

    return result
