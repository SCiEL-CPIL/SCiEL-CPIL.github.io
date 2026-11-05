import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start = time.time()
    DEADLINE = start + 55.0

    # All 4-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        m = 0
        for x in combo:
            m |= (1 << x)
        all_sets.append(m)
    
    n_all = len(all_sets)  # 1820
    
    def mask_to_list(m):
        return sorted((i+1) for i in range(16) if (m >> i) & 1)
    
    def build_pair_inter(F):
        """Build dict mapping (i,j) -> F[i]&F[j] for i<j"""
        pairs = {}
        for i in range(len(F)):
            for j in range(i+1, len(F)):
                pairs[(i,j)] = F[i] & F[j]
        return pairs
    
    def can_add_with_pairs(C, F, pair_inter):
        """Check if C can be added, using precomputed pair intersections."""
        for (i,j), v in pair_inter.items():
            if (F[i] & C) == v and (F[j] & C) == v:
                return False, (i, j)
        return True, None
    
    def can_add_except_with_pairs(C, F, pair_inter, excluded_idx):
        """Check if C can be added when excluding element at excluded_idx."""
        for (i,j), v in pair_inter.items():
            if i == excluded_idx or j == excluded_idx:
                continue
            if (F[i] & C) == v and (F[j] & C) == v:
                return False, (i, j)
        return True, None
    
    def can_add_two_except(C, F, pair_inter, excl1, excl2):
        """Check if C can be added when excluding two elements."""
        for (i,j), v in pair_inter.items():
            if i in (excl1, excl2) or j in (excl1, excl2):
                continue
            if (F[i] & C) == v and (F[j] & C) == v:
                return False, (i, j)
        return True, None
    
    def update_pair_inter_after_swap(F, pair_inter, old_idx, new_elem):
        """Update pair_inter after replacing F[old_idx] with new_elem."""
        n = len(F)
        for i in range(n):
            if i == old_idx:
                continue
            if i < old_idx:
                pair_inter[(i, old_idx)] = F[i] & new_elem
                pair_inter[(old_idx, i)] = F[i] & new_elem
            else:
                pair_inter[(old_idx, i)] = F[i] & new_elem
                pair_inter[(i, old_idx)] = F[i] & new_elem
    
    def update_pair_inter_after_add(F, pair_inter, new_elem):
        """Update pair_inter after appending new_elem to F."""
        n_old = len(F) - 1
        for i in range(n_old):
            pair_inter[(i, n_old)] = F[i] & new_elem
            pair_inter[(n_old, i)] = F[i] & new_elem
    
    def update_pair_inter_after_remove(F, pair_inter, rem_idx):
        """Update pair_inter after removing element at rem_idx from F.
        Note: F has already been modified (element removed)."""
        n = len(F)
        # Remove all pairs involving rem_idx
        to_del = []
        for key in pair_inter:
            if key[0] == rem_idx or key[1] == rem_idx:
                to_del.append(key)
        for key in to_del:
            del pair_inter[key]
        # Re-index pairs where both indices > rem_idx
        new_pairs = {}
        for (i, j), v in pair_inter.items():
            ni = i if i < rem_idx else i - 1
            nj = j if j < rem_idx else j - 1
            new_pairs[(ni, nj)] = v
        pair_inter.clear()
        pair_inter.update(new_pairs)
    
    def greedy_pass(order, deadline):
        family = []
        pair_inter = {}
        for C in order:
            if time.time() > deadline:
                break
            valid, _ = can_add_with_pairs(C, family, pair_inter)
            if valid:
                family.append(C)
                update_pair_inter_after_add(family, pair_inter, C)
        return family, pair_inter
    
    def greedy_pass_simple(order, deadline):
        """Simpler greedy for small families or when pair_inter isn't maintained."""
        family = []
        for C in order:
            if time.time() > deadline:
                break
            valid = True
            for i in range(len(family)):
                for j in range(i+1, len(family)):
                    v = family[i] & family[j]
                    if (family[i] & C) == v and (family[j] & C) == v:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                family.append(C)
        return family
    
    best_family = []
    best_size = 0
    
    # Phase 1: Multiple greedy passes with different orderings
    # Ordering strategies:
    # 1. Lexicographic (natural)
    # 2. Reverse lexicographic
    # 3. Random
    # 4. Sorted by "diversity" - sets that differ from each other
    
    # Natural order
    fam, pi = greedy_pass(all_sets, DEADLINE - 10.0)
    if len(fam) > best_size:
        best_family = fam
        best_size = len(fam)
    
    # Reverse order
    if time.time() < DEADLINE - 10.0:
        fam, pi = greedy_pass(all_sets[::-1], DEADLINE - 10.0)
        if len(fam) > best_size:
            best_family = fam
            best_size = len(fam)
    
    # Random orderings
    for _ in range(8):
        if time.time() > DEADLINE - 10.0:
            break
        order = all_sets[:]
        random.shuffle(order)
        fam, pi = greedy_pass(order, DEADLINE - 10.0)
        if len(fam) > best_size:
            best_family = fam
            best_size = len(fam)
    
    # Phase 2: Local search on best family
    def local_search(family, pair_inter, deadline):
        family = list(family)
        pair_inter = dict(pair_inter)  # copy
        in_family = set(family)
        candidates = [s for s in all_sets if s not in in_family]
        random.shuffle(candidates)
        
        no_improve_count = 0
        max_no_improve = 300
        temp = 1.0  # temperature for simulated annealing-like behavior
        best_local = len(family)
        
        while time.time() < deadline and candidates:
            if no_improve_count > max_no_improve:
                # Restart with fresh candidate ordering
                random.shuffle(candidates)
                no_improve_count = 0
                if len(family) < best_local:
                    # Accept a worse state temporarily (annealing)
                    pass
                max_no_improve = max(100, max_no_improve // 2)
            
            idx = random.randint(0, len(candidates) - 1)
            C = candidates[idx]
            
            valid, block_pair = can_add_with_pairs(C, family, pair_inter)
            
            if valid:
                # Add it
                old_len = len(family)
                family.append(C)
                in_family.add(C)
                update_pair_inter_after_add(family, pair_inter, C)
                candidates.pop(idx)
                no_improve_count = 0
                if len(family) > best_local:
                    best_local = len(family)
                    max_no_improve = 300  # Reset patience
            else:
                # Try to resolve the conflict
                if block_pair is None:
                    no_improve_count += 1
                    continue
                    
                A_idx, B_idx = block_pair
                A = family[A_idx]
                B = family[B_idx]
                
                improved = False
                
                # Try swapping A with C
                valid_swap, _ = can_add_except_with_pairs(C, family, pair_inter, A_idx)
                if valid_swap:
                    family[A_idx] = C
                    in_family.discard(A)
                    in_family.add(C)
                    update_pair_inter_after_swap(family, pair_inter, A_idx, C)
                    candidates.pop(idx)
                    candidates.append(A)
                    improved = True
                    no_improve_count = 0
                    if len(family) > best_local:
                        best_local = len(family)
                else:
                    # Try swapping B with C
                    valid_swap, _ = can_add_except_with_pairs(C, family, pair_inter, B_idx)
                    if valid_swap:
                        family[B_idx] = C
                        in_family.discard(B)
                        in_family.add(C)
                        update_pair_inter_after_swap(family, pair_inter, B_idx, C)
                        candidates.pop(idx)
                        candidates.append(B)
                        improved = True
                        no_improve_count = 0
                        if len(family) > best_local:
                            best_local = len(family)
                    else:
                        # Try 2-for-1 swap: remove both A and B, add C
                        if len(family) >= 3:
                            valid_2for1, _ = can_add_two_except(C, family, pair_inter, A_idx, B_idx)
                            if valid_2for1:
                                # Remove B first (higher index), then A
                                # Actually, let's do it carefully
                                # Save the elements
                                A_save = A
                                B_save = B
                                
                                # Remove B_idx first
                                del family[B_idx]
                                in_family.discard(B)
                                update_pair_inter_after_remove(family, pair_inter, B_idx)
                                
                                # Adjust A_idx if needed
                                if A_idx > B_idx:
                                    A_idx -= 1
                                
                                # Remove A_idx
                                del family[A_idx]
                                in_family.discard(A)
                                update_pair_inter_after_remove(family, pair_inter, A_idx)
                                
                                # Add C
                                family.append(C)
                                in_family.add(C)
                                update_pair_inter_after_add(family, pair_inter, C)
                                
                                candidates.pop(idx)
                                candidates.append(A_save)
                                candidates.append(B_save)
                                improved = True
                                no_improve_count = 0
                                # Note: family size decreased by 1, so this is a "worse" move
                                # but might unlock better configurations
                        else:
                            # Try random swap with any member
                            if no_improve_count % 3 == 0 and len(family) > 2:
                                rand_idx = random.randint(0, len(family)-1)
                                if rand_idx != A_idx and rand_idx != B_idx:
                                    R = family[rand_idx]
                                    valid_swap, _ = can_add_except_with_pairs(C, family, pair_inter, rand_idx)
                                    if valid_swap:
                                        family[rand_idx] = C
                                        in_family.discard(R)
                                        in_family.add(C)
                                        update_pair_inter_after_swap(family, pair_inter, rand_idx, C)
                                        candidates.pop(idx)
                                        candidates.append(R)
                                        improved = True
                                        no_improve_count = 0
                        if not improved:
                            no_improve_count += 1
                            
        return family, pair_inter
    
    # Run local search on best family
    if best_family:
        # Rebuild pair_inter for best_family
        best_pair_inter = build_pair_inter(best_family)
        improved_family, improved_pi = local_search(best_family, best_pair_inter, DEADLINE - 5.0)
        if len(improved_family) > len(best_family):
            best_family = improved_family
            best_size = len(improved_family)
    
    # Phase 3: More greedy + local search rounds
    if time.time() < DEADLINE - 8.0:
        for _ in range(6):
            if time.time() > DEADLINE - 8.0:
                break
            order = all_sets[:]
            random.shuffle(order)
            fam, pi = greedy_pass(order, DEADLINE - 8.0)
            if len(fam) > len(best_family):
                best_family = fam
                best_size = len(fam)
            
            # Try to improve this family
            if time.time() < DEADLINE - 5.0:
                improved_fam, improved_pi = local_search(best_family, build_pair_inter(best_family), DEADLINE - 5.0)
                if len(improved_fam) > len(best_family):
                    best_family = improved_fam
                    best_size = len(improved_fam)
    
    # Phase 4: Final polish - try to add any remaining candidates
    if time.time() < DEADLINE - 2.0 and best_family:
        in_family = set(best_family)
        remaining = [s for s in all_sets if s not in in_family]
        random.shuffle(remaining)
        best_pair_inter = build_pair_inter(best_family)
        for C in remaining:
            if time.time() > DEADLINE - 1.0:
                break
            valid, _ = can_add_with_pairs(C, best_family, best_pair_inter)
            if valid:
                best_family.append(C)
                update_pair_inter_after_add(best_family, best_pair_inter, C)
    
    return [mask_to_list(m) for m in best_family]
