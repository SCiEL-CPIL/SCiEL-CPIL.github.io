"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import random
import numpy as np

def construct():
    """Find largest subset of {1..128} with no three distinct elements a,b,c
    such that a+b, a+c, b+c are all in the set."""
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Start with the known good solution: {63, ..., 128}, size 66
    best_set = set(range(63, 129))
    best_size = len(best_set)
    
    # Bitset-based O(N^2) validity checking
    # For a set S, we build a membership array of size 257 (sums go up to 256)
    # Then for each pair (a,b) in S with a<b, check if a+b, and for each c in S
    # with c>b, check if a+c and b+c are in S.
    # Actually, the condition is: there exist distinct a,b,c in S such that
    # a+b in S, a+c in S, b+c in S.
    
    def build_membership(S):
        """Build a numpy boolean array where index i indicates if i is in S."""
        mem = np.zeros(257, dtype=np.uint8)
        for x in S:
            mem[x] = 1
        return mem
    
    def is_valid_fast(S):
        """O(N^2) check: for each pair (a,b) in S, check if a+b in S,
        and if so, check if there exists c in S, c != a, c != b, with a+c in S and b+c in S."""
        mem = build_membership(S)
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                s_ab = a + b
                if s_ab > 256:
                    continue
                if mem[s_ab] == 0:
                    continue
                # Now check if there's a c in S, c != a, c != b, such that a+c in S and b+c in S
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    s_ac = a + c
                    s_bc = b + c
                    if s_ac > 256 or s_bc > 256:
                        continue
                    if mem[s_ac] == 1 and mem[s_bc] == 1:
                        return False
        return True
    
    def try_add_fast(S, x):
        """Try adding x to S. Returns new_set if valid, else None."""
        if x in S:
            return S
        # Only need to check triples involving x
        # Triple (a, b, x): need a+b in S∪{x}, a+x in S∪{x}, b+x in S∪{x}
        # Since x is being added, a+x and b+x are in the new set.
        # a+b must be in S∪{x}. If a+b == x, that's fine. Otherwise a+b must be in S.
        # Also need a,b,x all distinct.
        mem = build_membership(S)
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                # Check triple (a, b, x)
                # a+b in new set? a+b in S or a+b == x
                s_ab = a + b
                if s_ab == x or (s_ab <= 256 and mem[s_ab] == 1):
                    # a+x in new set? a+x <= 256 and (a+x in S or a+x == x which means a=0, impossible)
                    s_ax = a + x
                    s_bx = b + x
                    if s_ax > 256 or s_bx > 256:
                        continue
                    if s_ax == x:  # a=0, impossible
                        continue
                    if s_bx == x:  # b=0, impossible
                        continue
                    # s_ax and s_bx must be in S (since they're not x)
                    if s_ax in S and s_bx in S:
                        return None
        # Also need to check if x creates issues where x plays the role of a sum
        # Actually, the triple condition is about elements a,b,c in the set.
        # If x is in the set, then triples can include x as one of a,b,c.
        # We checked (a,b,x). We don't need to check other orderings since
        # the condition is symmetric in a,b,c.
        return S | {x}
    
    def find_removals_for_add(S, x):
        """Find minimal removals to allow adding x. Returns set of removals or None."""
        if x in S:
            return set()
        new_S = S | {x}
        mem = build_membership(new_S)
        S_list = sorted(new_S)
        n = len(S_list)
        
        # Find all violating triples
        violations = []
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                s_ab = a + b
                if s_ab > 256:
                    continue
                if mem[s_ab] == 0:
                    continue
                for k in range(j+1, n):
                    c = S_list[k]
                    s_ac = a + c
                    s_bc = b + c
                    if s_ac > 256 or s_bc > 256:
                        continue
                    if mem[s_ac] == 1 and mem[s_bc] == 1:
                        violations.append((a, b, c))
        
        if not violations:
            return set()
        
        # Greedy hitting set
        from collections import Counter
        elem_count = Counter()
        for v in violations:
            for e in v:
                elem_count[e] += 1
        
        removals = set()
        remaining = set(violations)
        
        while remaining and len(removals) < 4:
            best_elem = None
            best_cnt = 0
            for e in elem_count:
                if e in removals:
                    continue
                cnt = sum(1 for v in remaining if e in v)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_elem = e
            if best_elem is None or best_cnt == 0:
                break
            removals.add(best_elem)
            remaining = {v for v in remaining if best_elem not in v}
        
        if not remaining:
            test_S = new_S - removals
            if is_valid_fast(test_S):
                return removals
        return None
    
    # Phase 1: Try adding elements to the base set
    candidates = list(range(1, 63))
    random.seed(42)
    
    for _ in range(30):
        if time.time() > deadline - 3:
            break
        improved = False
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 3:
                break
            if x in best_set:
                continue
            
            # Try direct addition
            new_S = try_add_fast(best_set, x)
            if new_S is not None:
                best_set = new_S
                best_size = len(best_set)
                improved = True
                continue
            
            # Try with removals
            removals = find_removals_for_add(best_set, x)
            if removals is not None and len(best_set) - len(removals) + 1 > best_size:
                new_S = (best_set - removals) | {x}
                if is_valid_fast(new_S):
                    best_set = new_S
                    best_size = len(best_set)
                    improved = True
        
        if not improved:
            break
    
    # Phase 2: Try different initial structures
    if time.time() < deadline - 5:
        # Try intervals [k, 128]
        for k in range(1, 80):
            if time.time() > deadline - 3:
                break
            S = set(range(k, 129))
            if is_valid_fast(S) and len(S) > best_size:
                best_set = S
                best_size = len(S)
        
        # Try odd/even
        for S_cand in [set(range(1, 129, 2)), set(range(2, 129, 2))]:
            if time.time() > deadline - 3:
                break
            if is_valid_fast(S_cand) and len(S_cand) > best_size:
                best_set = S_cand
                best_size = len(S_cand)
    
    # Phase 3: Aggressive local search with swaps
    if time.time() < deadline - 5:
        current_set = set(best_set)
        current_size = best_size
        all_elements = list(range(1, 129))
        
        for iteration in range(5000):
            if time.time() > deadline - 3:
                break
            
            # Try random swap
            to_remove = random.choice(list(current_set))
            to_add = random.choice([x for x in all_elements if x not in current_set])
            
            new_set = (current_set - {to_remove}) | {to_add}
            if is_valid_fast(new_set):
                current_set = new_set
                current_size = len(new_set)
                if current_size > best_size:
                    best_set = set(current_set)
                    best_size = current_size
            
            # Try adding an element (increase size)
            to_add = random.choice([x for x in all_elements if x not in current_set])
            new_set = try_add_fast(current_set, to_add)
            if new_set is not None:
                current_set = new_set
                current_size = len(new_set)
                if current_size > best_size:
                    best_set = set(current_set)
                    best_size = current_size
    
    # Final verification
    if not is_valid_fast(best_set):
        best_set = set(range(63, 129))
        best_size = 66
    
    return list(best_set)

# EVOLVE-BLOCK-END
