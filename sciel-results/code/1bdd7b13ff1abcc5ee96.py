# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 192
MAX_CARD = 48

def _is_valid_add(c, S):
    """
    Check if c can be added to set S (list of ints).
    Uses residue-class counting for O(n) checks instead of O(n^3).
    """
    n = len(S)
    if n < 3:
        return True
    
    # Check 1: c must NOT divide sum of any 3 distinct elements in S
    # For each residue r mod c, count how many elements of S have that residue.
    # A triple (x,y,z) with x+y+z ≡ 0 mod c exists iff there are residues
    # r1, r2, r3 (with r1+r2+r3 ≡ 0 mod c) such that we can pick distinct
    # elements. For counting purposes, we check: does there exist a multiset
    # of 3 residues summing to 0 mod c with sufficient multiplicity in S?
    # Simpler: for each pair (x,y) in S, check if there's z in S with (x+y+z)%c==0
    # But that's O(n^2). Let's use: for each x in S, for each y in S with y>x,
    # we need z in S\{x,y} with z ≡ -(x+y) mod c.
    # Precompute: for each residue r mod c, how many elements of S have that residue?
    # And which specific elements?
    
    # Actually, let's do it more carefully. For check 1:
    # We need: exist distinct x,y,z in S such that (x+y+z) % c == 0
    # Group S elements by their residue mod c.
    from collections import defaultdict
    res_groups = defaultdict(list)
    for v in S:
        res_groups[v % c].append(v)
    
    # For each pair of residues (r1, r2), we need r3 = (-r1-r2) % c
    # and we need to pick distinct elements.
    # To avoid O(n^2) over elements, iterate over residue pairs.
    residues_in_S = list(res_groups.keys())
    found = False
    for idx1, r1 in enumerate(residues_in_S):
        for idx2, r2 in enumerate(residues_in_S):
            r3 = (-r1 - r2) % c
            if r3 not in res_groups:
                continue
            # Now we need to check if we can pick distinct x from group r1,
            # y from group r2, z from group r3 such that they are all distinct elements.
            # Case analysis based on whether r1, r2, r3 are equal.
            if r1 == r2 == r3:
                # Need 3 distinct elements all with residue r1
                if len(res_groups[r1]) >= 3:
                    found = True
                    break
            elif r1 == r2:
                # x,y from group r1 (need 2 distinct), z from group r3 (need 1, distinct from x,y)
                # Since r1 != r3, any z from group r3 is distinct from x,y.
                if len(res_groups[r1]) >= 2 and len(res_groups[r3]) >= 1:
                    found = True
                    break
            elif r2 == r3:
                # x from group r1 (1), y,z from group r2 (2 distinct)
                if len(res_groups[r1]) >= 1 and len(res_groups[r2]) >= 2:
                    found = True
                    break
            else:
                # All different residues, just need at least 1 from each
                if len(res_groups[r1]) >= 1 and len(res_groups[r2]) >= 1 and len(res_groups[r3]) >= 1:
                    found = True
                    break
        if found:
            break
    if found:
        return False
    
    # Check 2: For each a in S, a must NOT divide (c + x + y) for any distinct x,y in S\{a}
    S_set = set(S)
    for a in S:
        target = (-c) % a
        # Need distinct x, y in S\{a} such that (x + y) % a == target
        # Group S\{a} by residue mod a
        res_groups_a = defaultdict(list)
        for v in S:
            if v != a:
                res_groups_a[v % a].append(v)
        
        residues_a = list(res_groups_a.keys())
        bad = False
        for idx1, r1 in enumerate(residues_a):
            for idx2, r2 in enumerate(residues_a):
                # Need r1 + r2 ≡ target (mod a)
                if (r1 + r2) % a != target:
                    continue
                # Check distinctness
                if r1 == r2:
                    # Need 2 distinct elements from group r1
                    if len(res_groups_a[r1]) >= 2:
                        bad = True
                        break
                else:
                    # Need 1 from group r1, 1 from group r2
                    if len(res_groups_a[r1]) >= 1 and len(res_groups_a[r2]) >= 1:
                        bad = True
                        break
            if bad:
                break
        if bad:
            return False
    
    return True

def _is_valid_set(S):
    """Check if the entire set S is valid."""
    S_list = list(S)
    n = len(S_list)
    
    for i, c in enumerate(S_list):
        others = [S_list[j] for j in range(n) if j != i]
        if not _is_valid_add(c, others):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Strategy 1: Descending Greedy with optimized check
    selected = []
    candidates = list(range(N, 0, -1))
    
    for c in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= MAX_CARD:
            break
            
        if _is_valid_add(c, selected):
            selected.append(c)
            
    best_set = list(selected)
    best_size = len(best_set)
    
    # Strategy 2: Local search with swaps and additions
    if time.time() < deadline - 5.0:
        rng = random.Random(42)
        
        # Multiple random restarts
        for restart in range(30):
            if time.time() > deadline - 3.0:
                break
            
            # Create a biased ordering: mix of descending and random
            cand = list(range(1, N + 1))
            # Bias towards larger numbers
            cand.sort(key=lambda x: (-x, rng.random()))
            
            S = []
            for c in cand:
                if len(S) >= MAX_CARD:
                    break
                if _is_valid_add(c, S):
                    S.append(c)
                
            if len(S) > best_size:
                best_set = list(S)
                best_size = len(S)
                
            # Local search: try adding any remaining element that fits
            if time.time() > deadline - 3.0:
                break
                
            remaining = [c for c in range(1, N + 1) if c not in S]
            rng.shuffle(remaining)
            for c in remaining:
                if len(S) >= MAX_CARD:
                    break
                if _is_valid_add(c, S):
                    S.append(c)
                    
            if len(S) > best_size:
                best_set = list(S)
                best_size = len(S)
            
            # Local search: 1-for-1 swaps to potentially enable more additions
            if time.time() > deadline - 3.0:
                break
                
            improved = True
            while improved and time.time() < deadline - 3.0:
                improved = False
                S_set = set(S)
                remaining = [c for c in range(1, N + 1) if c not in S_set]
                rng.shuffle(remaining)
                
                for c in remaining:
                    if len(S) >= MAX_CARD:
                        break
                    # Try to find an element to remove so c can be added
                    for idx_a, a in enumerate(S):
                        if time.time() > deadline - 3.0:
                            break
                        # Remove a, check if c can be added
                        S_test = S[:idx_a] + S[idx_a+1:]
                        if _is_valid_add(c, S_test):
                            S = S_test
                            S.append(c)
                            S_set = set(S)
                            improved = True
                            # After swap, try to add more
                            remaining2 = [c2 for c2 in range(1, N + 1) if c2 not in S_set]
                            rng.shuffle(remaining2)
                            for c2 in remaining2:
                                if len(S) >= MAX_CARD:
                                    break
                                if _is_valid_add(c2, S):
                                    S.append(c2)
                                    S_set = set(S)
                            break
                    if improved:
                        break
                
                if len(S) > best_size:
                    best_set = list(S)
                    best_size = len(S)
    
    return best_set
# EVOLVE-BLOCK-END
