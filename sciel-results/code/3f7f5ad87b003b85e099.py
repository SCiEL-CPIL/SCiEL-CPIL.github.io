"""Rewritten program with bitset-accelerated conflict detection and greedy set packing."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 192
    MAX_K = 48
    
    # Precompute conflict bitsets
    # conflict[a][b] = bitset of c such that a | (b+c+d) for some d, OR b | (a+c+d), OR c | (a+b+d)
    # Actually, let's define it more carefully.
    # For a set S to be valid: for every x in S, and every 3 distinct a,b,c in S\{x}, x does not divide a+b+c.
    # 
    # Conflict between a and b: if there exists c such that adding {a,b,c} creates a violation.
    # A violation in {a,b,c} occurs if:
    #   a | (b+c) + something... no wait.
    #   The condition is: x does not divide sum of any 3 OTHER selected.
    #   So for triple {a,b,c}: 
    #     a must not divide b+c? No, "sum of 3 distinct OTHER". 
    #     Wait, if we have only 3 elements {a,b,c}, then for x=a, the "3 distinct others" don't exist (only 2 others).
    #     So a set of size 3 is always valid? 
    #     Let's re-read: "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
    #     If |S| < 4, there are no 3 distinct others for any x. So any set of size <= 3 is valid.
    #     Violations only occur when |S| >= 4.
    #
    # So, when adding a new element 'a' to a set S (where |S| >= 3):
    # 1. a must not divide sum of any 3 distinct elements in S.
    # 2. For each x in S, x must not divide (a + b + c) for any 2 distinct b,c in S\{x}.
    #
    # Let's precompute for each pair (a,b) in 1..N, the set of c in 1..N such that {a,b,c} would be part of a violation
    # when a fourth element is added? No, that's complex.
    #
    # Better approach: Precompute "bad_triples".
    # A triple (a,b,c) is "bad" for x if x | (a+b+c).
    # But x is not in {a,b,c}.
    #
    # Let's precompute: for each x in 1..N, and for each pair (a,b) with a<b, 
    # find all c such that (a+b+c) % x == 0.
    # This tells us: if x is in the set, and a,b are in the set, then c CANNOT be in the set (if c is distinct from x,a,b).
    #
    # So, conflict[x][a][b] = set of c that are forbidden if x,a,b are all present.
    #
    # For efficient checking:
    # When adding 'a' to S:
    # 1. Check if a divides any triple sum in S.
    #    For each pair (b,c) in S, check if there exists d in S (d!=b,c) such that (b+c+d)%a==0.
    #    Precompute: for each a, and for each residue r mod a, the count of elements in S with that residue.
    #    Then for pair (b,c), need d % a == (-(b+c))%a. If count > 0 and the element is not b or c, then conflict.
    #    But we need to ensure d is distinct from b and c.
    #
    # Let's use a different precomputation:
    # For each x in 1..N, precompute a 2D array or use bitsets.
    # Let's use bitsets for the elements 1..N.
    # 
    # Precompute: divisors_of_sum[x][s] = bitset of pairs? No.
    #
    # Let's try a different angle. Precompute for each x, a bitset B_x where B_x has bit i set if i % x == 0? No.
    #
    # Let's precompute: for each x, for each pair (i,j) with i<j, the residue r = (-(i+j)) % x.
    # Then if we have a set S, and we want to check if x divides sum of 3 elements in S:
    # For each pair (i,j) in S, check if there is k in S (k!=i,j) with k % x == r.
    #
    # To make this fast, for each x, we can maintain a count of residues for the current set S.
    # residue_count[x][r] = number of elements in S with element % x == r.
    # Then for pair (i,j), we need residue_count[x][r] >= 1, and if the element providing that residue is i or j, we need count >= 2.
    # Actually, if i % x == r or j % x == r, we need to be careful.
    #
    # Let's precompute:
    # For each x in 1..N:
    #   For each i in 1..N: residue_i = i % x
    #
    # Then in can_add(a, S):
    #   Check 1: a | (b+c+d) for b,c,d in S.
    #   For each pair (b,c) in S:
    #       r = (-(b+c)) % a
    #       Need an element d in S, d!=b,c, with d % a == r.
    #       We can precompute for the current S, for each x, a list of elements by residue, or just use a set.
    #
    # Given the small N=192 and max size 48, O(n^3) per check is 48^3 = 110k ops, which is fast.
    # The bottleneck is the number of checks.
    #
    # Let's optimize the check itself.
    
    # Precompute residues for all x and all values
    # residues[x][v] = v % x
    residues = np.zeros((N+1, N+1), dtype=np.int8)
    for x in range(1, N+1):
        for v in range(1, N+1):
            residues[x][v] = v % x
    
    def can_add(a, selected):
        """Check if we can add element 'a' to 'selected' (a not in selected)."""
        n = len(selected)
        
        # Check 1: a does not divide sum of any 3 distinct elements in selected
        if n >= 3:
            # For each pair (b,c), check if there exists d in selected, d!=b,c, with (b+c+d) % a == 0
            # i.e., d % a == (-(b+c)) % a
            # Build a frequency map of residues for 'a'
            freq = {}
            for v in selected:
                r = residues[a][v]
                freq[r] = freq.get(r, 0) + 1
            
            for i in range(n):
                b = selected[i]
                for j in range(i+1, n):
                    c = selected[j]
                    need_r = (-(b + c)) % a
                    cnt = freq.get(need_r, 0)
                    if cnt > 0:
                        # Check if the element with residue need_r is b or c
                        # If b % a == need_r or c % a == need_r, we need cnt >= 2 (to have a third element)
                        # Actually, if b % a == need_r, then b is a candidate for d, but d must be distinct from b and c.
                        # So if b % a == need_r, we need another element with that residue.
                        # If neither b nor c has residue need_r, then any element with that residue works (cnt >= 1).
                        # If one of them has it, we need cnt >= 2.
                        # If both have it, we need cnt >= 3.
                        b_r = residues[a][b]
                        c_r = residues[a][c]
                        if b_r == need_r and c_r == need_r:
                            if cnt < 3:
                                continue
                        elif b_r == need_r or c_r == need_r:
                            if cnt < 2:
                                continue
                        else:
                            if cnt < 1:
                                continue
                        return False
        
        # Check 2: For each x in selected, x does not divide (a + b + c) for distinct b,c in selected\{x}
        for x in selected:
            # Need: no distinct b,c in selected\{x} such that (a + b + c) % x == 0
            # i.e., (b + c) % x == (-a) % x
            need_r = (-a) % x
            # Build freq for x, excluding x itself
            freq = {}
            for v in selected:
                if v == x:
                    continue
                r = residues[x][v]
                freq[r] = freq.get(r, 0) + 1
            
            # Now check pairs in selected\{x}
            others = [v for v in selected if v != x]
            m = len(others)
            if m >= 2:
                for i in range(m):
                    b = others[i]
                    for j in range(i+1, m):
                        c = others[j]
                        if (b + c) % x == need_r:
                            return False
        
        return True
    
    # Phase 1: Greedy from large to small
    selected = []
    for x in range(N, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, selected):
            selected.append(x)
    
    best_selected = list(selected)
    best_size = len(selected)
    
    # Phase 2: Try to add more elements in different orders
    if time.time() < deadline:
        for x in range(1, N+1):
            if time.time() > deadline:
                break
            if x not in best_selected:
                if can_add(x, best_selected):
                    best_selected.append(x)
                    best_size = len(best_selected)
    
    # Phase 3: Local search with swaps
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_selected)):
                if time.time() > deadline:
                    break
                for replacement in range(N, 0, -1):
                    if time.time() > deadline:
                        break
                    if replacement in best_selected:
                        continue
                    new_selected = best_selected[:i] + best_selected[i+1:] + [replacement]
                    if can_add(replacement, [x for x in best_selected if x != best_selected[i]]):
                        # Need to verify the whole set is valid
                        # Actually, can_add only checks if replacement can be added to the rest.
                        # We also need to ensure the rest is still valid (it was, and we removed one element, so it's still valid)
                        # And replacement doesn't conflict with the rest (can_add checks this)
                        # So the new set is valid.
                        best_selected = new_selected
                        best_size = len(best_selected)
                        improved = True
                        # Try to add more after swap
                        for x in range(N, 0, -1):
                            if time.time() > deadline:
                                break
                            if x not in best_selected:
                                if can_add(x, best_selected):
                                    best_selected.append(x)
                                    best_size = len(best_selected)
                        break
    
    # Phase 4: Final attempt to add any remaining elements
    if time.time() < deadline:
        for x in range(N, 0, -1):
            if time.time() > deadline:
                break
            if x not in best_selected:
                if can_add(x, best_selected):
                    best_selected.append(x)
                    best_size = len(best_selected)
    
    return best_selected
# EVOLVE-BLOCK-END
