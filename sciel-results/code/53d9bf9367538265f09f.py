"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    
    N = 192
    MAX_K = 48
    
    # Precompute: for each x in 1..N, and for each pair (a,b) with a<b,
    # check if x divides (a+b+c) for some c. Actually, let's think differently.
    # 
    # For incremental check: when adding element 'a' to set S:
    # 1. a must not divide sum of any 3 distinct elements in S
    # 2. For each x in S: x must not divide (a + b + c) where b,c are distinct from x, a, and each other, all in S
    #
    # Let's precompute triple_sums[x] = set of all sums of 3 distinct elements from 1..N that are divisible by x
    # Actually that's too large. Let me think of a smarter approach.
    #
    # Alternative: For the check "does x divide sum of 3 distinct others", 
    # we can iterate over pairs and check if there exists a third element.
    #
    # Let's precompute for efficiency:
    # For each x, precompute which values v in 1..N could be part of a triple summing to multiple of x
    # with two other specific values.
    
    # Precompute: divisors[x] = list of x's that divide x (i.e., just x itself for the "x divides sum" check)
    # Actually, let me precompute: for each x in 1..N, and for each pair (p,q) with p<q, p,q in 1..N,
    # find all r such that (p+q+r) % x == 0, i.e., r % x == (-p-q) % x
    # Then r can be any value in 1..N with r % x == (-p-q) % x
    
    # Let's precompute: for each x, for each residue class mod x, which values have that residue
    # residues[x][r] = sorted list of values in 1..N with value % x == r
    residues = {}
    for x in range(1, N+1):
        residues[x] = {}
        for v in range(1, N+1):
            r = v % x
            if r not in residues[x]:
                residues[x][r] = []
            residues[x][r].append(v)
    
    def can_add(a, selected):
        """Check if we can add element 'a' to 'selected' (a not in selected)."""
        # Check 1: a does not divide sum of any 3 distinct elements in selected
        # For each triple (b,c,d) in selected, check if (b+c+d) % a == 0
        # Optimize: for each pair (b,c), need d such that d % a == (-(b+c)) % a
        sel = selected
        n = len(sel)
        
        # Check 1: a | (b+c+d) for some distinct b,c,d in sel
        if n >= 3:
            # For each pair (b,c) in sel, check if there exists d in sel, d!=b,c, with d % a == (-(b+c))%a
            target_res = lambda b, c: (-(b + c)) % a
            # Build a set of (value, index) for quick lookup by residue
            # Actually, let's just iterate pairs and check
            for i in range(n):
                b = sel[i]
                for j in range(i+1, n):
                    c = sel[j]
                    need_res = (-(b + c)) % a
                    # Check if any sel[k] with k != i,j has sel[k] % a == need_res
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if sel[k] % a == need_res:
                            return False
        
        # Check 2: For each x in sel, x does not divide (a + b + c) for distinct b,c in sel, b!=c, b!=x, c!=x
        for x in sel:
            # Need: no distinct b,c in sel\{x} such that (a + b + c) % x == 0
            # i.e., (b + c) % x == (-a) % x
            need_res = (-a) % x
            # Find all elements in sel except x
            # For each pair (b,c) from sel\{x}, check if (b+c) % x == need_res
            others = [v for v in sel if v != x]
            m = len(others)
            if m >= 2:
                for i in range(m):
                    b = others[i]
                    for j in range(i+1, m):
                        c = others[j]
                        if (b + c) % x == need_res:
                            return False
        
        return True
    
    def full_check(candidate_set):
        """Full validation of a set."""
        n = len(candidate_set)
        for i, x in enumerate(candidate_set):
            others = [candidate_set[j] for j in range(n) if j != i]
            if len(others) < 3:
                continue
            for combo in combinations(others, 3):
                if sum(combo) % x == 0:
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
        # Try adding from small to large on top of what we have
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
                    if full_check(new_selected):
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
