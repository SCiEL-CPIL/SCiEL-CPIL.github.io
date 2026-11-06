"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 192
    MAX_K = 48
    
    # Precompute residues: for each x in 1..N, residues[x][r] = list of values in 1..N with v % x == r
    residues = {}
    for x in range(1, N + 1):
        residues[x] = {}
        for v in range(1, N + 1):
            r = v % x
            if r not in residues[x]:
                residues[x][r] = []
            residues[x][r].append(v)
    
    def can_add(a, selected):
        """Check if we can add element 'a' to 'selected' (a not in selected)."""
        sel = selected
        n = len(sel)
        
        # Check 1: a does not divide sum of any 3 distinct elements in selected
        if n >= 3:
            # For each pair (b,c) in sel, check if there exists d in sel, d!=b,c, with d % a == (-(b+c))%a
            # Build a map from residue to count of elements in sel with that residue mod a
            res_count = {}
            for v in sel:
                r = v % a
                res_count[r] = res_count.get(r, 0) + 1
            
            for i in range(n):
                b = sel[i]
                for j in range(i + 1, n):
                    c = sel[j]
                    need_res = (-(b + c)) % a
                    # Count of elements with this residue, minus b and c if they have this residue
                    count = res_count.get(need_res, 0)
                    if b % a == need_res:
                        count -= 1
                    if c % a == need_res:
                        count -= 1
                    if count > 0:
                        return False
        
        # Check 2: For each x in sel, x does not divide (a + b + c) for distinct b,c in sel, b!=c, b!=x, c!=x
        for x in sel:
            # Need: no distinct b,c in sel\{x} such that (a + b + c) % x == 0
            # i.e., (b + c) % x == (-a) % x
            need_res = (-a) % x
            # Build a map from residue to count of elements in sel\{x} with that residue mod x
            others_res_count = {}
            for v in sel:
                if v == x:
                    continue
                r = v % x
                others_res_count[r] = others_res_count.get(r, 0) + 1
            
            # For each b in sel\{x}, check if there exists c in sel\{x}, c!=b, with c % x == (need_res - b%x) % x
            for v in sel:
                if v == x:
                    continue
                b = v
                c_need_res = (need_res - b % x) % x
                count = others_res_count.get(c_need_res, 0)
                if b % x == c_need_res:
                    count -= 1
                if count > 0:
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
    
    def greedy_build(order):
        """Build a set greedily following the given order."""
        selected = []
        for x in order:
            if time.time() > deadline:
                break
            if x not in selected:
                if can_add(x, selected):
                    selected.append(x)
        return selected
    
    def try_add_more(selected, order):
        """Try to add more elements from order to selected."""
        sel = list(selected)
        for x in order:
            if time.time() > deadline:
                break
            if x not in sel:
                if can_add(x, sel):
                    sel.append(x)
        return sel
    
    # Phase 1: Multiple greedy starts with different orderings
    best_selected = []
    best_size = 0
    
    # Order 1: Descending
    selected = greedy_build(range(N, 0, -1))
    if len(selected) > best_size:
        best_selected = list(selected)
        best_size = len(selected)
    
    # Order 2: Ascending
    if time.time() < deadline:
        selected = greedy_build(range(1, N + 1))
        if len(selected) > best_size:
            best_selected = list(selected)
            best_size = len(selected)
    
    # Order 3: Primes first, then composites
    if time.time() < deadline:
        primes = [x for x in range(2, N + 1) if all(x % p != 0 for p in range(2, int(x**0.5) + 1))]
        composites = [x for x in range(1, N + 1) if x not in primes and x > 1]
        order3 = primes + [1] + composites
        selected = greedy_build(order3)
        if len(selected) > best_size:
            best_selected = list(selected)
            best_size = len(selected)
    
    # Order 4: Randomized
    if time.time() < deadline:
        random.seed(42)
        for trial in range(5):
            if time.time() > deadline:
                break
            order4 = list(range(1, N + 1))
            random.shuffle(order4)
            selected = greedy_build(order4)
            if len(selected) > best_size:
                best_selected = list(selected)
                best_size = len(selected)
    
    # Phase 2: Local search with swaps
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_selected)):
                if time.time() > deadline:
                    break
                # Try removing best_selected[i] and adding a better element
                base = best_selected[:i] + best_selected[i + 1:]
                base_set = set(base)
                found = False
                # Try candidates in descending order
                for replacement in range(N, 0, -1):
                    if time.time() > deadline:
                        break
                    if replacement in base_set:
                        continue
                    new_selected = base + [replacement]
                    if full_check(new_selected):
                        best_selected = new_selected
                        best_size = len(best_selected)
                        improved = True
                        found = True
                        # Try to add more after successful swap
                        best_selected = try_add_more(best_selected, list(range(N, 0, -1)))
                        break
                if found:
                    break  # Restart the i loop after a swap
    
    # Phase 3: Final attempt to add any remaining elements
    if time.time() < deadline:
        best_selected = try_add_more(best_selected, list(range(N, 0, -1)))
        best_selected = try_add_more(best_selected, list(range(1, N + 1)))
    
    # Final validation
    if not full_check(best_selected):
        # Fallback: rebuild from scratch
        selected = greedy_build(range(N, 0, -1))
        if full_check(selected):
            best_selected = selected
    
    return best_selected
# EVOLVE-BLOCK-END
