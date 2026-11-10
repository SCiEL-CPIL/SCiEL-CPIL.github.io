import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # We want to select moduli from 2..128 and assign residues
    # such that for all pairs (r_i, m_i), (r_j, m_j): r_i % gcd(m_i,m_j) != r_j % gcd(m_i,m_j)
    
    # Strategy: greedy with backtracking
    # Start with a good initial set, then try to add more
    
    best = []
    
    # Try different orderings of moduli
    # Key insight: moduli with many common factors with others are harder to place
    # Sort moduli by number of "conflicts" (pairs with gcd>1)
    
    moduli_range = list(range(2, 129))
    
    # For each modulus, count how many other moduli it shares a factor with
    def conflict_count(m):
        count = 0
        for n in moduli_range:
            if n != m and gcd(m, n) > 1:
                count += 1
        return count
    
    # Sort by conflict count descending (hardest first) or ascending (easiest first)
    # Let's try hardest first
    sorted_moduli = sorted(moduli_range, key=lambda m: conflict_count(m), reverse=True)
    
    def is_compatible(residues, moduli, new_r, new_m):
        for i, (r, m) in enumerate(zip(residues, moduli)):
            g = gcd(new_m, m)
            if new_r % g == r % g:
                return False
        return True
    
    def backtrack(idx, residues, moduli, current_set):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current_set) > len(best):
            best = list(current_set)
        
        if idx == len(sorted_moduli):
            return
        
        m = sorted_moduli[idx]
        
        # Try to add this modulus
        for r in range(m):
            if is_compatible(residues, moduli, r, m):
                residues.append(r)
                moduli.append(m)
                current_set.append([r, m])
                backtrack(idx + 1, residues, moduli, current_set)
                current_set.pop()
                moduli.pop()
                residues.pop()
        
        # Skip this modulus
        backtrack(idx + 1, residues, moduli, current_set)
    
    # This naive backtracking will be too slow. Let's use a greedy approach first.
    
    # Greedy: process moduli in some order, for each try to find a valid residue
    def greedy(order):
        residues = []
        moduli = []
        result = []
        for m in order:
            found = False
            for r in range(m):
                if is_compatible(residues, moduli, r, m):
                    residues.append(r)
                    moduli.append(m)
                    result.append([r, m])
                    found = True
                    break
            if time.time() > deadline:
                break
        return result
    
    # Try multiple orderings
    # Ordering 1: by conflict count descending
    r1 = greedy(sorted_moduli)
    if len(r1) > len(best):
        best = r1
    
    # Ordering 2: by conflict count ascending
    sorted_moduli_asc = sorted(moduli_range, key=lambda m: conflict_count(m))
    r2 = greedy(sorted_moduli_asc)
    if len(r2) > len(best):
        best = r2
    
    # Ordering 3: by modulus value
    r3 = greedy(moduli_range)
    if len(r3) > len(best):
        best = r3
    
    # Ordering 4: reverse modulus value
    r4 = greedy(list(reversed(moduli_range)))
    if len(r4) > len(best):
        best = r4
    
    # Try to improve: add moduli that weren't in the greedy result
    def try_improve(result):
        used_moduli = set(m for _, m in result)
        residues = [r for r, _ in result]
        moduli = [m for _, m in result]
        improved = list(result)
        
        # Try adding each unused modulus
        for m in moduli_range:
            if m in used_moduli:
                continue
            if time.time() > deadline:
                break
            for r in range(m):
                if is_compatible(residues, moduli, r, m):
                    residues.append(r)
                    moduli.append(m)
                    improved.append([r, m])
                    used_moduli.add(m)
                    break
        
        return improved
    
    best = try_improve(best)
    
    # Multiple passes of improvement
    for _ in range(5):
        if time.time() > deadline:
            break
        new_best = try_improve(best)
        if len(new_best) > len(best):
            best = new_best
        else:
            break
    
    if not best:
        # Fallback to the original solution
        best = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
    
    return best
