import math

def construct():
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute prime factors for each modulus
    def prime_factors(n):
        pf = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                pf.add(d)
                n //= d
            d += 1
        if n > 1:
            pf.add(n)
        return pf
    
    pf_map = {m: prime_factors(m) for m in moduli}
    
    # Precompute gcd between all pairs of moduli
    gcd_map = {}
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            m1, m2 = moduli[i], moduli[j]
            gcd_map[(m1, m2)] = math.gcd(m1, m2)
    
    # AC-3 arc consistency: check if for every value in D_i, there's a supporting value in D_j
    # Constraint: r_i not congruent r_j mod g, where g = gcd(m_i, m_j)
    # For fixed r_i, forbidden r_j are those with r_j ≡ r_i (mod g)
    # So D_j must contain at least one value not ≡ r_i (mod g)
    # Since |D_j| = m_j >= g (because g | m_j), there are always values not ≡ r_i (mod g)
    # unless m_j == g, in which case all values in D_j are ≡ some fixed value mod g... 
    # Actually if m_j == g, then D_j = {0, ..., g-1} and every value mod g is covered exactly once.
    # So for any r_i, exactly one value in D_j is ≡ r_i (mod g), and m_j - 1 >= 1 others are not.
    # So arc consistency is always satisfied for g >= 2 and m_j >= 2.
    # This means AC-3 doesn't prune anything here. The real constraint is on the modulus selection.
    
    # The key insight: if gcd(m1, m2) == 1, the classes always intersect, so we can't have both.
    # If gcd(m1, m2) > 1, we can always find residues that make them disjoint.
    # So the problem reduces to: find the largest subset of {2,...,128} where pairwise gcd > 1.
    # This is a maximum clique problem in the graph where edges connect moduli with gcd > 1.
    
    # Greedy approach with multiple orderings to find a large clique
    def greedy_clique(order):
        selected = []
        for m in order:
            can_add = True
            for m_s in selected:
                if math.gcd(m, m_s) == 1:
                    can_add = False
                    break
            if can_add:
                selected.append(m)
        return selected
    
    # Try multiple orderings and keep the best
    best_clique = []
    
    # Ordering 1: by number of prime factors (descending), then smallest prime factor (ascending)
    order1 = sorted(moduli, key=lambda m: (-len(pf_map[m]), min(pf_map[m]) if pf_map[m] else 999, m))
    best_clique = greedy_clique(order1)
    
    # Ordering 2: by smallest prime factor (ascending)
    order2 = sorted(moduli, key=lambda m: (min(pf_map[m]) if pf_map[m] else 999, m))
    clique2 = greedy_clique(order2)
    if len(clique2) > len(best_clique):
        best_clique = clique2
    
    # Ordering 3: by total number of divisors (descending) - more composite numbers first
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
        return count
    
    order3 = sorted(moduli, key=lambda m: (-num_divisors(m), m))
    clique3 = greedy_clique(order3)
    if len(clique3) > len(best_clique):
        best_clique = clique3
    
    # Ordering 4: random-ish shuffle based on prime factor sum
    order4 = sorted(moduli, key=lambda m: (sum(pf_map[m]), m))
    clique4 = greedy_clique(order4)
    if len(clique4) > len(best_clique):
        best_clique = clique4
    
    # Ordering 5: by largest prime factor (descending)
    order5 = sorted(moduli, key=lambda m: (-max(pf_map[m]) if pf_map[m] else 0, m))
    clique5 = greedy_clique(order5)
    if len(clique5) > len(best_clique):
        best_clique = clique5
    
    # Now assign residues to the selected moduli using backtracking
    def assign_residues(selected_moduli):
        n = len(selected_moduli)
        if n == 0:
            return []
        
        residues = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            m = selected_moduli[idx]
            for r in range(m):
                valid = True
                for j in range(idx):
                    m_j = selected_moduli[j]
                    g = math.gcd(m, m_j)
                    if (r - residues[j]) % g == 0:
                        valid = False
                        break
                if valid:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = None
            
            return False
        
        if backtrack(0):
            return [[residues[i], selected_moduli[i]] for i in range(n)]
        return None
    
    result = assign_residues(best_clique)
    
    if result is None:
        # Fallback: just use the first modulus
        return [[1, 2]]
    
    return result
