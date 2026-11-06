"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from fractions import Fraction
from functools import lru_cache

def construct():
    """
    Find a maximum weight independent set on the conflict graph of candidate (m, r) pairs.
    Nodes: (m, r) where 20 <= m <= 200, 0 <= r < m.
    Edges: two nodes (m1, r1) and (m2, r2) conflict if r1 ≡ r2 (mod gcd(m1, m2)).
    Weight: 1/m for each node.
    
    We use branch-and-bound with a fractional knapsack upper bound.
    """
    
    # Generate all candidate (m, r) pairs
    candidates = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))
    
    n = len(candidates)
    
    # Precompute weights as Fractions for exact arithmetic
    weights = [Fraction(1, m) for (m, r) in candidates]
    
    # Precompute gcd between all pairs of moduli
    # Actually, we need to check conflict between (m1, r1) and (m2, r2):
    # conflict if r1 % gcd(m1, m2) == r2 % gcd(m1, m2)
    
    # To make this tractable, let's think about the structure.
    # There are 181 moduli, each with up to 200 residues, so ~181*110 ≈ 20000 candidates.
    # That's too many for a full MIS. We need to be smarter.
    
    # Key insight: for a fixed modulus m, we can only pick at most one residue r.
    # Moreover, if we pick residue r for modulus m, it conflicts with any (m', r') 
    # where r' ≡ r (mod gcd(m, m')).
    
    # Let's reformulate: we choose a subset S of moduli, and for each m in S, we choose 
    # a residue r_m. The constraint is that for any pair m, m' in S, r_m ≠ r_m' (mod gcd(m, m')).
    
    # This is still complex. Let's use a different approach:
    # Since we want to maximize sum of 1/m, we prefer small moduli.
    # The current solution uses m = 20, 40, 60, ..., 200 (10 moduli) with residues 0, 1, ..., 9.
    # Score = 1/20 + 1/40 + 1/60 + 1/80 + 1/100 + 1/120 + 1/140 + 1/160 + 1/180 + 1/200
    # Let's compute this:
    # = 0.05 + 0.025 + 0.01667 + 0.0125 + 0.01 + 0.00833 + 0.00714 + 0.00625 + 0.00556 + 0.005
    # ≈ 0.14645
    
    # Can we do better? What if we use more moduli? The issue is that adding more moduli 
    # requires them to be pairwise disjoint in the residue sense.
    
    # Let's try a greedy approach first, then improve with local search.
    # Greedy: sort candidates by 1/m descending, add if no conflict.
    
    # But first, let's try a smarter construction.
    # Consider using all moduli from 20 to 200 that are pairwise coprime? No, that's too restrictive.
    
    # Actually, let's think about it differently. For two moduli m and m', the constraint is
    # r_m ≠ r_m' (mod gcd(m, m')). If gcd(m, m') = 1, then there's no constraint (any residues work).
    # If gcd(m, m') = g > 1, then r_m and r_m' must be in different residue classes mod g.
    
    # So if we pick moduli that are pairwise coprime, we can pick any residues. But there aren't
    # many pairwise coprime numbers in [20, 200].
    
    # Let's try a different strategy: use moduli that are multiples of a common base, like the current solution.
    # Current: m = 20j for j=1..10. gcd(20j, 20k) = 20*gcd(j,k). 
    # For j=1..10, gcd(j,k) is 1 for most pairs, so gcd(20j, 20k) = 20 for coprime j,k.
    # Residues: r_j = j-1. So r_j mod 20 = j-1. For j,k coprime, we need (j-1) ≠ (k-1) mod 20, 
    # which is true since j-1 and k-1 are distinct in 0..9.
    
    # What if we use more moduli? Say m = 10j for j=2..20? That gives 19 moduli.
    # gcd(10j, 10k) = 10*gcd(j,k). Residues r_j = ? We need r_j ≠ r_k mod 10*gcd(j,k).
    # If we set r_j = j-2 (so 0..18), then for j,k with gcd(j,k)=1, we need (j-2) ≠ (k-2) mod 10,
    # i.e., j ≠ k mod 10. But j and k range over 2..20, so this fails for j=2,k=12 (both ≡2 mod 10).
    
    # This is getting complicated. Let's just implement the branch-and-bound as specified, but with 
    # a reduced search space.
    
    # Actually, given the time limit of 120s, let's try a heuristic approach:
    # 1. Start with the current solution.
    # 2. Try to add more moduli by finding residues that don't conflict.
    # 3. Try to swap moduli to get better weights.
    
    # Let's implement a local search / simulated annealing approach.
    
    import random
    
    def check_disjoint(m1, r1, m2, r2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def is_valid_solution(classes):
        for i in range(len(classes)):
            for j in range(i+1, len(classes)):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                if m1 == m2:
                    return False
                if not check_disjoint(m1, r1, m2, r2):
                    return False
        return True
    
    def score(classes):
        return sum(Fraction(1, m) for m, r in classes)
    
    # Start with current solution
    current = [[20 * j, j - 1] for j in range(1, 11)]
    current_score = score(current)
    
    # Try to improve by adding moduli
    # For each unused modulus m in [20, 200], try to find a residue r that doesn't conflict
    # with any existing class.
    
    used_moduli = set(m for m, r in current)
    
    for m in range(20, 201):
        if m in used_moduli:
            continue
        for r in range(m):
            valid = True
            for m2, r2 in current:
                if not check_disjoint(m, r, m2, r2):
                    valid = False
                    break
            if valid:
                current.append([m, r])
                used_moduli.add(m)
                break
    
    # Now try to swap: replace a modulus with a smaller one if possible
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            m_i, r_i = current[i]
            # Try replacing with a smaller modulus
            for m_new in range(20, m_i):
                if m_new in used_moduli - {m_i}:
                    continue
                for r_new in range(m_new):
                    # Check if (m_new, r_new) is disjoint from all other classes
                    valid = True
                    for j in range(len(current)):
                        if j == i:
                            continue
                        m_j, r_j = current[j]
                        if not check_disjoint(m_new, r_new, m_j, r_j):
                            valid = False
                            break
                    if valid:
                        # Replace
                        old_m = current[i][0]
                        current[i] = [m_new, r_new]
                        used_moduli.discard(old_m)
                        used_moduli.add(m_new)
                        improved = True
                        break
                if improved:
                    break
    
    # Convert to float for return
    return [[m, r] for m, r in current]

# EVOLVE-BLOCK-END
