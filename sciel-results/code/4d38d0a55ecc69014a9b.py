import time
import math
from math import gcd

MAXIMUM = 128
DEADLINE = 90.0
SAFETY_MARGIN = 1.0

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Moduli that could be pairwise coprime with the selected set
    # Strategy: greedily select moduli from 2..128 such that all are pairwise coprime
    # and their product is as small as possible (to allow CRT combinations later if needed)
    # But the task says "distinct moduli" and "every two infinite residue classes must be disjoint"
    # The exact gcd criterion for two residue classes (r1, m1) and (r2, m2) to be disjoint:
    # They intersect iff r1 ≡ r2 (mod gcd(m1, m2))
    # So they are disjoint iff r1 % gcd(m1,m2) != r2 % gcd(m1,m2)
    
    # If we pick moduli that are pairwise coprime, then gcd(m_i, m_j) = 1 for all i != j
    # Then r1 % 1 = 0 and r2 % 1 = 0, so they ALWAYS intersect!
    # Wait, that means pairwise coprime moduli CANNOT be used for disjoint classes.
    
    # Let me reconsider. Two residue classes {x : x ≡ r1 (mod m1)} and {x : x ≡ r2 (mod m2)}
    # intersect iff the system x ≡ r1 (mod m1), x ≡ r2 (mod m2) has a solution.
    # By CRT, this has a solution iff r1 ≡ r2 (mod gcd(m1, m2)).
    # So for them to be DISJOINT, we need r1 NOT≡ r2 (mod gcd(m1, m2)).
    
    # If gcd(m1, m2) = 1, then r1 ≡ r2 (mod 1) is always true (both are 0 mod 1),
    # so they always intersect. Thus we need gcd(m1, m2) >= 2 for any pair.
    
    # This means no two moduli can be coprime! We need all pairs to share a common factor.
    # One strategy: use moduli that all share a common prime factor, say all even moduli.
    # Then gcd(m_i, m_j) >= 2 for all pairs.
    
    # For even moduli m_i, m_j, gcd(m_i, m_j) >= 2.
    # We need r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j) for all i != j.
    
    # A simpler approach: use moduli that are powers of 2.
    # If m_i = 2^a_i with distinct a_i, then gcd(2^a_i, 2^a_j) = 2^min(a_i, a_j).
    # We need r_i % 2^min(a_i,a_j) != r_j % 2^min(a_i,a_j).
    
    # The seed program uses [(2^k - 1, 2^(k+1)) for k in range(7)]
    # That's residues 1,3,7,15,31,63,127 with moduli 4,8,16,32,64,128,256
    # But modulus 256 > 128, so that's invalid per the task constraint (moduli 2..128).
    
    # Let me re-read: "distinct moduli 2..maximum=128"
    # So moduli must be in [2, 128].
    
    # The seed has modulus 256 which is > 128, so it should be invalid?
    # But it scored 1.0000, so maybe I'm misunderstanding.
    
    # Let me try a different approach: use moduli that are all multiples of 2.
    # Pick a set of even moduli from 2 to 128, and assign residues such that
    # for every pair (i,j), r_i mod gcd(m_i,m_j) != r_j mod gcd(m_i,m_j).
    
    # Greedy approach:
    # 1. Select a set of moduli from 2..128
    # 2. For each modulus, try all residues 0..m-1
    # 3. Check if the residue is compatible with all previously selected classes
    
    best_solution = []
    
    # Try selecting moduli greedily
    # We want to maximize the number of classes
    # Start with a candidate list of moduli
    # For efficiency, try moduli in a smart order
    
    # Let's try all even moduli from 128 down to 2
    candidate_moduli = [m for m in range(2, 129) if m % 2 == 0]
    
    # Try different orderings
    for mod_order in [candidate_moduli, list(reversed(candidate_moduli)), 
                      sorted(candidate_moduli, key=lambda m: -m)]:
        if time.time() > deadline:
            break
        
        solution = []
        used_moduli = set()
        
        for m in mod_order:
            if m in used_moduli:
                continue
            
            # Try to find a residue for modulus m that works with all existing classes
            found = False
            for r in range(m):
                compatible = True
                for (r_prev, m_prev) in solution:
                    g = gcd(m, m_prev)
                    if r % g == r_prev % g:
                        compatible = False
                        break
                if compatible:
                    solution.append([r, m])
                    used_moduli.add(m)
                    found = True
                    break
            
            if not found:
                continue
        
        if len(solution) > len(best_solution):
            best_solution = solution
        
        if time.time() > deadline:
            break
    
    # Also try with odd moduli mixed in
    # Actually, let's be more systematic. Try all moduli 2..128
    all_moduli = list(range(2, 129))
    
    # Try a few different orderings
    orderings = [
        all_moduli,
        list(reversed(all_moduli)),
        sorted(all_moduli, key=lambda m: -m),
        sorted(all_moduli, key=lambda m: m)
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        
        solution = []
        used_moduli = set()
        
        for m in mod_order:
            if m in used_moduli:
                continue
            
            found = False
            for r in range(m):
                compatible = True
                for (r_prev, m_prev) in solution:
                    g = gcd(m, m_prev)
                    if r % g == r_prev % g:
                        compatible = False
                        break
                if compatible:
                    solution.append([r, m])
                    used_moduli.add(m)
                    found = True
                    break
        
        if len(solution) > len(best_solution):
            best_solution = solution
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
