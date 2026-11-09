"""Rewritten program: formulate as maximum clique on compatibility graph + residue assignment."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    MAX_MOD = 128
    all_moduli = list(range(2, MAX_MOD + 1))
    n = len(all_moduli)  # 127 moduli
    
    # Step 1: Build compatibility graph.
    # Two moduli m1, m2 are "compatible" if there exist residues r1, r2 such that
    # the classes {r1 mod m1} and {r2 mod m2} are disjoint.
    # The classes are disjoint iff for all integers x, y: x ≡ r1 (mod m1) and x ≡ r2 (mod m2) has no solution.
    # By CRT, a solution exists iff r1 ≡ r2 (mod gcd(m1, m2)).
    # So the classes are disjoint iff r1 NOT≡ r2 (mod gcd(m1, m2)).
    # For a given pair (m1, m2) with g = gcd(m1, m2), we need to choose r1 ∈ [0, m1) and r2 ∈ [0, m2)
    # such that r1 % g != r2 % g.
    # This is always possible as long as g < m1 and g < m2 (which is true since m1, m2 > g unless one divides the other).
    # Wait: if m1 divides m2, then g = m1. Then r1 % m1 = r1 and r2 % m1 must be != r1.
    # Since r2 can be any value in [0, m2), and m2 is a multiple of m1, r2 % m1 can be any value in [0, m1).
    # So we can always pick r2 such that r2 % m1 != r1. Similarly for any pair.
    # Actually, for ANY pair of moduli m1, m2 >= 2, we can always find residues making them disjoint.
    # So the compatibility graph is a complete graph! That means the "maximum clique" is all 127 moduli.
    # But wait - we need ALL pairs to be simultaneously disjoint. This is a global constraint.
    # The pairwise compatibility doesn't guarantee global compatibility.
    # We need to assign residues r_m for each modulus m such that for every pair (m1, m2),
    # r_{m1} % gcd(m1,m2) != r_{m2} % gcd(m1,m2).
    
    # This is a constraint satisfaction problem. Let's use a smarter approach.
    # Key insight: if we group moduli by their prime factor structure, we can simplify.
    # For two moduli sharing a common prime p, the constraint is on r mod p (and higher powers).
    
    # Let's try a different structural approach:
    # Assign residues based on a "coloring" of the prime factors.
    # For modulus m with prime factorization, assign r such that r mod p is "different" for each prime p dividing m.
    
    # Actually, let's use a greedy + local search approach with a better initial solution.
    # First, let's find a large feasible set using a constraint-based approach.
    
    def is_pairwise_disjoint(residues, moduli):
        """Check if the assignment is valid: all pairs of residue classes are disjoint."""
        for i in range(len(moduli)):
            for j in range(i + 1, len(moduli)):
                m1, m2 = moduli[i], moduli[j]
                r1, r2 = residues[i], residues[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True
    
    def try_greedy_order(order):
        """Try greedy assignment in the given order of moduli."""
        assignment = {}
        for m in order:
            found = False
            # Try all residues, prefer ones that conflict least with existing
            best_r = -1
            best_conflicts = m + 1  # max possible conflicts + 1
            for r in range(m):
                conflicts = 0
                for m2 in assignment:
                    g = gcd(m, m2)
                    if r % g == assignment[m2] % g:
                        conflicts += 1
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_r = r
                if conflicts == 0:
                    best_r = r
                    break
            if best_r >= 0 and best_conflicts == 0:
                assignment[m] = best_r
                found = True
            else:
                # Can't add this modulus without conflict; skip it
                continue
        return assignment
    
    def local_search(assignment, max_iters=20000):
        """Improve assignment by reassigning residues to reduce conflicts, then try to add more moduli."""
        current = dict(assignment)
        moduli_in_set = set(current.keys())
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try to add a modulus not in the set
            added = False
            for m in range(2, MAX_MOD + 1):
                if m in moduli_in_set:
                    continue
                # Try to find a residue for m that doesn't conflict
                for r in range(m):
                    ok = True
                    for m2 in moduli_in_set:
                        g = gcd(m, m2)
                        if r % g == current[m2] % g:
                            ok = False
                            break
                    if ok:
                        current[m] = r
                        moduli_in_set.add(m)
                        added = True
                        break
                if added:
                    break
            
            if not added:
                # Try to reassign a residue to free up space for another modulus
                improved = False
                # Pick a random modulus to reassign
                import random
                if not hasattr(local_search, '_rng'):
                    local_search._rng = random.Random(42)
                rng = local_search._rng
                
                candidates = list(moduli_in_set)
                rng.shuffle(candidates)
                
                for m_reassign in candidates[:5]:  # try a few
                    old_r = current[m_reassign]
                    # Try alternative residues
                    for r_new in range(m_reassign):
                        if r_new == old_r:
                            continue
                        # Check if r_new is compatible with all others
                        ok = True
                        for m2 in moduli_in_set:
                            if m2 == m_reassign:
                                continue
                            g = gcd(m_reassign, m2)
                            if r_new % g == current[m2] % g:
                                ok = False
                                break
                        if ok:
                            # Now check if this allows adding a new modulus
                            current[m_reassign] = r_new
                            # Try to add
                            for m_new in range(2, MAX_MOD + 1):
                                if m_new in moduli_in_set:
                                    continue
                                for r_new2 in range(m_new):
                                    ok2 = True
                                    for m3 in moduli_in_set:
                                        g = gcd(m_new, m3)
                                        if r_new2 % g == current[m3] % g:
                                            ok2 = False
                                            break
                                    if ok2:
                                        current[m_new] = r_new2
                                        moduli_in_set.add(m_new)
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                            else:
                                current[m_reassign] = old_r  # revert
                    if improved:
                        break
                
                if not improved and not added:
                    # No improvement possible in this iteration
                    break
        
        return current
    
    # Strategy: Use a structured approach based on prime factors.
    # Key insight: If all moduli in our set are multiples of the same prime p, then the constraint
    # is that r_m % p must be all different (since gcd(m1,m2) >= p for all pairs).
    # Wait, that's not quite right. gcd(m1,m2) could be p, p^2, etc.
    
    # Better structural approach:
    # Partition moduli by their "radical" (product of distinct prime factors).
    # Moduli with the same radical share the same set of prime factors.
    # For moduli m1, m2 with the same set of primes, gcd(m1,m2) contains all those primes.
    
    # Let's try: pick a prime p, and take all multiples of p up to 128.
    # For multiples of p: m = k*p. gcd(m1, m2) is divisible by p.
    # So we need r1 % p != r2 % p for all pairs. But there are only p distinct values mod p.
    # So we can have at most p moduli that are multiples of p (if gcd is exactly p for all pairs).
    # But if some pairs have gcd = p^2, the constraint is tighter.
    
    # Actually, the cleanest approach: use moduli that are all powers of 2, or all multiples of a single prime with controlled structure.
    
    # Let's try a completely different structural approach:
    # Use the fact that if we pick moduli m1, m2, ..., mk such that for every pair, gcd(mi, mj) = 1,
    # then any residues work (since gcd=1 means r1%1=0=r2%1 always, so they'd always be disjoint? No!)
    # Wait: if gcd(m1,m2)=1, then r1%1 = 0 and r2%1 = 0, so r1%1 == r2%1, meaning the classes INTERSECT.
    # So coprime moduli always have intersecting classes! We CANNOT use coprime moduli together.
    
    # This is a crucial insight: we need gcd(m1,m2) > 1 for all pairs in our set.
    # So all moduli in our set must share a common factor... or at least pairwise share some factor.
    
    # The simplest structure: all moduli are even (multiples of 2).
    # Then gcd(mi,mj) >= 2 for all pairs.
    # We need r_i % gcd(mi,mj) != r_j % gcd(mi,mj) for all pairs.
    
    # For all-even moduli, let's try to maximize the count.
    # There are 63 even numbers from 2 to 128.
    
    # Let's use a SAT-like or constraint propagation approach.
    # Actually, let's just do a thorough search with the local search approach.
    
    # First, get a good initial solution using greedy with multiple orderings.
    best_assignment = {}
    
    # Try different orderings
    orderings = [
        sorted(all_moduli),
        sorted(all_moduli, reverse=True),
        [m for m in all_moduli if m % 2 == 0],  # evens
        [m for m in all_moduli if m % 2 == 0][::-1],
        [m for m in all_moduli if m % 3 == 0],  # multiples of 3
        [m for m in all_moduli if m % 3 == 0][::-1],
        [m for m in all_moduli if m % 4 == 0],
        [m for m in all_moduli if m % 5 == 0],
        # Mixed: prioritize smaller moduli
        sorted(all_moduli, key=lambda m: (m % 2 != 0, m)),
    ]
    
    for order in orderings:
        if time.time() > deadline - 30:
            break
        result = try_greedy_order(order)
        if len(result) > len(best_assignment):
            best_assignment = result
    
    # Now do local search to improve
    if time.time() < deadline - 10:
        best_assignment = local_search(best_assignment, max_iters=50000)
    
    # Try multiple local search restarts from different initial solutions
    import random
    rng = random.Random(123)
    
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        # Create a perturbed initial solution
        perturbed = dict(best_assignment)
        # Remove a few random elements
        keys = list(perturbed.keys())
        rng.shuffle(keys)
        for k in keys[:3]:
            del perturbed[k]
        
        # Try to rebuild
        result = try_greedy_order(sorted(perturbed.keys()) + [m for m in all_moduli if m not in perturbed])
        if len(result) > len(best_assignment):
            best_assignment = result
        
        if time.time() < deadline - 10:
            improved = local_search(result, max_iters=20000)
            if len(improved) > len(best_assignment):
                best_assignment = improved
    
    # Final verification
    moduli_list = list(best_assignment.keys())
    residues_list = [best_assignment[m] for m in moduli_list]
    
    if not is_pairwise_disjoint(residues_list, moduli_list):
        # Fallback: use a known valid small solution
        # Two even moduli with different residues mod 2... wait, they'd both be even.
        # m=2, r=0: class is all even numbers
        # m=4, r=1: class is 1,5,9,... (odd numbers ≡1 mod 4)
        # gcd(2,4)=2, r1%2=0, r2%2=1, 0!=1, so disjoint. Good.
        best_assignment = {2: 0, 4: 1}
        moduli_list = list(best_assignment.keys())
        residues_list = [best_assignment[m] for m in moduli_list]
    
    # Ensure canonical residues
    result = []
    for m in moduli_list:
        r = best_assignment[m] % m
        result.append([r, m])
    
    # Sort by modulus for determinism
    result.sort(key=lambda x: x[1])
    
    return result
# EVOLVE-BLOCK-END
