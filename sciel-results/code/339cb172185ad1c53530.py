import math
import random
from functools import lru_cache

def construct():
    max_mod = 128
    
    # Precompute prime factorizations and prime powers for each modulus
    # For each m, we need to know its distinct prime factors and their powers
    
    def prime_factorization(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    # For each modulus, get its prime factorization
    factorizations = {}
    for m in range(2, max_mod + 1):
        factorizations[m] = prime_factorization(m)
    
    # Two moduli m1 and m2 are "incompatible" if they share a prime factor
    # We need to select moduli such that any two selected moduli are coprime
    # (since if gcd(m1, m2) > 1, we need (r1 % g) != (r2 % g) for some g > 1)
    
    # Actually, the key insight: if m1 and m2 are coprime, then gcd(m1, m2) = 1
    # and (r1 % 1) == (r2 % 1) == 0 always, so they are NEVER disjoint!
    # Wait, let me re-read: "Every two infinite residue classes must be disjoint"
    # are_disjoint returns (r1 % g) != (r2 % g) where g = gcd(m1, m2)
    # If g = 1, then r1 % 1 = 0 and r2 % 1 = 0, so they are NOT disjoint.
    # So coprime moduli give NON-disjoint classes!
    
    # This means we need moduli that are NOT coprime (share a prime factor)
    # and we need to choose residues such that for each shared prime power,
    # the residues differ mod that prime power.
    
    # Let me reconsider: g = gcd(m1, m2). We need (r1 % g) != (r2 % g).
    # This is equivalent to: there exists some prime p dividing both m1 and m2
    # such that r1 != r2 (mod p^k) where p^k is the highest power of p in both.
    
    # Strategy: Group moduli by their "prime signature" - the set of distinct primes
    # Moduli that share at least one prime can potentially be disjoint.
    # Moduli that are coprime cannot be disjoint.
    
    # So we want to select a set of moduli where every pair shares at least one prime,
    # AND we can assign residues so that for every pair, they differ mod gcd.
    
    # This is a complex constraint satisfaction problem.
    
    # Let's use a backtracking approach with good heuristics.
    
    def are_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    # Precompute gcd for all pairs
    gcd_table = {}
    for m1 in range(2, max_mod + 1):
        for m2 in range(m1 + 1, max_mod + 1):
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
            gcd_table[(m2, m1)] = math.gcd(m1, m2)
    
    # Sort moduli by number of distinct prime factors (more flexible first?)
    # Or by number of prime factors total
    def distinct_primes(m):
        return len(factorizations[m])
    
    def total_prime_factors(m):
        return sum(factorizations[m].values())
    
    # Strategy: Use maximum bipartite matching style greedy
    # First, let's try to find a large set of moduli where every pair shares a prime.
    # This is a clique problem in a graph where edges connect moduli that share a prime.
    
    # Build compatibility graph: edge between m1, m2 if gcd(m1, m2) > 1
    moduli_list = list(range(2, max_mod + 1))
    
    # For each modulus, compute which other moduli it's compatible with
    compat = {m: set() for m in moduli_list}
    for i, m1 in enumerate(moduli_list):
        for m2 in moduli_list[i+1:]:
            if gcd_table[(m1, m2)] > 1:
                compat[m1].add(m2)
                compat[m2].add(m1)
    
    # Find a large clique in this compatibility graph
    # Use greedy + local search
    
    def greedy_clique():
        """Find a large clique using greedy approach."""
        remaining = set(moduli_list)
        clique = set()
        
        # Sort by degree descending
        for m in sorted(remaining, key=lambda x: -len(compat[x])):
            if m not in remaining:
                continue
            # Check if m can join clique (compatible with all in clique)
            if all(c in compat[m] for c in clique):
                clique.add(m)
                remaining.discard(m)
        
        return clique
    
    best_clique = greedy_clique()
    
    # Try to improve with local search
    for _ in range(50):
        new_clique = set(best_clique)
        improved = False
        
        # Try adding a node
        for m in moduli_list:
            if m not in new_clique:
                if all(c in compat[m] for c in new_clique):
                    new_clique.add(m)
                    improved = True
                    break
        
        if not improved:
            # Try swapping: remove one, add another
            for m_remove in list(new_clique):
                for m_add in moduli_list:
                    if m_add in new_clique:
                        continue
                    temp = new_clique - {m_remove}
                    if all(c in compat[m_add] for c in temp):
                        new_clique = temp | {m_add}
                        improved = True
                        break
                if improved:
                    break
        
        if len(new_clique) > len(best_clique):
            best_clique = new_clique
        else:
            if not improved:
                break
    
    # Now we have a clique of moduli. Assign residues.
    # For each pair of moduli in the clique, we need residues that are disjoint.
    
    # Use backtracking to assign residues
    clique_moduli = sorted(best_clique, key=lambda m: -total_prime_factors(m))
    
    def backtrack_assign(idx, assignment):
        """
        assignment: dict from modulus to residue
        idx: index in clique_moduli
        Returns True if successful, modifies assignment
        """
        if idx == len(clique_moduli):
            return True
        
        m = clique_moduli[idx]
        # Try residues in order of "best" - maybe residues that are 0 mod small primes first?
        # Actually, we want to leave flexibility for future assignments.
        # Try all residues, but use a heuristic order.
        
        # Sort residues by how many constraints they violate/satisfy
        # For each residue r, count how many already-assigned moduli it's disjoint from
        # We want residues that are disjoint from all assigned ones
        
        candidates = []
        for r in range(m):
            ok = True
            for m2, r2 in assignment.items():
                if not are_disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                candidates.append(r)
        
        if not candidates:
            return False
        
        # Try candidates in random order to break symmetry
        random.shuffle(candidates)
        
        for r in candidates:
            assignment[m] = r
            if backtrack_assign(idx + 1, assignment):
                return True
            del assignment[m]
        
        return False
    
    # Try multiple times with different orderings
    best_assignment = None
    for trial in range(100):
        random.shuffle(clique_moduli)
        assignment = {}
        if backtrack_assign(0, assignment):
            if best_assignment is None or len(assignment) > len(best_assignment):
                best_assignment = dict(assignment)
            if len(assignment) >= 20:  # Good enough
                break
    
    if best_assignment is None:
        # Fallback: use greedy clique without residue assignment check
        # Just return moduli with residue 0
        best_assignment = {m: 0 for m in best_clique}
    
    # Convert to output format
    result = [[r, m] for m, r in best_assignment.items()]
    
    if not result:
        return [[0, 2]]
    
    return result

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Number of classes: {len(res)}")
    print(res[:10])
