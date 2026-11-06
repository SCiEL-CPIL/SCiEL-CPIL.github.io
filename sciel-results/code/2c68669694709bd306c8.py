"""Solve: maximize pairwise-disjoint residue classes with distinct moduli in [2,128].

Two residue classes (r1,m1) and (r2,m2) are disjoint iff gcd(m1,m2) does NOT divide (r1-r2).
Equivalently, for each pair we need (r1 - r2) % gcd(m1,m2) != 0.

Strategy:
1. Enumerate all moduli 2..128.
2. Use a backtracking search with strong pruning:
   - Sort moduli by a heuristic (e.g., number of prime factors descending, or value ascending).
   - For each modulus, try residues 0..m-1.
   - Prune: for a candidate residue r with modulus m, check against all already-assigned (r_i, m_i):
     let g = gcd(m, m_i). We need (r - r_i) % g != 0.
   - Use forward checking / constraint propagation to order residues.
3. To maximize count, do a best-first or iterative-deepening search:
   - Start by trying to find a solution of size N = 127 (all moduli), then N=126, etc.
   - Actually, that's too slow. Instead, use a greedy + local search or a smarter backtracking.

Better approach:
- Key insight: if two moduli share a prime factor p, then their residues must differ mod p.
- If gcd(m1,m2) = g, then residues must differ mod g (i.e., not congruent mod g).
- This is a constraint satisfaction problem.

For practicality within 60s:
- Precompute gcd table.
- Use a greedy heuristic: process moduli in order of "constraint tightness" (highest gcd with others first).
- Then do a limited backtracking.

Alternative: model as graph coloring where each modulus is a vertex, and edges represent constraints.
Actually, the constraint is: for each pair (i,j), r_i != r_j mod gcd(m_i, m_j).

Let's try a backtracking with:
- Order moduli by descending "density" of constraints (sum of gcd with all other moduli).
- For each modulus, try residues in a smart order.
- Keep the best solution found.

Since 127 moduli is a lot, but many moduli are coprime to each other (no constraint), we can:
- Group moduli by their prime factor structure.
- Moduli that are pairwise coprime have no constraint between them.

Let me think about this differently. The maximum number of classes is at most 127.
The seed gives 7. We want to do much better.

Key observation: if we pick moduli that are all powers of 2, then gcd(2^a, 2^b) = 2^min(a,b).
We need residues r_a (mod 2^a) such that for a < b, r_a != r_b mod 2^a.
This is like a "Chinese Remainder" style constraint.

Actually, let me just do a backtracking search with good heuristics and a time limit.
"""

from math import gcd
from functools import lru_cache
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s to leave safety margin

    # All moduli from 2 to 128
    all_moduli = list(range(2, 129))
    
    # Precompute gcd for all pairs
    n = len(all_moduli)
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i):
            g = gcd(all_moduli[i], all_moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Heuristic: order moduli by how "constrained" they are.
    # A modulus is more constrained if it has large gcds with many others.
    # Compute sum of gcds with all other moduli for each modulus
    constraint_scores = [0] * n
    for i in range(n):
        s = 0
        for j in range(n):
            if i != j:
                s += gcd_table[i][j]
        constraint_scores[i] = s
    
    # Sort indices by descending constraint score (most constrained first)
    order = sorted(range(n), key=lambda i: -constraint_scores[i])
    
    # Residue assignments: res[i] = residue for modulus all_moduli[i], or -1 if unassigned
    res = [-1] * n
    
    best_solution = []
    best_count = 0
    
    # For the backtracking, we'll process in the order defined above.
    # assigned[k] = index in all_moduli that is the k-th assigned
    # We'll use a recursive backtracking with pruning.
    
    def is_compatible(idx, r, assigned_indices):
        """Check if assigning residue r to modulus all_moduli[idx] is compatible with all assigned."""
        for ai in assigned_indices:
            g = gcd_table[idx][ai]
            if g > 1:
                # Need (r - res[ai]) % g != 0
                if (r - res[ai]) % g == 0:
                    return False
        return True
    
    def backtrack(pos, assigned_indices):
        nonlocal best_solution, best_count
        
        if time.time() > deadline:
            return
        
        # Try to assign the next modulus in our order
        if pos >= n:
            # All assigned
            if len(assigned_indices) > best_count:
                best_count = len(assigned_indices)
                best_solution = [(res[all_moduli.index(m)], m) for m in all_moduli if res[all_moduli.index(m)] >= 0]
            return
        
        idx = order[pos]
        m = all_moduli[idx]
        
        # Try residues 0 to m-1
        for r in range(m):
            if is_compatible(idx, r, assigned_indices):
                res[idx] = r
                assigned_indices.append(idx)
                backtrack(pos + 1, assigned_indices)
                assigned_indices.pop()
                res[idx] = -1
        
        # If we didn't find a valid residue for this modulus, skip it
        # (this allows us to find a large subset)
    
    # The full backtracking above is too slow for 127 moduli.
    # Let's use a greedy approach with backtracking for conflicts.
    
    def greedy_with_backtracking():
        nonlocal best_solution, best_count
        
        # Process moduli in order of constraint (most constrained first)
        # But actually, for maximizing count, we should try to include as many as possible.
        # Greedy: for each modulus in order, try to assign a residue. If can't, skip.
        
        assigned = []  # list of (modulus_index, residue)
        assigned_set = set()
        
        for idx in order:
            m = all_moduli[idx]
            found = False
            for r in range(m):
                compatible = True
                for (ai, ar) in assigned:
                    g = gcd_table[idx][ai]
                    if g > 1 and (r - ar) % g == 0:
                        compatible = False
                        break
                if compatible:
                    assigned.append((idx, r))
                    assigned_set.add(idx)
                    found = True
                    break
            
            if time.time() > deadline:
                break
        
        if len(assigned) > best_count:
            best_count = len(assigned)
            best_solution = [(r, all_moduli[idx]) for (idx, r) in assigned]
        
        return len(assigned)
    
    greedy_with_backtracking()
    
    # Try a local search / improvement: try to add more moduli or reassign residues
    # Actually, let's try a different ordering and see if we get better results
    
    # Try ascending order of modulus
    order2 = sorted(range(n), key=lambda i: all_moduli[i])
    assigned2 = []
    for idx in order2:
        m = all_moduli[idx]
        found = False
        for r in range(m):
            compatible = True
            for (ai, ar) in assigned2:
                g = gcd_table[idx][ai]
                if g > 1 and (r - ar) % g == 0:
                    compatible = False
                    break
            if compatible:
                assigned2.append((idx, r))
                found = True
                break
        if time.time() > deadline:
            break
    
    if len(assigned2) > best_count:
        best_count = len(assigned2)
        best_solution = [(r, all_moduli[idx]) for (idx, r) in assigned2]
    
    # Try descending order of modulus
    order3 = sorted(range(n), key=lambda i: -all_moduli[i])
    assigned3 = []
    for idx in order3:
        m = all_moduli[idx]
        found = False
        for r in range(m):
            compatible = True
            for (ai, ar) in assigned3:
                g = gcd_table[idx][ai]
                if g > 1 and (r - ar) % g == 0:
                    compatible = False
                    break
            if compatible:
                assigned3.append((idx, r))
                found = True
                break
        if time.time() > deadline:
            break
    
    if len(assigned3) > best_count:
        best_count = len(assigned3)
        best_solution = [(r, all_moduli[idx]) for (idx, r) in assigned3]
    
    # Try: prioritize moduli with many prime factors (more constrained)
    def num_prime_factors(m):
        count = 0
        d = 2
        while d * d <= m:
            while m % d == 0:
                count += 1
                m //= d
            d += 1
        if m > 1:
            count += 1
        return count
    
    order4 = sorted(range(n), key=lambda i: -num_prime_factors(all_moduli[i]))
    assigned4 = []
    for idx in order4:
        m = all_moduli[idx]
        found = False
        for r in range(m):
            compatible = True
            for (ai, ar) in assigned4:
                g = gcd_table[idx][ai]
                if g > 1 and (r - ar) % g == 0:
                    compatible = False
                    break
            if compatible:
                assigned4.append((idx, r))
                found = True
                break
        if time.time() > deadline:
            break
    
    if len(assigned4) > best_count:
        best_count = len(assigned4)
        best_solution = [(r, all_moduli[idx]) for (idx, r) in assigned4]
    
    # Try a combined approach: use the best greedy result, then try to improve
    # by doing a local search on residues
    
    if best_solution and time.time() < deadline - 10:
        # Local search: try to reassign residues to allow adding more moduli
        # For simplicity, try to add one more modulus at a time by reassigning
        # This is complex; skip for now.
        
        # Try a different heuristic: group by largest prime factor
        def largest_prime_factor(m):
            lp = 2
            d = 2
            while d * d <= m:
                while m % d == 0:
                    lp = d
                    m //= d
                d += 1
            if m > 1:
                lp = m
            return lp
        
        order5 = sorted(range(n), key=lambda i: (largest_prime_factor(all_moduli[i]), -all_moduli[i]))
        assigned5 = []
        for idx in order5:
            m = all_moduli[idx]
            found = False
            for r in range(m):
                compatible = True
                for (ai, ar) in assigned5:
                    g = gcd_table[idx][ai]
                    if g > 1 and (r - ar) % g == 0:
                        compatible = False
                        break
                if compatible:
                    assigned5.append((idx, r))
                    found = True
                    break
            if time.time() > deadline:
                break
        
        if len(assigned5) > best_count:
            best_count = len(assigned5)
            best_solution = [(r, all_moduli[idx]) for (idx, r) in assigned5]
    
    # Verify the solution
    def verify(solution):
        if not solution:
            return False
        moduli = [m for r, m in solution]
        if len(moduli) != len(set(moduli)):
            return False
        if any(m < 2 or m > 128 for m in moduli):
            return False
        if any(r < 0 or r >= m for r, m in solution):
            return False
        for i in range(len(solution)):
            for j in range(i + 1, len(solution)):
                r1, m1 = solution[i]
                r2, m2 = solution[j]
                g = gcd(m1, m2)
                if (r1 - r2) % g == 0:
                    return False
        return True
    
    if best_solution and verify(best_solution):
        return best_solution
    
    # Fallback: return a known valid small solution
    # Powers of 2: 2,4,8,16,32,64,128 with residues 1, 3, 7, 15, 31, 63, 127
    # Check: gcd(2^a, 2^b) = 2^min(a,b). Residues: 2^k - 1.
    # (2^a - 1) - (2^b - 1) = 2^a - 2^b = 2^b(2^(a-b) - 1) for a > b
    # gcd = 2^b. Need 2^b(2^(a-b)-1) not divisible by 2^b => 2^(a-b)-1 not divisible by 1, which is fine.
    # Actually: (r_a - r_b) % 2^b = (2^a - 2^b) % 2^b = 0. That's bad!
    # So that doesn't work. Let me use a different set.
    
    # Use moduli 3, 5, 7, 11, 13, 17, 19, 23, 29, 31 (primes)
    # For primes, gcd is 1, so no constraint. All pairs are compatible.
    prime_moduli = [m for m in range(2, 129) if all(m % d != 0 for d in range(2, int(m**0.5) + 1))]
    solution = [(0, m) for m in prime_moduli]
    if verify(solution):
        return solution
    
    # Ultimate fallback
    return [[0, 2]]

BET: {mechanism: "Multiple greedy heuristics with different ordering strategies (constraint-based, ascending, descending, prime-factor-based) explore diverse solution spaces; the best among them is returned, likely exceeding the seed's 7 classes by finding 10+ valid disjoint residue classes", q: 0.6, delta: [0.1, 0.5]}
