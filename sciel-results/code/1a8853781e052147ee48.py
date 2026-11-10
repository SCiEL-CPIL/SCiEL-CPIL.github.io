"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START

def _factorize(n):
    """Return dict of prime -> exponent for n."""
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

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline to be safe within 90s
    
    MAX_MOD = 128
    
    # Precompute factorizations for all moduli 2..128
    factorizations = {}
    for m in range(2, MAX_MOD + 1):
        factorizations[m] = _factorize(m)
    
    # A modulus m is "compatible" with a set of selected moduli if gcd(m, m_i) == 1 for all selected m_i
    # Actually, the criterion is: two residue classes (r1, m1) and (r2, m2) are disjoint iff gcd(m1, m2) == 1
    # OR if gcd(m1, m2) > 1, then we need r1 ≢ r2 (mod gcd(m1, m2))
    # Wait, the task says "the verifier uses the exact gcd criterion". Let me re-read.
    # "Every two infinite residue classes must be disjoint; the verifier uses the exact gcd criterion."
    # For two arithmetic progressions a1 + m1*k and a2 + m2*k to be disjoint:
    # The system x ≡ a1 (mod m1), x ≡ a2 (mod m2) has a solution iff a1 ≡ a2 (mod gcd(m1, m2)).
    # So they are disjoint iff a1 ≢ a2 (mod gcd(m1, m2)).
    # If gcd(m1, m2) == 1, then there's always a solution (CRT), so they are NEVER disjoint!
    # Wait, that's wrong. If gcd(m1, m2) == 1, then by CRT, for ANY a1, a2, there exists x such that
    # x ≡ a1 (mod m1) and x ≡ a2 (mod m2). So the two classes INTERSECT.
    # So for disjointness, we NEED gcd(m1, m2) > 1 AND a1 ≢ a2 (mod gcd(m1, m2)).
    # 
    # Actually wait, let me reconsider. The current solution uses moduli 4, 8, 16, 32, 64, 128, 256... no wait.
    # Current: [(1<<k)-1, 1<<(k+1)] for k in range(7) -> residues 1,3,7,15,31,63,127 with moduli 2,4,8,16,32,64,128.
    # gcd(2,4)=2, 1 mod 2 = 1, 3 mod 2 = 1. So 1 ≡ 3 (mod 2). They INTERSECT!
    # Hmm, but the score is 1.0 which is valid. Let me re-read the task.
    
    # "Many disjoint residue classes with distinct moduli"
    # "Every two infinite residue classes must be disjoint"
    
    # Wait, maybe I'm misunderstanding. Let me check: class (1, 2) is all odd numbers. Class (3, 4) is 3, 7, 11, 15, ...
    # 3 is odd, so 3 is in both classes. They intersect!
    
    # But the current program has score 1.0, meaning it's valid. So either my understanding is wrong,
    # or the task definition is different.
    
    # Let me re-read: "the verifier uses the exact gcd criterion"
    # Maybe the criterion is: classes are disjoint iff gcd(m1, m2) == 1? No, that's the opposite.
    
    # Actually, maybe the task means something different by "disjoint". Or maybe the current program
    # is actually invalid but scores 1.0 as a baseline? No, "1.0 equals the seed program".
    
    # Let me reconsider. Perhaps the criterion is that the classes must be pairwise disjoint,
    # and the "exact gcd criterion" is the standard one: (r1,m1) and (r2,m2) are disjoint iff
    # r1 ≢ r2 (mod gcd(m1,m2)).
    
    # For the current solution: (1,2) and (3,4): gcd(2,4)=2, 1 mod 2 = 1, 3 mod 2 = 1. 1 ≡ 1 (mod 2). NOT disjoint.
    
    # This is confusing. Let me just assume the current solution is valid and try to improve it.
    # Maybe the task is asking for something else, or maybe I'm wrong about the math.
    
    # Actually, wait. Let me re-read more carefully. Maybe "disjoint" in this context means something specific.
    # Or maybe the current program is a baseline that happens to score 1.0 but isn't optimal.
    
    # Let me try a different approach: use moduli that are pairwise coprime. If gcd(m1,m2)=1 for all pairs,
    # then by CRT, any two classes intersect. So that's NOT what we want.
    
    # We want gcd(m1,m2) > 1 for all pairs, and r1 ≢ r2 (mod gcd(m1,m2)).
    
    # The current solution uses powers of 2: 2,4,8,16,32,64,128. All have gcd > 1 with each other.
    # But as I showed, (1,2) and (3,4) intersect.
    
    # I'm going to assume the task is as stated and try to find a better solution.
    # Let me try using moduli that share a common factor, e.g., all even, and choose residues carefully.
    
    # Actually, let me just try to maximize the number of classes using a greedy approach.
    
    # Strategy: Select moduli and residues such that for every pair (i,j), r_i ≢ r_j (mod gcd(m_i, m_j)).
    
    # Greedy approach:
    # 1. Sort moduli in some order.
    # 2. For each modulus, try to find a residue that is compatible with all already-selected classes.
    
    # Let's try this:
    
    def is_compatible(residue, modulus, selected):
        """Check if (residue, modulus) is disjoint from all in selected."""
        for (r_s, m_s) in selected:
            g = _gcd(modulus, m_s)
            if g > 1:
                if residue % g == r_s % g:
                    return False
        return True
    
    def solve_greedy(moduli_order, residue_strategy):
        """Greedy selection with given modulus order and residue strategy."""
        selected = []
        for m in moduli_order:
            if m < 2 or m > MAX_MOD:
                continue
            # Try residues in order determined by strategy
            if residue_strategy == 'ascending':
                residues = range(m)
            elif residue_strategy == 'descending':
                residues = range(m - 1, -1, -1)
            else:
                residues = list(range(m))
                random.shuffle(residues)
            
            found = False
            for r in residues:
                if is_compatible(r, m, selected):
                    selected.append((r, m))
                    found = True
                    break
            if not found:
                # Try all residues more carefully
                for r in range(m):
                    if is_compatible(r, m, selected):
                        selected.append((r, m))
                        found = True
                        break
        return selected
    
    # Try different orders
    moduli_list = list(range(2, MAX_MOD + 1))
    
    # Sort by number of prime factors (ascending) - fewer shared factors might be easier
    def prime_factor_count(m):
        return sum(factorizations[m].values())
    
    orders = [
        sorted(moduli_list, key=prime_factor_count),  # Fewest prime factors first
        sorted(moduli_list, key=lambda m: -prime_factor_count(m)),  # Most prime factors first
        sorted(moduli_list, reverse=True),  # Largest first
        sorted(moduli_list),  # Smallest first
    ]
    
    best_solution = []
    best_count = 0
    
    strategies = ['ascending', 'descending']
    
    for order in orders:
        for strategy in strategies:
            if time.time() > deadline:
                break
            solution = solve_greedy(order, strategy)
            if len(solution) > best_count:
                best_count = len(solution)
                best_solution = solution
    
    # Also try with randomized orders
    for _ in range(50):
        if time.time() > deadline:
            break
        random_order = moduli_list[:]
        random.shuffle(random_order)
        solution = solve_greedy(random_order, 'ascending')
        if len(solution) > best_count:
            best_count = len(solution)
            best_solution = solution
    
    # Local search: try to swap or improve
    if time.time() < deadline - 5:
        # Try to add more classes by removing some and re-adding
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            current = best_solution[:]
            # Try removing a random element and re-solving
            if len(current) > 1:
                idx = random.randint(0, len(current) - 1)
                removed = current.pop(idx)
                # Try to re-add with different residue
                m = removed[1]
                for r in range(m):
                    if is_compatible(r, m, current):
                        current.append((r, m))
                        break
                # Try adding new moduli
                used_moduli = set(m for (_, m) in current)
                for m in range(2, MAX_MOD + 1):
                    if m not in used_moduli:
                        for r in range(m):
                            if is_compatible(r, m, current):
                                current.append((r, m))
                                break
                if len(current) > len(best_solution):
                    best_solution = current
                    best_count = len(current)
    
    # Convert to required format
    return [[r, m] for (r, m) in best_solution]

# EVOLVE-BLOCK-END
