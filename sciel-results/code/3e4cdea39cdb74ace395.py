import math

def construct():
    max_mod = 128
    
    def are_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)

    def solve_residues(moduli):
        n = len(moduli)
        if n == 0:
            return []
        
        # Backtracking to assign residues
        residues = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            for r in range(moduli[idx]):
                residues[idx] = r
                ok = True
                for j in range(idx):
                    if not are_disjoint(r, moduli[idx], residues[j], moduli[j]):
                        ok = False
                        break
                if ok:
                    if backtrack(idx + 1):
                        return True
            residues[idx] = None
            return False

        if backtrack(0):
            return [residues[i] for i in range(n)]
        return None

    best_solution = []
    
    # Strategy: Iterate over base moduli K
    # For each K, consider multiples K*i <= 128
    # We want to select a subset of these multiples.
    # To make it easy, we select multipliers i that are pairwise coprime.
    # This ensures gcd(K*i, K*j) = K * gcd(i,j) = K.
    # Then we just need distinct residues mod K.
    # If n <= K, we can always assign distinct residues mod K (e.g., 0..n-1).
    # Wait, if we assign 0..n-1, do they satisfy the condition?
    # Yes, if n <= K, 0..n-1 are distinct mod K.
    # So if we find a set of pairwise coprime multipliers of size n, and n <= K,
    # then the set of moduli is valid with residues 0..n-1.
    
    # But can we do better by using non-coprime multipliers?
    # If multipliers are not coprime, gcd > K.
    # Then we need distinct residues mod (K*d).
    # This is harder.
    # So the "pairwise coprime multipliers" strategy is a safe lower bound.
    
    # Let's find the max n for each K using pairwise coprime multipliers.
    
    def max_pairwise_coprime_count(limit):
        # Max number of pairwise coprime integers in 1..limit
        # 1 is coprime to all.
        # Then distinct primes.
        # Then products? No, if we pick primes, we are good.
        # Count of primes <= limit + 1 (for 1).
        # But we can also pick composite numbers if they don't share factors with others.
        # Greedy: pick 1, then smallest primes.
        count = 0
        used_primes = set()
        # Include 1
        count += 1
        d = 2
        while d <= limit:
            is_prime = True
            for p in used_primes:
                if d % p == 0:
                    is_prime = False
                    break
            if is_prime:
                count += 1
                used_primes.add(d)
            d += 1
        return count

    for K in range(2, max_mod + 1):
        limit = max_mod // K
        if limit < 1:
            continue
        
        # Get pairwise coprime multipliers
        # We want the largest set.
        # Simple greedy: 1, then primes.
        multipliers = [1]
        used_primes = set()
        d = 2
        while d <= limit:
            is_prime = True
            for p in used_primes:
                if d % p == 0:
                    is_prime = False
                    break
            if is_prime:
                multipliers.append(d)
                used_primes.add(d)
            d += 1
        
        n = len(multipliers)
        # We need n <= K for the simple residue assignment to work?
        # Actually, if we use residues 0..n-1, they are distinct mod K if n <= K.
        # If n > K, we can't have distinct residues mod K.
        # But we can use a backtracking solver to see if a solution exists.
        
        moduli = [K * i for i in multipliers]
        
        # Try to solve with backtracking
        sol = solve_residues(moduli)
        if sol:
            if len(sol) > len(best_solution):
                best_solution = list(zip(sol, moduli))
        
        # Also, maybe we can add more multipliers that are NOT pairwise coprime?
        # The backtracking solver will handle the constraints.
        # But the search space for multipliers is large.
        # However, for small K, limit is large.
        # For K=2, limit=64.
        # Pairwise coprime multipliers: 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61.
        # Count ~ 18.
        # K=2. Need distinct mod 2? No, gcd is 2.
        # If multipliers are pairwise coprime, gcd is 2.
        # Need distinct mod 2. Max 2.
        # So my logic "n <= K" is correct for the simple case.
        # If n > K, the simple assignment fails, but backtracking might find a solution
        # if the actual gcds are larger for some pairs?
        # No, if multipliers are pairwise coprime, gcds are exactly K.
        # So if n > K, it's impossible.
        
        # So for pairwise coprime multipliers, we are capped at K.
        
        # What if we don't require pairwise coprime multipliers?
        # Then gcds can be larger.
        # Example K=2.
        # Multipliers 2, 4. Moduli 4, 8.
        # gcd(4,8)=4.
        # Need distinct mod 4.
        # We can have 4 moduli if we pick multipliers with pairwise gcd 1?
        # No, if we pick 2,4,6,8.
        # gcd(4,8)=4. gcd(4,6)=2. gcd(6,8)=2.
        # This is mixed.
        
        # Let's just run backtracking on the set of ALL multiples of K?
        # That set is too large for backtracking (2^limit).
        
        # Let's stick to the pairwise coprime strategy for a baseline,
        # and then try to expand the set by adding one more modulus at a time?
        
        # Actually, the current best score is 2.5714.
        # The seed program likely gets around 10-15 classes.
        # My K=24 example gave 5.
        # K=6 gave 6.
        # K=30?
        # Limit 4. Multipliers 1, 2, 3, 4?
        # 1,2,3 are pairwise coprime. 4 shares with 2.
        # Set {1, 2, 3}. Moduli 30, 60, 90.
        # gcds 30.
        # Need distinct mod 30.
        # 3 <= 30. Works.
        
        # Can we get more?
        # Try K=12.
        # Limit 10.
        # Pairwise coprime: 1, 2, 3, 5, 7.
        # Count 5.
        # Moduli 12, 24, 36, 60, 84.
        # gcds 12.
        # Need distinct mod 12.
        # 5 <= 12. Works.
        
        # Try K=6.
        # Limit 21.
        # Pairwise coprime: 1, 2, 3, 5, 7, 11, 13, 17, 19.
        # Count 9.
        # 9 <= 6? No.
        # So simple assignment fails.
        # But backtracking might work if some gcds are larger?
        # No, pairwise coprime means gcds are exactly 6.
        # So max 6.
        
        # So the max size is roughly max K such that we can find K pairwise coprime multipliers?
        # No, we are capped by K.
        # So we want to maximize min(K, count_pairwise_coprime(128//K)).
        
        # Let's just run the backtracking solver on the set of pairwise coprime multipliers
        # and see if it finds a solution even if n > K (it shouldn't, but just in case).
        
    if not best_solution:
        return [[0, 2]]
        
    return [[r, m] for r, m in best_solution]
