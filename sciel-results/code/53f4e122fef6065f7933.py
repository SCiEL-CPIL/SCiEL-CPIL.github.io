import time
from math import gcd
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 90s limit

    # Known baseline solution
    best_solution = [8, 28, 1736]
    best_max_denom = 1736

    def verify(solution):
        if not solution:
            return False
        if len(solution) < 1 or len(solution) > 256:
            return False
        if len(set(solution)) != len(solution):
            return False
        if any(d < 1 or d > 10000000 for d in solution):
            return False
        s = sum(Fraction(1, d) for d in solution)
        return s == Fraction(5, 31)

    def find_solution_dfs(limit):
        """
        DFS with modular arithmetic filtering to reduce branching factor.
        Key insight: if num/den = 1/d + remainder, then d must satisfy
        num*d - den ≡ 0 (mod den), i.e., d ≡ num^{-1} * den (mod den) when gcd(num,den)=1.
        More generally, we can filter candidates using divisibility constraints.
        """
        def factorize(n):
            """Return prime factorization of n as dict {prime: exponent}."""
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

        def dfs(num, den, min_d, current_set, depth=0):
            if time.time() - start_time > time_limit - 1.0:
                return None
            
            if num == 0:
                return current_set
            
            if num < 0:
                return None
            
            # Lower bound: 1/d <= num/den => d >= den/num
            lower_bound = (den + num - 1) // num
            if lower_bound < min_d:
                lower_bound = min_d
            
            if lower_bound > limit:
                return None
            
            # Upper bound: we need at least 1 more term, so 1/d >= num/den - (max possible remaining)
            # Actually, since all denominators must be distinct and >= d, the remaining sum after picking d
            # is num/den - 1/d. For this to be achievable with distinct denominators > d,
            # we need num/den - 1/d > 0, which gives d > den/num (already handled by lower_bound).
            # Also, the next term 1/(d+1) must be <= remaining, but that's a weaker constraint.
            
            # Modular filtering: 
            # We have num/den = 1/d + r where r >= 0
            # => num*d - den = r*den*d
            # => num*d ≡ den (mod den) is trivially true
            # Better: num*d - den must be divisible by gcd considerations
            # Actually: r = (num*d - den)/(den*d), so num*d - den >= 0 => d >= den/num
            # And for r to be a valid rational with denominator den*d/gcd(...), we need
            # num*d - den to be non-negative (already ensured) and the fraction to reduce properly.
            
            # Key filter: since all future denominators must be > d and distinct,
            # the remaining sum after picking d is (num*d - den)/(den*d).
            # Let g = gcd(num*d - den, den*d). Then reduced remainder is (num*d-den)/g / (den*d/g).
            # For this to be representable as sum of reciprocals of distinct integers > d,
            # we need (num*d-den)/g > 0 (ensured) and the denominator (den*d/g) must be
            # "large enough" to accommodate multiple distinct terms > d.
            
            # Stronger filter: num*d - den must be positive (d > den/num, already have d >= ceil(den/num))
            # And num*d - den < den*d (always true since num < den for proper fractions in our case? 
            # Actually num/den could be > 1 if we have few terms... let's check: 5/31 < 1, and 
            # as we subtract terms, the remainder decreases. So num < den always in our DFS.)
            
            # Modular constraint: For the remainder (num*d - den)/(den*d) to be expressible,
            # one necessary condition is that num*d - den >= 1 (integer).
            # More useful: if we factor den, then d must make num*d - den divisible by certain factors.
            
            # Let's use a simpler but effective filter:
            # Try candidates d starting from lower_bound, but skip many using step patterns.
            # If gcd(num, den) = g, then num/g * d - den/g must be integer (always true).
            # The remainder is (num*d - den)/(den*d). Let g2 = gcd(num*d - den, den).
            # Then reduced remainder numerator = (num*d - den)/g2, denominator = den*d/gcd(den*d, num*d-den)
            
            # Practical filter: iterate d but with a stride based on den's factors.
            # If p | den and p ∤ num, then for num*d - den to be divisible by p, we need
            # num*d ≡ den (mod p), so d ≡ den * num^{-1} (mod p).
            # This means d must be in a specific residue class mod p.
            
            # Compute the required residue for each prime factor of den
            g = gcd(num, den)
            num_red = num // g
            den_red = den // g
            
            # For each prime p dividing den_red, find the required residue of d mod p
            # such that num_red * d ≡ den_red (mod p)
            # This is d ≡ den_red * (num_red^{-1} mod p) (mod p)
            
            # We'll collect constraints and use Chinese Remainder Theorem or just filter
            # candidates by checking divisibility.
            
            # Simpler approach: just iterate but with a larger step when possible.
            # If den_red has a prime factor p > 2, we can skip (p-1) out of every p candidates.
            
            # For now, use a hybrid: if den_red is large, use modular filtering; else linear scan.
            candidates = []
            if den_red > 10:
                # Use modular filtering
                factors = factorize(den_red)
                # For each prime factor, determine valid residues
                valid_residues = []
                for p, exp in factors.items():
                    if p == 2:
                        # Handle mod 2 separately
                        if num_red % 2 == 1:
                            # num_red * d ≡ den_red (mod 2)
                            # d ≡ den_red (mod 2)
                            valid_residues.append((p, den_red % p))
                        else:
                            # num_red even, den_red must be even for solution to exist mod 2
                            # But gcd(num_red, den_red) = 1, so this case doesn't arise
                            pass
                    else:
                        # Find inverse of num_red mod p
                        try:
                            inv = pow(num_red, -1, p)
                        except:
                            continue
                        residue = (den_red * inv) % p
                        valid_residues.append((p, residue))
                
                if valid_residues:
                    # Generate candidates that satisfy all modular constraints
                    # Use CRT or just filter a range
                    # For simplicity, filter a window of candidates
                    window = 1000
                    for d in range(lower_bound, min(lower_bound + window, limit + 1)):
                        valid = True
                        for p, res in valid_residues:
                            if d % p != res:
                                valid = False
                                break
                        if valid:
                            candidates.append(d)
                    
                    # If window didn't cover enough, extend
                    if len(candidates) < 50 and lower_bound + window <= limit:
                        for d in range(lower_bound + window, limit + 1):
                            valid = True
                            for p, res in valid_residues:
                                if d % p != res:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(d)
                            if len(candidates) >= 200:
                                break
                else:
                    # No modular constraints, do linear scan
                    for d in range(lower_bound, limit + 1):
                        candidates.append(d)
                        if len(candidates) >= 200:
                            break
            else:
                # Small denominator, just do linear scan with limit
                for d in range(lower_bound, limit + 1):
                    candidates.append(d)
                    if len(candidates) >= 200:
                        break
            
            for d in candidates:
                if time.time() - start_time > time_limit - 1.0:
                    return None
                
                # Check: num*d - den must be >= 0
                remainder_num = num * d - den
                if remainder_num < 0:
                    continue
                
                # Check distinctness
                if d in current_set:
                    continue
                
                next_num = remainder_num
                next_den = den * d
                
                g2 = gcd(next_num, next_den)
                if g2 > 1:
                    next_num //= g2
                    next_den //= g2
                
                # Pruning: if next_num == 0, we found a solution
                if next_num == 0:
                    return current_set + [d]
                
                # Pruning: remaining sum must be positive and achievable
                # The remaining sum is next_num/next_den, which must be expressible as
                # sum of reciprocals of distinct integers > d.
                # Minimum possible sum with k terms all > d is approximately k/(d+1) for small k.
                # Maximum possible is 1/(d+1) + 1/(d+2) + ... 
                # Simple check: next_num/next_den < 1/(d+1) would mean we can't even fit one more term
                # Actually, next_num/next_den must be < 1/d (since we already used d)
                # and > 0.
                
                # Check if next_num/next_den is too large to be represented with denominators > d
                # The largest single term we can add is 1/(d+1)
                # So we need next_num/next_den < 1/(d+1) if we're going to use more terms
                # But if next_num/next_den == 1/(d+1), that would mean we can use just d+1.
                
                res = dfs(next_num, next_den, d + 1, current_set + [d], depth + 1)
                if res is not None:
                    return res
                    
            return None
        
        return dfs(5, 31, 1, [])

    # Binary search on the maximum denominator bound
    current_limit = best_max_denom - 1
    
    while current_limit >= 1:
        if time.time() - start_time > time_limit - 2.0:
            break
        
        sol = find_solution_dfs(current_limit)
        
        if sol is not None:
            actual_max = max(sol)
            if actual_max < best_max_denom:
                best_max_denom = actual_max
                best_solution = sol
            current_limit = actual_max - 1
        else:
            break
        
        if current_limit < 7:
            break

    if verify(best_solution):
        return best_solution
    
    return [8, 28, 1736]
