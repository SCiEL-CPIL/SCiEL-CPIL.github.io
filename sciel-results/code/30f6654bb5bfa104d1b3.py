import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Start with the largest antichain: (128, 256]
    # This ensures no a | b for distinct a, b in S.
    # This reduces the problem to checking "split" divisibility a | bc.
    S = set(range(129, 257))
    
    # We need to remove elements to ensure no a | bc for distinct a,b,c.
    
    def get_conflicts(S_set):
        """
        Returns a dictionary: element -> list of tuples (role, other1, other2)
        Or simpler: count how many violations each element is part of.
        """
        counts = {x: 0 for x in S_set}
        S_list = list(S_set)
        n = len(S_list)
        
        # O(N^3) is too slow for N=128? 128^3 = 2M.
        # 2M operations is fast in Python if optimized, but let's be careful.
        # We can optimize by precomputing divisibility.
        
        # For each a, find b,c such that a | bc.
        # a | bc <=> (a / gcd(a,b)) | c.
        
        for i, a in enumerate(S_list):
            # Check a as divisor
            # We need to count pairs (b, c) with b != c, b != a, c != a
            # such that a | b*c.
            
            # Optimization:
            # If a is prime, a | bc implies a|b or a|c.
            # Since S is an antichain, a does not divide any b in S.
            # So if a is prime, it can NEVER divide bc for b,c in S (antichain).
            # Wait.
            # If a is prime p.
            # p | bc. Since p is prime, p|b or p|c.
            # But S is an antichain, so p does not divide any b in S.
            # Thus, primes in S are SAFE from being the divisor a.
            
            # What if a is composite?
            # a = p1 * p2.
            # p1 | b and p2 | c (or vice versa).
            
            # So we only need to check composite a.
            if a < 2: continue # not possible here
            # Simple primality check or just try to find factors.
            
            # Let's just iterate b, c for composites.
            # Identify composites in S.
            
            # To speed up, precompute for each a, the set of b in S such that gcd(a,b) > 1?
            # No, we need a | b*c.
            
            # Let's stick to O(N^3) but with early exits or numpy?
            # N=128. 128*128*128 = 2,097,152.
            # In Python, a loop of 2M iterations takes ~0.2s.
            # We might do this a few times. It's acceptable.
            
            pass
            
        # Let's implement the counting loop directly.
        # For efficiency, we'll just count violations for each element.
        
        violation_count = {x: 0 for x in S_set}
        
        # We check all triples (a,b,c) distinct.
        # To avoid duplicates, we can assume a is the "divisor" role.
        # But b and c are symmetric.
        # We iterate a, then b, then c.
        
        # Optimization:
        # For fixed a, we want to count pairs (b,c) such that a | bc.
        # Let g_b = gcd(a, b).
        # Condition: (a // g_b) | c.
        
        for a in S_list:
            # If a is prime, skip?
            # Check if a is prime.
            is_prime = True
            for p in range(2, int(a**0.5) + 1):
                if a % p == 0:
                    is_prime = False
                    break
            if is_prime:
                continue # Primes are safe as divisors in an antichain
            
            # For composite a
            # Count pairs (b,c)
            # We can iterate b in S_list, b != a
            # And check if (a // gcd(a,b)) divides any c in S_list, c != b, c != a.
            
            # To make this faster, precompute divisors?
            # Let req_c = a // gcd(a, b)
            # We need to know if there exists c in S such that c % req_c == 0.
            
            # We can precompute for each possible divisor d (up to 256),
            # which elements in S are divisible by d.
            
            # This precomputation depends on S, which changes.
            # So we do it inside the loop or update it.
            
            # Given the small size, let's just do the inner loop.
            
            for b in S_list:
                if b == a:
                    continue
                g = __import__('math').gcd(a, b)
                req = a // g
                if req == 1:
                    # a | b, which is impossible in antichain.
                    continue
                
                # Check if there is a c in S, c != b, c != a, such that c % req == 0
                found_c = False
                for c in S_list:
                    if c == b or c == a:
                        continue
                    if c % req == 0:
                        found_c = True
                        break
                
                if found_c:
                    # Violation found: a | b*c
                    # Increment counts for a, b, and the found c?
                    # We don't know which c exactly if we break early, but we know at least one exists.
                    # For greedy removal, we just need a score.
                    violation_count[a] += 1
                    violation_count[b] += 1
                    # We should also penalize c.
                    # But finding the specific c adds cost.
                    # Let's just penalize a and b for now.
                    # Or better, count the number of such c's.
                    # Let's refine: count all valid c's.
                    # This is expensive.
                    
        return violation_count

    # The above is O(N^3).
    # Let's try a simpler heuristic first:
    # Just check validity of the current S.
    # If invalid, remove the element with the highest "divisibility pressure".
    
    def is_valid(S_set):
        S_list = list(S_set)
        for i, a in enumerate(S_list):
            # Skip primes?
            # Check if a is prime
            is_prime = True
            for p in range(2, int(a**0.5) + 1):
                if a % p == 0:
                    is_prime = False
                    break
            if is_prime:
                continue
                
            for j, b in enumerate(S_list):
                if i == j: continue
                g = __import__('math').gcd(a, b)
                req = a // g
                if req == 1: continue
                for k, c in enumerate(S_list):
                    if k == i or k == j: continue
                    if c % req == 0:
                        return False, (a, b, c)
        return True, None

    # Iterative removal
    while time.time() < deadline:
        valid, triple = is_valid(S)
        if valid:
            break
        
        # Triple (a, b, c) is a violation.
        # We need to remove one of a, b, or c.
        # Which one?
        # Heuristic: Remove the one that is "most central" or "most composite"?
        # Or the one that is in the most violations.
        
        # Let's compute a quick conflict score for a, b, c.
        # Or just remove 'a' (the divisor) as it's the "active" constraint?
        # If we remove a, we fix this violation.
        # But a might be needed for other things? No, we just want max size.
        # Removing any one fixes this specific triple.
        
        # Let's pick the one with the smallest "utility".
        # Utility of x: how many other elements does x "protect" or "conflict with"?
        
        # Simple heuristic: Remove 'a'.
        # Why? Because a is the divisor. If a is composite, it's "heavier".
        # Primes are never divisors in antichain.
        
        S.remove(triple[0])
        
    return list(S)

construct()
