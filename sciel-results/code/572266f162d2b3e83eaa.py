import time
from itertools import combinations
from math import gcd

def construct():
    deadline = time.time() + 80  # safety margin before 90s limit
    
    def gcds_equal(a, b, c):
        """Check if three pairwise gcds are all equal."""
        g1 = gcd(a, b)
        g2 = gcd(a, c)
        g3 = gcd(b, c)
        return g1 == g2 == g3
    
    def count_conflicts(selected):
        """Count number of bad triples in selected."""
        conflicts = 0
        n = len(selected)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if gcds_equal(selected[i], selected[j], selected[k]):
                        conflicts += 1
        return conflicts
    
    def is_valid_triple(a, b, c):
        """Return True if triple (a,b,c) is valid (not a bad triple)."""
        return not gcds_equal(a, b, c)
    
    def can_add(selected, x):
        """Check if x can be added to selected without creating a bad triple."""
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                if not is_valid_triple(selected[i], selected[j], x):
                    return False
        return True
    
    def greedy_build(order):
        """Greedy build following the given order."""
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        return selected
    
    def local_search(selected):
        """Try to improve by removing elements from bad triples and re-adding."""
        selected = list(selected)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Find all bad triples
            bad_triples = []
            n = len(selected)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if not is_valid_triple(selected[i], selected[j], selected[k]):
                            bad_triples.append((i, j, k))
            
            if not bad_triples:
                break
            
            # For each bad triple, try removing one element
            for (i, j, k) in bad_triples:
                if time.time() > deadline:
                    break
                for idx in [i, j, k]:
                    candidate = selected[:idx] + selected[idx+1:]
                    # Try to add back other removed elements or new ones
                    # Just check if removing this element helps
                    old_size = len(selected)
                    new_size = len(candidate)
                    # Count conflicts in candidate
                    conf_candidate = count_conflicts(candidate)
                    conf_selected = count_conflicts(selected)
                    if conf_candidate < conf_selected or (conf_candidate == conf_selected and new_size >= old_size - 1):
                        selected = candidate
                        improved = True
                        break
    
        # Try to add back elements from [1,128] not in selected
        for x in range(1, 129):
            if time.time() > deadline:
                break
            if x not in selected and can_add(selected, x):
                selected.append(x)
        
        return selected
    
    # Strategy 1: Start with numbers that are powers of primes or have few prime factors
    # These tend to have distinct gcd patterns
    
    # Build initial candidate pool with emphasis on numbers with unique properties
    # Numbers that are prime powers: 2,4,8,16,32,64, 3,9,27,81, 5,25, 7,49, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113
    
    # Actually, let's use a smarter initial set
    # Include numbers with distinct prime factorizations
    
    # Start with a broad set and prune
    initial = list(range(1, 129))
    
    # Greedy build in different orders
    best_set = []
    
    # Order 1: by number of distinct prime factors (ascending)
    def num_distinct_primes(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order1 = sorted(range(1, 129), key=lambda x: (num_distinct_primes(x), x))
    set1 = greedy_build(order1)
    if len(set1) > len(best_set):
        best_set = set1
    
    # Order 2: by value descending (larger numbers have more divisors, might be safer)
    order2 = sorted(range(1, 129), reverse=True)
    set2 = greedy_build(order2)
    if len(set2) > len(best_set):
        best_set = set2
    
    # Order 3: by number of divisors
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    order3 = sorted(range(1, 129), key=lambda x: num_divisors(x))
    set3 = greedy_build(order3)
    if len(set3) > len(best_set):
        best_set = set3
    
    # Local search on the best set found
    if time.time() < deadline:
        improved_set = local_search(best_set)
        if len(improved_set) > len(best_set):
            best_set = improved_set
    
    # Try another approach: start with specific patterns
    # Numbers that are products of exactly 2 distinct primes (semiprimes) might work well
    # But actually, let's try including numbers with a common factor structure
    
    # Try: all even numbers + some odd numbers
    evens = list(range(2, 129, 2))
    odds = list(range(1, 129, 2))
    
    # Greedy with evens first, then odds
    order4 = evens + odds
    set4 = greedy_build(order4)
    if len(set4) > len(best_set):
        best_set = set4
        if time.time() < deadline:
            improved_set = local_search(best_set)
            if len(improved_set) > len(best_set):
                best_set = improved_set
    
    # Try: numbers divisible by 2 but not by 4, plus numbers divisible by 4
    # This creates a hierarchy
    
    # Final: ensure we return a nonempty valid set
    if not best_set:
        best_set = [1]
    
    return best_set
