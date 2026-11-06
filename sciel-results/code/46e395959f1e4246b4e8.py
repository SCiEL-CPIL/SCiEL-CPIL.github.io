# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    def time_left():
        return deadline - time.time()
    
    # Precompute pair sum residues for efficiency
    # For each a in 1..192, and for each possible sum s, precompute s % a
    # Actually, let's precompute for each a, a list of (i, j, (i+j)%a) for all i<j
    # This is too large. Instead, we'll compute on the fly but cache.
    
    # Key insight: for large numbers (close to 192), they're less likely to divide
    # sums of three others (max sum of 3 others is 3*191 = 573, so only divisors of
    # numbers up to 573 matter). Large numbers > 573/2 = 286.5 can't divide any sum
    # of three numbers <= 192. But 192 < 286.5, so this doesn't help directly.
    # However, if x > 573, x can't divide any sum. But x <= 192, so x <= 192 < 573.
    # The sum of three distinct numbers from 1..192 is at most 192+191+190 = 573.
    # So x can divide a sum only if x <= 573, which is always true for x <= 192.
    # But x must be a divisor of the sum. For large x, there are fewer multiples.
    
    def can_add(S, x):
        """Check if x can be added to set S (list of ints)."""
        n = len(S)
        
        # Check 1: x does not divide sum of any 3 distinct elements in S
        if n >= 3:
            for combo in combinations(S, 3):
                if sum(combo) % x == 0:
                    return False
        
        # Check 2: For each a in S, no triple (x, u, v) with u,v in S\{a}, u!=v
        # has (x+u+v) % a == 0
        for a in S:
            if n - 1 < 2:
                continue
            others = [y for y in S if y != a]
            target = (-x) % a
            # Check all pairs in others
            for i in range(len(others)):
                oi = others[i]
                for j in range(i+1, len(others)):
                    if (oi + others[j]) % a == target:
                        return False
        return True
    
    def greedy_construct(order):
        """Build a set using the given order of candidates."""
        S = []
        for x in order:
            if time_left() < 0.3:
                break
            if can_add(S, x):
                S.append(x)
        return S
    
    def local_search(S, max_rounds=10):
        """Improve S by trying to add or swap elements."""
        S = list(S)
        S_set = set(S)
        
        for _ in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            
            # Try adding
            for x in range(192, 0, -1):
                if x in S_set:
                    continue
                if can_add(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping
            for i in range(len(S)):
                if time_left() < 2.0:
                    break
                S_without_i = S[:i] + S[i+1:]
                S_without_set = S_set - {S[i]}
                for x in range(192, 0, -1):
                    if x in S_without_set:
                        continue
                    if can_add(S_without_i, x):
                        S[i] = x
                        S_set.remove(S[i])
                        S_set.add(x)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return S
    
    # Generate diverse orders
    orders = []
    orders.append(list(range(192, 0, -1)))  # Descending
    orders.append(list(range(1, 193)))      # Ascending
    
    random.seed(42)
    
    # Descending with perturbations
    for _ in range(20):
        perm = list(range(192, 0, -1))
        for i in range(len(perm)-1, 0, -1):
            if random.random() < 0.15:
                j = random.randint(0, i)
                perm[i], perm[j] = perm[j], perm[i]
        orders.append(perm)
    
    # Large-first with random small fill
    for _ in range(15):
        large = list(range(192, 96, -1))
        small = list(range(96, 0, -1))
        random.shuffle(small)
        orders.append(large + small)
    
    # Random
    for _ in range(15):
        perm = list(range(1, 193))
        random.shuffle(perm)
        orders.append(perm)
    
    best_set = []
    best_size = 0
    
    for order in orders:
        if time_left() < 2.0:
            break
        S = greedy_construct(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Local search on best
    if time_left() > 3.0:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Local search on a few other good solutions
    if time_left() > 5.0:
        for order in orders[:8]:
            if time_left() < 3.0:
                break
            S = greedy_construct(order)
            if len(S) >= best_size - 3:
                S = local_search(S, max_rounds=5)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S
    
    return best_set
# EVOLVE-BLOCK-END
