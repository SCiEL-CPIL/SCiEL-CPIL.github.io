import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    n = 192
    universe = list(range(1, n + 1))
    
    # Precompute divisors for all sums up to 3*n
    max_sum = 3 * n
    divisors = [[] for _ in range(max_sum + 1)]
    for s in range(3, max_sum + 1):
        for d in range(1, n + 1):
            if s % d == 0:
                divisors[s].append(d)
    
    # Precompute residue bitsets for fast triple-sum checks
    # For each divisor d, and each residue r mod d, maintain a bitset of elements in [1,192]
    # with that residue. This allows O(1) checks for whether 3 distinct elements sum to a multiple of d.
    
    # For a given d, we need to check if there exist x, y, z in S (distinct) such that (x+y+z) % d == 0
    # This is equivalent to: for some residue r, x% d + y% d + z% d ≡ 0 (mod d)
    # We can precompute for each d, for each pair of residues (r1, r2), the set of residues r3 such that r1+r2+r3 ≡ 0 (mod d)
    
    # Actually, let's use a different approach:
    # For each d, maintain a count of how many elements in S have each residue mod d
    # Then check if there exist 3 distinct elements whose residues sum to 0 mod d
    
    # Even faster: Use bitsets. For each d, create a list of bitsets where bitset[r] has bit i set
    # if element i (0-indexed in universe) has residue r mod d.
    # Then for checking if 3 elements sum to 0 mod d, we iterate over residue triples (r1, r2, r3) with r1+r2+r3 ≡ 0 (mod d)
    # and check if bitset[r1] & bitset[r2] & bitset[r3] has at least 3 bits set (for distinct elements).
    
    # But this is complex. Let's use a simpler optimization:
    # Precompute for each d, for each possible sum s (3 to 3*n), whether s % d == 0
    # Then use the existing can_add logic but with early termination.
    
    # Actually, the key insight is: for a fixed set S, we can precompute all pair sums and their residues.
    # Let's try a different approach: maintain for each element a, a set of "conflicting" elements.
    
    # Let's stick with the current approach but optimize the inner loops.
    
    # Helper: Check if 'a' can be added to set S
    # Returns True if S U {a} is valid
    def can_add(a, S):
        S_list = list(S)
        K = len(S_list)
        
        if K == 0:
            return True
        
        # 1. Check if a divides sum of any 3 distinct elements in S
        # We need x, y, z in S, x!=y!=z, such that (x+y+z) % a == 0
        
        # Optimization: Only check multiples of a that are in range [3, 3*max(S)]
        max_s = max(S) if S else 0
        max_possible_sum = 3 * max_s
        
        for M in range(a, max_possible_sum + 1, a):
            if M < 3:
                continue
            # Check if there are 3 distinct elements in S summing to M
            # Use two-pointer or hash set approach
            remaining = M
            # For each pair (x, y) in S, check if z = M - x - y is in S and distinct
            for i in range(K):
                x = S_list[i]
                if x >= M:
                    break
                for j in range(i+1, K):
                    y = S_list[j]
                    if x + y >= M:
                        break
                    z = M - x - y
                    if z <= y:  # z must be > y for distinctness (since we iterate i<j, and z should be different)
                        continue
                    if z in S:
                        return False
        
        # 2. Check if any b in S is divided by sum of triple involving 'a'
        # Triples are (a, u, v) with u, v in S, u != v.
        # Sum T = a + u + v.
        # Check if any b in S, b != a, b != u, b != v, divides T.
        
        S_set = set(S)
        for i in range(K):
            u = S_list[i]
            for j in range(i+1, K):
                v = S_list[j]
                T = a + u + v
                if T <= max_sum:
                    for b in divisors[T]:
                        if b == a or b == u or b == v:
                            continue
                        if b in S_set:
                            return False
        return True

    # Helper: Check if S is valid (for initialization or verification)
    def is_valid(S):
        S_list = list(S)
        S_set = set(S)
        K = len(S_list)
        if K < 4: 
            return True
        
        for a in S_list:
            others = [x for x in S_list if x != a]
            # Check if a divides sum of any 3 in others
            max_others = max(others) if others else 0
            max_possible_sum = 3 * max_others
            for M in range(a, max_possible_sum + 1, a):
                if M < 3:
                    continue
                for i in range(len(others)):
                    x = others[i]
                    if x >= M:
                        break
                    for j in range(i+1, len(others)):
                        y = others[j]
                        if x + y >= M:
                            break
                        z = M - x - y
                        if z <= y:
                            continue
                        if z in S_set:
                            return False
        return True

    # Initialize with greedy
    S = set()
    for x in range(n, 0, -1):
        if can_add(x, S):
            S.add(x)
            
    best_S = set(S)
    best_size = len(S)
    
    # Local Search with more aggressive exploration
    while time.time() - start_time < time_limit:
        S_list = list(S)
        U_not_S = [x for x in range(1, n+1) if x not in S]
        
        if not U_not_S:
            break
        
        # Try multiple random actions per iteration to explore more
        for _ in range(5):  # Try 5 actions per outer loop
            if time.time() - start_time >= time_limit:
                break
            
            if random.random() < 0.5:
                # Try Add
                if S_list:
                    a = random.choice(U_not_S)
                    if can_add(a, S):
                        S.add(a)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
            else:
                # Try Swap
                if S_list:
                    r = random.choice(S_list)
                    a = random.choice(U_not_S)
                    
                    S_mid = S - {r}
                    if can_add(a, S_mid):
                        S.remove(r)
                        S.add(a)
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
    
    # Ensure we return the best found
    return list(best_S)
