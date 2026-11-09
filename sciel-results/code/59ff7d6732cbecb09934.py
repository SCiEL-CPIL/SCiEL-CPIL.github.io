import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
            
    # Precompute conflict bitsets C[a][b]
    # C[a][b] is a bitmask of c such that (a,b,c) is a bad triple
    # i.e., lcm(a,b) == lcm(a,c) == lcm(b,c)
    C = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L = lcm_table[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == L and lcm_table[b][c] == L:
                    mask |= (1 << c)
            C[a][b] = mask
            C[b][a] = mask
            
    def popcount(m):
        return bin(m).count('1')
        
    def is_valid_add(x, S):
        """Check if x can be added to set S (bitmask)."""
        # For all a in S, check if C[a][x] intersects S
        # If C[a][x] & S != 0, then there exists c in S such that (a,x,c) is bad.
        # Since we are adding x, we check conflicts with existing elements.
        # Note: C[a][x] contains c. If c is in S, then {a,x,c} is a bad triple.
        # We must ensure no such c exists in S.
        temp_S = S
        while temp_S:
            # Extract lowest set bit
            lsb = temp_S & -temp_S
            a = lsb.bit_length() - 1
            if C[a][x] & S:
                return False
            temp_S ^= lsb
        return True

    def is_valid_set(S):
        """Check if set S is valid (no bad triples)."""
        temp_S = S
        while temp_S:
            lsb_a = temp_S & -temp_S
            a = lsb_a.bit_length() - 1
            temp_S ^= lsb_a
            
            temp_S2 = temp_S
            while temp_S2:
                lsb_b = temp_S2 & -temp_S2
                b = lsb_b.bit_length() - 1
                temp_S2 ^= lsb_b
                
                # Check if C[a][b] intersects S
                if C[a][b] & S:
                    return False
        return True

    def get_list(S):
        return [i for i in range(1, N + 1) if S & (1 << i)]

    # Greedy Initialization
    def greedy_init(order):
        S = 0
        for x in order:
            if is_valid_add(x, S):
                S |= (1 << x)
        return S

    # Local Search: 1-swap and Add
    def local_search(S, deadline_ls):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline_ls:
                break
                
            # Try adding
            for x in range(1, N + 1):
                if not (S & (1 << x)):
                    if is_valid_add(x, S):
                        S |= (1 << x)
                        improved = True
                        break
                        
            if improved:
                continue
                
            # Try 1-swap: remove a, add x
            selected = get_list(S)
            for a in selected:
                if time.time() > deadline_ls:
                    break
                new_S = S & ~(1 << a)
                for x in range(1, N + 1):
                    if not (new_S & (1 << x)):
                        if is_valid_add(x, new_S):
                            # Check if size increased? 
                            # Since we removed 1 and added 1, size is same.
                            # We only accept if it leads to a better state? 
                            # For max size, we only accept if size increases.
                            # But 1-swap keeps size same. It's for escaping local optima.
                            # In SA, we might accept. In ILS, we use it for perturbation.
                            # For deterministic LS, we only accept if size increases.
                            # So 1-swap alone doesn't increase size.
                            # We need 2-swap or just use 1-swap to enable further adds.
                            pass
                            
            # Actually, let's implement a proper 2-swap local search
            # Remove a, b and add x, y such that size increases by 0? No, we want max size.
            # If we remove 2 and add 2, size is same.
            # If we remove 1 and add 2, size increases by 1.
            
            # Let's try "Remove 1, Add 2"
            # For each a in S:
            #   new_S = S - {a}
            #   Find pair x,y not in new_S such that {x,y} can be added to new_S
            #   If found, S = new_S | {x,y}
            
            found_improvement = False
            for a in selected:
                if time.time() > deadline_ls:
                    break
                new_S = S & ~(1 << a)
                # Try to add two elements x, y
                # Iterate x
                for x in range(1, N + 1):
                    if new_S & (1 << x):
                        continue
                    if not is_valid_add(x, new_S):
                        continue
                    temp_S = new_S | (1 << x)
                    # Try to add y
                    for y in range(x + 1, N + 1):
                        if temp_S & (1 << y):
                            continue
                        if is_valid_add(y, temp_S):
                            # Improvement: size +1
                            S = temp_S | (1 << y)
                            improved = True
                            found_improvement = True
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
                    
        return S

    # Initial Greedy with different orders
    best_S = 0
    best_size = 0
    
    # Order 1: Primes first
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
            
    primes = [x for x in numbers if is_prime(x)]
    others = [x for x in numbers if x not in primes]
    
    order1 = primes + others
    S1 = greedy_init(order1)
    S1 = local_search(S1, deadline - 5)
    if popcount(S1) > best_size:
        best_S = S1
        best_size = popcount(S1)
        
    # Order 2: Random
    random.seed(42)
    for i in range(50):
        if time.time() > deadline - 10:
            break
        random.shuffle(numbers)
        S = greedy_init(numbers)
        S = local_search(S, deadline - 10)
        sz = popcount(S)
        if sz > best_size:
            best_S = S
            best_size = sz
            
    # Iterated Local Search / Simulated Annealing for refinement
    # We will try to improve best_S further using random restarts from perturbations
    current_S = best_S
    current_size = best_size
    
    # Simple Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # We need a move generator that allows same-size moves to escape local optima
    # Move types:
    # 1. Add (size +1)
    # 2. Remove (size -1)
    # 3. Swap 1-1 (size 0)
    # 4. Swap 1-2 (size +1) - already in local search
    # 5. Swap 2-2 (size 0)
    
    # Given the time, let's just run more random restarts with local search
    # and try to find a larger set.
    
    # Try to add more random restarts
    for i in range(200):
        if time.time() > deadline - 2:
            break
        random.shuffle(numbers)
        S = greedy_init(numbers)
        S = local_search(S, deadline - 2)
        sz = popcount(S)
        if sz > best_size:
            best_S = S
            best_size = sz
            
    return get_list(best_S)
