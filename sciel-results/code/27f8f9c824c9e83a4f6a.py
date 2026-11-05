import time
from math import gcd
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    # For each pair (i,j), store their GCD
    # A triple (a,b,c) is bad if gcd(a,b)=gcd(a,c)=gcd(b,c)
    # Let's precompute for each pair, which third elements would form a bad triple
    
    # conflicts_pair[pair] = set of elements x such that {pair[0], pair[1], x} is bad
    # But this is expensive. Instead, let's use a different approach.
    
    # Key insight: For a fixed g, consider the set S_g = {x : g | x}
    # For x,y,z in S_g, gcd(x,y)=gcd(x,z)=gcd(y,z)=g iff x/g, y/g, z/g are pairwise coprime
    # So for each g, we need to avoid triples in S_g/g that are pairwise coprime
    
    # Let's use a different structural approach: 
    # For each number, compute its "conflict set" - numbers that cannot be in the same set with it
    # given some other number already in the set
    
    # Actually, let's try a completely different optimization: 
    # Use a maximum independent set in a 3-uniform hypergraph with a better local search
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in combinations(numbers, 3):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        if g_ab == g_ac == g_bc:
            bad_triples.append((a, b, c))
    
    # For each element, store which bad triples contain it
    element_triples = [set() for _ in range(N + 1)]
    for a, b, c in bad_triples:
        element_triples[a].add((b, c))
        element_triples[b].add((a, c))
        element_triples[c].add((a, b))
    
    # Degree of each element (number of bad triples it's in)
    degrees = [len(element_triples[v]) for v in range(N + 1)]
    
    best_size = 0
    best_set = set()
    
    # Use simulated annealing with a different neighborhood structure
    # State: set S
    # Energy: -|S| (we want to maximize, so minimize negative)
    
    # Start with a random feasible set
    def is_feasible(S):
        """Check if S is feasible (no bad triples)"""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                # Check if there's a c in S such that {a,b,c} is bad
                for c in S_list[j + 1:]:
                    if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                        return False
        return True
    
    def can_add(S, v):
        """Check if v can be added to S without creating a bad triple"""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                if gcd_table[a][b] == gcd_table[a][v] == gcd_table[b][v]:
                    return False
        return True
    
    # Greedy initialization with randomization
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Randomized greedy initialization
        random.shuffle(numbers)
        S = set()
        
        # Sort by degree with some randomness
        candidates = sorted(numbers, key=lambda x: (degrees[x], random.random()))
        
        for v in candidates:
            if time.time() > deadline:
                break
            if can_add(S, v):
                S.add(v)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = set(S)
        
        # Simulated annealing local search
        T = 1.0  # Temperature
        T_min = 0.01
        cooling = 0.995
        max_iter = 5000
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            if T < T_min:
                break
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try to add a random element
                remaining = [v for v in numbers if v not in S]
                if not remaining:
                    continue
                v = random.choice(remaining)
                if can_add(S, v):
                    S.add(v)
                    delta = 1  # Size increases
                    if delta > 0 or random.random() < pow(2.718, -(-delta) / T):
                        pass  # Accept
                else:
                    continue
            elif move_type < 0.6:
                # Try to remove a random element
                if not S:
                    continue
                v = random.choice(list(S))
                S.remove(v)
                delta = -1  # Size decreases
                if delta > 0 or random.random() < pow(2.718, -(-delta) / T):
                    pass  # Accept
                else:
                    S.add(v)  # Reject, restore
            else:
                # Try a swap: remove one, add one
                if not S:
                    continue
                u = random.choice(list(S))
                remaining = [v for v in numbers if v not in S]
                if not remaining:
                    continue
                v = random.choice(remaining)
                
                # Check if swap is feasible
                S.remove(u)
                if can_add(S, v):
                    S.add(v)
                    delta = 0  # Size unchanged
                    # Accept with some probability to allow exploration
                    if random.random() < 0.5:
                        pass  # Accept
                    else:
                        S.remove(v)
                        S.add(u)  # Reject, restore
                else:
                    S.add(u)  # Restore
                
            T *= cooling
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        # Try 1-2 swaps (remove 1, add 2) to increase size
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            remaining = [v for v in numbers if v not in S]
            if len(remaining) < 2:
                break
            
            for u in list(S):
                if time.time() > deadline:
                    break
                
                S.remove(u)
                
                # Find all v that can be added
                valid_v = [v for v in remaining if can_add(S, v)]
                
                # Find pair v1, v2 in valid_v such that {v1, v2} can both be added
                found = False
                for i in range(len(valid_v)):
                    for j in range(i + 1, len(valid_v)):
                        v1, v2 = valid_v[i], valid_v[j]
                        # Check if v1 and v2 can both be added
                        S.add(v1)
                        if can_add(S, v2):
                            S.add(v2)
                            found = True
                            improved = True
                            break
                        else:
                            S.remove(v1)
                    
                    if found:
                        break
                
                if not found:
                    S.add(u)  # Restore u
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
        
        # Try 2-1 swaps (remove 2, add 1) to explore different local optima
        # This is less useful for maximizing, but can help escape local optima
    
    return list(best_set)
