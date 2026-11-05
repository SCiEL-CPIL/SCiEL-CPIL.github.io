import time
import random
import math

def construct():
    deadline = time.time() + 85
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    # For each a, find divisors d of a^2 such that d < a.
    # b = a + d, c = a + a^2/d
    # Check b < c and c <= N
    
    # conflict_pairs[x] will store a list of tuples (y, z)
    # such that {x, y, z} is a forbidden triple.
    # Note: The triple is undirected in the set context.
    # So if (a,b,c) is a triple, then:
    # conflict_pairs[a] contains (b,c)
    conflict_pairs = [list() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    # Triple (a, b, c)
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Bitset representation for fast membership tests
    # bitset[x] is a bitmask where bit i is 1 if i is in the set
    # We use Python's arbitrary-precision integers as bitsets
    
    def make_bitset(S):
        bits = 0
        for x in S:
            bits |= (1 << x)
        return bits
    
    def bitset_has(bits, x):
        return (bits >> x) & 1
    
    def can_add_bitset(x, bits):
        """Check if x can be added to the set represented by bits."""
        for y, z in conflict_pairs[x]:
            if bitset_has(bits, y) and bitset_has(bits, z):
                return False
        return True

    def init_solution(seed):
        """Create an initial solution using a given seed."""
        random.seed(seed)
        # Start with upper half, then greedily add
        S = set(range(91, N + 1))
        for x in range(1, 91):
            if can_add_bitset(x, make_bitset(S)):
                S.add(x)
        return S

    best_S = None
    best_size = 0
    
    # Try multiple initial seeds to find a good starting point
    for seed in [42, 123, 456, 789, 321]:
        S = init_solution(seed)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    all_nums = set(range(1, N + 1))
    
    # Local Search with 1-swap and 2-swap moves
    random.seed(42)
    
    # Use the best initial solution
    S = set(best_S)
    bits = make_bitset(S)
    
    # Run a quick local search to polish the start
    for iteration in range(50000):
        if time.time() > deadline - 5:
            break
        excluded = all_nums - S
        if not excluded:
            break
        x = random.choice(tuple(excluded))
        conflicts = []
        for y, z in conflict_pairs[x]:
            if bitset_has(bits, y) and bitset_has(bits, z):
                conflicts.append((y, z))
        if not conflicts:
            S.add(x)
            bits |= (1 << x)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            continue
        candidates = set()
        for y, z in conflicts:
            candidates.add(y)
            candidates.add(z)
        cand_list = list(candidates)
        random.shuffle(cand_list)
        found_swap = False
        for u in cand_list:
            S.remove(u)
            bits ^= (1 << u)
            if can_add_bitset(x, bits):
                S.add(x)
                bits |= (1 << x)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                found_swap = True
                break
            else:
                S.add(u)
                bits |= (1 << u)
        if found_swap:
            continue

    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    # Energy is -|S|. We minimize Energy.
    # Current Energy
    E = -len(S)
    best_E = E
    
    for iteration in range(1000000):
        if time.time() > deadline - 2:
            break
        if T < T_min:
            break
            
        # Choose a move
        r = random.random()
        if r < 0.4:
            # Try to add an element
            # Pick random x in [1, N]
            x = random.randint(1, N)
            if not bitset_has(bits, x):
                if can_add_bitset(x, bits):
                    S.add(x)
                    bits |= (1 << x)
                    E_new = E - 1
                    # Always accept improvement
                    E = E_new
                    if E < best_E:
                        best_E = E
                        best_S = set(S)
        elif r < 0.8:
            # Try to swap
            if S:
                # Pick u in S
                u = random.randint(1, N)
                while not bitset_has(bits, u):
                    u = random.randint(1, N)
                # Pick v not in S
                v = random.randint(1, N)
                while bitset_has(bits, v):
                    v = random.randint(1, N)
                
                # Remove u, add v
                S.remove(u)
                bits ^= (1 << u)
                if can_add_bitset(v, bits):
                    S.add(v)
                    bits |= (1 << v)
                    E_new = E
                    # Accept swap to diversify
                    E = E_new
                else:
                    # Revert
                    S.add(u)
                    bits |= (1 << u)
        else:
            # Try to remove an element (to escape local minima)
            if S:
                u = random.randint(1, N)
                while not bitset_has(bits, u):
                    u = random.randint(1, N)
                
                S.remove(u)
                bits ^= (1 << u)
                E_new = E + 1
                # Accept with probability exp(-(E_new - E) / T) = exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    E = E_new
                else:
                    # Revert
                    S.add(u)
                    bits |= (1 << u)
                    
        T *= cooling
        
    return sorted(best_S)
