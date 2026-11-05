import time
import random
import math

def construct():
    deadline = time.time() + 85
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  <=> (b-a)(c-a) = a^2
    conflict_pairs = [list() for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a2 = a * a
        for d in range(1, a):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    conflict_pairs[a].append((b, c))
                    conflict_pairs[b].append((a, c))
                    conflict_pairs[c].append((a, b))
    
    # Bitset helpers
    def make_bitset(S):
        bits = 0
        for x in S:
            bits |= (1 << x)
        return bits
    
    def bitset_has(bits, x):
        return (bits >> x) & 1
    
    def can_add_bitset(x, bits):
        for y, z in conflict_pairs[x]:
            if bitset_has(bits, y) and bitset_has(bits, z):
                return False
        return True
    
    def get_conflicts_bitset(x, bits):
        conflicts = []
        for y, z in conflict_pairs[x]:
            if bitset_has(bits, y) and bitset_has(bits, z):
                conflicts.append((y, z))
        return conflicts
    
    def greedy_add(S, bits, start, end):
        """Greedily add elements from [start, end] if possible."""
        for x in range(start, end + 1):
            if not bitset_has(bits, x):
                if can_add_bitset(x, bits):
                    S.add(x)
                    bits |= (1 << x)
        return S, bits
    
    def local_search_1swap(S, bits, iterations, deadline):
        """Local search with 1-swap moves."""
        all_nums = set(range(1, N + 1))
        for _ in range(iterations):
            if time.time() > deadline - 3:
                break
            excluded = all_nums - S
            if not excluded:
                break
            x = random.choice(tuple(excluded))
            conflicts = get_conflicts_bitset(x, bits)
            if not conflicts:
                S.add(x)
                bits |= (1 << x)
                continue
            candidates = set()
            for y, z in conflicts:
                candidates.add(y)
                candidates.add(z)
            cand_list = list(candidates)
            random.shuffle(cand_list)
            found = False
            for u in cand_list:
                S.remove(u)
                bits ^= (1 << u)
                if can_add_bitset(x, bits):
                    S.add(x)
                    bits |= (1 << x)
                    found = True
                    break
                else:
                    S.add(u)
                    bits |= (1 << u)
            if not found:
                # Try 2-removal kick
                if random.random() < 0.1 and len(cand_list) >= 2:
                    u, v = random.sample(cand_list, 2)
                    S.remove(u)
                    S.remove(v)
                    bits ^= (1 << u)
                    bits ^= (1 << v)
                    if can_add_bitset(x, bits):
                        S.add(x)
                        bits |= (1 << x)
                    else:
                        S.add(u)
                        S.add(v)
                        bits |= (1 << u)
                        bits |= (1 << v)
        return S, bits
    
    best_S = None
    best_size = 0
    
    # Multiple diverse initial solutions
    init_strategies = [
        lambda: set(range(91, N + 1)),  # Upper half
        lambda: set(range(121, N + 1)), # Upper third
        lambda: set(x for x in range(1, N + 1) if x % 2 == 1),  # Odds
        lambda: set(range(1, N + 1, 3)),  # Every third
        lambda: set(range(1, N + 1)) - set(range(1, 61)),  # Exclude lower third
    ]
    
    random.seed(42)
    
    for strat_idx, strategy in enumerate(init_strategies):
        if time.time() > deadline - 5:
            break
        
        S = strategy()
        bits = make_bitset(S)
        
        # Greedily add remaining elements
        S, bits = greedy_add(S, bits, 1, N)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        # Local search
        S, bits = local_search_1swap(S, bits, 50000, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Final polish on best solution
    if best_S and time.time() < deadline - 2:
        S = set(best_S)
        bits = make_bitset(S)
        S, bits = local_search_1swap(S, bits, 100000, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    return sorted(best_S)
