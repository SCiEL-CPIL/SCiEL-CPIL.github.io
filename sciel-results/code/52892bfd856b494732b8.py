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
    
    # Precompute for each x, a bitmask of all y such that (x,y) appears in any conflict pair
    # and a bitmask of all z such that (x,z) appears
    # Actually, let's precompute conflict_mask[x] = bitmask of all elements that conflict with x
    # But we need pairs, so let's keep the pair structure but also make fast checks
    
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
    
    def init_solution():
        """Create a good initial solution using greedy approach."""
        # Start with upper half
        S = set(range(91, N + 1))
        bits = make_bitset(S)
        
        # Try to add elements from 1 to 90 greedily
        for x in range(1, 91):
            if can_add_bitset(x, bits):
                S.add(x)
                bits |= (1 << x)
        
        # Also try adding from the bottom up with different orderings
        # Try a few different greedy strategies
        best_S = S
        best_size = len(S)
        
        for start in range(50, 131):
            S2 = set(range(start, N + 1))
            bits2 = make_bitset(S2)
            for x in range(1, start):
                if can_add_bitset(x, bits2):
                    S2.add(x)
                    bits2 |= (1 << x)
            if len(S2) > best_size:
                best_size = len(S2)
                best_S = S2
        
        return best_S
    
    # Get initial solution
    S = init_solution()
    bits = make_bitset(S)
    best_S = set(S)
    best_size = len(S)
    
    all_nums = list(range(1, N + 1))
    random.seed(42)
    
    # Local Search with add and swap moves
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 3:
        improved = False
        
        # Try to add random elements
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            x = random.randint(1, N)
            if not bitset_has(bits, x):
                if can_add_bitset(x, bits):
                    S.add(x)
                    bits |= (1 << x)
                    improved = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
        
        # Try swaps: remove u from S, add v not in S
        for _ in range(300):
            if time.time() > deadline - 3:
                break
            if not S:
                break
            u = random.choice(list(S))
            v = random.randint(1, N)
            if bitset_has(bits, v):
                continue
            
            # Try removing u and adding v
            S.remove(u)
            bits ^= (1 << u)
            
            if can_add_bitset(v, bits):
                S.add(v)
                bits |= (1 << v)
                improved = True
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
            else:
                # Revert
                S.add(u)
                bits |= (1 << u)
        
        # Try 2-exchanges: remove u1, u2 from S, add v1, v2 not in S
        for _ in range(100):
            if time.time() > deadline - 3:
                break
            if len(S) < 2:
                break
            S_list = list(S)
            u1, u2 = random.sample(S_list, 2)
            v1 = random.randint(1, N)
            v2 = random.randint(1, N)
            if bitset_has(bits, v1) or bitset_has(bits, v2) or v1 == v2:
                continue
            
            # Remove u1, u2
            S.remove(u1)
            S.remove(u2)
            bits ^= (1 << u1)
            bits ^= (1 << u2)
            
            # Try adding v1, v2
            if can_add_bitset(v1, bits):
                bits_temp = bits | (1 << v1)
                S_temp = set(S)
                S_temp.add(v1)
                if can_add_bitset(v2, bits_temp):
                    S.add(v1)
                    S.add(v2)
                    bits |= (1 << v1) | (1 << v2)
                    improved = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                else:
                    # Revert
                    S.add(u1)
                    S.add(u2)
                    bits |= (1 << u1) | (1 << u2)
            else:
                # Revert
                S.add(u1)
                S.add(u2)
                bits |= (1 << u1) | (1 << u2)
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart with a different initial solution
                S = init_solution()
                bits = make_bitset(S)
                no_improve_count = 0
        else:
            no_improve_count = 0
    
    # Final polish: try to add any remaining elements
    for x in range(1, N + 1):
        if not bitset_has(bits, x):
            if can_add_bitset(x, bits):
                S.add(x)
                bits |= (1 << x)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
    
    return sorted(best_S)
