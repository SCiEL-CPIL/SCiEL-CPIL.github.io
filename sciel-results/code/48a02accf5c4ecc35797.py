import random
import time

N = 128
U = list(range(65, N + 1))
U_set = set(U)
L_range = range(1, 65)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_L_size = 0
    best_L = set()
    
    # Helper to check if x can be added to L
    def can_add_L(x, L):
        # Check Case 1: a, b, x in L
        # Triples {a, b, x} with a, b in L, a != b
        for a in L:
            for b in L:
                if a == b: continue
                # Sums: a+b, a+x, b+x
                # Check if all in S = L U U
                # a+b in S?
                if a + b > 64:
                    if a + b not in U_set: continue # Not in S
                else:
                    if a + b not in L: continue # Not in S
                
                # a+x in S?
                if a + x > 64:
                    if a + x not in U_set: continue
                else:
                    if a + x not in L: continue
                    
                # b+x in S?
                if b + x > 64:
                    if b + x not in U_set: continue
                else:
                    if b + x not in L: continue
                    
                # If we reach here, all 3 sums are in S. Violation!
                return False
        
        # Check Case 2: a, x in L, c in U
        # Triples {a, x, c} with a in L, c in U
        # Note: x is being added, so x in L_new. a in L_old.
        # We need to check if adding x creates a violation with existing a in L and c in U.
        # The triple is {a, x, c}.
        # Sums: a+x, a+c, x+c.
        for a in L:
            # a+x in S?
            if a + x > 64:
                if a + x not in U_set: continue
            else:
                if a + x not in L: # Note: x is not in L yet, but a+x could be x? No a>=1.
                    # Wait, a+x could be in L?
                    # If a+x in L, it's in S.
                    # If a+x not in L and <= 64, it's NOT in S.
                    continue
            
            # Now check for all c in U?
            # We need to exist c in U such that a+c in S AND x+c in S.
            # a+c in S: a+c > 64 always (a>=1, c>=65). So a+c in U_set if a+c <= 128.
            # x+c in S: x+c > 64 always. So x+c in U_set if x+c <= 128.
            
            # So we need:
            # 1. a+x in S (checked above)
            # 2. There exists c in U such that:
            #    a+c <= 128 and a+c in U_set (always true if <=128)
            #    x+c <= 128 and x+c in U_set (always true if <=128)
            
            # So we just need to check if there is a c in U such that
            # a+c <= 128 AND x+c <= 128.
            # c <= 128 - a
            # c <= 128 - x
            # c >= 65
            
            max_c = min(128 - a, 128 - x)
            if max_c >= 65:
                # Such c exists (e.g., c=65 if 65 <= max_c)
                # And since U contains all 65..128, c is in U.
                # And a+c, x+c are in U (and thus S).
                return False
        
        return True

    # Greedy + Local Search on L
    for _ in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        L = set()
        candidates = list(L_range)
        random.shuffle(candidates)
        
        # Greedy add
        for x in candidates:
            if can_add_L(x, L):
                L.add(x)
                
        # Local search: try to swap
        # We want to maximize size. Swaps don't change size.
        # But we can try to remove one and add two? No, simple moves.
        # Actually, if we are stuck, we can try to remove a random element and see if we can add a different one.
        
        current_L = set(L)
        
        # Try to improve by random swaps/removals
        for _ in range(1000):
            if time.time() > deadline:
                break
            r = random.random()
            if r < 0.5:
                # Try to add a random candidate not in L
                # Pick from 1..64 not in L
                not_in_L = [i for i in L_range if i not in L]
                if not_in_L:
                    x = random.choice(not_in_L)
                    if can_add_L(x, L):
                        L.add(x)
            else:
                # Try to swap
                if L:
                    a = random.choice(list(L))
                    not_in_L = [i for i in L_range if i not in L]
                    if not_in_L:
                        b = random.choice(not_in_L)
                        # Try L - {a} + {b}
                        new_L = set(L)
                        new_L.remove(a)
                        new_L.add(b)
                        # Check validity of new_L
                        # This is expensive to check from scratch.
                        # But N=64, so O(N^3) is small.
                        # However, we can just check if b can be added to L-{a}
                        L_temp = set(L)
                        L_temp.remove(a)
                        if can_add_L(b, L_temp):
                            L = new_L
                            
        if len(L) > best_L_size:
            best_L_size = len(L)
            best_L = set(L)
            
    result = list(best_L) + U
    return result
