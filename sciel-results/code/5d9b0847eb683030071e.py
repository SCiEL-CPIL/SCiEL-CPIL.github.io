"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    # Precompute reciprocals as Fractions
    recip = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = Fraction(1, i)
        
    # in_S is a list of bools
    def can_add(x, in_S):
        """Check if x can be added to the set represented by in_S."""
        # x is in 1..N
        # Check 3-APs involving x
        # 1. x is middle: a, x, c -> 2x = a+c -> c = 2x-a
        # 2. x is end: a, b, x or x, b, a -> 2b = a+x -> b = (a+x)/2
        
        # Iterate through all a in 1..N that are in S
        for a in range(1, N + 1):
            if not in_S[a]:
                continue
            
            # Case 1: x is middle (a, x, c)
            c = 2 * x - a
            if c != a and 1 <= c <= N and in_S[c]:
                return False
            
            # Case 2: x is end (a, b, x) or (x, b, a)
            # b = (a + x) / 2
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b != a and b != x and 1 <= b <= N and in_S[b]:
                    return False
                    
        return True

    def get_score(in_S):
        s = Fraction(0)
        for i in range(1, N + 1):
            if in_S[i]:
                s += recip[i]
        return s

    def greedy_init():
        in_S = [False] * (N + 1)
        for x in range(1, N + 1):
            if can_add(x, in_S):
                in_S[x] = True
        return in_S

    def local_search(in_S):
        score = get_score(in_S)
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try all additions
            for x in range(1, N + 1):
                if not in_S[x]:
                    if can_add(x, in_S):
                        in_S[x] = True
                        score += recip[x]
                        improved = True
                        
            # 2. Try all swaps
            # We iterate u in S, v not in S, v < u
            # To speed up, we can break early if no improvement found in a while?
            # No, we want the best improvement or first improvement.
            # Let's do first improvement to be faster.
            
            for u in range(1, N + 1):
                if not in_S[u]:
                    continue
                for v in range(1, u):
                    if in_S[v]:
                        continue
                    
                    # Tentatively remove u
                    in_S[u] = False
                    if can_add(v, in_S):
                        in_S[v] = True
                        # Check if score improves
                        # New score = score - recip[u] + recip[v]
                        # Since v < u, recip[v] > recip[u], so it always improves!
                        # Wait, is it always valid?
                        # If can_add(v, S\u) is True, then the new set is valid.
                        # And the score increases.
                        # So we accept immediately.
                        score = score - recip[u] + recip[v]
                        improved = True
                        break # Break inner loop, move to next u
                    else:
                        # Revert
                        in_S[u] = True
                        continue
                    # If accepted, u is False, v is True.
                    # We break inner loop.
                    
            # 3. If no improvement, try a kick
            if not improved:
                if time.time() > deadline - 2:
                    break
                # Remove 5 random elements
                elems = [i for i in range(1, N + 1) if in_S[i]]
                if len(elems) > 10:
                    to_remove = random.sample(elems, min(5, len(elems)//4))
                    for r in to_remove:
                        in_S[r] = False
                        score -= recip[r]
                    # Re-greedy fill
                    for x in range(1, N + 1):
                        if not in_S[x] and can_add(x, in_S):
                            in_S[x] = True
                            score += recip[x]
                    improved = True # Force another pass
        
        return in_S, score

    # Initial construction
    in_S = greedy_init()
    in_S, score = local_search(in_S)
    
    best_in_S = in_S[:]
    best_score = score
    
    # Try multiple restarts if time permits
    while time.time() < deadline - 10:
        # Take current best, remove random 10%, re-greedy fill, local search
        elems = [i for i in range(1, N + 1) if best_in_S[i]]
        if len(elems) > 20:
            to_remove = random.sample(elems, 10)
            in_S = best_in_S[:]
            for r in to_remove:
                in_S[r] = False
            
            # Re-greedy
            for x in range(1, N + 1):
                if not in_S[x] and can_add(x, in_S):
                    in_S[x] = True
            
            in_S, new_score = local_search(in_S)
            
            if new_score > best_score:
                best_score = new_score
                best_in_S = in_S[:]
        
        # Check time
        if time.time() > deadline - 5:
            break

    result = [i for i in range(1, N + 1) if best_in_S[i]]
    return result
# EVOLVE-BLOCK-END
