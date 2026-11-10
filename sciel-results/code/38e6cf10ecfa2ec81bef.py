"""Rewritten program: ILS approach for large sets avoiding reciprocal Schur triples."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    N = 180
    
    # Precompute forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = bc / (b+c)
    # Condition: (b+c) divides (b*c)
    
    a_to_bc = {i: [] for i in range(N + 1)}
    b_to_ac = {i: [] for i in range(N + 1)}
    c_to_ab = {i: [] for i in range(N + 1)}
    
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b:
                    triples.append((a, b, c))
                    a_to_bc[a].append((b, c))
                    b_to_ac[b].append((a, c))
                    c_to_ab[c].append((a, b))
    
    # Use sets for fast membership testing
    # S is a set of integers
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a forbidden triple."""
        # x as 'a'
        for (b, c) in a_to_bc[x]:
            if b in S and c in S:
                return False
        # x as 'b'
        for (a, c) in b_to_ac[x]:
            if a in S and c in S:
                return False
        # x as 'c'
        for (a, b) in c_to_ab[x]:
            if a in S and b in S:
                return False
        return True

    def is_valid(S):
        """Verify S has no forbidden triple."""
        for a in S:
            for (b, c) in a_to_bc[a]:
                if b in S and c in S:
                    return False
        return True

    def maximize(S):
        """Greedy addition until no more elements can be added."""
        # Use a list of candidates to iterate, checking each time
        # To be efficient, we can just loop 1..N repeatedly
        # Or maintain a list of "blocked" elements?
        # Simple loop is O(N * cost(can_add)). cost(can_add) is small.
        # N=180. 180 * 180 = 32400 ops per pass. Very fast.
        improved = True
        while improved:
            if time.time() > deadline:
                return S
            improved = False
            # Randomize order? Or fixed?
            # Fixed order might be faster but less diverse.
            # Let's use fixed order for speed, rely on kicks for diversity.
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        improved = True
                        # Do not break, continue to see if more can be added
                        # But adding x might block others, or unblock? No, adding only blocks.
                        # So if we add x, we don't need to re-check previously failed elements?
                        # Actually, if x was blocked by y, and y is still there, x is still blocked.
                        # If x was blocked by y, and we added z which doesn't interact, x is still blocked.
                        # So we can just continue.
        return S

    best_set = set()
    best_size = 0
    rng = random.Random(42)
    
    # Initial Greedy
    S = set()
    maximize(S)
    if len(S) > best_size:
        best_set = set(S)
        best_size = len(S)
    
    # Iterated Local Search
    while time.time() < deadline:
        # Perturb: Remove k random elements from current S (or best if S is empty)
        if not S:
            S = set(best_set)
        
        # Determine kick size
        # Start with 1, increase if stuck?
        # Let's just pick a random kick size 1-3
        kick_size = rng.randint(1, 3)
        
        # Remove kick_size elements
        if len(S) > kick_size:
            elements_to_remove = rng.sample(list(S), kick_size)
            for e in elements_to_remove:
                S.remove(e)
        elif len(S) > 0:
            # If S is very small, clear it or remove all?
            # If we remove all, we restart from scratch.
            # Let's just remove all if it's too small to kick effectively
            if len(S) <= kick_size:
                 S.clear()
        
        # Maximize again
        maximize(S)
        
        # Update best
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
            # Optional: If we hit a new best, maybe restart from scratch?
            # No, continue ILS from this new best.
            
        # If we have used all elements, stop
        if best_size == N:
            break

    # Final verification
    if not is_valid(best_set):
        # Fallback
        best_set = set(range(91, N + 1))
        if not is_valid(best_set):
            best_set = {180}
            
    return list(best_set)

# EVOLVE-BLOCK-END
