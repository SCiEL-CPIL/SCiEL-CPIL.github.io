import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n = 16
    points = list(range(n))
    
    def calc_alpha(triples):
        # Returns the size of the largest independent set
        # We check k from 8 down to 5
        for k in range(8, 4, -1):
            for subset in combinations(points, k):
                s = set(subset)
                # Check if any triple is contained in s
                independent = True
                for t in triples:
                    if t[0] in s and t[1] in s and t[2] in s:
                        independent = False
                        break
                if independent:
                    return k
        # Check k=4
        for subset in combinations(points, 4):
            s = set(subset)
            independent = True
            for t in triples:
                if t[0] in s and t[1] in s and t[2] in s:
                    independent = False
                    break
            if independent:
                return 4
        return 3

    def find_max_independent_set(triples, k_target):
        # Find one independent set of size k_target if exists
        for subset in combinations(points, k_target):
            s = set(subset)
            independent = True
            for t in triples:
                if t[0] in s and t[1] in s and t[2] in s:
                    independent = False
                    break
            if independent:
                return list(subset)
        return None

    best_triples = []
    best_alpha = 100
    
    # Pre-generate all possible triples
    all_triples = list(combinations(points, 3))
    
    while time.time() - start_time < DEADLINE - 5:
        triples = []
        blocked_pairs = set()
        
        # Greedy Init
        # Shuffle all pairs
        all_pairs = list(combinations(points, 2))
        random.shuffle(all_pairs)
        
        for p1, p2 in all_pairs:
            # Try to complete pair (p1, p2) to a triple
            # Pick random p3
            candidates = [p for p in points if p != p1 and p != p2]
            random.shuffle(candidates)
            for p3 in candidates:
                # Check if pairs (p1,p3) and (p2,p3) are free
                if ((p1,p3) in blocked_pairs or (p3,p1) in blocked_pairs): continue
                if ((p2,p3) in blocked_pairs or (p3,p2) in blocked_pairs): continue
                # Add triple
                triples.append((p1, p2, p3))
                blocked_pairs.add((p1,p2)); blocked_pairs.add((p2,p1))
                blocked_pairs.add((p1,p3)); blocked_pairs.add((p3,p1))
                blocked_pairs.add((p2,p3)); blocked_pairs.add((p3,p2))
                break
        
        alpha = calc_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples.copy()
            if best_alpha <= 5:
                break
                
        # Local Search
        # We want to reduce alpha.
        # Strategy: Find a max independent set S.
        # Try to swap a triple to add a new triple that hits S.
        
        for _ in range(100): # Max swaps per restart
            if time.time() - start_time > DEADLINE - 2:
                break
                
            # Find current alpha
            # If alpha is low, we might stop
            if alpha <= 5:
                break
                
            # Find an independent set of size alpha
            S = find_max_independent_set(triples, alpha)
            if not S:
                # Recalculate alpha?
                alpha = calc_alpha(triples)
                continue
                
            # We want to add a triple T_new that intersects S in at least 2 points.
            # Let S = {s1, ..., sk}
            # Possible T_new: {s_i, s_j, x}
            
            # To add T_new, we must remove some triples to free up pairs.
            # Specifically, pairs in T_new must be free.
            # If a pair (u,v) in T_new is blocked by T_old, we must remove T_old.
            
            # Let's try a specific move:
            # Pick two points s_i, s_j from S.
            # Pick a third point x (random).
            # T_candidate = {s_i, s_j, x}
            
            # Check which pairs in T_candidate are blocked.
            # If (s_i, s_j) is blocked by T1, remove T1.
            # If (s_i, x) is blocked by T2, remove T2.
            # If (s_j, x) is blocked by T3, remove T3.
            # Then add T_candidate.
            
            # This is a "remove k, add 1" move.
            # We need to ensure we don't remove too many or create conflicts.
            
            # Simplified: Just try to swap one triple.
            # Pick a random triple T_old to remove.
            # Then try to add a T_new that hits S.
            
            idx = random.randint(0, len(triples) - 1)
            T_old = triples[idx]
            
            # Remove T_old
            triples.pop(idx)
            a, b, c = T_old
            blocked_pairs.discard((a,b)); blocked_pairs.discard((b,a))
            blocked_pairs.discard((a,c)); blocked_pairs.discard((c,a))
            blocked_pairs.discard((b,c)); blocked_pairs.discard((c,b))
            
            # Try to add a triple hitting S
            # Generate candidates from S
            # Pick 2 from S, 1 from outside S (or S)
            # Actually, if we pick 3 from S, it hits S in 3.
            
            # Let's try to add a triple with at least 2 points in S.
            added = False
            # Shuffle S for diversity
            S_copy = S[:]
            random.shuffle(S_copy)
            
            for i in range(len(S_copy)):
                for j in range(i+1, len(S_copy)):
                    s1, s2 = S_copy[i], S_copy[j]
                    # Pick x
                    # Try random x
                    x_candidates = points[:]
                    random.shuffle(x_candidates)
                    for x in x_candidates:
                        if x == s1 or x == s2: continue
                        # Check if (s1,s2), (s1,x), (s2,x) are free
                        if ((s1,s2) in blocked_pairs or (s2,s1) in blocked_pairs): continue
                        if ((s1,x) in blocked_pairs or (x,s1) in blocked_pairs): continue
                        if ((s2,x) in blocked_pairs or (x,s2) in blocked_pairs): continue
                        
                        # Add it
                        triples.append((s1, s2, x))
                        blocked_pairs.add((s1,s2)); blocked_pairs.add((s2,s1))
                        blocked_pairs.add((s1,x)); blocked_pairs.add((x,s1))
                        blocked_pairs.add((s2,x)); blocked_pairs.add((x,s2))
                        added = True
                        break
                    if added: break
                if added: break
                
            if added:
                new_alpha = calc_alpha(triples)
                if new_alpha < alpha:
                    alpha = new_alpha
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_triples = triples.copy()
                        if best_alpha <= 5:
                            break
                else:
                    # Revert
                    # Remove new triple
                    triples.pop()
                    s1, s2, x = triples[-1] if triples else (0,0,0) # Careful
                    # Actually, we stored s1,s2,x in loop vars
                    blocked_pairs.discard((s1,s2)); blocked_pairs.discard((s2,s1))
                    blocked_pairs.discard((s1,x)); blocked_pairs.discard((x,s1))
                    blocked_pairs.discard((s2,x)); blocked_pairs.discard((x,s2))
                    
                    # Re-add T_old
                    triples.append(T_old)
                    blocked_pairs.add((a,b)); blocked_pairs.add((b,a))
                    blocked_pairs.add((a,c)); blocked_pairs.add((c,a))
                    blocked_pairs.add((b,c)); blocked_pairs.add((c,b))
            else:
                # Re-add T_old
                triples.append(T_old)
                blocked_pairs.add((a,b)); blocked_pairs.add((b,a))
                blocked_pairs.add((a,c)); blocked_pairs.add((c,a))
                blocked_pairs.add((b,c)); blocked_pairs.add((c,b))

    return [list(t) for t in best_triples]
