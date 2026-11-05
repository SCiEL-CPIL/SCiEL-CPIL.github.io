import random
import time
import math

def construct():
    N = 180
    
    # Precompute the third element for any pair (m, M) where m < M
    # If m, M, w form a reciprocal Schur triple (1/m = 1/M + 1/w or 1/m = 1/w + 1/M), 
    # then w is unique.
    # Note: In 1/a = 1/b + 1/c, a is the smallest.
    # So for pair (m, M) with m < M, m is 'a'.
    # 1/m = 1/M + 1/w  =>  1/w = 1/m - 1/M = (M-m)/(mM)  =>  w = mM / (M-m)
    
    # w_val[m][M] stores w if valid, else 0
    w_val = [[0] * (N + 1) for _ in range(N + 1)]
    
    for m in range(1, N + 1):
        for M in range(m + 1, N + 1):
            num = m * M
            den = M - m
            if num % den == 0:
                w = num // den
                if 1 <= w <= N and w != m and w != M:
                    w_val[m][M] = w
                    w_val[M][m] = w # Symmetric storage for quick lookup? 
                                    # Actually, we need to know which is m and which is M.
                                    # We can just store in a dict or 2D array and check order.
                                    # Let's stick to w_val[min][max] = w.

    # Helper to get the third element for pair v, u
    # Returns 0 if no such w exists or w is invalid
    def get_third(v, u):
        if v == u:
            return 0
        m = v if v < u else u
        M = u if v < u else v
        return w_val[m][M]

    # Initial Solution: Greedy
    # Sort by some heuristic? Random order is fine for initialization, 
    # but SA will refine it.
    # Let's use a simple greedy to get a decent start.
    current_set = set()
    vertices = list(range(1, N + 1))
    random.shuffle(vertices)
    
    for v in vertices:
        can_add = True
        # Check if adding v creates a conflict with existing u in current_set
        for u in current_set:
            w = get_third(v, u)
            if w != 0 and w in current_set:
                can_add = False
                break
        if can_add:
            current_set.add(v)
            
    best_set = current_set.copy()
    best_size = len(current_set)
    
    start_time = time.time()
    T = 5.0
    decay = 0.999
    min_T = 0.1
    
    # Precompute list of vertices for sampling
    all_vertices = list(range(1, N + 1))
    
    while time.time() - start_time < 50:
        if T < min_T:
            T = min_T
            
        # Pick a random vertex v not in current_set
        # To be efficient, we can maintain a list of available vertices, 
        # but N is small (180), so scanning is fine.
        v = random.choice(all_vertices)
        
        if v in current_set:
            # If v is already in set, we might want to try removing it?
            # Or just pick another. 
            # Let's try a "swap" or "remove" move occasionally?
            # The standard SA move "Add v, remove conflicts" works even if v is in S?
            # If v in S, R will contain v? No, we check u in S.
            # If v in S, we don't add it. 
            # Let's just re-pick or handle "remove v" move.
            # A remove move: pick v in S, remove it.
            # Delta = -1.
            # This helps escape local optima where we are stuck.
            if random.random() < 0.1:
                # Try removing v
                new_set = current_set.copy()
                new_set.remove(v)
                delta = -1
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_set = new_set
            continue

        # Calculate conflicts R
        R = set()
        for u in current_set:
            w = get_third(v, u)
            if w != 0 and w in current_set:
                R.add(u)
                
        # New set: remove R, add v
        # Note: v is not in current_set, so no conflict with itself.
        # Are there conflicts between elements in R? 
        # Removing R might break other triples, but we are only checking 
        # if the NEW set is valid.
        # The new set is S' = (S \ R) U {v}.
        # We must ensure S' is valid.
        # Since we removed all u in S that formed a triple with v and some w in S,
        # are there any remaining triples in S'?
        # Any triple in S' must involve v or be a triple in S \ R.
        # Triples in S \ R: Since S was valid, any triple in S \ R would have been in S.
        # But wait, if we remove u, we might break a triple, but we don't create one.
        # The only new triples can involve v.
        # We ensured that for any u in S \ R, there is no w in S \ R such that {v, u, w} is a triple.
        # Wait, the condition for u to be in R was: there EXISTS w in S such that {v, u, w} is a triple.
        # If w was in R, then w is removed.
        # If w was NOT in R, then w is in S \ R.
        # So if there exists w in S \ R such that {v, u, w} is a triple, then u should have been in R.
        # Therefore, for all u in S \ R, there is NO w in S \ R such that {v, u, w} is a triple.
        # So S' is valid.
        
        new_size = len(current_set) - len(R) + 1
        delta = new_size - len(current_set)
        
        if delta > 0 or random.random() < math.exp(delta / T):
            # Apply move
            current_set = current_set - R
            current_set.add(v)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
                
        T *= decay

    return sorted(best_set)
