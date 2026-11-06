import time
from math import gcd

def construct():
    start_time = time.time()
    # Use 145s to be safe against 150s limit
    limit = 145.0
    
    N = 512
    
    def get_score(A):
        # A is a list of ints
        s = 0.0
        for a in A:
            s += 1.0 / (N - a)
        return s

    def is_coprime_to_all(a, A):
        for b in A:
            if gcd(a, b) != 1:
                return False
        return True
        
    def greedy_fill(A, candidates):
        # A is a list. candidates is an iterable.
        current = list(A)
        for c in candidates:
            if c in current:
                continue
            if is_coprime_to_all(c, current):
                current.append(c)
        return current

    # Initial solution: Seed strategy (descending a)
    best_A = []
    for a in range(N - 1, 0, -1):
        if is_coprime_to_all(a, best_A):
            best_A.append(a)
            
    best_score = get_score(best_A)
    
    # Local Search: Remove one element, re-greedy
    improved = True
    while improved and (time.time() - start_time) < limit:
        improved = False
        # Try removing each element
        for i in range(len(best_A)):
            if (time.time() - start_time) > limit:
                break
            # Remove best_A[i]
            A_temp = [x for j, x in enumerate(best_A) if j != i]
            # Re-greedy with all candidates sorted by weight (descending a)
            A_new = greedy_fill(A_temp, range(N - 1, 0, -1))
            
            score_new = get_score(A_new)
            if score_new > best_score:
                best_A = A_new
                best_score = score_new
                improved = True
                break
                
    # Random Perturbations
    if (time.time() - start_time) < limit:
        import random
        random.seed(42)
        
        for _ in range(200): # Try more perturbations
            if (time.time() - start_time) > limit:
                break
            
            if len(best_A) < 2:
                continue
                
            A_pert = list(best_A)
            # Remove 2 random distinct elements
            idx1 = random.randint(0, len(A_pert) - 1)
            idx2 = random.randint(0, len(A_pert) - 1)
            if idx1 == idx2:
                continue
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            A_pert.pop(idx1)
            A_pert.pop(idx2)
            
            # Re-greedy
            A_new = greedy_fill(A_pert, range(N - 1, 0, -1))
            score_new = get_score(A_new)
            
            if score_new > best_score:
                best_A = A_new
                best_score = score_new
                # Run local search on new best
                improved = True
                while improved and (time.time() - start_time) < limit:
                    improved = False
                    for i in range(len(best_A)):
                        if (time.time() - start_time) > limit:
                            break
                        A_temp = [x for j, x in enumerate(best_A) if j != i]
                        A_new2 = greedy_fill(A_temp, range(N - 1, 0, -1))
                        score_new2 = get_score(A_new2)
                        if score_new2 > best_score:
                            best_A = A_new2
                            best_score = score_new2
                            improved = True
                            break

    return best_A
