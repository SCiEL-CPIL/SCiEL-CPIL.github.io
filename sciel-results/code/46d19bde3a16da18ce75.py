"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 180
    
    # Check if a set is valid: no three distinct a,b,c with 1/a = 1/b + 1/c
    # Equivalent: for distinct a,b,c, a != bc/(b+c)
    def is_valid(S):
        S_set = set(S)
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                # c = a*b/(b-a) must be integer and in S and distinct from a,b
                num = a * b
                den = b - a
                if den == 0:
                    continue
                if num % den == 0:
                    c = num // den
                    if c != a and c != b and c in S_set and c <= N:
                        return False
        return True
    
    # Start with the seed solution: [91, 180]
    best_set = set(range(91, 181))
    
    # Try to add elements from [1, 90] one by one, greedily
    # For each candidate, check if adding it maintains validity
    for candidate in range(1, 91):
        if time.time() > deadline:
            break
        new_set = best_set | {candidate}
        if is_valid(new_set):
            best_set = new_set
    
    # Try to add more by removing some and adding others
    # Simple local search: try swapping elements
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        for remove_candidate in list(best_set):
            if time.time() > deadline:
                break
            for add_candidate in range(1, N+1):
                if add_candidate in best_set:
                    continue
                new_set = (best_set - {remove_candidate}) | {add_candidate}
                if is_valid(new_set):
                    best_set = new_set
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return list(best_set)
# EVOLVE-BLOCK-END
