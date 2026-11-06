import math
import time
import random

def construct():
    # Deadline guard: 150s limit, use 140s to be safe
    deadline = time.time() + 140
    
    min_m = 20
    max_m = 200
    
    # Precompute gcd table for speed
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    # Moduli sorted by weight (1/m) descending, i.e., m ascending
    moduli = list(range(min_m, max_m + 1))
    
    # Precompute suffix sums of weights for bounding
    weights = [1.0 / m for m in moduli]
    suffix_sums = [0.0] * (len(moduli) + 1)
    for i in range(len(moduli) - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + weights[i]
    
    best_score = 0.0
    best_solution = []
    
    # Helper to check validity of (m, r) against current selection
    def is_valid(m, r, selection):
        for m2, r2 in selection:
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def dfs(idx, current_score, selection):
        nonlocal best_score, best_solution
        
        # Time check
        if time.time() > deadline:
            return
            
        # Bound check
        if current_score + suffix_sums[idx] <= best_score:
            return
            
        if idx == len(moduli):
            if current_score > best_score:
                best_score = current_score
                best_solution = list(selection)
            return
            
        m = moduli[idx]
        
        # Try to include m with some residue r
        for r in range(m):
            if is_valid(m, r, selection):
                selection.append((m, r))
                dfs(idx + 1, current_score + weights[idx], selection)
                selection.pop()
                
        # Try to skip m
        dfs(idx + 1, current_score, selection)

    # Initial greedy solution to set a good lower bound for pruning
    greedy_sel = []
    for m in moduli:
        for r in range(m):
            if is_valid(m, r, greedy_sel):
                greedy_sel.append((m, r))
                break
    if greedy_sel:
        best_score = sum(1.0 / m for m, _ in greedy_sel)
        best_solution = list(greedy_sel)
        
    # Run DFS
    dfs(0, 0.0, [])
    
    # Convert to required format
    if best_solution:
        return [[m, r] for m, r in best_solution]
    return []

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
