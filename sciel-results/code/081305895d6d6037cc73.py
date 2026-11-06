"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 80  # 80s budget, 10s safety margin
    LIMIT = 512
    
    # Precompute LCM compatibility
    # comp[i][j] is True if lcm(i,j) <= LIMIT
    # Use a list of sets for faster checking?
    # Or a 512x512 boolean matrix.
    comp = [[False] * (LIMIT + 1) for _ in range(LIMIT + 1)]
    for i in range(1, LIMIT + 1):
        for j in range(i, LIMIT + 1):
            l = i * j // math.gcd(i, j)
            if l <= LIMIT:
                comp[i][j] = True
                comp[j][i] = True
            else:
                comp[i][j] = False
                comp[j][i] = False
                
    def get_divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return divs
    
    def is_valid_set(S):
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                if not comp[S_list[i]][S_list[j]]:
                    return False
        return True

    def greedy_expand(initial_set):
        current = set(initial_set)
        # Sort candidates by degree in remaining graph?
        # Just iterate 1..512
        for candidate in range(1, LIMIT + 1):
            if time.time() > deadline:
                break
            if candidate not in current:
                # Check if compatible with all in current
                valid = True
                for x in current:
                    if not comp[candidate][x]:
                        valid = False
                        break
                if valid:
                    current.add(candidate)
        return current

    def local_search(initial_set):
        current = set(initial_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to add
            added = False
            for candidate in range(1, LIMIT + 1):
                if time.time() > deadline:
                    break
                if candidate not in current:
                    valid = True
                    for x in current:
                        if not comp[candidate][x]:
                            valid = False
                            break
                    if valid:
                        current.add(candidate)
                        improved = True
                        added = True
                        break
            if added:
                continue
            
            # Try to swap
            # Remove y, add x
            for y in list(current):
                if time.time() > deadline:
                    break
                temp_set = current - {y}
                for x in range(1, LIMIT + 1):
                    if time.time() > deadline:
                        break
                    if x == y or x in current:
                        continue
                    # Check x compatible with temp_set
                    valid = True
                    for z in temp_set:
                        if not comp[x][z]:
                            valid = False
                            break
                    if valid:
                        # Check if this swap is "good"
                        # Heuristic: if x has higher degree than y in the context of the set?
                        # Or just accept if it allows future additions?
                        # For now, accept if it maintains size (swap) and we hope for more.
                        # To avoid cycling, we can keep a history or just rely on time limit.
                        # Better: only swap if x is "better" than y?
                        # Let's just do random swaps if stuck?
                        # Or systematic: try all swaps.
                        current = temp_set | {x}
                        improved = True
                        break
                if improved:
                    break
        return current

    best_set = set()
    best_size = 0
    
    # Candidate starting sets
    candidates = []
    for m in range(1, LIMIT + 1):
        divs = get_divisors(m)
        divs = {x for x in divs if 1 <= x <= LIMIT}
        if len(divs) > best_size:
             # Only keep promising ones
             pass
             
    # Start with highly composite numbers
    hcn = [360, 420, 480, 504, 336, 240, 300, 420, 480, 360, 210, 180, 120, 60, 24, 12, 6, 4, 2, 1]
    
    for m in hcn:
        if time.time() > deadline:
            break
        initial = get_divisors(m)
        initial = {x for x in initial if 1 <= x <= LIMIT}
        
        # Verify initial set is valid (should be)
        if not is_valid_set(initial):
            continue
            
        expanded = greedy_expand(initial)
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_set = expanded
            
        # Local search on the expanded set
        refined = local_search(expanded)
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined

    # Random restarts if time remains
    while time.time() < deadline - 5:
        # Pick random starting set
        m = random.randint(1, LIMIT)
        initial = get_divisors(m)
        initial = {x for x in initial if 1 <= x <= LIMIT}
        if not initial:
            continue
        if not is_valid_set(initial):
            continue
        expanded = greedy_expand(initial)
        refined = local_search(expanded)
        if len(refined) > best_size:
            best_size = len(refined)
            best_set = refined

    if not best_set:
        best_set = {1}
        
    return list(best_set)
# EVOLVE-BLOCK-END
