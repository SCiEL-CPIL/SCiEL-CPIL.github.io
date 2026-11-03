import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Lower bound
    diffs_per_row = (k + 1) * k // 2
    lower_bound = n_rows * diffs_per_row
    
    # Fallback: Simple Greedy
    def greedy_fallback():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # Try to find optimal solution using Clique on Random Pool
    best_rows = greedy_fallback()
    best_scope = max(max(row) for row in best_rows)
    
    # We will search for M in [lower_bound, best_scope)
    # If we find a solution with M < best_scope, we update best_rows
    
    current_M = lower_bound
    max_M_to_try = min(best_scope - 1, lower_bound + 100) # Cap the search range
    
    while current_M < max_M_to_try and time.time() < deadline:
        # Try to find a solution with max entry <= current_M
        solution = search_for_solution(n_rows, k, current_M, deadline)
        if solution:
            # Validate and update
            scope = max(max(row) for row in solution)
            if scope < best_scope:
                best_rows = solution
                best_scope = scope
                # If we hit the lower bound, we can stop (it's optimal)
                if scope == lower_bound:
                    break
            # Even if not better, if we found one, we can try smaller M?
            # No, we are iterating M upwards.
            # If we found one for M, we should try M-1? 
            # Actually, if we found one for M, it's valid. 
            # We want the smallest M.
            # So if we find a solution for M, we should check if a smaller M works?
            # But we started from lower_bound and went up.
            # So the first M we find a solution for is the minimum (or close to it).
            # However, random search might miss a solution for a smaller M.
            # So we should keep trying smaller M? No, we go up.
            # If we find a solution for M=120, we return it.
            # But maybe M=115 works too.
            # Since we start from lower_bound, the first success is the best we found.
            # But random search is not guaranteed to find it.
            # So we should try multiple times for each M?
            # Or just accept the first found M as "good enough" and stop?
            # Given time constraints, accepting the first found M is reasonable.
            return solution
        else:
            current_M += 1
            
    return best_rows

def search_for_solution(n_rows, k, M, deadline):
    # Generate pool of random Sidon rows
    pool_size = 1000
    max_pool_size = 5000
    
    while time.time() < deadline:
        pool = []
        for _ in range(pool_size):
            if time.time() > deadline:
                break
            row = generate_random_sidon(k, M)
            if row:
                # Compute mask
                mask = 0
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        diff = row[j] - row[i]
                        mask |= (1 << diff)
                pool.append((row, mask))
                
        if len(pool) < n_rows:
            return None
            
        # Shuffle pool to randomize search
        random.shuffle(pool)
        
        # Find clique
        sol = find_clique(pool, n_rows)
        if sol:
            return [row for row, mask in sol]
            
        # If not found, increase pool size
        if pool_size < max_pool_size:
            pool_size = int(pool_size * 1.5)
        else:
            # If max pool size reached, maybe M is too small or we are unlucky
            # Break to let outer loop increase M
            break
            
    return None

def generate_random_sidon(k, M):
    row = [0]
    used = set()
    while len(row) < k + 1:
        last = row[-1]
        remaining = k + 1 - len(row)
        min_candidate = last + 1
        max_candidate = M - (remaining - 1)
        
        if min_candidate > max_candidate:
            return None
            
        candidates = []
        # Optimization: Instead of checking all, maybe sample?
        # For small M, checking all is fine.
        for x in range(min_candidate, max_candidate + 1):
            new_diffs = [x - a for a in row]
            if any(d in used for d in new_diffs):
                continue
            candidates.append(x)
            
        if not candidates:
            return None
            
        x = random.choice(candidates)
        row.append(x)
        for a in row[:-1]:
            used.add(x - a)
            
    return row

def find_clique(pool, n_rows):
    # Backtracking
    selected = []
    
    def backtrack(start_idx):
        if len(selected) == n_rows:
            return True
        
        # Pruning
        if start_idx + (n_rows - len(selected)) > len(pool):
            return False
            
        for i in range(start_idx, len(pool)):
            row, mask = pool[i]
            conflict = False
            for _, sel_mask in selected:
                if mask & sel_mask:
                    conflict = True
                    break
            if conflict:
                continue
                
            selected.append((row, mask))
            if backtrack(i + 1):
                return True
            selected.pop()
            
        return False

    if backtrack(0):
        return selected
    return None
