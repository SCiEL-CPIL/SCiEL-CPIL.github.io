import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin for 90s limit
    
    N = 256
    
    # Precompute for each x, the list of "conflict pairs" - pairs (a,b) such that 
    # {a,b,x} forms a 3-AP. When checking if x can be added, we just need to check
    # if any conflict pair is fully in S.
    # Actually, more useful: for each x, precompute the list of (i, j) pairs where
    # adding x would create a 3-AP iff both i and j are in S.
    
    # For x to form a 3-AP:
    # Case 1: x is middle: (x-d, x, x+d) for d=1..min(x-1, N-x)
    # Case 2: x is largest: (x-2d, x-d, x) for d=1..(x-1)//2
    # Case 3: x is smallest: (x, x+d, x+2d) for d=1..(N-x)//2
    
    # Precompute conflict_pairs[x] = list of (a, b) pairs
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x is middle
        for d in range(1, min(x, N - x + 1)):
            a, b = x - d, x + d
            if a >= 1 and b <= N:
                conflict_pairs[x].append((a, b))
        # x is largest
        for d in range(1, (x + 1) // 2 + 1):
            a, b = x - 2*d, x - d
            if a >= 1 and b <= N:
                conflict_pairs[x].append((a, b))
        # x is smallest
        for d in range(1, (N - x) // 2 + 1):
            a, b = x + d, x + 2*d
            if a <= N and b <= N:
                conflict_pairs[x].append((a, b))
    
    def can_add(x, in_set):
        """Check if x can be added given current set represented by bool array (1-indexed)."""
        for a, b in conflict_pairs[x]:
            if in_set[a] and in_set[b]:
                return False
        return True
    
    # Greedy initialization: add numbers in order 1, 2, ..., N
    # Since 1/x decreases with x, we prefer smaller numbers
    in_set = [False] * (N + 1)
    S = set()
    for x in range(1, N + 1):
        if can_add(x, in_set):
            in_set[x] = True
            S.add(x)
    
    def compute_score(S):
        return sum(1.0 / x for x in S)
    
    best_set = set(S)
    best_score = compute_score(best_set)
    best_in_set = in_set[:]
    
    # Local search
    improved = True
    round_count = 0
    no_improve_rounds = 0
    
    while improved and time.time() < deadline:
        improved = False
        round_count += 1
        
        # Strategy 1: Remove-and-readd (remove x, greedily re-fill)
        # Sort by value (1/x) descending to try removing valuable elements first
        sorted_by_value = sorted(best_set, key=lambda x: -1.0/x)
        
        for x in sorted_by_value:
            if time.time() > deadline:
                break
            
            in_set_temp = best_in_set[:]
            in_set_temp[x] = False
            S_temp = best_set - {x}
            
            new_S = set(S_temp)
            new_in_set = in_set_temp[:]
            
            for y in range(1, N + 1):
                if new_in_set[y]:
                    continue
                if can_add(y, new_in_set):
                    new_in_set[y] = True
                    new_S.add(y)
            
            new_score = compute_score(new_S)
            if new_score > best_score + 1e-15:
                best_set = new_S
                best_score = new_score
                best_in_set = new_in_set
                improved = True
                no_improve_rounds = 0
        
        # Strategy 2: Single swap - remove x, add y
        if time.time() < deadline:
            for x in sorted(best_set):
                if time.time() > deadline:
                    break
                
                in_set_temp = best_in_set[:]
                in_set_temp[x] = False
                S_temp = best_set - {x}
                
                best_y = -1
                best_new_score = best_score
                
                for y in range(1, N + 1):
                    if in_set_temp[y]:
                        continue
                    if can_add(y, in_set_temp):
                        new_score = best_score - 1.0/x + 1.0/y
                        if new_score > best_new_score:
                            best_new_score = new_score
                            best_y = y
                
                if best_y > 0 and best_new_score > best_score + 1e-15:
                    best_set = S_temp | {best_y}
                    best_score = best_new_score
                    best_in_set = in_set_temp[:]
                    best_in_set[best_y] = True
                    improved = True
                    no_improve_rounds = 0
        
        # Strategy 3: Remove two elements, re-fill greedily
        # Try removing pairs of larger elements (less valuable) to free up space
        if time.time() < deadline:
            elements = sorted(best_set, reverse=True)  # largest first (least valuable)
            n_elem = len(elements)
            max_pairs_to_try = min(50, n_elem)
            
            for i in range(max_pairs_to_try):
                if time.time() > deadline:
                    break
                for j in range(i + 1, min(i + 20, n_elem)):
                    if time.time() > deadline:
                        break
                    x1, x2 = elements[i], elements[j]
                    
                    in_set_temp = best_in_set[:]
                    in_set_temp[x1] = False
                    in_set_temp[x2] = False
                    S_temp = best_set - {x1, x2}
                    
                    new_S = set(S_temp)
                    new_in_set = in_set_temp[:]
                    
                    for y in range(1, N + 1):
                        if new_in_set[y]:
                            continue
                        if can_add(y, new_in_set):
                            new_in_set[y] = True
                            new_S.add(y)
                    
                    new_score = compute_score(new_S)
                    if new_score > best_score + 1e-15:
                        best_set = new_S
                        best_score = new_score
                        best_in_set = new_in_set
                        improved = True
                        no_improve_rounds = 0
        
        # Strategy 4: Multi-swap - remove x, try adding multiple y's
        if time.time() < deadline:
            for x in sorted(best_set):
                if time.time() > deadline:
                    break
                
                in_set_temp = best_in_set[:]
                in_set_temp[x] = False
                S_temp = best_set - {x}
                
                # Try adding up to 2 elements after removing x
                best_improvement = 0
                best_additions = []
                
                # Find all candidates that can be added
                candidates = []
                for y in range(1, N + 1):
                    if in_set_temp[y]:
                        continue
                    if can_add(y, in_set_temp):
                        candidates.append(y)
                
                # Try single additions
                for y in candidates:
                    new_score = best_score - 1.0/x + 1.0/y
                    if new_score > best_score + best_improvement:
                        best_improvement = new_score - best_score
                        best_additions = [y]
                
                # Try double additions (if time permits)
                if len(candidates) > 1 and time.time() < deadline - 5:
                    for i in range(min(10, len(candidates))):
                        for j in range(i + 1, min(10, len(candidates))):
                            y1, y2 = candidates[i], candidates[j]
                            # Check if both can be added
                            test_in_set = in_set_temp[:]
                            test_in_set[y1] = True
                            if can_add(y2, test_in_set):
                                new_score = best_score - 1.0/x + 1.0/y1 + 1.0/y2
                                if new_score > best_score + best_improvement:
                                    best_improvement = new_score - best_score
                                    best_additions = [y1, y2]
                
                if best_additions and best_improvement > 1e-15:
                    new_set = S_temp | set(best_additions)
                    new_in_set = in_set_temp[:]
                    for y in best_additions:
                        new_in_set[y] = True
                    best_set = new_set
                    best_score += best_improvement
                    best_in_set = new_in_set
                    improved = True
                    no_improve_rounds = 0
        
        if not improved:
            no_improve_rounds += 1
            if no_improve_rounds >= 3:
                break
    
    # Final polish pass: try all single removals with greedy refill one more time
    if time.time() < deadline - 2:
        final_improved = True
        while final_improved and time.time() < deadline - 1:
            final_improved = False
            for x in sorted(best_set):
                if time.time() > deadline - 1:
                    break
                in_set_temp = best_in_set[:]
                in_set_temp[x] = False
                S_temp = best_set - {x}
                
                new_S = set(S_temp)
                new_in_set = in_set_temp[:]
                
                for y in range(1, N + 1):
                    if new_in_set[y]:
                        continue
                    if can_add(y, new_in_set):
                        new_in_set[y] = True
                        new_S.add(y)
                
                new_score = compute_score(new_S)
                if new_score > best_score + 1e-15:
                    best_set = new_S
                    best_score = new_score
                    best_in_set = new_in_set
                    final_improved = True
    
    return list(best_set)
