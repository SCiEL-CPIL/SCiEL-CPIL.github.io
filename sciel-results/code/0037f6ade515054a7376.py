import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 90s limit
    
    N = 192
    MAX_SIZE = 48
    
    def is_valid(S):
        """Check if set S satisfies the constraint."""
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            # Check if a divides any triple sum of other elements
            others = [S_list[j] for j in range(n) if j != i]
            m = len(others)
            for u_idx in range(m):
                u = others[u_idx]
                for v_idx in range(u_idx + 1, m):
                    v = others[v_idx]
                    for w_idx in range(v_idx + 1, m):
                        w = others[w_idx]
                        if (u + v + w) % a == 0:
                            return False
        return True
    
    def is_valid_addition(x, S):
        """Check if adding x to S maintains validity."""
        # 1. Check if x divides any triple in S
        if len(S) >= 3:
            S_list = list(S)
            for i in range(len(S_list)):
                a = S_list[i]
                others = [S_list[j] for j in range(len(S_list)) if j != i]
                for u in others:
                    for v in others:
                        if u == v:
                            continue
                        for w in others:
                            if w == u or w == v:
                                continue
                            if (u + v + w) % x == 0:
                                return False
        
        # 2. Check if any a in S is violated by triples involving x
        S_list = list(S)
        for a in S_list:
            # Triples are {x, u, v} where u, v in S, u != v, u != a, v != a
            target = (-x) % a
            others = [v for v in S_list if v != a]
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if (others[i] + others[j]) % a == target:
                        return False
        return True

    # Strategy 1: Greedy Removal from High Range
    # Large numbers are less likely to divide sums of other numbers.
    # Start with a large pool and remove conflicts.
    
    def solve_removal():
        # Start with top 64 numbers? Or all 192?
        # Let's start with a reasonably sized pool to allow removal to work fast.
        # If we start with 192, removal is expensive.
        # Let's try starting with numbers 97..192 (96 numbers).
        # Or even higher: 145..192 (48 numbers).
        # The lesson said "Greedy removal from the densest valid-likely pool (144-192) ... found valid set of size 48".
        
        # Let's try multiple pools.
        best_S = set()
        
        pools = [
            list(range(192, 143, -1)), # 48 numbers
            list(range(192, 120, -1)), # 72 numbers
            list(range(192, 90, -1)),  # 102 numbers
        ]
        
        for pool in pools:
            S = set(pool)
            # Greedy removal
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                # Try to remove each element to see if it resolves a conflict?
                # No, we want to KEEP as many as possible.
                # So we check if current S is valid.
                # If not, find a conflict and remove one of the elements involved.
                
                if is_valid(S):
                    if len(S) > len(best_S):
                        best_S = set(S)
                    break # Found valid, stop removing for this pool
                
                # Find a conflict
                S_list = list(S)
                n = len(S_list)
                conflict_found = False
                for i in range(n):
                    a = S_list[i]
                    others = [S_list[j] for j in range(n) if j != i]
                    m = len(others)
                    for u_idx in range(m):
                        u = others[u_idx]
                        for v_idx in range(u_idx + 1, m):
                            v = others[v_idx]
                            for w_idx in range(v_idx + 1, m):
                                w = others[w_idx]
                                if (u + v + w) % a == 0:
                                    # Conflict: a divides u+v+w
                                    # We can remove a, u, v, or w.
                                    # Heuristic: Remove the smallest? Or the one that causes most conflicts?
                                    # For now, remove 'a' (the divisor) as it might be "bad".
                                    # Or remove one of the summands.
                                    # Let's try removing 'a' first.
                                    S.remove(a)
                                    changed = True
                                    conflict_found = True
                                    break
                        if conflict_found:
                            break
                    if conflict_found:
                        break
                
        return best_S

    # Strategy 2: Greedy Addition
    def solve_addition():
        S = set()
        # Try adding from high to low? Or low to high?
        # High to low is often better for this type of problem.
        candidates = list(range(N, 0, -1))
        random.shuffle(candidates)
        
        # Actually, deterministic high-to-low might be better for reproducibility, 
        # but randomized helps escape local minima if we restart.
        
        for x in candidates:
            if len(S) >= MAX_SIZE:
                break
            if is_valid_addition(x, S):
                S.add(x)
        
        return S

    # Strategy 3: Local Search / Hill Climbing
    def local_search(S):
        S = set(S)
        best_S = set(S)
        
        # Try to add
        candidates = [i for i in range(1, N+1) if i not in S]
        random.shuffle(candidates)
        
        # Try swaps
        # If we can't add, maybe swap a bad element for a good one?
        
        # For now, just try to add more if possible.
        for x in candidates:
            if len(S) >= MAX_SIZE:
                break
            if is_valid_addition(x, S):
                S.add(x)
                if len(S) > len(best_S):
                    best_S = set(S)
        
        # Try swapping to improve? 
        # This is complex to evaluate "improvement" unless we define a fitness.
        # Since the goal is just size, any valid set of max size is optimal.
        
        return best_S

    # Execute strategies
    
    # 1. Try Greedy Removal (most promising based on lessons)
    S_remove = solve_removal()
    best_S = S_remove
    
    # 2. Try Greedy Addition
    S_add = solve_addition()
    if len(S_add) > len(best_S):
        best_S = S_add
    
    # 3. Try to improve best_S with local search (addition)
    # But careful: is_valid_addition is O(N^3) roughly. 
    # With N=48, 48^3 = 110k ops. 192 candidates * 110k = 20M ops. 
    # This might take ~1-2 seconds in Python. 
    # We have ~50s left. So we can do multiple random restarts of addition.
    
    if time.time() < deadline - 10:
        for _ in range(20): # 20 restarts
            if time.time() > deadline - 5:
                break
            S_add = solve_addition()
            # Simple local search: try to add remaining candidates
            S_add = local_search(S_add)
            if len(S_add) > len(best_S):
                best_S = S_add
    
    # Final check: ensure best_S is valid
    if not is_valid(best_S):
        # Fallback: try a very simple greedy
        S = set()
        for x in range(N, 0, -1):
            if is_valid_addition(x, S):
                S.add(x)
        if len(S) > len(best_S):
            best_S = S
    
    return list(best_S)
