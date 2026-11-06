import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s limit
    
    n = 256
    best_A = []
    best_B = []
    best_size = 0
    
    def get_sums(S):
        return {a + b for a, b in combinations_with_replacement(S, 2)}
    
    def get_diffs(S):
        return {b - a for a, b in combinations(S, 2)}
        
    def can_add_sidon(S, sums, x):
        new_sums = {x + s for s in S}
        new_sums.add(2 * x)
        if new_sums & sums:
            return False
        return True

    def can_add_diff(S, diffs, D_other, x):
        new_diffs = {x - s for s in S}
        new_diffs.update({s - x for s in S})
        if new_diffs & D_other:
            return False
        return True

    def is_valid_state(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        sums_A = get_sums(A)
        if len(sums_A) != len(list(combinations_with_replacement(A, 2))):
            return False
        sums_B = get_sums(B)
        if len(sums_B) != len(list(combinations_with_replacement(B, 2))):
            return False
        diffs_A = get_diffs(A)
        diffs_B = get_diffs(B)
        if diffs_A & diffs_B:
            return False
        return True

    def local_search(A, B):
        A = set(A)
        B = set(B)
        
        sums_A = get_sums(list(A))
        diffs_A = get_diffs(list(A))
        sums_B = get_sums(list(B))
        diffs_B = get_diffs(list(B))
        
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add elements
            for x in range(1, n + 1):
                if x in A or x in B:
                    continue
                
                # Try adding to A
                if len(A) < 64:
                    if can_add_sidon(A, sums_A, x) and can_add_diff(A, diffs_A, diffs_B, x):
                        A.add(x)
                        sums_A.update({x + s for s in A if s != x})
                        sums_A.add(2 * x)
                        diffs_A.update({x - s for s in A if s != x})
                        diffs_A.update({s - x for s in A if s != x})
                        improved = True
                        break
                
                # Try adding to B
                if len(B) < 64:
                    if can_add_sidon(B, sums_B, x) and can_add_diff(B, diffs_B, diffs_A, x):
                        B.add(x)
                        sums_B.update({x + s for s in B if s != x})
                        sums_B.add(2 * x)
                        diffs_B.update({x - s for s in B if s != x})
                        diffs_B.update({s - x for s in B if s != x})
                        improved = True
                        break
            
            if improved:
                continue
            
            # 2. Try kicks (Remove one, Add one) if stuck
            if time.time() - start_time > time_limit - 5:
                break
                
            kicked = False
            # Try kicking from A
            if A and not kicked:
                for _ in range(5):
                    if time.time() - start_time > time_limit - 5:
                        break
                    x_to_remove = random.choice(list(A))
                    A_old = A.copy()
                    B_old = B.copy()
                    sums_A_old = sums_A.copy()
                    diffs_A_old = diffs_A.copy()
                    sums_B_old = sums_B.copy()
                    diffs_B_old = diffs_B.copy()
                    
                    A.remove(x_to_remove)
                    sums_A_new = get_sums(list(A))
                    diffs_A_new = get_diffs(list(A))
                    
                    found_add = False
                    for y in range(1, n + 1):
                        if y in A or y in B:
                            continue
                        if len(A) < 64 and can_add_sidon(A, sums_A_new, y) and can_add_diff(A, diffs_A_new, diffs_B, y):
                            A.add(y)
                            found_add = True
                            break
                        if len(B) < 64 and can_add_sidon(B, sums_B, y) and can_add_diff(B, diffs_B, diffs_A_new, y):
                            B.add(y)
                            found_add = True
                            break
                    
                    if found_add:
                        sums_A = get_sums(list(A))
                        diffs_A = get_diffs(list(A))
                        sums_B = get_sums(list(B))
                        diffs_B = get_diffs(list(B))
                        improved = True
                        kicked = True
                        break
                    else:
                        A = A_old
                        B = B_old
                        sums_A = sums_A_old
                        diffs_A = diffs_A_old
                        sums_B = sums_B_old
                        diffs_B = diffs_B_old
            
            # Try kicking from B
            if B and not kicked:
                for _ in range(5):
                    if time.time() - start_time > time_limit - 5:
                        break
                    x_to_remove = random.choice(list(B))
                    A_old = A.copy()
                    B_old = B.copy()
                    sums_A_old = sums_A.copy()
                    diffs_A_old = diffs_A.copy()
                    sums_B_old = sums_B.copy()
                    diffs_B_old = diffs_B.copy()
                    
                    B.remove(x_to_remove)
                    sums_B_new = get_sums(list(B))
                    diffs_B_new = get_diffs(list(B))
                    
                    found_add = False
                    for y in range(1, n + 1):
                        if y in A or y in B:
                            continue
                        if len(A) < 64 and can_add_sidon(A, sums_A, y) and can_add_diff(A, diffs_A, diffs_B_new, y):
                            A.add(y)
                            found_add = True
                            break
                        if len(B) < 64 and can_add_sidon(B, sums_B_new, y) and can_add_diff(B, diffs_B_new, diffs_A, y):
                            B.add(y)
                            found_add = True
                            break
                    
                    if found_add:
                        sums_A = get_sums(list(A))
                        diffs_A = get_diffs(list(A))
                        sums_B = get_sums(list(B))
                        diffs_B = get_diffs(list(B))
                        improved = True
                        kicked = True
                        break
                    else:
                        A = A_old
                        B = B_old
                        sums_A = sums_A_old
                        diffs_A = diffs_A_old
                        sums_B = sums_B_old
                        diffs_B = diffs_B_old
            
            if kicked:
                continue

        return list(A), list(B)

    num_restarts = 50
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        A = set()
        B = set()
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
                
            if len(A) < 64:
                if can_add_sidon(A, sums_A, x) and can_add_diff(A, diffs_A, diffs_B, x):
                    A.add(x)
                    sums_A.update({x + s for s in A if s != x})
                    sums_A.add(2 * x)
                    diffs_A.update({x - s for s in A if s != x})
                    diffs_A.update({s - x for s in A if s != x})
                    continue
            
            if len(B) < 64:
                if can_add_sidon(B, sums_B, x) and can_add_diff(B, diffs_B, diffs_A, x):
                    B.add(x)
                    sums_B.update({x + s for s in B if s != x})
                    sums_B.add(2 * x)
                    diffs_B.update({x - s for s in B if s != x})
                    diffs_B.update({s - x for s in B if s != x})
                    continue
            
        A, B = local_search(list(A), list(B))
        
        if is_valid_state(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A = A
                best_B = B

    if not best_A or not best_B:
        return [[1], [2]]
        
    return [best_A, best_B]
