import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    N = 192
    
    def rebuild_bad_res(S):
        bad_res = {}
        S_list = list(S)
        for a in S_list:
            residues = set()
            others = [x for x in S_list if x != a]
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    s = others[i] + others[j]
                    residues.add(s % a)
            bad_res[a] = residues
        return bad_res

    def can_add(y, S, bad_res):
        S_list = list(S)
        if len(S_list) >= 3:
            for t in combinations(S_list, 3):
                if sum(t) % y == 0:
                    return False
        for a in S_list:
            if y % a in bad_res[a]:
                return False
        return True

    def add_element(y, S, bad_res):
        S_list = list(S)
        for a in S_list:
            if a == y: continue
            others = [x for x in S_list if x != a and x != y]
            for a2 in others:
                bad_res[a].add((y + a2) % a)
        
        others = [x for x in S_list if x != y]
        residues_y = set()
        for i in range(len(others)):
            for j in range(i+1, len(others)):
                residues_y.add((others[i] + others[j]) % y)
        bad_res[y] = residues_y
        
        S.add(y)
        return S, bad_res

    # Init
    S = set()
    bad_res = {}
    for x in range(N, 0, -1):
        if can_add(x, S, bad_res):
            S, bad_res = add_element(x, S, bad_res)
            
    best_S = set(S)
    
    while time.time() < deadline:
        # Try Add
        candidates = [x for x in range(1, N+1) if x not in S]
        random.shuffle(candidates)
        added = False
        for y in candidates:
            if can_add(y, S, bad_res):
                S, bad_res = add_element(y, S, bad_res)
                if len(S) > len(best_S):
                    best_S = set(S)
                added = True
                break
        
        if not added and len(S) > 0:
            # Try Swap
            S_list = list(S)
            random.shuffle(S_list)
            x = S_list[0]
            
            candidates = [x for x in range(1, N+1) if x not in S]
            random.shuffle(candidates)
            
            swapped = False
            for y in candidates:
                S.remove(x)
                bad_res_tmp = rebuild_bad_res(S)
                
                if can_add(y, S, bad_res_tmp):
                    S, bad_res = add_element(y, S, bad_res_tmp)
                    if len(S) > len(best_S):
                        best_S = set(S)
                    swapped = True
                    break
                else:
                    S.add(x)
                    # bad_res is stale here, but we will rebuild if we try another y or exit loop
                    # To be safe, if we continue trying y, we need to rebuild for S\{x} again?
                    # No, S is back to original. The next y will remove x again and rebuild.
                    # So we don't need to rebuild here.
            
            if not swapped:
                # S is original, bad_res is stale (from last failed swap's rebuild or previous state)
                # Rebuild to ensure consistency for next iteration
                bad_res = rebuild_bad_res(S)

    return list(best_S)

# EVOLVE-BLOCK-START
# Code is in the function above
# EVOLVE-BLOCK-END
