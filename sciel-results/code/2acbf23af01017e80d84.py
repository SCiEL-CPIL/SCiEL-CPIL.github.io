from fractions import Fraction
import time
import itertools

LCM = 232792560  # LCM(1..22)
# verify: let me compute
from math import gcd
def _lcm(a,b): return a*b//gcd(a,b)
l=1
for i in range(1,23): l=_lcm(l,i)
# l should be 232792560

def construct():
    start = time.time()
    deadline = start + 50  # safety margin under 60s
    # weights
    w = {i: LCM // i for i in range(1,23)}
    
    def subset_sums_distinct(S):
        # S: list of elements
        sums = [0]
        for i in S:
            wi = w[i]
            # append s+wi for s in current sums
            cur = len(sums)
            for j in range(cur):
                sums.append(sums[j] + wi)
        # check distinct
        return len(set(sums)) == len(sums), sums
    
    # Phase 1: start with all, remove to fix
    def fix_set(S, strategy):
        S = list(S)
        while time.time() < deadline:
            ok, sums = subset_sums_distinct(S)
            if ok:
                return S
            # find a collision: find a sum that appears twice
            # need the two subsets; recompute with tracking
            # Instead, find duplicate value
            seen = {}
            sums2 = [0]
            for i in S:
                wi = w[i]
                cur = len(sums2)
                for j in range(cur):
                    sums2.append(sums2[j] + wi)
            # find dup
            dup_val = None
            for v in sums2:
                if v in seen:
                    dup_val = v
                    break
                seen[v] = True
            if dup_val is None:
                return S
            # find two subsets with this sum
            # recompute with subset masks
            n = len(S)
            # map sum -> mask
            mask_of = {}
            sums3 = [0]
            masks = [0]
            for idx, i in enumerate(S):
                wi = w[i]
                cur = len(sums3)
                for j in range(cur):
                    sums3.append(sums3[j] + wi)
                    masks.append(masks[j] | (1 << idx))
            # find two masks with dup_val
            m1 = None
            m2 = None
            for mm, vv in zip(masks, sums3):
                if vv == dup_val:
                    if m1 is None:
                        m1 = mm
                    else:
                        m2 = mm
                        break
            # symmetric difference
            D_mask = m1 ^ m2
            # elements in D
            D_elems = [S[idx] for idx in range(n) if (D_mask >> idx) & 1]
            # choose which to remove based on strategy
            if strategy == 0:
                e = max(D_elems)
            elif strategy == 1:
                e = min(D_elems)
            elif strategy == 2:
                e = max(D_elems, key=lambda x: w[x])
            else:
                e = min(D_elems, key=lambda x: w[x])
            S.remove(e)
        return S
    
    def try_add(S, e):
        # check if S + [e] is dissociated
        T = S + [e]
        ok, _ = subset_sums_distinct(T)
        return ok
    
    # Backtracking search for maximum cardinality dissociated set
    def backtrack_search():
        """
        Backtracking search that tries to include elements in order of decreasing weight.
        Uses the LCM-based integer representation for exact arithmetic.
        """
        # Order elements by decreasing weight (LCM//i) - larger weights first
        # This helps find collisions early
        elements = list(range(1, 23))
        elements.sort(key=lambda x: -w[x])
        
        best_set = []
        best_size = 0
        
        def check_dissociated(sums, new_w):
            """Check if adding new_w to existing sums keeps all sums distinct."""
            # new sums are sums + new_w
            for s in sums:
                if (s + new_w) in sums:
                    return None  # collision
            # also check new sums against each other (but they're all sums+new_w, so distinct if sums are)
            return sums + [s + new_w for s in sums]
        
        def dfs(idx, current_sums, current_set):
            nonlocal best_set, best_size
            if time.time() >= deadline:
                return
            if idx == len(elements):
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = list(current_set)
                return
            
            # Pruning: even if we add all remaining elements, can we beat best?
            remaining = len(elements) - idx
            if len(current_set) + remaining <= best_size:
                return
            
            e = elements[idx]
            we = w[e]
            
            # Try including e
            new_sums = check_dissociated(current_sums, we)
            if new_sums is not None:
                current_set.append(e)
                dfs(idx + 1, new_sums, current_set)
                current_set.pop()
            
            # Try excluding e
            dfs(idx + 1, current_sums, current_set)
        
        dfs(0, [0], [])
        return best_set
    
    best = None
    for strategy in range(4):
        if time.time() >= deadline:
            break
        S = fix_set(list(range(1,23)), strategy)
        # greedily re-add
        removed = [i for i in range(1,23) if i not in S]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for e in removed:
                if e in S:
                    continue
                if try_add(S, e):
                    S.append(e)
                    removed.remove(e)
                    changed = True
        if best is None or len(S) > len(best):
            best = S
        # also try to improve by swapping? skip for now
    
    # Try backtracking search if we have time
    if time.time() < deadline - 5:  # leave some margin
        bt_result = backtrack_search()
        if bt_result and (best is None or len(bt_result) > len(best)):
            best = bt_result
    
    if best is None:
        best = list(range(1,23))
    # final verification
    ok, _ = subset_sums_distinct(best)
    if not ok:
        # fallback to seed
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    return best
