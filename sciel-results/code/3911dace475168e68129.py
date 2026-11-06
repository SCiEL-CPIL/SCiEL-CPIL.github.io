import time
import random
import bisect
import numpy as np

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0

    def can_add_descending(S, a):
        """Check if 'a' can be added to S, where all elements in S are > a.
        Condition: No pair b, c in S (b < c) such that a divides (b+c)."""
        if not S:
            return True
        
        residues = [x % a for x in S]
        
        counts = [0] * a
        for r in residues:
            counts[r] += 1
            
        for r in range(a):
            partner = (-r) % a
            if r == partner:
                if counts[r] >= 2:
                    return False
            else:
                if counts[r] > 0 and counts[partner] > 0:
                    return False
        return True

    def can_add_two_descending(S, v, w):
        """Check if v and w can be added to S, where all elements in S are > max(v, w).
        Assumes v < w.
        """
        if not can_add_descending(S, v):
            return False
        if not can_add_descending(S, w):
            return False
        
        # Check v against w and S
        for b in S:
            if (w + b) % v == 0:
                return False
        return True

    def greedy_descending():
        S = []
        for a in range(N, 0, -1):
            if can_add_descending(S, a):
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
        return S

    best_set = []
    best_len = 0
    
    S = greedy_descending()
    best_set = S[:]
    best_len = len(S)
    
    S_set = set(S)
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        
        # 1. Try to add any missing element
        for a in range(1, N + 1):
            if a in S_set:
                continue
            if can_add_descending(S, a):
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
                S_set.add(a)
                best_len = len(S)
                improved = True
                break
        
        if improved:
            continue
            
        # 2. Try Remove 1, Add 2
        missing = [x for x in range(1, N + 1) if x not in S_set]
        
        for u_idx in range(len(S)):
            u = S[u_idx]
            S_new = S[:]
            S_new.pop(u_idx)
            S_new_set = S_set.copy()
            S_new_set.discard(u)
            
            found = False
            for i in range(len(missing)):
                v = missing[i]
                if v in S_new_set: continue
                for j in range(i + 1, len(missing)):
                    w = missing[j]
                    if w in S_new_set: continue
                    
                    if v > w:
                        v, w = w, v
                        
                    if can_add_two_descending(S_new, v, w):
                        pos_v = bisect.bisect_left(S_new, v)
                        S_new.insert(pos_v, v)
                        pos_w = bisect.bisect_left(S_new, w)
                        S_new.insert(pos_w, w)
                        
                        S = S_new
                        S_set = S_new_set.copy()
                        S_set.add(v)
                        S_set.add(w)
                        best_len = len(S)
                        improved = True
                        found = True
                        break
                if found:
                    break
            if improved:
                break
        
        # 3. Random perturbation if stuck
        if not improved:
            if len(S) > 1 and time.time() - start_time < deadline:
                rem_idx = random.randint(0, len(S) - 1)
                a_candidate = random.randint(1, N)
                if a_candidate in S_set:
                    continue
                S_new = S[:]
                S_new.pop(rem_idx)
                if can_add_descending(S_new, a_candidate):
                    pos = bisect.bisect_left(S_new, a_candidate)
                    S_new.insert(pos, a_candidate)
                    S = S_new
                    S_set = set(S)
                    improved = True

    if len(S) > best_len:
        best_set = S[:]
        best_len = len(S)
        
    return best_set
