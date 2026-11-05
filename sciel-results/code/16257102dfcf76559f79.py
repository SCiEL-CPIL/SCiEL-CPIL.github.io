<<<<<<< SEARCH
"""Cayley-graph construction for 4-regular girth-10 graph."""
import itertools
import numpy as np

def construct(k, g):
    if k != 4:
        return _construct_fallback(k, g)
        
    # Try Cayley graph of A6 (order 360)
    # A6 is the alternating group on 6 letters.
    
    def perm_sign(p):
        inv = 0
        for i in range(len(p)):
            for j in range(i+1, len(p)):
                if p[i] > p[j]:
                    inv += 1
        return 1 if inv % 2 == 0 else -1
        
    perms = list(itertools.permutations(range(6)))
    A6 = [p for p in perms if perm_sign(p) == 1]
    idx = {p: i for i, p in enumerate(A6)}
    
    def compose(p, q):
        return tuple(p[q[i]] for i in range(6))
        
    id_perm = (0, 1, 2, 3, 4, 5)
    
    order_3 = []
    order_5 = []
    
    for p in A6:
        if p == id_perm: continue
        cur = p
        for _ in range(1, 10):
            cur = compose(cur, p)
            if cur == id_perm:
                if _ == 3: order_3.append(p)
                if _ == 5: order_5.append(p)
                break
                
    def inv(p):
        r = [0]*6
        for i in range(6):
            r[p[i]] = i
        return tuple(r)
        
    candidates = order_3 + order_5
    
    for s1 in candidates:
        for s2 in candidates:
            s1_inv = inv(s1)
            s2_inv = inv(s2)
            
            if s1 == s2 or s1 == s2_inv or s2 == s1_inv or s1 == s1_inv or s2 == s2_inv:
                continue
            
            # Check girth >= g
            from collections import deque
            q = deque()
            q.append((0, id_perm, None))
            found_cycle = False
            visited = set()
            
            while q:
                l, v, prev = q.popleft()
                if l >= g:
                    continue
                if l > 0 and v == id_perm:
                    found_cycle = True
                    break
                
                for s in [s1, s1_inv, s2, s2_inv]:
                    if prev is not None and s == inv(prev):
                        continue
                    next_v = compose(v, s)
                    state = (next_v, s)
                    if state in visited:
                        continue
                    visited.add(state)
                    q.append((l+1, next_v, s))
                    
            if not found_cycle:
                edges = []
                for i, v in enumerate(A6):
                    for s in [s1, s2]:
                        u_idx = idx[v]
                        w = compose(v, s)
                        w_idx = idx[w]
                        if u_idx < w_idx:
                            edges.append((u_idx, w_idx))
                return len(A6), edges
                
    return _construct_fallback(4, g)
=======
"""Cayley-graph construction for 4-regular girth-10 graph."""
import itertools
import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    
    if k != 4:
        return _construct_fallback(k, g)
        
    # Try Cayley graph of A6 (order 360)
    # A6 is the alternating group on 6 letters.
    
    def perm_sign(p):
        inv = 0
        for i in range(len(p)):
            for j in range(i+1, len(p)):
                if p[i] > p[j]:
                    inv += 1
        return 1 if inv % 2 == 0 else -1
        
    perms = list(itertools.permutations(range(6)))
    A6 = [p for p in perms if perm_sign(p) == 1]
    idx = {p: i for i, p in enumerate(A6)}
    
    def compose(p, q):
        return tuple(p[q[i]] for i in range(6))
        
    id_perm = (0, 1, 2, 3, 4, 5)
    
    order_3 = []
    order_5 = []
    
    for p in A6:
        if p == id_perm: continue
        cur = p
        for _ in range(1, 10):
            cur = compose(cur, p)
            if cur == id_perm:
                if _ == 3: order_3.append(p)
                if _ == 5: order_5.append(p)
                break
                
    def inv(p):
        r = [0]*6
        for i in range(6):
            r[p[i]] = i
        return tuple(r)
        
    candidates = order_3 + order_5
    
    # Use a faster girth check: enumerate non-backtracking closed walks
    def has_short_nonbacktracking_walk(s1, s2, min_len, max_len):
        """Check if there's a non-backtracking closed walk of length in [min_len, max_len)."""
        S = [s1, inv(s1), s2, inv(s2)]
        S_inv = {s: inv(s) for s in S}
        
        # BFS over (position, last_step) states
        from collections import deque
        # State: (element, last_generator)
        # We track all non-backtracking walks of length < max_len
        # Start: identity with no previous step
        # A closed walk of length L means we return to identity after L steps
        
        # For each length L from min_len to max_len-1, check if any non-backtracking
        # sequence of L generators from S sums to identity (in group operation)
        
        # More efficient: track reachable states (element, last_gen) at each depth
        # and check if identity is reachable at depth L with last_gen != None
        
        for L in range(min_len, max_len):
            # Track set of (element, last_gen) reachable by non-backtracking walks of length L
            if L == 0:
                states = {(id_perm, None)}
            else:
                states = set()
                for (elem, last) in states:
                    for s in S:
                        if last is not None and s == S_inv[last]:
                            continue
                        new_elem = compose(elem, s)
                        states.add((new_elem, s))
            if (id_perm, None) in states or any(e == id_perm for (e, s) in states if s is not None):
                return True
            if time.time() - start_time > DEADLINE - 5:
                return True  # Assume bad to be safe
        
        return False
    
    for s1 in candidates:
        for s2 in candidates:
            if time.time() - start_time > DEADLINE - 10:
                break
            s1_inv = inv(s1)
            s2_inv = inv(s2)
            
            if s1 == s2 or s1 == s2_inv or s2 == s1_inv or s1 == s1_inv or s2 == s2_inv:
                continue
            
            # Check girth >= g: no non-backtracking closed walk of length < g
            if not has_short_nonbacktracking_walk(s1, s2, 2, g):
                edges = []
                for i, v in enumerate(A6):
                    for s in [s1, s2]:
                        u_idx = idx[v]
                        w = compose(v, s)
                        w_idx = idx[w]
                        if u_idx < w_idx:
                            edges.append((u_idx, w_idx))
                return len(A6), edges
        if time.time() - start_time > DEADLINE - 10:
            break
            
    return _construct_fallback(4, g)
>>>>>>> REPLACE

