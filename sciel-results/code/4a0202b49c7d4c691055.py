import time
import numpy as np
import random

def edge_index(i, j, n):
    # i < j
    # index = i*n - i*(i+1)/2 + (j-i-1)
    # Simpler:
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def get_edge(u, v, n):
    if u > v:
        u, v = v, u
    return edge_index(u, v, n)

def precompute_contexts(n):
    contexts = [None] * (n * (n - 1) // 2)
    for u in range(n):
        for v in range(u + 1, n):
            idx = edge_index(u, v, n)
            ctx_list = []
            # Pairs P1={u, u'}, P2={v, v'}, P3={w, w'}
            # u' != u, v
            # v' != u, v, u'
            # w, w' != u, v, u', v'
            for u_prime in range(n):
                if u_prime == u or u_prime == v:
                    continue
                for v_prime in range(n):
                    if v_prime == u or v_prime == v or v_prime == u_prime:
                        continue
                    # Choose w, w' from remaining
                    # Remaining: n - 4 vertices
                    # We need to iterate over pairs {w, w'}
                    # To avoid duplicates in w, w', assume w < w'
                    for w in range(n):
                        if w in (u, v, u_prime, v_prime):
                            continue
                        for w_prime in range(w + 1, n):
                            if w_prime in (u, v, u_prime, v_prime):
                                continue
                            
                            # Edges in this K2,2,2 involving (u,v)
                            # P1={u, u'}, P2={v, v'}, P3={w, w'}
                            # Edges:
                            # (u,v) is the one being flipped.
                            # Other 11:
                            # P1-P2: (u,v'), (u',v), (u',v')
                            # P1-P3: (u,w), (u,w'), (u',w), (u',w')
                            # P2-P3: (v,w), (v,w'), (v',w), (v',w')
                            
                            e1 = get_edge(u, v_prime, n)
                            e2 = get_edge(u_prime, v, n)
                            e3 = get_edge(u_prime, v_prime, n)
                            e4 = get_edge(u, w, n)
                            e5 = get_edge(u, w_prime, n)
                            e6 = get_edge(u_prime, w, n)
                            e7 = get_edge(u_prime, w_prime, n)
                            e8 = get_edge(v, w, n)
                            e9 = get_edge(v, w_prime, n)
                            e10 = get_edge(v_prime, w, n)
                            e11 = get_edge(v_prime, w_prime, n)
                            
                            ctx_list.append((e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11))
            contexts[idx] = ctx_list
    return contexts

def count_violations(colors, n, contexts):
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            idx = edge_index(u, v, n)
            c = colors[idx]
            for ctx in contexts[idx]:
                # Check if all edges in ctx are color c
                # Optimization: break early
                all_c = True
                for e in ctx:
                    if colors[e] != c:
                        all_c = False
                        break
                if all_c:
                    total += 1
    return total

def sa_search(n, time_limit):
    start_time = time.time()
    m = n * (n - 1) // 2
    contexts = precompute_contexts(n)
    
    colors = np.random.randint(0, 2, size=m)
    current_energy = count_violations(colors, n, contexts)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    best_energy = current_energy
    best_colors = colors.copy()
    
    while time.time() - start_time < time_limit:
        if current_energy == 0:
            return colors
        
        # Pick random edge
        idx = np.random.randint(0, m)
        
        # Calculate Delta E
        c = colors[idx]
        new_c = 1 - c
        delta = 0
        for ctx in contexts[idx]:
            # Check if all edges in ctx are color c
            # If yes, flipping removes a violation (delta -= 1)
            # If all edges in ctx are color new_c (1-c), flipping adds a violation (delta += 1)
            
            # We need to know if all are c, or all are 1-c
            # Since we are flipping c to 1-c:
            # If all ctx are c: Before: c==c (match), After: 1-c != c (no match). Violation removed.
            # If all ctx are 1-c: Before: c != 1-c (no match), After: 1-c == 1-c (match). Violation added.
            
            # We can just check against c and 1-c
            all_c = True
            all_1_c = True
            for e in ctx:
                val = colors[e]
                if val != c:
                    all_c = False
                if val != 1 - c:
                    all_1_c = False
                if not all_c and not all_1_c:
                    break
            
            if all_c:
                delta -= 1
            if all_1_c:
                delta += 1
        
        new_energy = current_energy + delta
        
        if new_energy < current_energy or np.random.rand() < np.exp(-delta / T):
            colors[idx] = new_c
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = colors.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_colors

def construct():
    # Try n=30
    try:
        colors_30 = sa_search(30, 60)
        # Verify
        contexts_30 = precompute_contexts(30)
        if count_violations(colors_30, 30, contexts_30) == 0:
            # Convert to list of ints
            return {'n': 30, 'colors': [int(c) for c in colors_30]}
    except Exception:
        pass

    # Try n=29
    try:
        colors_29 = sa_search(29, 60)
        contexts_29 = precompute_contexts(29)
        if count_violations(colors_29, 29, contexts_29) == 0:
            return {'n': 29, 'colors': [int(c) for c in colors_29]}
    except Exception:
        pass

    # Fallback to Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
