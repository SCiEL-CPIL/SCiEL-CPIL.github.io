"""Hypergraph Ramsey R(4,5;3): maximise n for a 2-colouring of triples with no red K4^(3) and no blue K5^(3)."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import time

def _triples(n):
    """Generate all triples (a,b,c) with a<b<c in lexicographic order."""
    return list(itertools.combinations(range(n), 3))

def _triples_to_idx(triples):
    """Map triple tuple to index."""
    return {t: i for i, t in enumerate(triples)}

def _build_quads(triples, tidx):
    """For each quad (4 vertices), return list of 4 triple indices that form it."""
    n = max(max(t) for t in triples) + 1 if triples else 0
    quads = []
    quad_triples = []
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            for c in range(b + 1, n - 1):
                for d in range(c + 1, n):
                    tlist = [
                        tidx[(a, b, c)],
                        tidx[(a, b, d)],
                        tidx[(a, c, d)],
                        tidx[(b, c, d)]
                    ]
                    quad_triples.append(tlist)
    return np.array(quad_triples, dtype=np.int32)

def _build_pents(triples, tidx):
    """For each pent (5 vertices), return list of 10 triple indices that form it."""
    n = max(max(t) for t in triples) + 1 if triples else 0
    pent_triples = []
    for a in range(n - 4):
        for b in range(a + 1, n - 3):
            for c in range(b + 1, n - 2):
                for d in range(c + 1, n - 1):
                    for e in range(d + 1, n):
                        verts = [a, b, c, d, e]
                        tlist = [tidx[t] for t in itertools.combinations(verts, 3)]
                        pent_triples.append(tlist)
    return np.array(pent_triples, dtype=np.int32)

def _count_violations(colors, quads, pents):
    """Count red K4 violations (all 4 triples red=0) and blue K5 violations (all 10 triples blue=1)."""
    colors_arr = np.array(colors, dtype=np.int8)
    # Red K4: all 4 triples are 0 (red)
    red_viol = np.sum(np.all(colors_arr[quads] == 0, axis=1))
    # Blue K5: all 10 triples are 1 (blue)
    blue_viol = np.sum(np.all(colors_arr[pents] == 1, axis=1))
    return red_viol + blue_viol, red_viol, blue_viol

def _local_search(n, colors, quads, pents, tidx, triples, max_time=100.0, seed=42):
    """Local search: flip triple colors to reduce violations. Returns best (n, colors)."""
    random.seed(seed)
    colors_arr = np.array(colors, dtype=np.int8)
    best_viol, best_red, best_blue = _count_violations(colors_arr, quads, pents)
    best_colors = colors_arr.copy()
    
    start = time.time()
    deadline = start + max_time
    
    # Precompute for each triple which quads and pents it belongs to
    tri_quads = [[] for _ in range(len(triples))]
    for qi, qt in enumerate(quads):
        for t in qt:
            tri_quads[t].append(qi)
    
    tri_pents = [[] for _ in range(len(triples))]
    for pi, pt in enumerate(pents):
        for t in pt:
            tri_pents[t].append(pi)
    
    # Convert to numpy arrays for speed
    tri_quads_arr = [np.array(q, dtype=np.int32) for q in tri_quads]
    tri_pents_arr = [np.array(p, dtype=np.int32) for p in tri_pents]
    
    # Build mapping: for each quad, the 4 triple indices; for each pent, the 10 triple indices
    # For delta computation when flipping triple t:
    # - For each quad containing t: check if other 3 are all red (0) -> flipping t from red to blue removes a violation
    #   or if other 3 are all blue (1) -> flipping t from blue to red adds a violation
    # - Similarly for pents
    
    current_colors = colors_arr.copy()
    current_viol, current_red, current_blue = best_viol, best_red, best_blue
    
    # Precompute: for each quad, which triples; for delta we need to know the state of the other 3
    # Let's use a smarter approach: maintain quad_red_count and pent_blue_count
    
    quad_red = np.zeros(len(quads), dtype=np.int32)  # number of red (0) triples in each quad
    pent_blue = np.zeros(len(pents), dtype=np.int32)  # number of blue (1) triples in each pent
    
    for qi in range(len(quads)):
        for t in quads[qi]:
            if current_colors[t] == 0:
                quad_red[qi] += 1
    
    for pi in range(len(pents)):
        for t in pents[pi]:
            if current_colors[t] == 1:
                pent_blue[pi] += 1
    
    # A quad is a red violation if quad_red[qi] == 4
    # A pent is a blue violation if pent_blue[pi] == 10
    
    def _delta_violation(t, new_color):
        """Compute change in total violations if we set triple t to new_color."""
        delta = 0
        old_color = current_colors[t]
        if old_color == new_color:
            return 0
        
        # For quads containing t
        for qi in tri_quads[t]:
            old_qr = quad_red[qi]
            if old_color == 0:  # was red
                new_qr = old_qr - 1
            else:  # was blue
                new_qr = old_qr + 1
            # Before: violation if old_qr == 4
            # After: violation if new_qr == 4
            before_v = 1 if old_qr == 4 else 0
            after_v = 1 if new_qr == 4 else 0
            delta += (after_v - before_v)
        
        # For pents containing t
        for pi in tri_pents[t]:
            old_pb = pent_blue[pi]
            if old_color == 1:  # was blue
                new_pb = old_pb - 1
            else:  # was red
                new_pb = old_pb + 1
            before_v = 1 if old_pb == 10 else 0
            after_v = 1 if new_pb == 10 else 0
            delta += (after_v - before_v)
        
        return delta
    
    # Random restarts with local search
    best_result = (n, best_colors.tolist())
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Start from a random valid-ish coloring
        if restart == 0:
            cur = colors_arr.copy()
        else:
            cur = np.random.randint(0, 2, size=len(triples), dtype=np.int8)
        
        # Recompute quad_red and pent_blue
        quad_red[:] = 0
        pent_blue[:] = 0
        for qi in range(len(quads)):
            for t in quads[qi]:
                if cur[t] == 0:
                    quad_red[qi] += 1
        for pi in range(len(pents)):
            for t in pents[pi]:
                if cur[t] == 1:
                    pent_blue[pi] += 1
        
        cur_viol = np.sum(quad_red == 4) + np.sum(pent_blue == 10)
        cur_red = np.sum(quad_red == 4)
        cur_blue = np.sum(pent_blue == 10)
        
        # Local search
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try flipping random triples
            indices = list(range(len(triples)))
            random.shuffle(indices)
            for t in indices:
                if time.time() > deadline - 2:
                    break
                old_color = cur[t]
                new_color = 1 - old_color
                d = _delta_violation(t, new_color)
                if d < 0:
                    # Apply flip
                    cur[t] = new_color
                    for qi in tri_quads[t]:
                        if old_color == 0:
                            quad_red[qi] -= 1
                        else:
                            quad_red[qi] += 1
                    for pi in tri_pents[t]:
                        if old_color == 1:
                            pent_blue[pi] -= 1
                        else:
                            pent_blue[pi] += 1
                    cur_viol += d
                    improved = True
                    if cur_viol < best_viol:
                        best_viol = cur_viol
                        best_colors = cur.copy()
        
        # Track best
        if cur_viol < best_viol:
            best_viol = cur_viol
            best_colors = cur.copy()
        
        # Check if we have a valid coloring
        if best_viol == 0:
            # Try to increase n
            break
    
    # Now try to grow n from the best valid coloring
    # Actually, let's try different n values
    
    return best_result

def _random_valid(n, max_time=50.0, seed=0):
    """Try to find a valid coloring for n vertices."""
    triples = _triples(n)
    tidx = _triples_to_idx(triples)
    quads = _build_quads(triples, tidx)
    pents = _build_pents(triples, tidx)
    
    best_colors = None
    best_viol = float('inf')
    
    start = time.time()
    deadline = start + max_time
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        colors = np.random.randint(0, 2, size=len(triples), dtype=np.int8)
        viol, rv, bv = _count_violations(colors, quads, pents)
        if viol < best_viol:
            best_viol = viol
            best_colors = colors.copy()
            if viol == 0:
                return colors.tolist()
    
    # Local search from best
    if best_colors is not None:
        result = _local_search(n, best_colors.tolist(), quads, pents, tidx, triples, max_time=max(10, deadline - time.time()), seed=seed)
        # result is (n, colors_list) but we just need colors
        # Actually _local_search returns best_result which is (n, colors)
        # Let's just do a quick local search inline
        pass
    
    return None

def construct():
    """Construct a valid 2-coloring of triples maximizing n."""
    start_time = time.time()
    total_time = 145.0  # safety margin from 150s limit
    
    # Try to find valid colorings for increasing n
    # Published construction has n=34
    # Start with a reasonable base and grow
    
    best_n = 1
    best_colors = None
    
    # Try n values from high to low, but start with a heuristic
    # The seed uses sum mod 3 which gives n=9
    # Let's try to find larger valid colorings
    
    # First, try the modular coloring for various n
    for n_test in range(34, 8, -1):
        if time.time() - start_time > total_time - 30:
            break
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n_test), 3)]
        triples = _triples(n_test)
        tidx = _triples_to_idx(triples)
        quads = _build_quads(triples, tidx)
        pents = _build_pents(triples, tidx)
        viol, rv, bv = _count_violations(colors, quads, pents)
        if viol == 0:
            best_n = n_test
            best_colors = colors
            break
    
    # If modular didn't work for large n, try local search
    if best_n < 34:
        for n_test in range(34, best_n, -1):
            if time.time() - start_time > total_time - 20:
                break
            
            triples = _triples(n_test)
            tidx = _triples_to_idx(triples)
            quads = _build_quads(triples, tidx)
            pents = _build_pents(triples, tidx)
            
            # Try multiple random starts with local search
            for seed in range(10):
                if time.time() - start_time > total_time - 10:
                    break
                
                colors = np.random.randint(0, 2, size=len(triples), dtype=np.int8)
                viol, rv, bv = _count_violations(colors, quads, pents)
                
                if viol < 100:  # Only do local search if reasonably close
                    result = _local_search(n_test, colors.tolist(), quads, pents, tidx, triples, 
                                           max_time=min(15, total_time - (time.time() - start_time) - 5), seed=seed)
                    # Check if we got a valid coloring
                    r_n, r_colors = result
                    triples_check = _triples(r_n)
                    tidx_check = _triples_to_idx(triples_check)
                    quads_check = _build_quads(triples_check, tidx_check)
                    pents_check = _build_pents(triples_check, tidx_check)
                    v, _, _ = _count_violations(r_colors, quads_check, pents_check)
                    if v == 0:
                        best_n = r_n
                        best_colors = r_colors
                        break
            
            if best_n >= n_test:
                break
    
    # Final safety: if we still have a small n, use the modular coloring for n=9
    if best_colors is None or best_n < 9:
        n_final = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n_final), 3)]
    else:
        n_final = best_n
    
    return {'n': n_final, 'colors': best_colors}
# EVOLVE-BLOCK-END
