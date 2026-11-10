# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 16
_EDGES = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_EDGE_INDEX = {(i, j): k for k, (i, j) in enumerate(_EDGES)}
_QUADS = []
for a in range(_N):
    for b in range(a + 1, _N):
        for c in range(b + 1, _N):
            for d in range(c + 1, _N):
                _QUADS.append((a, b, c, d))

# Precompute edge indices for each quad
_QUAD_EDGE_IDX = []
for q in _QUADS:
    a, b, c, d = q
    _QUAD_EDGE_IDX.append([
        _EDGE_INDEX[(a, b)], _EDGE_INDEX[(a, c)], _EDGE_INDEX[(a, d)],
        _EDGE_INDEX[(b, c)], _EDGE_INDEX[(b, d)], _EDGE_INDEX[(c, d)]
    ])

# Precompute, for each edge, the list of quads containing it.
_EDGE_QUADS = [[] for _ in range(len(_EDGES))]
for qi, q in enumerate(_QUADS):
    for e in _QUAD_EDGE_IDX[qi]:
        _EDGE_QUADS[e].append(qi)

def _base_colours():
    return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]

def _count_violations(colours):
    colours = np.asarray(colours)
    bad = 0
    for eidx in _QUAD_EDGE_IDX:
        cs = colours[eidx]
        if len(set(cs.tolist())) < 5:
            bad += 1
    return bad

def construct():
    deadline = time.time() + 140.0
    rng = random.Random(42)
    n_edges = len(_EDGES)
    n_quads = len(_QUADS)
    
    base = _base_colours()
    best = list(base)
    best_viol = _count_violations(base)
    best_pal = len(set(base))
    best_key = (best_viol, best_pal)
    
    def palette_size(c):
        return len(set(c))
    
    # Efficient violation counting using numpy
    def compute_violations(colours_arr):
        """Returns (viol_count, quad_distinct array)"""
        quad_dist = np.zeros(n_quads, dtype=np.int8)
        for qi, eidx in enumerate(_QUAD_EDGE_IDX):
            cs = colours_arr[eidx]
            quad_dist[qi] = len(set(cs.tolist()))
        viol = int(np.sum(quad_dist < 5))
        return viol, quad_dist
    
    # Try to find valid colourings with minimal palette
    # Start with the known valid 17-colour base
    for target_T in (15, 14, 13):
        if time.time() > deadline - 10.0:
            break
        
        # Project base colours into [0, target_T)
        c = np.array([x % target_T for x in base], dtype=np.int32)
        viol, quad_dist = compute_violations(c)
        
        # Simulated annealing to fix violations
        T_curr = 100.0
        T_min = 0.1
        cooling = 0.999
        steps = 0
        max_steps = 500000
        stuck = 0
        
        while viol > 0 and steps < max_steps and time.time() < deadline - 10.0:
            steps += 1
            e = rng.randrange(n_edges)
            newc = rng.randrange(target_T)
            oldc = c[e]
            if oldc == newc:
                continue
            
            # Compute delta violation
            old_viol = viol
            old_qd = quad_dist.copy()
            
            c[e] = newc
            for qi in _EDGE_QUADS[e]:
                eidx = _QUAD_EDGE_IDX[qi]
                cs = c[eidx]
                quad_dist[qi] = len(set(cs.tolist()))
            new_viol = int(np.sum(quad_dist < 5))
            
            delta = new_viol - old_viol
            
            if delta <= 0:
                if delta < 0:
                    stuck = 0
                viol = new_viol
            else:
                # Accept with probability exp(-delta/T_curr)
                if rng.random() < np.exp(-delta / max(T_curr, 0.01)):
                    viol = new_viol
                else:
                    # Revert
                    c[e] = oldc
                    quad_dist = old_qd
                    viol = old_viol
                stuck += 1
            
            T_curr *= cooling
            if stuck > 5000:
                # Restart with fresh projection
                c = np.array([x % target_T for x in base], dtype=np.int32)
                viol, quad_dist = compute_violations(c)
                T_curr = 100.0
                stuck = 0
                steps = 0
        
        if viol == 0:
            c_list = c.tolist()
            pal = palette_size(c_list)
            key = (0, pal)
            if key < best_key:
                best = c_list
                best_key = key
                best_pal = pal
                best_viol = 0
            continue
        else:
            # Couldn't fix for this T; smaller T is harder, break
            break
    
    # If we found a valid solution, return it
    if best_viol == 0:
        return best
    
    # Fallback: try direct local search on base to reduce palette
    # while maintaining validity
    c = np.array(base, dtype=np.int32)
    viol, quad_dist = compute_violations(c)
    pal = palette_size(c.tolist())
    
    # Try to reduce palette by remapping colours
    for T in (15, 14, 13):
        if time.time() > deadline - 5.0:
            break
        
        c2 = np.array([x % T for x in base], dtype=np.int32)
        viol2, quad_dist2 = compute_violations(c2)
        
        # Greedy fix: repeatedly pick a violating quad and change one edge
        it = 0
        while viol2 > 0 and it < 100000 and time.time() < deadline - 5.0:
            it += 1
            # Find a violating quad
            viol_qis = [qi for qi in range(n_quads) if quad_dist2[qi] < 5]
            if not viol_qis:
                break
            qi = rng.choice(viol_qis)
            eidx = _QUAD_EDGE_IDX[qi]
            e = rng.choice(eidx)
            newc = rng.randrange(T)
            oldc = c2[e]
            if oldc == newc:
                continue
            
            old_viol = viol2
            old_qd = quad_dist2.copy()
            c2[e] = newc
            for qi2 in _EDGE_QUADS[e]:
                eidx2 = _QUAD_EDGE_IDX[qi2]
                cs = c2[eidx2]
                quad_dist2[qi2] = len(set(cs.tolist()))
            new_viol = int(np.sum(quad_dist2 < 5))
            
            if new_viol < old_viol:
                viol2 = new_viol
            else:
                c2[e] = oldc
                quad_dist2 = old_qd
                viol2 = old_viol
        
        if viol2 == 0:
            c2_list = c2.tolist()
            pal2 = palette_size(c2_list)
            if pal2 < pal:
                best = c2_list
                pal = pal2
                best_key = (0, pal2)
                best_viol = 0
                best_pal = pal2
            break
    
    if best_viol == 0:
        return best
    
    return base
# EVOLVE-BLOCK-END
