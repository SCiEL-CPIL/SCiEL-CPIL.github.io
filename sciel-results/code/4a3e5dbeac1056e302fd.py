import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_DEADLINE = 150.0

def _t_idx(a, b, c, n):
    cnt = 0
    for x in range(a):
        cnt += (n - x - 1) * (n - x - 2) // 2
    for y in range(a + 1, b):
        cnt += (n - y - 1)
    cnt += (c - b - 1)
    return cnt

def _build_data(n):
    triples = list(combinations(range(n), 3))
    m = len(triples)
    
    quads = []
    for q in combinations(range(n), 4):
        a, b, c, d = q
        idxs = [_t_idx(a, b, c, n), _t_idx(a, b, d, n),
                _t_idx(a, c, d, n), _t_idx(b, c, d, n)]
        quads.append(idxs)
    
    pents = []
    for p in combinations(range(n), 5):
        idxs = []
        for t in combinations(p, 3):
            idxs.append(_t_idx(t[0], t[1], t[2], n))
        pents.append(idxs)
    
    return triples, m, quads, pents

def _is_valid(colors, quads, pents):
    m = len(colors)
    # Check Red K4
    for qi, idxs in enumerate(quads):
        if all(colors[i] == 0 for i in idxs):
            return False
    # Check Blue K5
    for pi, idxs in enumerate(pents):
        if all(colors[i] == 1 for i in idxs):
            return False
    return True

def _find_valid_cdcl(n, time_budget, seed=0):
    """
    CDCL-style search:
    1. Start with a random or algebraic seed.
    2. Identify violations (Red K4s or Blue K5s).
    3. Pick a violation.
    4. Generate a clause: "At least one edge in this structure must be flipped".
    5. Use a simple SAT-like propagation/backtracking to satisfy clauses.
    """
    triples, m, quads, pents = _build_data(n)
    rng = random.Random(seed)
    
    # Initialize with a biased random coloring
    # Red is "forbidden" in K4, Blue is "forbidden" in K5.
    # We want to avoid all-Red quads and all-Blue pents.
    # A good heuristic: color edges Red if they are part of a "rare" structure, Blue otherwise?
    # Let's try a random start with p_red=0.4
    colors = [rng.random() < 0.4 for _ in range(m)]
    colors = np.array(colors, dtype=np.int8)
    
    t_end = time.time() + time_budget
    
    # Precompute adjacency for faster violation checks?
    # For n<=40, m ~ 9880. Quads ~ 91390, Pents ~ 658008.
    # Checking all is too slow. We need incremental updates.
    
    # Let's use a local search with "clause learning" inspired logic.
    # If we find a Red K4, we must flip at least one edge.
    # If we find a Blue K5, we must flip at least one edge.
    
    # We maintain a set of "active clauses" (violations).
    # We pick a clause, pick an edge in it, flip it.
    # We check if this creates new violations or resolves old ones.
    
    # To speed up, we maintain counts of Red edges in each Quad and Blue edges in each Pent.
    quad_red = np.zeros(len(quads), dtype=np.int8)
    pent_blue = np.zeros(len(pents), dtype=np.int8)
    
    # Initialize counts
    for qi, idxs in enumerate(quads):
        quad_red[qi] = np.sum(colors[idxs] == 0)
    for pi, idxs in enumerate(pents):
        pent_blue[pi] = np.sum(colors[idxs] == 1)
    
    red_k4 = np.sum(quad_red == 4)
    blue_k5 = np.sum(pent_blue == 10)
    
    if red_k4 == 0 and blue_k5 == 0:
        return colors.tolist()
    
    # Build reverse maps: edge -> quads/pents containing it
    tri_quads = [[] for _ in range(m)]
    for qi, idxs in enumerate(quads):
        for i in idxs:
            tri_quads[i].append(qi)
            
    tri_pents = [[] for _ in range(m)]
    for pi, idxs in enumerate(pents):
        for i in idxs:
            tri_pents[i].append(pi)
    
    # Convert to numpy arrays for speed
    tri_quads_np = [np.array(qs, dtype=np.int32) if qs else np.array([], dtype=np.int32) for qs in tri_quads]
    tri_pents_np = [np.array(ps, dtype=np.int32) if ps else np.array([], dtype=np.int32) for ps in tri_pents]
    
    best_colors = colors.copy()
    best_energy = red_k4 + blue_k5
    
    stall = 0
    MAX_STALL = 100
    
    while time.time() < t_end:
        if red_k4 == 0 and blue_k5 == 0:
            return colors.tolist()
        
        # Pick a random violating structure
        # Prioritize structures with more "potential" to be fixed?
        # Or just random.
        
        # Let's pick a random violating Quad or Pent
        # We need a list of violating quads and pents.
        # Recalculating this list every step is O(Q + P) which is ~750k. 
        # For 150s, we can do maybe 1000-5000 steps if each step is fast.
        # But building the list is the bottleneck.
        
        # Optimization: Maintain a set of violating indices.
        # When we flip an edge, update the set.
        
        # For simplicity in this implementation, we'll just scan if n is small,
        # but for n=34, scanning 750k items per step is too slow for many steps.
        
        # Alternative: Randomly sample edges, check if they are in a violation.
        # Or, maintain a list of violating quads/pents and update it incrementally.
        
        # Let's try the incremental update approach.
        
        # We need to track which quads/pents are violating.
        # viol_quads = set()
        # viol_pents = set()
        
        # Since we don't have them initialized, let's initialize them.
        # This is O(Q + P) once.
        
        # Actually, let's just use a simple local search that is very fast per step,
        # but picks moves intelligently.
        
        # Pick a random edge that is part of a violation?
        # We don't have the violation list handy without scanning.
        
        # Let's try a different tactic: 
        # Randomly pick an edge. Flip it. 
        # If it reduces total violations, keep it.
        # This is just hill climbing.
        
        # The "CDCL" idea is to pick a *specific* violation.
        # To do this efficiently, we can maintain a heap or list of violations.
        
        # Given the time constraints and the need for a structural change,
        # let's implement a "Greedy Clause Solving" where we maintain the list of violations.
        
        # This implementation will be a hybrid: 
        # 1. Find all violations (slow, do once or rarely).
        # 2. Pick one.
        # 3. Flip an edge in it.
        # 4. Update counts.
        # 5. Update violation list incrementally.
        
        # To avoid full scans, we'll just do a few full scans per second if needed,
        # or accept that for n=34, a full scan is ~0.1s in Python? 
        # 750k iterations in Python is ~0.05-0.1s. 
        # So we can do ~10-20 full scans per second.
        # That's not enough for a good search.
        
        # We MUST use numpy for the scan.
        
        # Let's vectorize the violation finding.
        # quad_red is an array. viol_quads = np.where(quad_red == 4)[0]
        # This is fast in numpy.
        
        viol_quads = np.where(quad_red == 4)[0]
        viol_pents = np.where(pent_blue == 10)[0]
        
        if len(viol_quads) == 0 and len(viol_pents) == 0:
            return colors.tolist()
        
        # Pick a random violation
        if len(viol_quads) > 0 and (len(viol_pents) == 0 or rng.random() < 0.5):
            # Pick a violating quad
            qi = viol_quads[rng.randint(len(viol_quads))]
            idxs = quads[qi]
            # Pick an edge in this quad to flip
            # Prefer edges that are in few other violations?
            # For now, random edge in the quad
            t = idxs[rng.randint(4)]
            new_color = 1 - colors[t]
            
            # Calculate delta
            # Flipping t affects quad_red for quads containing t
            # and pent_blue for pents containing t
            
            aq = tri_quads_np[t]
            ap = tri_pents_np[t]
            
            # We need to check if this flip is beneficial.
            # It resolves the chosen quad (qi) if we flip to Blue (1).
            # If we flip to Red (0), it worsens it.
            # So if we picked a Red K4, we MUST flip an edge to Blue.
            # So new_color must be 1.
            
            if new_color != 1:
                # If we randomly picked an edge that was already Blue? 
                # In a Red K4, all edges are Red (0). So flipping makes it Blue (1).
                # So new_color is always 1 here.
                pass
            
            # Apply flip
            colors[t] = 1
            
            # Update counts
            if len(aq) > 0:
                quad_red[aq] -= 1 # Red count decreases by 1
            if len(ap) > 0:
                pent_blue[ap] += 1 # Blue count increases by 1
            
            # Check for new violations
            # This is the expensive part.
            # We can just recalculate red_k4 and blue_k5 using numpy sums.
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            
            energy = red_k4 + blue_k5
            if energy < best_energy:
                best_energy = energy
                best_colors = colors.copy()
                stall = 0
            else:
                stall += 1
                # If stall is too high, backtrack or random restart
                if stall > MAX_STALL:
                    # Revert to best
                    colors = best_colors.copy()
                    quad_red = np.zeros(len(quads), dtype=np.int8)
                    pent_blue = np.zeros(len(pents), dtype=np.int8)
                    for qi, idxs in enumerate(quads):
                        quad_red[qi] = np.sum(colors[idxs] == 0)
                    for pi, idxs in enumerate(pents):
                        pent_blue[pi] = np.sum(colors[idxs] == 1)
                    red_k4 = int(np.sum(quad_red == 4))
                    blue_k5 = int(np.sum(pent_blue == 10))
                    stall = 0
                    # Perturb
                    for _ in range(5):
                        t2 = rng.randint(m)
                        colors[t2] = 1 - colors[t2]
                        aq2 = tri_quads_np[t2]
                        ap2 = tri_pents_np[t2]
                        if len(aq2) > 0:
                            quad_red[aq2] += 1 if colors[t2] == 0 else -1
                        if len(ap2) > 0:
                            pent_blue[ap2] += 1 if colors[t2] == 1 else -1
        else:
            # Pick a violating pent
            pi = viol_pents[rng.randint(len(viol_pents))]
            idxs = pents[pi]
            t = idxs[rng.randint(10)]
            
            # Must flip to Red (0)
            colors[t] = 0
            
            aq = tri_quads_np[t]
            ap = tri_pents_np[t]
            
            if len(aq) > 0:
                quad_red[aq] += 1
            if len(ap) > 0:
                pent_blue[ap] -= 1
            
            red_k4 = int(np.sum(quad_red == 4))
            blue_k5 = int(np.sum(pent_blue == 10))
            
            energy = red_k4 + blue_k5
            if energy < best_energy:
                best_energy = energy
                best_colors = colors.copy()
                stall = 0
            else:
                stall += 1
                if stall > MAX_STALL:
                    colors = best_colors.copy()
                    quad_red = np.zeros(len(quads), dtype=np.int8)
                    pent_blue = np.zeros(len(pents), dtype=np.int8)
                    for qi, idxs in enumerate(quads):
                        quad_red[qi] = np.sum(colors[idxs] == 0)
                    for pi, idxs in enumerate(pents):
                        pent_blue[pi] = np.sum(colors[idxs] == 1)
                    red_k4 = int(np.sum(quad_red == 4))
                    blue_k5 = int(np.sum(pent_blue == 10))
                    stall = 0
                    for _ in range(5):
                        t2 = rng.randint(m)
                        colors[t2] = 1 - colors[t2]
                        aq2 = tri_quads_np[t2]
                        ap2 = tri_pents_np[t2]
                        if len(aq2) > 0:
                            quad_red[aq2] += 1 if colors[t2] == 0 else -1
                        if len(ap2) > 0:
                            pent_blue[ap2] += 1 if colors[t2] == 1 else -1

    return best_colors.tolist()

def construct():
    deadline = _START + _DEADLINE - 5.0
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try to find valid colorings for increasing n
    for n in range(10, 35):
        if time.time() >= deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
            
        # Allocate time based on difficulty
        # Larger n is harder
        time_budget = min(remaining * 0.5, 20.0)
        
        colors = _find_valid_cdcl(n, time_budget, seed=42 + n)
        if colors is not None:
            best_n = n
            best_colors = colors
        else:
            # If we fail for n, we might fail for n+1 too, but let's try one more
            # or break if we've spent too much time
            if n > 20:
                # If we can't even get to 20, stop
                break
                
    return {'n': best_n, 'colors': best_colors}
