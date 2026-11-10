import numpy as np
import time
import random
from math import gcd
from itertools import product

def normalize_line(A, B, C):
    """Normalize line coefficients to ensure uniqueness and integer validity."""
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g
    B //= g
    C //= g
    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C
    return (A, B, C)

def count_incidences(points, lines):
    """Count incidences: (point, line) pairs where A*x + B*y + C == 0."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    vals = (P[:,0,None]*L[None,:,0] + P[:,1,None]*L[None,:,1] + L[None,:,2])
    return int(np.sum(vals == 0))

def build_lines_from_families(v_params, h_params, d_params, d2_params=None):
    """
    Build 27 lines from families:
    - v_params: list of x-coordinates for vertical lines x = v (A=1,B=0,C=-v)
    - h_params: list of y-coordinates for horizontal lines y = h (A=0,B=1,C=-h)  
    - d_params: list of c values for diagonal lines x - y = c (A=1,B=-1,C=-c)
    - d2_params: list of c values for anti-diagonal lines x + y = c (A=1,B=1,C=-c)
    Returns (lines_array, family_assignments) where family_assignments[i] is 0=V,1=H,2=D,3=D2
    """
    lines = []
    fams = []
    for v in v_params:
        lines.append((1, 0, -v))
        fams.append(0)
    for h in h_params:
        lines.append((0, 1, -h))
        fams.append(1)
    for d in d_params:
        lines.append((1, -1, -d))
        fams.append(2)
    if d2_params:
        for d2 in d2_params:
            lines.append((1, 1, -d2))
            fams.append(3)
    # Pad if needed
    while len(lines) < 27:
        lines.append((1, 0, -len(lines)))
        fams.append(0)
    return np.array(lines[:27], dtype=np.int64), fams[:27]

def find_intersection(x1, y1, x2, y2):
    """Find integer intersection of line through (x1,y1) and line through (x2,y2).
    Lines: x-y=c1 and x-y=c2 have no intersection (parallel).
    For general: solve the 2x2 system."""
    # This is for specific families; use direct computation instead
    pass

def compute_incidence_grid(v_params, h_params, d_params, d2_params=None):
    """
    For given line families, compute incidence counts at all grid points.
    A point (x,y) has incidence:
      +1 for each v in v_params where x == v
      +1 for each h in h_params where y == h
      +1 for each d in d_params where x - y == d
      +1 for each d2 in d2_params where x + y == d2
    """
    # Determine bounding box
    all_x = list(v_params) + [d + h for d in d_params for h in h_params]
    all_y = list(h_params) + [v - d for v in v_params for d in d_params]
    if d2_params:
        all_x.extend([d2 - h for d2 in d2_params for h in h_params])
        all_y.extend([v - d2 for v in v_params for d2 in d2_params])
    
    if not all_x or not all_y:
        return {}, 0
    
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    
    # Use sets for fast lookup
    v_set = set(v_params)
    h_set = set(h_params)
    d_set = set(d_params)
    d2_set = set(d2_params) if d2_params else set()
    
    # Compute incidences at candidate points (intersections)
    # Key insight: high-incidence points are at intersections of lines from different families
    # V-H intersection: (v, h) -> incidence = 1+1 + (1 if v-h in d_set) + (1 if v+h in d2_set)
    # V-D intersection: (v, v-d) for d in d_params -> incidence = 1+1 + (1 if v-(v-d)=d in d_set which is true) + ...
    # Actually let's just enumerate all candidate points efficiently
    
    candidates = set()
    # V-H intersections
    for v in v_params:
        for h in h_params:
            candidates.add((v, h))
    # V-D intersections: x=v, x-y=d => y=v-d
    for v in v_params:
        for d in d_params:
            candidates.add((v, v-d))
    # H-D intersections: y=h, x-y=d => x=h+d
    for h in h_params:
        for d in d_params:
            candidates.add((h+d, h))
    if d2_params:
        # V-D2 intersections: x=v, x+y=d2 => y=d2-v
        for v in v_params:
            for d2 in d2_params:
                candidates.add((v, d2-v))
        # H-D2 intersections: y=h, x+y=d2 => x=d2-h
        for h in h_params:
            for d2 in d2_params:
                candidates.add((d2-h, h))
        # D-D2 intersections: x-y=d, x+y=d2 => x=(d+d2)//2, y=(d2-d)//2 (if even)
        for d in d_params:
            for d2 in d2_params:
                if (d + d2) % 2 == 0 and (d2 - d) % 2 == 0:
                    x = (d + d2) // 2
                    y = (d2 - d) // 2
                    candidates.add((x, y))
    
    # Compute incidence for each candidate
    incidences = {}
    for (x, y) in candidates:
        inc = 0
        if x in v_set:
            inc += 1
        if y in h_set:
            inc += 1
        if (x - y) in d_set:
            inc += 1
        if d2_params and (x + y) in d2_set:
            inc += 1
        if inc > 0:
            incidences[(x, y)] = inc
    
    return incidences, len(incidences)

def greedy_select_points_from_incidence(incidences, k=54):
    """Select k points with highest incidence from the incidence dict."""
    if not incidences:
        return []
    sorted_items = sorted(incidences.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    return [item[0] for item in sorted_items[:k]]

def optimize_families(v_range, h_range, d_range, d2_range=None, deadline=None):
    """
    Try different parameter sets for the 4 line families.
    v_range: (min, max, count) for vertical line parameters
    Returns best (lines, points, I).
    """
    best_I = -1
    best_lines = None
    best_pts = None
    
    # Try various configurations
    configs = []
    
    # Symmetric configurations
    for offset in range(-5, 6):
        v_params = [offset + i for i in range(9)]
        h_params = [offset + i for i in range(9)]
        d_params = [offset + i for i in range(9)]
        configs.append((v_params, h_params, d_params, None))
    
    # With anti-diagonals
    for offset in range(-5, 6):
        v_params = [offset + i for i in range(7)]
        h_params = [offset + i for i in range(7)]
        d_params = [offset + i for i in range(7)]
        d2_params = [offset + i for i in range(6)]
        configs.append((v_params, h_params, d_params, d2_params))
    
    # Asymmetric
    for vo in range(-3, 4):
        for ho in range(-3, 4):
            v_params = [vo + i for i in range(9)]
            h_params = [ho + i for i in range(9)]
            d_params = [0 + i for i in range(9)]
            configs.append((v_params, h_params, d_params, None))
    
    # Random perturbations
    random.seed(42)
    for _ in range(200):
        v_params = sorted(random.sample(range(-10, 11), 9))
        h_params = sorted(random.sample(range(-10, 11), 9))
        d_params = sorted(random.sample(range(-10, 11), 9))
        configs.append((v_params, h_params, d_params, None))
    
    for v_params, h_params, d_params, d2_params in configs:
        if deadline and time.time() > deadline:
            break
        incidences, _ = compute_incidence_grid(v_params, h_params, d_params, d2_params)
        if not incidences:
            continue
        pts_list = greedy_select_points_from_incidence(incidences, 54)
        if len(pts_list) < 54:
            # Pad with nearby points
            existing = set(pts_list)
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if len(pts_list) >= 54:
                        break
                    p = (dx, dy)
                    if p not in existing:
                        pts_list.append(p)
                        existing.add(p)
                if len(pts_list) >= 54:
                    break
        
        pts = np.array(pts_list[:54], dtype=np.int64)
        lines, _ = build_lines_from_families(v_params, h_params, d_params, d2_params)
        I = count_incidences(pts, lines)
        
        if I > best_I:
            best_I = I
            best_lines = lines.copy()
            best_pts = pts.copy()
    
    return best_lines, best_pts, best_I

def local_search_families(v_params, h_params, d_params, d2_params, deadline):
    """
    Local search: perturb one parameter at a time in the family representation.
    """
    best_I = -1
    best_lines = None
    best_pts = None
    
    def evaluate(vp, hp, dp, dp2):
        incidences, _ = compute_incidence_grid(vp, hp, dp, dp2)
        if not incidences:
            return None, None, 0
        pts_list = greedy_select_points_from_incidence(incidences, 54)
        if len(pts_list) < 54:
            existing = set(pts_list)
            for dx in range(-15, 16):
                for dy in range(-15, 16):
                    if len(pts_list) >= 54:
                        break
                    p = (dx, dy)
                    if p not in existing:
                        pts_list.append(p)
                        existing.add(p)
                if len(pts_list) >= 54:
                    break
        pts = np.array(pts_list[:54], dtype=np.int64)
        lines, _ = build_lines_from_families(vp, hp, dp, dp2)
        I = count_incidences(pts, lines)
        return lines, pts, I
    
    # Initial evaluation
    bl, bp, bi = evaluate(v_params, h_params, d_params, d2_params)
    if bl is not None:
        best_I = bi
        best_lines = bl
        best_pts = bp
    
    # Local search
    while time.time() < deadline:
        improved = False
        # Try perturbing each parameter
        for param_idx in range(len(v_params)):
            for delta in [-2, -1, 1, 2]:
                if time.time() > deadline:
                    break
                new_vp = list(v_params)
                new_vp[param_idx] += delta
                new_vp = sorted(new_vp)
                # Check for duplicates
                if len(set(new_vp)) != len(new_vp):
                    continue
                bl2, bp2, bi2 = evaluate(new_vp, h_params, d_params, d2_params)
                if bl2 is not None and bi2 > best_I:
                    best_I = bi2
                    best_lines = bl2
                    best_pts = bp2
                    v_params = new_vp
                    improved = True
        for param_idx in range(len(h_params)):
            for delta in [-2, -1, 1, 2]:
                if time.time() > deadline:
                    break
                new_hp = list(h_params)
                new_hp[param_idx] += delta
                new_hp = sorted(new_hp)
                if len(set(new_hp)) != len(new_hp):
                    continue
                bl2, bp2, bi2 = evaluate(v_params, new_hp, d_params, d2_params)
                if bl2 is not None and bi2 > best_I:
                    best_I = bi2
                    best_lines = bl2
                    best_pts = bp2
                    h_params = new_hp
                    improved = True
        for param_idx in range(len(d_params)):
            for delta in [-2, -1, 1, 2]:
                if time.time() > deadline:
                    break
                new_dp = list(d_params)
                new_dp[param_idx] += delta
                new_dp = sorted(new_dp)
                if len(set(new_dp)) != len(new_dp):
                    continue
                bl2, bp2, bi2 = evaluate(v_params, h_params, new_dp, d2_params)
                if bl2 is not None and bi2 > best_I:
                    best_I = bi2
                    best_lines = bl2
                    best_pts = bp2
                    d_params = new_dp
                    improved = True
        if d2_params:
            for param_idx in range(len(d2_params)):
                for delta in [-2, -1, 1, 2]:
                    if time.time() > deadline:
                        break
                    new_dp2 = list(d2_params)
                    new_dp2[param_idx] += delta
                    new_dp2 = sorted(new_dp2)
                    if len(set(new_dp2)) != len(new_dp2):
                        continue
                    bl2, bp2, bi2 = evaluate(v_params, h_params, d_params, new_dp2)
                    if bl2 is not None and bi2 > best_I:
                        best_I = bi2
                        best_lines = bl2
                        best_pts = bp2
                        d2_params = new_dp2
                        improved = True
        if not improved:
            break
    
    return best_lines, best_pts, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 590  # Safety margin
    
    best_pts_global = None
    best_lines_global = None
    best_I_global = -1
    
    # Strategy 1: Combinatorial search over family parameters
    bl, bp, bi = optimize_families(None, None, None, None, deadline)
    if bl is not None:
        best_I_global = bi
        best_lines_global = bl
        best_pts_global = bp
    
    # Strategy 2: Local search from best found
    if best_I_global > 0 and time.time() < deadline - 60:
        # Extract current parameters (approximate)
        # Just do local search from a few good starting points
        for start_offset in range(-3, 4):
            if time.time() > deadline - 30:
                break
            v_params = [start_offset + i for i in range(9)]
            h_params = [start_offset + i for i in range(9)]
            d_params = [start_offset + i for i in range(9)]
            bl2, bp2, bi2 = local_search_families(v_params, h_params, d_params, None, deadline - 30)
            if bl2 is not None and bi2 > best_I_global:
                best_I_global = bi2
                best_lines_global = bl2
                best_pts_global = bp2
    
    # Strategy 3: Try with anti-diagonals
    if time.time() < deadline - 30:
        for start_offset in range(-3, 4):
            if time.time() > deadline - 15:
                break
            v_params = [start_offset + i for i in range(7)]
            h_params = [start_offset + i for i in range(7)]
            d_params = [start_offset + i for i in range(7)]
            d2_params = [start_offset + i for i in range(6)]
            bl2, bp2, bi2 = local_search_families(v_params, h_params, d_params, d2_params, deadline - 15)
            if bl2 is not None and bi2 > best_I_global:
                best_I_global = bi2
                best_lines_global = bl2
                best_pts_global = bp2
    
    # Fallback
    if best_pts_global is None:
        lines = np.zeros((27,3), dtype=np.int64)
        for i in range(9):
            lines[i,0:3] = (1,0,-i)
        for i in range(9):
            lines[9+i,0:3] = (0,1,-i)
        for i in range(9):
            lines[18+i,0:3] = (1,-1,-(i-4))
        # Find best 54 points
        incidences, _ = compute_incidence_grid(list(range(9)), list(range(9)), list(range(9)), None)
        pts_list = greedy_select_points_from_incidence(incidences, 54)
        while len(pts_list) < 54:
            pts_list.append((len(pts_list), 0))
        best_lines_global = lines
        best_pts_global = np.array(pts_list[:54], dtype=np.int64)
        best_I_global = count_incidences(best_pts_global, best_lines_global)
    
    # Final validation and normalization
    # Ensure 54 unique points
    uniq, idx = np.unique(best_pts_global, axis=0, return_index=True)
    if len(uniq) < 54:
        added = 0
        existing = set(map(tuple, uniq))
        for dx in range(-20, 21):
            for dy in range(-20, 21):
                if added >= 54 - len(uniq):
                    break
                if (dx, dy) not in existing:
                    uniq = np.vstack([uniq, [[dx, dy]]])
                    existing.add((dx, dy))
                    added += 1
            if added >= 54 - len(uniq):
                break
        best_pts_global = uniq[:54]
    else:
        best_pts_global = uniq[idx]
    
    # Ensure 27 unique valid lines
    ul = []
    for l in best_lines_global:
        nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if nl and nl not in ul:
            ul.append(nl)
    while len(ul) < 27:
        ul.append(normalize_line(1, 0, len(ul)))
    best_lines_global = np.array(ul[:27], dtype=np.int64)
    
    return best_pts_global, best_lines_global
