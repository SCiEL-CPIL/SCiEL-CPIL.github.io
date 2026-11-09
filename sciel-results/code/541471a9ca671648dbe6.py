import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    d = 7
    n = 2 ** d
    m = d * (n // 2)  # 448
    deadline = time.time() + 115.0

    # Generate edges
    edges = []
    edge_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_idx[(x, y)] = idx
                edge_idx[(y, x)] = idx

    # Generate faces and edge-face incidence
    faces = []
    edge_faces = [[] for _ in range(m)]

    for b in range(d):
        for c in range(b + 1, d):
            for a in range(n):
                if (a & (1 << b)) or (a & (1 << c)):
                    continue
                v0 = a
                v1 = a ^ (1 << b)
                v2 = a ^ (1 << c)
                v3 = a ^ (1 << b) ^ (1 << c)

                e0 = edge_idx[(min(v0, v1), max(v0, v1))]
                e1 = edge_idx[(min(v1, v3), max(v1, v3))]
                e2 = edge_idx[(min(v2, v3), max(v2, v3))]
                e3 = edge_idx[(min(v0, v2), max(v0, v2))]

                face_edges = [e0, e1, e2, e3]
                faces.append(face_edges)
                f_idx = len(faces) - 1
                for e in face_edges:
                    edge_faces[e].append(f_idx)

    F = len(faces)  # 672

    # Precompute face edge lists as tuples for fast access
    face_edge_lists = [tuple(f) for f in faces]

    # --- MILP Approach ---
    try:
        # Objective: minimize sum(x_e)
        c = np.ones(m)
        
        # Constraints: for each face, sum(x_e for e in face) >= 1
        A = np.zeros((F, m))
        for f in range(F):
            for e in face_edge_lists[f]:
                A[f, e] = 1
        
        cons = LinearConstraint(A, lb=np.ones(F), ub=np.inf)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(m)
        
        res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, 
                   options={'time_limit': 100})
        
        if res.status == 0 and res.x is not None:
            x = np.round(res.x).astype(int)
            
            valid = True
            for f in range(F):
                count = sum(x[e] for e in face_edge_lists[f])
                if count < 1:
                    valid = False
                    break
            
            if valid:
                removed = set(np.nonzero(x)[0].tolist())
                kept_edges = [edges[i] for i in range(m) if i not in removed]
                return kept_edges
    except Exception:
        pass

    # --- Heuristic Fallback with improved local search ---
    
    def greedy_hitting_set(rng):
        removed = set()
        unhit = set(range(F))
        
        while unhit:
            best_e = -1
            max_cov = -1
            for e in range(m):
                cov = 0
                for f in edge_faces[e]:
                    if f in unhit:
                        cov += 1
                if cov > max_cov:
                    max_cov = cov
                    best_e = e
            
            if best_e == -1:
                break
            
            removed.add(best_e)
            for f in edge_faces[best_e]:
                if f in unhit:
                    unhit.remove(f)
        
        return removed

    def local_search(removed, rng, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try deletions - remove redundant edges
            for e in list(removed):
                redundant = True
                for f in edge_faces[e]:
                    count = sum(1 for ee in face_edge_lists[f] if ee in removed)
                    if count == 1:
                        redundant = False
                        break
                if redundant:
                    removed.discard(e)
                    improved = True

            if time.time() >= time_limit:
                break

            # 2. Try 1-swaps
            if removed:
                for e in list(removed):
                    critical_faces = []
                    for f in edge_faces[e]:
                        count = sum(1 for ee in face_edge_lists[f] if ee in removed)
                        if count == 1:
                            critical_faces.append(f)

                    if not critical_faces:
                        continue

                    inter = set(face_edge_lists[critical_faces[0]])
                    for f in critical_faces[1:]:
                        inter.intersection_update(face_edge_lists[f])

                    candidates = inter - removed
                    if candidates:
                        e_new = rng.choice(list(candidates))
                        removed.discard(e)
                        removed.add(e_new)
                        improved = True
                        break

            if time.time() >= time_limit:
                break

            # 3. Try 2-remove, 1-add moves
            if len(removed) >= 2:
                removed_list = list(removed)
                found_improvement = False
                num_attempts = 50
                for _ in range(num_attempts):
                    if time.time() >= time_limit:
                        break
                    e1, e2 = rng.sample(removed_list, 2)
                    
                    # Check if removing e1, e2 leaves any face uncovered
                    ok = True
                    for f in edge_faces[e1] + edge_faces[e2]:
                        remaining = sum(1 for ee in face_edge_lists[f] if ee in removed and ee != e1 and ee != e2)
                        if remaining == 0:
                            ok = False
                            break
                    
                    if not ok:
                        continue
                    
                    # Find faces not hit by (removed - {e1, e2})
                    unhit_by_new = set()
                    for f in range(F):
                        if not any(ee in removed for ee in face_edge_lists[f] if ee != e1 and ee != e2):
                            unhit_by_new.add(f)
                    
                    if not unhit_by_new:
                        non_removed = [e for e in range(m) if e not in removed]
                        if non_removed:
                            e3 = rng.choice(non_removed)
                            removed.discard(e1)
                            removed.discard(e2)
                            removed.add(e3)
                            improved = True
                            found_improvement = True
                            break
                        continue

                    inter = set(face_edge_lists[unhit_by_new.pop()])
                    for f in unhit_by_new:
                        inter.intersection_update(face_edge_lists[f])
                    
                    candidates = inter - removed
                    if candidates:
                        e3 = rng.choice(list(candidates))
                        removed.discard(e1)
                        removed.discard(e2)
                        removed.add(e3)
                        improved = True
                        found_improvement = True
                        break

                if found_improvement:
                    break

            # 4. Try 3-remove, 1-add moves (more aggressive)
            if len(removed) >= 3 and time.time() < time_limit:
                removed_list = list(removed)
                found_improvement = False
                num_attempts = 20
                for _ in range(num_attempts):
                    if time.time() >= time_limit:
                        break
                    e1, e2, e3 = rng.sample(removed_list, 3)
                    
                    ok = True
                    for f in edge_faces[e1] + edge_faces[e2] + edge_faces[e3]:
                        remaining = sum(1 for ee in face_edge_lists[f] if ee in removed and ee not in (e1, e2, e3))
                        if remaining == 0:
                            ok = False
                            break
                    
                    if not ok:
                        continue
                    
                    unhit_by_new = set()
                    for f in range(F):
                        if not any(ee in removed for ee in face_edge_lists[f] if ee not in (e1, e2, e3)):
                            unhit_by_new.add(f)
                    
                    if not unhit_by_new:
                        non_removed = [e for e in range(m) if e not in removed]
                        if non_removed:
                            e4 = rng.choice(non_removed)
                            removed.discard(e1)
                            removed.discard(e2)
                            removed.discard(e3)
                            removed.add(e4)
                            improved = True
                            found_improvement = True
                            break
                        continue

                    inter = set(face_edge_lists[unhit_by_new.pop()])
                    for f in unhit_by_new:
                        inter.intersection_update(face_edge_lists[f])
                    
                    candidates = inter - removed
                    if candidates:
                        e4 = rng.choice(list(candidates))
                        removed.discard(e1)
                        removed.discard(e2)
                        removed.discard(e3)
                        removed.add(e4)
                        improved = True
                        found_improvement = True
                        break

                if found_improvement:
                    break

        return removed

    def solve_once(seed):
        rng = random.Random(seed)
        removed = greedy_hitting_set(rng)
        removed = local_search(removed, rng, deadline - 5.0)
        return removed

    best_removed = None
    best_size = m

    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        removed = solve_once(i)
        if len(removed) < best_size:
            best_size = len(removed)
            best_removed = removed.copy()

    if best_removed is not None:
        rng = random.Random(99999)
        best_removed = local_search(best_removed, rng, deadline - 2.0)
        best_size = len(best_removed)

    kept_edges = [edges[i] for i in range(m) if i not in best_removed]
    return kept_edges
