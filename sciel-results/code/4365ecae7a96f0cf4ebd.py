import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 290 # Safety margin
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            dim = self.dim
            n_obj = self.n_obj
            
            # State
            X = []
            F = []
            evals = 0
            
            def evaluate(x):
                nonlocal evals
                if evals >= self.budget:
                    return None
                fx = problem(x)
                evals += 1
                return fx

            # 1. Initialization
            pop_size = 30
            X = np.random.uniform(lo, hi, (pop_size, dim))
            F = np.array([evaluate(x) for x in X])
            
            # Helper: Calculate HV contribution of a point given a set of other points
            # For 2 objectives, HV contribution of point i is:
            # (max(0, ref_f1 - f_i1)) * (max(0, ref_f2 - f_i2)) - (HV of points dominating i)
            # Simpler: Just compute HV of the set with and without i? Too slow.
            # For 2D, we can sort by f1 and compute.
            
            def hypervolume_2d(points, ref):
                # points: list of (f1, f2)
                # ref: (ref1, ref2)
                if not points:
                    return 0.0
                # Filter points strictly better than ref
                pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
                if not pts:
                    return 0.0
                # Sort by f1 ascending
                pts.sort(key=lambda x: x[0])
                hv = 0.0
                max_f2 = -np.inf
                for f1, f2 in pts:
                    if f2 > max_f2:
                        hv += (ref[0] - f1) * (f2 - max_f2) if max_f2 != -np.inf else (ref[0] - f1) * (ref[1] - f2)
                        # Correction: The area is a rectangle from f1 to ref1, and f2 to max_f2?
                        # No, standard 2D HV calculation:
                        # Sort by f1.
                        # Keep track of max f2 seen so far? No, min f2?
                        # We want the area under the curve (f2 vs f1) relative to ref.
                        # Since we minimize, the "front" is the lower-left envelope.
                        # Sort by f1 ascending.
                        # The contribution of a point is width * height.
                        # Width = next_f1 - current_f1 (or ref1 - current_f1 for last).
                        # Height = ref2 - current_f2.
                        # But only if current_f2 is lower than all previous f2s?
                        # Yes, we only care about non-dominated points.
                        # So, filter non-dominated first?
                        # Or just sort and track min f2?
                        # If we sort by f1 ascending, a point is non-dominated if its f2 is lower than all f2s of points with smaller f1.
                        
                        pass
                # Let's implement a proper 2D HV calculation
                return 0.0

            # Proper 2D HV
            def calc_hv_2d(pts, ref):
                # pts: np.array of shape (N, 2)
                # ref: np.array of shape (2,)
                if len(pts) == 0:
                    return 0.0
                # Filter
                mask = (pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])
                pts = pts[mask]
                if len(pts) == 0:
                    return 0.0
                # Sort by f1
                idx = np.argsort(pts[:, 0])
                pts = pts[idx]
                
                hv = 0.0
                prev_f2 = ref[1]
                # Iterate from left to right (small f1 to large f1)
                # The height of the rectangle for point i is prev_f2 - pts[i, 1]
                # The width is pts[i+1, 0] - pts[i, 0] (or ref[0] - pts[i, 0] for the last)
                # Wait, this logic assumes we are integrating the "skyline".
                # Let's trace:
                # Point A (1, 1), Point B (2, 0.5). Ref (3, 2).
                # Sorted: A, B.
                # i=0 (A): f1=1, f2=1.
                #   Next f1 is 2.
                #   Width = 2 - 1 = 1.
                #   Height = ref[1] - f2? No.
                #   The area contributed by A is the rectangle from x=1 to x=2, y=1 to y=2?
                #   No, the HV is the union of boxes.
                #   Box A: [1, 3] x [1, 2]. Area = 2*1 = 2.
                #   Box B: [2, 3] x [0.5, 2]. Area = 1*1.5 = 1.5.
                #   Union:
                #   x in [1, 2]: y in [1, 2] (from A). Area 1*1=1.
                #   x in [2, 3]: y in [0.5, 2] (from B, since B is lower). Area 1*1.5=1.5.
                #   Total = 2.5.
                
                # Algorithm:
                # Sort by f1 ascending.
                # Keep track of min_f2_so_far? No, we want the "lower envelope".
                # Actually, just iterate and if current f2 < min_f2_so_far, it's a new point on the front.
                # But we need the area.
                
                # Let's use the standard sweep line.
                # Sort by f1.
                # h = ref[1]
                # for i in range(len(pts)):
                #   if pts[i, 1] < h:
                #      # This point extends the front
                #      # Width is determined by the next point's f1 or ref[0]
                #      # But we need to subtract the area already covered?
                #      # Easier:
                #      # The contribution of this point is (next_f1 - current_f1) * (h - current_f2)
                #      # where next_f1 is the f1 of the next point with a LOWER f2?
                #      # No, if we just sort by f1, and take the points that are non-dominated,
                #      # the width for point i is f1[i+1] - f1[i] (if i is last, ref[0] - f1[i])
                #      # and the height is ref[1] - f2[i].
                #      # BUT this only works if the points are sorted by f1 and f2 is decreasing?
                #      # Non-dominated points sorted by f1 ascending MUST have f2 descending.
                #      # So yes.
                
                pass
                
            # Implementation of calc_hv_2d
            def calc_hv_2d(pts, ref):
                if len(pts) == 0: return 0.0
                mask = (pts[:, 0] < ref[0]) & (pts[:, 1] < ref[1])
                pts = pts[mask]
                if len(pts) == 0: return 0.0
                idx = np.argsort(pts[:, 0])
                pts = pts[idx]
                
                # Filter to non-dominated in this sorted list
                # A point is dominated if there is a previous point with lower f1 AND lower f2.
                # Since sorted by f1, we just need to check if f2 is lower than the min f2 so far.
                nd_pts = []
                min_f2 = np.inf
                for p in pts:
                    if p[1] < min_f2:
                        nd_pts.append(p)
                        min_f2 = p[1]
                
                if not nd_pts: return 0.0
                nd_pts = np.array(nd_pts)
                
                hv = 0.0
                for i in range(len(nd_pts)):
                    f1, f2 = nd_pts[i]
                    if i == len(nd_pts) - 1:
                        w = ref[0] - f1
                    else:
                        w = nd_pts[i+1, 0] - f1
                    h = ref[1] - f2
                    hv += w * h
                return hv

            # Main Loop
            # We will maintain a population X, F.
            # We want to maximize HV of the non-dominated set.
            
            # To speed up, we keep the population size small, e.g., 20.
            # We replace the point with the lowest HV contribution.
            
            # Calculate HV contribution of each point in the current non-dominated set.
            # Contribution of i = HV(Set) - HV(Set \ {i})
            
            def get_hv_contributions(X_nd, F_nd, ref):
                # X_nd, F_nd are lists of indices or arrays
                # Return array of contributions
                if len(F_nd) == 0: return np.array([])
                hv_full = calc_hv_2d(F_nd, ref)
                contribs = np.zeros(len(F_nd))
                for i in range(len(F_nd)):
                    # Remove i
                    mask = np.ones(len(F_nd), dtype=bool)
                    mask[i] = False
                    hv_without = calc_hv_2d(F_nd[mask], ref)
                    contribs[i] = hv_full - hv_without
                return contribs

            ref = np.array([11.0, 11.0])
            
            # Evolution
            pop_size = 20
            # Init done above, but let's reset to be clean
            X = np.random.uniform(lo, hi, (pop_size, dim))
            F = np.array([evaluate(x) for x in X])
            
            while evals < self.budget - 50: # Reserve 50 for polish
                if time.time() > deadline: break
                
                # Find non-dominated set
                # For 2D, easy:
                # A point i is non-dominated if no j exists such that F[j,0] <= F[i,0] and F[j,1] <= F[i,1] and (F[j,0] < F[i,0] or F[j,1] < F[i,1])
                
                nd_mask = np.ones(pop_size, dtype=bool)
                for i in range(pop_size):
                    for j in range(pop_size):
                        if i == j: continue
                        if F[j, 0] <= F[i, 0] and F[j, 1] <= F[i, 1] and (F[j, 0] < F[i, 0] or F[j, 1] < F[i, 1]):
                            nd_mask[i] = False
                            break
                            
                nd_indices = np.where(nd_mask)[0]
                
                if len(nd_indices) == 0:
                    # Fallback: pick random
                    parent_idx = np.random.randint(pop_size)
                else:
                    # Calculate HV contributions for ND points
                    F_nd = F[nd_indices]
                    contribs = get_hv_contributions(nd_indices, F_nd, ref)
                    # Select parent with highest contribution? Or random from top?
                    # To maintain diversity, maybe random from top half?
                    # Or just pick the one with lowest contribution to replace?
                    # We need a parent for crossover.
                    # Let's pick a parent from the ND set.
                    parent_idx = nd_indices[np.random.randint(len(nd_indices))]
                    
                # Generate Child
                # Simple mutation
                child = X[parent_idx].copy()
                # Adaptive mutation step
                progress = evals / self.budget
                step = 0.5 * (1 - progress) # Decrease step size
                child += step * (hi - lo) * np.random.randn(dim)
                child = np.clip(child, lo, hi)
                
                fc = evaluate(child)
                if fc is None: break
                
                # Replacement
                # Find the point in the population to replace.
                # Strategy: Replace the point with the lowest HV contribution in the ND set.
                # If the new point is dominated, we might still want to replace a dominated point if it has worse "quality"?
                # Simpler: Add child to population, then prune to pop_size.
                
                X_new = np.vstack([X, child.reshape(1, -1)])
                F_new = np.vstack([F, fc.reshape(1, -1)])
                
                # Prune to pop_size
                # Calculate HV of the new non-dominated set?
                # Or just calculate contribution of each point in the new set?
                # That's O(N^2) or O(N^3). N=21. Fast enough.
                
                # Find ND set in new population
                n_new = len(X_new)
                nd_mask_new = np.ones(n_new, dtype=bool)
                for i in range(n_new):
                    for j in range(n_new):
                        if i == j: continue
                        if F_new[j, 0] <= F_new[i, 0] and F_new[j, 1] <= F_new[i, 1] and (F_new[j, 0] < F_new[i, 0] or F_new[j, 1] < F_new[i, 1]):
                            nd_mask_new[i] = False
                            break
                            
                nd_indices_new = np.where(nd_mask_new)[0]
                
                if len(nd_indices_new) > pop_size:
                    # We have more ND points than slots. Remove the one with lowest HV contribution.
                    F_nd_new = F_new[nd_indices_new]
                    contribs_new = get_hv_contributions(nd_indices_new, F_nd_new, ref)
                    # Find index in nd_indices_new with min contribution
                    min_idx_local = np.argmin(contribs_new)
                    min_idx_global = nd_indices_new[min_idx_local]
                    
                    # Remove it
                    mask_keep = np.ones(n_new, dtype=bool)
                    mask_keep[min_idx_global] = False
                    X = X_new[mask_keep]
                    F = F_new[mask_keep]
                else:
                    # Keep all ND points. What about dominated points?
                    # We should replace dominated points if we have space?
                    # Actually, we want to maintain pop_size.
                    # If len(nd_indices_new) <= pop_size, we can keep all ND points.
                    # We need to fill the rest with best dominated points?
                    # Or just keep the current population if the child was dominated?
                    # Better: Replace the worst dominated point if child is dominated?
                    # Or just replace the point with lowest HV contribution in the *entire* population?
                    # If child is dominated, its HV contribution is 0.
                    # So it would be removed if we just pruned based on HV.
                    
                    # Let's just keep the non-dominated points and fill the rest with random?
                    # No, let's keep the previous population if the child didn't improve the ND set size or HV.
                    # Simplest robust strategy:
                    # If child is non-dominated, add it. If size > pop_size, remove lowest HV contributor.
                    # If child is dominated, replace the worst dominated point?
                    
                    # Let's implement:
                    # If child is ND:
                    if nd_mask_new[len(X)]: # Child is the last one
                        X = X_new
                        F = F_new
                        # Prune if > pop_size
                        if len(X) > pop_size:
                             # Re-calculate ND and remove lowest HV
                             # (Code above handles this if we just always vstack and prune)
                             pass 
                    else:
                        # Child is dominated.
                        # Replace the point in the current population that is dominated and has the worst "quality" (e.g., highest sum of objectives, or lowest HV contribution if we consider the whole set).
                        # Let's just replace the point with the highest sum of objectives among dominated points.
                        dom_mask = ~nd_mask # From previous population
                        if np.any(dom_mask):
                            worst_dom_idx = np.argmax(F[dom_mask].sum(axis=1))
                            # Map back to original index
                            dom_indices = np.where(dom_mask)[0]
                            replace_idx = dom_indices[worst_dom_idx]
                            X[replace_idx] = child
                            F[replace_idx] = fc
                        else:
                            # No dominated points? Then all are ND.
                            # This implies the new population would be ND + Child.
                            # If Child is dominated by some ND point, it's bad.
                            # Just ignore or replace worst ND?
                            pass

            # Polish Phase
            # Use remaining budget to do local search on ND points.
            # Get current ND set.
            # For each point, try small perturbations.
            
            while evals < self.budget:
                if time.time() > deadline: break
                
                # Get ND set
                n_cur = len(X)
                nd_mask = np.ones(n_cur, dtype=bool)
                for i in range(n_cur):
                    for j in range(n_cur):
                        if i == j: continue
                        if F[j, 0] <= F[i, 0] and F[j, 1] <= F[i, 1] and (F[j, 0] < F[i, 0] or F[j, 1] < F[i, 1]):
                            nd_mask[i] = False
                            break
                nd_indices = np.where(nd_mask)[0]
                
                if len(nd_indices) == 0:
                    break
                    
                # Pick a random ND point
                target_idx = nd_indices[np.random.randint(len(nd_indices))]
                
                # Generate neighbor
                neighbor = X[target_idx].copy()
                neighbor += 0.05 * (hi - lo) * np.random.randn(dim)
                neighbor = np.clip(neighbor, lo, hi)
                
                fn = evaluate(neighbor)
                if fn is None: break
                
                # Check if neighbor is better than target (non-dominated by others)
                # If neighbor dominates target, replace.
                # If neighbor is non-dominated by the rest of the ND set, replace target.
                
                # Simple check: Does neighbor dominate target?
                if fn[0] <= F[target_idx, 0] and fn[1] <= F[target_idx, 1] and (fn[0] < F[target_idx, 0] or fn[1] < F[target_idx, 1]):
                    X[target_idx] = neighbor
                    F[target_idx] = fn
                else:
                    # Check if neighbor is non-dominated by the *other* ND points
                    is_dominated = False
                    for j in nd_indices:
                        if j == target_idx: continue
                        if F[j, 0] <= fn[0] and F[j, 1] <= fn[1] and (F[j, 0] < fn[0] or F[j, 1] < fn[1]):
                            is_dominated = True
                            break
                    if not is_dominated:
                        # Replace target with neighbor
                        X[target_idx] = neighbor
                        F[target_idx] = fn

            return X
