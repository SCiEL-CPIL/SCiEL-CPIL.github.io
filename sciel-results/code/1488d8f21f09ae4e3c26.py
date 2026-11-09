"""
Heuristic: Delaunay Triangulation-based geometric landscape + high-quality reference tour.
Uses MDS to recover coordinates, Delaunay for local structure, and robust local search
(2-opt + Or-opt) for a strong reference tour to anchor the heuristic.
"""

import numpy as np
import time
from scipy.spatial import Delaunay

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([[1.0]])

    # --- 1. Recover Coordinates via MDS ---
    # The input is a distance matrix of points in the unit square.
    # Classical MDS can recover the relative geometry (up to rotation/reflection/translation).
    pts = None
    try:
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        eigvals, eigvecs = np.linalg.eigh(B)
        # Take top 2 eigenvectors
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # Scale by sqrt(eigenvalues)
        pts = eigvecs[:, :2] * np.sqrt(np.maximum(eigvals[:2], 0))
    except Exception:
        pts = None

    # --- 2. Local Search Utilities ---
    
    def tour_len(t):
        idx = np.asarray(t, dtype=int)
        return float(np.sum(D[idx, np.roll(idx, -1)]))

    def two_opt(t, max_passes=50):
        t = np.asarray(t, dtype=int).copy()
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for i in range(n - 1):
                if i == 0:
                    j_start, j_end = 2, n - 1
                else:
                    j_start, j_end = i + 2, n
                if j_start >= j_end:
                    continue
                js = np.arange(j_start, j_end, dtype=int)
                a = t[i]
                b = t[i + 1]
                cs = t[js]
                ds = t[(js + 1) % n]
                deltas = (D[a, cs] + D[b, ds]) - (D[a, b] + D[cs, ds])
                mask = deltas < -1e-12
                if np.any(mask):
                    # Pick the best improvement in this pass step for speed? 
                    # Or just the first one? First one is faster to find but might be less effective.
                    # Let's pick the best one among candidates to reduce iterations.
                    # However, vectorized argmax on mask is fast.
                    j = js[np.argmax(mask)]
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    improved = True
        return t

    def or_opt(t, max_passes=10):
        # Moves a segment of length 1, 2, or 3 to a new position
        t = np.asarray(t, dtype=int).copy()
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for seg_len in [1, 2, 3]:
                for i in range(n - seg_len):
                    if time.time() > deadline:
                        break
                    # Extract segment
                    seg = t[i:i+seg_len]
                    # Remove segment
                    # Indices: i to i+seg_len-1
                    # Predecessor: t[i-1] (circular)
                    # Successor: t[i+seg_len] (circular)
                    pred_idx = i - 1
                    succ_idx = i + seg_len
                    pred = t[pred_idx % n]
                    succ = t[succ_idx % n]
                    
                    # Current cost of removing segment
                    # Edges: (pred, seg[0]) and (seg[-1], succ)
                    cost_remove = D[pred, seg[0]] + D[seg[-1], succ]
                    
                    # Try inserting segment between j and j+1
                    # j cannot be i-1, i, ..., i+seg_len-1
                    for j in range(n):
                        if j in range(i, i+seg_len) or j == (i-1)%n:
                            continue
                        # Insert between t[j] and t[(j+1)%n]
                        ins_pred = t[j]
                        ins_succ = t[(j+1) % n]
                        
                        # New edges: (ins_pred, seg[0]) and (seg[-1], ins_succ)
                        # Old edges to replace: (ins_pred, ins_succ)
                        cost_insert = D[ins_pred, seg[0]] + D[seg[-1], ins_succ]
                        cost_old_edge = D[ins_pred, ins_succ]
                        
                        delta = (cost_insert - cost_old_edge) - cost_remove
                        if delta < -1e-12:
                            # Perform move
                            # Shift array
                            # t[i:i+seg_len] is removed
                            # t[j] gets segment inserted after it
                            # This requires careful indexing with numpy
                            new_t = np.empty(n, dtype=int)
                            # Copy parts
                            # Case 1: j < i
                            # Case 2: j > i+seg_len-1
                            # We need to handle circularity
                            
                            # Let's just reconstruct the list
                            lst = t.tolist()
                            seg_list = lst[i:i+seg_len]
                            del lst[i:i+seg_len]
                            # Now insert after index j in the NEW list?
                            # Wait, j was an index in the OLD list.
                            # If j < i, the index j is unchanged in the new list (shifted left? no, i is removed).
                            # If j > i, the index j shifts left by seg_len.
                            
                            # This is getting complicated with in-place numpy.
                            # Let's use lists for Or-opt for simplicity and correctness, 
                            # then convert back. n=50 is small.
                            
                            # Actually, let's restart or_opt with list-based approach for safety
                            return or_opt_list(t.tolist(), max_passes)
            if improved:
                break
        return t

    def or_opt_list(t_list, max_passes=10):
        t = t_list[:]
        n = len(t)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            if time.time() > deadline:
                break
            for seg_len in [1, 2, 3]:
                for i in range(n):
                    if time.time() > deadline:
                        break
                    # Segment starts at i
                    # Predecessor is t[i-1], Successor is t[i+seg_len]
                    pred_idx = i - 1
                    succ_idx = i + seg_len
                    pred = t[pred_idx % n]
                    succ = t[succ_idx % n]
                    
                    seg = t[i:i+seg_len]
                    
                    cost_remove = D[pred, seg[0]] + D[seg[-1], succ]
                    
                    # Try inserting after j
                    for j in range(n):
                        # j is the index AFTER which we insert
                        # Cannot insert back into the gap we just opened
                        # The gap is between pred (at i-1) and succ (at i+seg_len)
                        # In the list, the gap is effectively at index i-1 (after removal)
                        # We skip j such that the insertion point is the same gap
                        # If we insert after j, the new edge is (t[j], seg[0]) and (seg[-1], t[j+1])
                        # We must not insert into the gap we just removed.
                        # The gap is between t[i-1] and t[i+seg_len] (original indices).
                        # In the modified list (after removal), the gap is between the element at i-1 and the element that was at i+seg_len.
                        
                        # Simple check: if j is i-1 (mod n) or j is i+seg_len-1 (mod n, which is the last element of segment, but segment is removed)
                        # Actually, if we insert after j, we are breaking edge (j, j+1).
                        # If j == i-1, we are breaking (i-1, i). This is part of the removal gap? No, removal breaks (i-1, i) and (i+seg_len-1, i+seg_len).
                        # If we insert after i-1, we break (i-1, i) again? No, the list has changed.
                        
                        # Let's just skip if j corresponds to the immediate neighbors in the original tour
                        # j == i-1 (mod n) -> inserting back into the start of the gap
                        # j == i+seg_len-1 (mod n) -> inserting back into the end of the gap? 
                        # Actually, if we remove seg [i, i+seg_len-1], the neighbors are i-1 and i+seg_len.
                        # If we insert after j, we break edge (j, j+1).
                        # If j == i-1, we break (i-1, i). This edge was broken during removal. Re-inserting here just puts it back.
                        # If j == i+seg_len-1, this index is no longer in the list (it was removed).
                        # So we just need to make sure we don't insert into the gap we just created.
                        # The gap is between index i-1 and index i (in the new list, index i is the old i+seg_len).
                        # So we skip j == i-1.
                        
                        if j == (i - 1) % n:
                            continue
                        # Also skip if j is one of the indices in the segment (they don't exist in the new list, but j is an index in the OLD list)
                        # Wait, we are iterating j in range(n) based on OLD list indices.
                        # If j is in [i, i+seg_len-1], it's removed. We can't insert after a removed node.
                        if i <= j < i + seg_len:
                            continue
                            
                        ins_pred = t[j]
                        ins_succ = t[(j + 1) % n]
                        
                        cost_insert = D[ins_pred, seg[0]] + D[seg[-1], ins_succ]
                        cost_old_edge = D[ins_pred, ins_succ]
                        
                        delta = (cost_insert - cost_old_edge) - cost_remove
                        if delta < -1e-12:
                            # Perform move
                            # Remove seg
                            del t[i:i+seg_len]
                            # Insert seg after j
                            # Need to find the new index of j
                            # If j < i, index is j
                            # If j > i, index is j - seg_len
                            # (Since we removed seg_len items before j)
                            # Note: j is not in the removed segment.
                            if j < i:
                                insert_idx = j + 1
                            else:
                                insert_idx = j - seg_len + 1
                                
                            t[insert_idx:insert_idx] = seg
                            improved = True
                            break # Restart inner loop? Or just break to next seg_len?
                            # Breaking to next seg_len is fine, we found an improvement.
                            # But we should restart the i loop? 
                            # Usually local search restarts the scan.
                            # Let's break out of all loops and restart passes.
                            # To simplify, we just break out of j loop and set a flag.
                            # Actually, `improved = True` and breaking j loop is enough if we break i loop too.
                            break
                    if improved:
                        break
                if improved:
                    break
        return np.array(t, dtype=int)

    def nearest_neighbor(start):
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]
        cur = start
        for _ in range(n - 1):
            d = D[cur, :].copy()
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            tour.append(nxt)
            visited[nxt] = True
            cur = nxt
        return tour

    # --- 3. Generate Reference Tour ---
    # Multi-start NN + 2-opt + Or-opt
    best_tour = None
    best_len = float('inf')
    rng = np.random.default_rng(42)
    
    # Try a few random starts as well
    starts = list(range(min(10, n)))
    if n > 10:
        starts.extend(rng.choice(n, size=5, replace=False).tolist())

    for s in starts:
        if time.time() > deadline:
            break
        t = nearest_neighbor(s)
        t = two_opt(t)
        t = or_opt_list(t.tolist())
        L = tour_len(t)
        if L < best_len:
            best_len = L
            best_tour = list(t)
            
    if best_tour is None:
        best_tour = two_opt(nearest_neighbor(0)).tolist()

    # --- 4. Construct Heuristic Matrix ---
    
    # Base: Exp(-distance)
    # Tuning the decay factor. 
    # Typical distances in unit square are ~0.1 to 1.0.
    # exp(-5*0.1) = 0.6, exp(-5*1.0) = 0.006.
    # This provides good discrimination.
    base_heur = np.exp(-D * 5.0)
    np.fill_diagonal(base_heur, 0.0)

    # Delaunay Boost
    delaunay_mask = np.zeros((n, n), dtype=bool)
    if pts is not None:
        try:
            tri = Delaunay(pts)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        u, v = simplex[i], simplex[j]
                        delaunay_mask[u, v] = True
                        delaunay_mask[v, u] = True
        except Exception:
            pass
    
    # Boost Delaunay edges
    # These are likely edges in the TSP tour.
    boost = 10.0
    base_heur[delaunay_mask] *= boost

    # Boost Reference Tour edges
    # Strongly anchor the search to the known good solution.
    # If we set these to a very high value, the ants will likely follow this tour initially,
    # then pheromone decay and other edges might allow exploration if this tour isn't optimal.
    # However, if the reference tour IS optimal, this guarantees convergence to it.
    # If it's not, it biases the search towards it.
    # Given the quality of 2-opt+Or-opt on 50 nodes, it's likely very close to optimal.
    ref_boost = 1000.0
    for k in range(n):
        i = best_tour[k]
        j = best_tour[(k + 1) % n]
        base_heur[i, j] = max(base_heur[i, j], ref_boost)
        base_heur[j, i] = max(base_heur[j, i], ref_boost)

    # Normalize
    max_val = np.max(base_heur)
    if max_val > 0:
        base_heur /= max_val
    
    # Scale to a reasonable range
    # ACO: prob ~ tau * eta.
    # If eta is ~1, and tau starts at 1, prob is ~1 for best edges.
    # If we scale eta to 100, it dominates pheromone initially.
    # This is usually good for fast convergence.
    base_heur *= 100.0
    
    # Ensure validity
    base_heur[~np.isfinite(base_heur)] = 1e-9
    base_heur = np.maximum(base_heur, 1e-9)
    np.fill_diagonal(base_heur, 1e-9)

    return base_heur
