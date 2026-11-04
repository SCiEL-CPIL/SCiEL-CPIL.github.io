"""Matrix multiplication 4x5x6 rank minimization."""
import time
import sys
import numpy as np
from itertools import combinations

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24
    num_points = n * m * p  # 120
    
    # Schoolbook baseline
    best_rank = num_points
    best_U = np.zeros((nm, best_rank))
    best_V = np.zeros((mp, best_rank))
    best_W = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pt = i * mp + j * p + k
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1
    
    # Generate box candidates: all non-empty subsets with size up to 3 in each dim
    boxes = []
    for I in range(1, 1 << n):
        if bin(I).count('1') > 3:
            continue
        for J in range(1, 1 << m):
            if bin(J).count('1') > 3:
                continue
            for K in range(1, 1 << p):
                if bin(K).count('1') > 3:
                    continue
                boxes.append((I, J, K))
    
    num_boxes = len(boxes)
    
    # Precompute point index lists for each box
    box_points = []
    for I, J, K in boxes:
        pts = []
        for i in range(n):
            if I & (1 << i):
                for j in range(m):
                    if J & (1 << j):
                        for k in range(p):
                            if K & (1 << k):
                                pts.append(i * mp + j * p + k)
        box_points.append(pts)
    
    # Precompute point counts per box
    box_sizes = [len(pts) for pts in box_points]
    
    def build_solution(selected_boxes):
        """Build U, V, W matrices from selected box indices."""
        rank = len(selected_boxes)
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for c_idx, b_idx in enumerate(selected_boxes):
            I, J, K = boxes[b_idx]
            for i in range(n):
                if I & (1 << i):
                    for j in range(m):
                        if J & (1 << j):
                            U[i * m + j, c_idx] = 1
            for j in range(m):
                if J & (1 << j):
                    for k in range(p):
                        if K & (1 << k):
                            V[j * p + k, c_idx] = 1
            for k in range(p):
                if K & (1 << k):
                    for i in range(n):
                        if I & (1 << i):
                            W[k * n + i, c_idx] = 1
        return U, V, W
    
    def verify_solution(U, V, W):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    def greedy_with_lookahead(residual_init, box_order, max_steps=200):
        """Greedy with lookahead: pick box that maximizes coverage of residual points."""
        residual = residual_init.copy()
        selected = []
        
        for _ in range(max_steps):
            if time.time() > deadline - 15:
                break
            if not np.any(residual > 1e-9):
                break
            
            best_box = None
            best_score = -1
            best_count = 0
            
            for b_idx in box_order:
                pts = box_points[b_idx]
                count = 0
                for pt in pts:
                    if residual[pt] >= 1 - 1e-9:
                        count += 1
                if count == 0:
                    continue
                # Score: prefer boxes that cover more residual points,
                # and among ties, prefer larger boxes (more efficient)
                score = count * 1000 + box_sizes[b_idx]
                if score > best_score:
                    best_score = score
                    best_box = b_idx
                    best_count = count
            
            if best_box is None or best_count == 0:
                break
            
            selected.append(best_box)
            for pt in box_points[best_box]:
                residual[pt] -= 1
        
        return selected
    
    def local_search(selected_boxes, residual_init, max_iter=50):
        """Try to improve by swapping one box for another that reduces total rank."""
        current_selected = list(selected_boxes)
        current_rank = len(current_selected)
        residual = residual_init.copy()
        for b_idx in current_selected:
            for pt in box_points[b_idx]:
                residual[pt] -= 1
        
        improved = True
        while improved and len(current_selected) > 0 and time.time() < deadline - 20:
            improved = False
            for idx in range(len(current_selected)):
                if time.time() > deadline - 20:
                    break
                b_out = current_selected[idx]
                # Try removing this box and see if we can add fewer boxes to cover the gap
                temp_residual = residual.copy()
                for pt in box_points[b_out]:
                    temp_residual[pt] += 1
                
                # Try to cover the newly uncovered points with one box
                for b_in in range(num_boxes):
                    if b_in == b_out:
                        continue
                    pts_in = box_points[b_in]
                    # Check if b_in can cover all points that became uncovered
                    can_cover = True
                    for pt in pts_in:
                        if temp_residual[pt] < 1 - 1e-9:
                            can_cover = False
                            break
                    if not can_cover:
                        continue
                    # Check that adding b_in doesn't over-cover any point
                    ok = True
                    for pt in pts_in:
                        if residual[pt] + 1 > 1 + 1e-9:  # would exceed 1
                            ok = False
                            break
                    if not ok:
                        continue
                    # This swap is valid: replace b_out with b_in
                    new_selected = current_selected[:idx] + current_selected[idx+1:] + [b_in]
                    if len(new_selected) < current_rank:
                        current_selected = new_selected
                        current_rank = len(current_selected)
                        residual = residual.copy()
                        for pt in box_points[b_out]:
                            residual[pt] += 1
                        for pt in box_points[b_in]:
                            residual[pt] -= 1
                        improved = True
                        break
                if improved:
                    break
        
        return current_selected
    
    # Try multiple greedy orderings
    orderings = [
        sorted(range(num_boxes), key=lambda idx: -box_sizes[idx]),
        sorted(range(num_boxes), key=lambda idx: box_sizes[idx]),
        list(range(num_boxes)),
        sorted(range(num_boxes), key=lambda idx: -box_sizes[idx] / (1 + bin(boxes[idx][0]).count('1'))),
    ]
    
    best_selected = None
    best_sel_rank = best_rank
    
    for order_idx, box_order in enumerate(orderings):
        if time.time() > deadline - 30:
            break
        
        residual_init = np.ones(num_points)
        selected = greedy_with_lookahead(residual_init, box_order)
        
        if len(selected) < best_sel_rank:
            # Try local search to improve
            improved_selected = local_search(selected, np.ones(num_points))
            if len(improved_selected) < len(selected):
                selected = improved_selected
        
        if len(selected) < best_sel_rank:
            U_new, V_new, W_new = build_solution(selected)
            if verify_solution(U_new, V_new, W_new):
                best_selected = selected
                best_sel_rank = len(selected)
                best_rank = best_sel_rank
                best_U = U_new
                best_V = V_new
                best_W = W_new
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
